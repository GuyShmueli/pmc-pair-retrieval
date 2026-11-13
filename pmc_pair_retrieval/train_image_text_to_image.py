"""Fine-tune BiomedCLIP for retrieval with image-plus-text queries (configuration ``image_text_to_image``).

Run through the training command::

    python -m pmc_pair_retrieval.train --config configs/image_text_to_image.yaml \\
        --data-dir DATA_DIR --output-dir runs/image_text_to_image

Training. Both towers are fine-tuned, except the first ``frozen_vision_blocks`` of the 12
vision transformer blocks; the patch, position and class-token embeddings stay trainable.
Every training pair (two training images linked by a pair record) is used at most once per
epoch; only full batches are formed, so the fewer than ``batch_size`` pairs left at the end
of a pass are skipped (548 batches of 128 use 70,144 of the 70,262 pairs). A hard-negative
batch sampler groups ``batch_size`` pairs per batch: it starts from a random pair and adds
pairs that contain cached hard negatives (``hard_negatives``) of the images already in the
batch and, when none can be added, the next unused pair. Each image is seen as its three
jittered view crops (``train_views``), whose embeddings are averaged. Two triplet losses
with margin ``triplet_margin``, each scoring a positive against the mean of the
``triplet_topk_negatives`` hardest negatives, are combined as
``lambda_image_image * image-image + lambda_image_text * image-text``:

- image-image: every image of the batch is an anchor against all 2 x ``batch_size``
  images of the batch. Its direct training partners and other copies of itself are
  positives. Images linked to it only through the transitive closure of the pairs are
  neither positives nor negatives;
- image-text: the normalised mean embedding of the two images of a pair is an anchor
  against the distinct descriptions of the batch, with the pair's own description as the
  positive, in both directions (images to descriptions and descriptions to images).

The model runs under bf16 autocast. The optimiser is AdamW (weight decay on weight
matrices and embeddings, not on biases and norms) with gradient clipping and a linear
warmup followed by a cosine decay over ``epochs`` x steps per epoch.

Validation and selection. After every epoch the validation images are encoded from their
three deterministic view crops (``eval_views``), and every validation image is a query
against the validation gallery. Its text vector is the mean of the unit embeddings of
all its distinct descriptions, not renormalised. The query vector is the normalised mean
of the image embedding and this text vector, so the gallery is ranked by
``cos(image, g) + text . g``, the score of the reported evaluation. The query's direct
partners are the relevant images; the other members of its connected component are left
out of the ranking. The checkpoint of the epoch with the highest validation
``early_stopping.metric`` is kept, and a later epoch must be strictly better. Training
stops once the metric has not improved for ``patience`` epochs, but not before
``min_epochs``. At the end the selected weights are loaded back from their file and
validated again.

Outputs in the output directory:

- ``config.yaml``: the configuration as run, i.e. the configuration file as written
  (repository-relative data paths) with the ``--epochs`` override; the weight sidecar and
  ``summary.json`` record the same;
- ``history.jsonl``: one record per epoch (training losses, batch statistics,
  validation summary, minutes, ``batch_sequence_sha256``);
- ``best_model.safetensors`` and ``best_model.json``: full-model weights of the selected
  epoch (see ``pmc_pair_retrieval.model.save_weights``);
- ``summary.json``.

Every ``log_every`` steps of an epoch a progress line on stdout reports the running mean
loss of the epoch, the image-image and image-text losses of the current step and the
learning rate after the step.

Reproducibility. The seed and ``PYTHONHASHSEED`` fix the data order and augmentations:

1. The training pairs are listed by iterating Python sets of image ids, then shuffled
   with ``random.Random(seed)``. Set iteration order depends on string hashing, so the
   run needs ``PYTHONHASHSEED=hash_seed`` and Python 3.11. The set construction in
   ``prepare_data`` also determines this order. The log and ``summary.json`` record the
   sha256 of the training-pair order.
2. The batch sampler draws from its own ``random.Random(seed)`` across all epochs.
3. Building the model advances the global torch generator. Its next draw is the base
   seed of the persistent training workers (logged as ``loader base seed``). Worker
   ``w`` seeds Python's and torch's generators with ``base seed + w`` and draws the swap
   coin, the description and the view jitter of the pairs of every ``num_workers``-th
   batch, starting with batch ``w``. ``num_workers`` therefore changes the numbers.
4. The text-tower dropout draws from the CUDA generator.

GPU training is not guaranteed to be bitwise reproducible (cuDNN autotuning, bf16
kernels).

``--max-steps N`` is for smoke tests only. It also sets the cosine horizon to N steps
(identical to a full run for N <= ``warmup_steps``, where all steps are warmup steps).
The epoch in which step N is reached is validated, and the run ends.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from .dataset import (Edge, ImageDataset, article_id, connected_components, load_pairs, load_pairs_and_captions,
                      load_split, validate_split)
from .metrics import SUMMARY_METRICS, bootstrap_ci, retrieval_for_query
from .model import ModelCacheError, load_biomedclip, load_weights, save_weights
from .train import config_as_run, hash_seed_problems, validation_line
from .views import (N_VIEWS, MultiViewEvalTransform, MultiViewTrainTransform, enable_vision_block_checkpointing,
                    encode_image_views)

CONFIGURATION = "image_text_to_image"
CONFIG_KEYS = frozenset({
    "configuration", "seed", "hash_seed", "split", "hard_negatives",
    "image_size", "train_views", "eval_views", "crop_scale",
    "epochs", "batch_size", "lr", "weight_decay", "betas", "eps", "warmup_steps", "clip_grad_norm",
    "frozen_vision_blocks", "precision", "grad_checkpointing",
    "lambda_image_image", "lambda_image_text", "triplet_margin", "triplet_topk_negatives",
    "hard_negatives_per_anchor", "hard_negatives_sampled_per_image",
    "query_text", "eval_batch_size", "text_batch_size", "early_stopping",
    "num_workers", "log_every",
})
# Batches each training worker loads ahead. Together with num_workers it sets how many
# batches the extra iterator of every epoch discards (see start_epoch), so it changes the numbers.
PREFETCH_FACTOR = 2
# Validation summaries: 200 bootstrap resamples of the queries per metric, with a fresh
# generator seeded with the configuration seed for each metric.
BOOTSTRAP_RESAMPLES = 200


# --------------------------------------------------------------------------------------
# Configuration, seeding, backend flags
# --------------------------------------------------------------------------------------
def check_config(config: Mapping) -> None:
    """Refuse a configuration that this trainer would not run exactly as written."""
    missing, unknown = sorted(CONFIG_KEYS - config.keys()), sorted(config.keys() - CONFIG_KEYS)
    if missing or unknown:
        raise SystemExit(f"{CONFIGURATION} configuration: missing keys {missing}, unknown keys {unknown}")
    if config["configuration"] != CONFIGURATION:
        raise SystemExit(f"configuration must be {CONFIGURATION!r}, got {config['configuration']!r}")
    if config["image_size"] != 224:
        raise SystemExit("image_size must be 224, the input size of the pretrained BiomedCLIP vision tower")
    for key in ("train_views", "eval_views"):
        if config[key] != N_VIEWS:
            raise SystemExit(f"{key} must be {N_VIEWS}, the views per image of the published configurations; "
                             f"got {config[key]!r}")
    if config["precision"] != "bf16":
        raise SystemExit("precision must be 'bf16' (bf16 autocast), the only precision of this trainer")
    if config["grad_checkpointing"] not in ("none", "trainable", "all"):
        raise SystemExit("grad_checkpointing must be 'none', 'trainable' or 'all'")
    if config["query_text"] != "mean":
        raise SystemExit("query_text must be 'mean': the query text of an image is the mean of the unit embeddings "
                         "of its distinct descriptions")
    if set(config["early_stopping"]) != {"metric", "min_epochs", "patience"}:
        raise SystemExit("early_stopping needs exactly the keys metric, min_epochs and patience")
    if config["early_stopping"]["metric"] not in SUMMARY_METRICS:
        raise SystemExit(f"early_stopping.metric must be one of {list(SUMMARY_METRICS)}")
    if config["num_workers"] < 1:
        raise SystemExit("num_workers must be at least 1: the training loader uses persistent worker processes")
    for key in ("epochs", "batch_size", "log_every"):
        if config[key] < 1:
            raise SystemExit(f"{key} must be a positive integer")


def seed_everything(seed: int) -> None:
    """Seed Python's, numpy's and torch's (CPU and CUDA) global generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_backend_flags() -> None:
    """Enable TF32 for fp32 matmuls and convolutions, and cuDNN autotuning."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _autocast(device: str | torch.device):
    """bf16 autocast on CUDA; a no-op context on the CPU."""
    device_type = torch.device(device).type
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda")


# --------------------------------------------------------------------------------------
# Data: images, pair graph, split, training pairs, hard negatives
# --------------------------------------------------------------------------------------
def known_positive_edges(all_ids: set[str], direct_edges: set[Edge]) -> set[Edge]:
    """Pairs ``(u, v)``, ``u < v``, of images in the same connected component that are not direct pairs.

    These are the pairs added by the full transitive closure of the pair edges. Such
    images are never used as negatives and are left out of the validation ranking, but
    they are not positives either.
    """
    members = defaultdict(list)
    for iid, comp in connected_components(all_ids, direct_edges).items():
        members[comp].append(iid)
    inferred = set()
    for group in members.values():
        m = sorted(group)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                p = (m[i], m[j])
                if p not in direct_edges:
                    inferred.add(p)
    return inferred


def build_adjacency(all_ids: set[str], direct_edges: set[Edge], inferred_edges: set[Edge]
                    ) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Neighbour sets of every image for the direct and the inferred edges.

    The direct neighbour sets are filled in the iteration order of ``direct_edges``; their
    own iteration order fixes the order in which the training pairs are listed.
    """
    adj_d = {i: set() for i in all_ids}
    adj_i = {i: set() for i in all_ids}
    for (u, v) in direct_edges:
        if u in adj_d and v in adj_d:
            adj_d[u].add(v)
            adj_d[v].add(u)
    for (u, v) in inferred_edges:
        if u in adj_i and v in adj_i:
            adj_i[u].add(v)
            adj_i[v].add(u)
    return adj_d, adj_i


def sanitize_hard_negatives(hard_negs: Mapping[str, Sequence[str]], train_ids: Sequence[str],
                            adj_known_direct: Mapping[str, set[str]], adj_known_infer: Mapping[str, set[str]],
                            top_k: int, id2comp_train: Mapping[str, str]
                            ) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Keep the cached hard negatives that are safe negatives, in their cached order, at most ``top_k`` per image.

    A candidate ``v`` of anchor ``u`` is kept if it is a training image, not a repeat,
    not ``u``, not a direct or inferred known positive of ``u``, not in ``u``'s connected
    training component and not from ``u``'s source article. Returns the cleaned lists for
    every training image and removal counts.
    """
    train_set = set(train_ids)
    comp_to_ids: dict[str, set[str]] = defaultdict(set)
    for iid in train_set:
        comp_to_ids[id2comp_train.get(iid, iid)].add(iid)

    out: dict[str, list[str]] = {}
    stats = {
        "anchors": 0,
        "kept": 0,
        "removed_total": 0,
        "removed_not_train": 0,
        "removed_self": 0,
        "removed_direct": 0,
        "removed_inferred": 0,
        "removed_same_component": 0,
        "removed_same_article": 0,
        "removed_duplicate": 0,
    }

    cap = max(0, int(top_k))
    for u in sorted(train_set):
        stats["anchors"] += 1
        seen = set()
        clean: list[str] = []
        direct_ban = set(adj_known_direct.get(u, set()))
        infer_ban = set(adj_known_infer.get(u, set()))
        comp_ban = set()
        comp = id2comp_train.get(u, None)
        if comp is not None:
            comp_ban = set(comp_to_ids.get(comp, set()))

        for v0 in hard_negs.get(u, []) or []:
            v = str(v0)
            remove_reason = None
            if v in seen:
                remove_reason = "removed_duplicate"
            elif v not in train_set:
                remove_reason = "removed_not_train"
            elif v == u:
                remove_reason = "removed_self"
            elif v in direct_ban:
                remove_reason = "removed_direct"
            elif v in infer_ban:
                remove_reason = "removed_inferred"
            elif v in comp_ban:
                remove_reason = "removed_same_component"
            elif article_id(v) == article_id(u):
                remove_reason = "removed_same_article"

            if remove_reason is not None:
                stats[remove_reason] += 1
                stats["removed_total"] += 1
                continue

            seen.add(v)
            clean.append(v)
            if len(clean) >= cap:
                break

        out[u] = clean
        stats["kept"] += len(clean)

    return out, stats


def train_pairs_sha256(train_pairs: Sequence[tuple[str, str]]) -> str:
    """sha256 of the training-pair order (lines ``a,b`` joined by newlines)."""
    return hashlib.sha256("\n".join(f"{a},{b}" for a, b in train_pairs).encode()).hexdigest()


@dataclass
class TrainingData:
    """Images, pair graph, split, training pairs and hard negatives as the trainer uses them."""

    id2path: dict[str, str]                       # image id -> JPEG path
    id2texts: dict[str, list[str]]                # image id -> sorted distinct descriptions of its pairs
    split: dict[str, str]                         # image id -> train / val / test, in image-set order
    adj_direct: dict[str, set[str]]               # direct pair partners
    adj_inferred: dict[str, set[str]]             # other members of the connected component
    val_ids: list[str]                            # sorted validation ids: the queries and the gallery
    train_pairs: list[tuple[str, str]]            # training pairs in training order
    pair2texts_train: dict[Edge, list[str]]       # training pair -> its sorted distinct descriptions
    adj_train_direct: dict[str, set[str]]         # direct partners inside the training split
    adj_train_inferred: dict[str, set[str]]       # inferred known positives inside the training split
    hard_negatives: dict[str, list[str]]          # training image -> sanitised hard negatives
    counts: dict = field(default_factory=dict)    # sizes reported in the log


def prepare_data(config: Mapping, data_dir: str, allow_missing_images: bool = False) -> TrainingData:
    """Load the public data, the split and the hard negatives, and build the training structures.

    Sets are built from pairs in file order. Their iteration order determines the
    training-pair order and depends on this construction order. Every image of the pairs
    must be present unless ``allow_missing_images``. Without them, pairs with a missing
    image are dropped.
    """
    pairs, captions, id2path_all = load_pairs(data_dir)
    all_ids, direct_edges, id2texts, pair2texts = load_pairs_and_captions(pairs, captions)

    present = set(id2path_all.keys()) & set(all_ids)
    missing = set(all_ids) - present
    if missing:
        if not allow_missing_images:
            raise SystemExit(f"{len(missing)} of {len(all_ids)} images of image_pairs.json are missing from the data "
                             f"directory (e.g. {min(missing)}); reproducing the published model needs all of them. "
                             "Pass --allow-missing-images to train without them.")
        print(f"[data] WARNING: training without {len(missing)} missing images and their pairs; results will differ "
              "from the published ones.", flush=True)
    id2path = {i: id2path_all[i] for i in present}
    all_ids = set(present)
    id2texts = {i: id2texts.get(i, []) for i in all_ids}
    direct_edges = {(u, v) for (u, v) in direct_edges if (u in all_ids and v in all_ids)}
    pair2texts = {(u, v): caps for (u, v), caps in pair2texts.items() if (u in all_ids and v in all_ids)}

    inferred_edges = known_positive_edges(all_ids, direct_edges)
    adj_direct, adj_inferred = build_adjacency(all_ids, direct_edges, inferred_edges)
    print(f"[data] direct edges: {len(direct_edges)}", flush=True)
    print(f"[data] known-positive edges of the transitive closure (not direct): {len(inferred_edges)}", flush=True)

    split_loaded = load_split(config["split"])
    validate_split(split_loaded, all_ids, direct_edges)
    split = {i: split_loaded[i] for i in all_ids}
    ids_train = {i for i in all_ids if split[i] == "train"}
    ids_val = {i for i in all_ids if split[i] == "val"}
    ids_test = {i for i in all_ids if split[i] == "test"}
    print(f"[data] split sizes: train={len(ids_train)} val={len(ids_val)} test={len(ids_test)}", flush=True)

    direct_edges_train = {(u, v) for (u, v) in direct_edges if (u in ids_train and v in ids_train)}
    id2comp_train = connected_components(set(ids_train), direct_edges_train)
    pair2texts_train = {(u, v): caps for (u, v), caps in pair2texts.items() if (u in ids_train and v in ids_train)}

    train_pairs: list[tuple[str, str]] = []
    for u in ids_train:
        for v in adj_direct[u] & ids_train:
            if u < v:
                train_pairs.append((u, v))
    random.Random(config["seed"]).shuffle(train_pairs)
    if not train_pairs:
        raise SystemExit("No training pairs found; check the split file.")
    order_sha256 = train_pairs_sha256(train_pairs)
    print(f"[data] training pairs: {len(train_pairs)} (order sha256 {order_sha256})", flush=True)

    # Training positives are the direct partners; the same sets mark the direct known positives.
    adj_train_direct = {i: (adj_direct[i] & ids_train) for i in ids_train}
    adj_train_inferred = {i: (adj_inferred[i] & ids_train) for i in ids_train}

    with open(config["hard_negatives"], "r", encoding="utf-8") as f:
        cache = json.load(f)
    if not isinstance(cache, dict):
        raise SystemExit("The hard-negative file must be a JSON object mapping image ids to lists of image ids.")
    hard_negs = {str(k): [str(x) for x in (v or [])] for k, v in cache.items()}
    hard_negs, hn_stats = sanitize_hard_negatives(hard_negs, sorted(list(ids_train)), adj_train_direct,
                                                  adj_train_inferred, top_k=config["hard_negatives_per_anchor"],
                                                  id2comp_train=id2comp_train)
    print(f"[data] hard negatives: {json.dumps(hn_stats, sort_keys=True)}", flush=True)

    counts = {"images": len(all_ids), "direct_edges": len(direct_edges), "inferred_edges": len(inferred_edges),
              "split_sizes": {"train": len(ids_train), "val": len(ids_val), "test": len(ids_test)},
              "train_pairs": len(train_pairs), "train_pairs_sha256": order_sha256, "hard_negatives": hn_stats}
    return TrainingData(id2path=id2path, id2texts=id2texts, split=split, adj_direct=adj_direct,
                        adj_inferred=adj_inferred, val_ids=sorted(ids_val), train_pairs=train_pairs,
                        pair2texts_train=pair2texts_train, adj_train_direct=adj_train_direct,
                        adj_train_inferred=adj_train_inferred, hard_negatives=hard_negs, counts=counts)


# --------------------------------------------------------------------------------------
# Training data loading
# --------------------------------------------------------------------------------------
class PairDataset(Dataset):
    """Training pairs as ``(id_a, id_b, views_a, views_b, description)``.

    Each item draws, in this order and from the generators of the DataLoader worker: a coin
    flip that swaps the two images, the description (``random.choice`` over the pair's
    sorted distinct descriptions, drawn even when there is only one), the training views
    of the first image, then those of the second image.
    """

    def __init__(self, pairs: Sequence[tuple[str, str]], id2path: Mapping[str, str], transform,
                 pair2texts: Mapping[Edge, Sequence[str]]):
        self.pairs = pairs
        self.id2path = id2path
        self.transform = transform
        self.pair2texts = pair2texts

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        a, b = self.pairs[idx]
        if random.random() < 0.5:
            a, b = b, a
        ia = Image.open(self.id2path[a]).convert("RGB")
        ib = Image.open(self.id2path[b]).convert("RGB")
        u, v = (a, b) if a < b else (b, a)
        t = random.choice(self.pair2texts[(u, v)])
        return a, b, self.transform(ia), self.transform(ib), t


class HardNegativeBatchSampler(Sampler[list[int]]):
    """Batches of pair indices grouped around cached hard negatives.

    Each pass shuffles all pairs with the sampler's own ``random.Random(seed)``, which
    persists across passes. A batch starts from the next unused pair in shuffled order.
    Images are visited breadth-first: the two images of the starting pair, then the images
    of the pairs added in each round. A visited image draws up to ``sample_per_img`` of
    its hard negatives and tries them in turn. For each one it makes up to 8 random draws
    (with replacement, at most as many as there are such pairs) from the pairs containing
    it, and the first unused pair joins the batch and queues its images. A visited image
    therefore adds at most one pair, and after the first addition of a round, each later
    image of that round tries only its first drawn negative. When a round adds no pair,
    the next unused pair in shuffled order is added. Only full batches are yielded when
    ``drop_last``.
    """

    def __init__(self, pairs: Sequence[tuple[str, str]], hard_negs: Mapping[str, Sequence[str]], batch_size: int,
                 seed: int, sample_per_img: int, drop_last: bool = True):
        self.pairs = pairs
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)
        self.hard_negs = hard_negs
        self.sample_per_img = max(1, int(sample_per_img))
        self.drop_last = bool(drop_last)

        self.img_to_pair_idxs = defaultdict(list)
        for idx, (a, b) in enumerate(pairs):
            self.img_to_pair_idxs[a].append(idx)
            self.img_to_pair_idxs[b].append(idx)

        self.num_pairs = len(pairs)
        self._len = ((self.num_pairs // self.batch_size) if self.drop_last
                     else ((self.num_pairs + self.batch_size - 1) // self.batch_size))

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[list[int]]:
        all_idxs = list(range(self.num_pairs))
        self.rng.shuffle(all_idxs)
        available = set(all_idxs)
        idx_ptr = 0

        def next_available() -> int | None:
            nonlocal idx_ptr
            while idx_ptr < self.num_pairs and all_idxs[idx_ptr] not in available:
                idx_ptr += 1
            if idx_ptr >= self.num_pairs:
                return None
            idx = all_idxs[idx_ptr]
            idx_ptr += 1
            return idx

        while len(available) >= self.batch_size or (not self.drop_last and len(available) > 0):
            first = next_available()
            if first is None:
                break
            if first not in available:
                continue

            batch: list[int] = [first]
            available.remove(first)
            a0, b0 = self.pairs[first]
            search_queue = [a0, b0]

            while len(batch) < self.batch_size:
                added = False
                next_queue: list[str] = []

                for img_id in search_queue:
                    if len(batch) >= self.batch_size:
                        break
                    hns = self.hard_negs.get(img_id, [])
                    if not hns:
                        continue

                    for hn in self.rng.sample(hns, min(self.sample_per_img, len(hns))):
                        if len(batch) >= self.batch_size:
                            break
                        cands = self.img_to_pair_idxs.get(hn, [])
                        if not cands:
                            continue

                        # A few random tries to find an unused pair that contains the hard negative.
                        tries = min(8, len(cands))
                        for _ in range(tries):
                            cand_idx = self.rng.choice(cands)
                            if cand_idx in available:
                                batch.append(cand_idx)
                                available.remove(cand_idx)
                                u, v = self.pairs[cand_idx]
                                next_queue.extend([u, v])
                                added = True
                                break
                        if added:
                            break

                if not added:
                    nxt = next_available()
                    if nxt is None:
                        break
                    if nxt in available:
                        batch.append(nxt)
                        available.remove(nxt)
                        u, v = self.pairs[nxt]
                        next_queue.extend([u, v])

                search_queue = next_queue

                if not search_queue and len(batch) < self.batch_size:
                    nxt = next_available()
                    if nxt is None:
                        break
                    if nxt in available:
                        batch.append(nxt)
                        available.remove(nxt)
                        u, v = self.pairs[nxt]
                        search_queue = [u, v]

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                yield batch


def make_train_loader(dataset: Dataset, sampler: Sampler, num_workers: int) -> DataLoader:
    """Training DataLoader with persistent workers.

    The workers are seeded once, when the first iterator is created, and keep their
    generators across epochs. Batch ``k`` of every pass goes to worker ``k mod num_workers``.
    """
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=True, prefetch_factor=PREFETCH_FACTOR)


def start_epoch(loader: DataLoader) -> Iterator:
    """Start the next training epoch with two DataLoader iterator calls.

    With persistent workers, the first ``iter(loader)`` call starts a pass of the batch
    sampler and queues ``PREFETCH_FACTOR x num_workers`` batches (8). The second call
    discards them and starts the pass that is trained on. Every epoch therefore advances
    the sampler's generator by one shuffle and 8 batches before the trained batches, and
    each worker loads its 2
    discarded batches, drawing their augmentations. These initial draws are part of the
    batch order and augmentation streams.
    """
    iter(loader)
    return iter(loader)


def loader_base_seed(loader: DataLoader) -> int | None:
    """Base seed of the loader's current worker processes (worker ``w`` is seeded with base seed + ``w``)."""
    return getattr(getattr(loader, "_iterator", None), "_base_seed", None)


def batch_sequence_line(ida: Sequence[str], idb: Sequence[str]) -> bytes:
    """One line of the per-epoch ``batch_sequence_sha256``: the batch's pairs as ``min,max`` joined by ``|``."""
    return ("|".join(f"{min(a, b)},{max(a, b)}" for a, b in zip(ida, idb)) + "\n").encode("utf-8")


# --------------------------------------------------------------------------------------
# Model and optimiser
# --------------------------------------------------------------------------------------
def freeze_vision_blocks(model, n_frozen: int) -> int:
    """Freeze the first ``n_frozen`` vision transformer blocks; returns the number of blocks."""
    blocks = model.visual.trunk.blocks
    if not 0 <= n_frozen <= len(blocks):
        raise SystemExit(f"frozen_vision_blocks must be between 0 and {len(blocks)}")
    for blk in list(blocks)[:n_frozen]:
        for p in blk.parameters():
            p.requires_grad = False
    return len(blocks)


def set_frozen_blocks_eval(model, n_frozen: int) -> None:
    """Put the frozen vision blocks in eval mode (called after every ``model.train()``)."""
    for blk in list(model.visual.trunk.blocks)[:n_frozen]:
        blk.eval()


def parameter_groups(model) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Trainable parameters with and without weight decay.

    No decay for parameters with at most one dimension, biases, and names containing
    ``bn``, ``ln`` or ``norm``; the class token, position embedding and patch-embedding
    weight are decayed.
    """
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        ln = n.lower()
        is_nd = (p.ndim <= 1) or n.endswith(".bias") or ("bn" in ln) or ("ln" in ln) or ("norm" in ln)
        (no_decay if is_nd else decay).append(p)
    return decay, no_decay


def build_optimizer(model, config: Mapping, total_steps: int):
    """AdamW over the decay and no-decay groups and the warmup-cosine ``LambdaLR`` schedule.

    The schedule multiplies the peak learning rate by ``(step + 1) / warmup_steps`` during
    warmup and by ``0.5 * (1 + cos(pi * progress))`` afterwards, with ``progress`` going
    from 0 to 1 at ``total_steps``.
    """
    decay, no_decay = parameter_groups(model)
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"]), "lr": float(config["lr"])},
            {"params": no_decay, "weight_decay": 0.0, "lr": float(config["lr"])},
        ],
        betas=(float(config["betas"][0]), float(config["betas"][1])),
        eps=float(config["eps"]),
    )
    warmup_steps = int(config["warmup_steps"])

    def warmcos(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=[warmcos, warmcos])
    groups = {"decay": {"tensors": len(decay), "parameters": sum(p.numel() for p in decay)},
              "no_decay": {"tensors": len(no_decay), "parameters": sum(p.numel() for p in no_decay)}}
    return opt, sched, groups


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------
def build_pos_mask_weights(ida: Sequence[str], idb: Sequence[str], adj_train_direct: Mapping[str, set[str]],
                           device) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive mask and weights ``[len(ida), len(idb)]``: direct training partners and the same image id."""
    N = len(ida)
    M = len(idb)
    pos_mask = torch.zeros((N, M), dtype=torch.bool, device=device)
    pos_weight = torch.zeros((N, M), dtype=torch.float, device=device)

    idb_to_idxs: dict[str, list[int]] = defaultdict(list)
    for j, iid in enumerate(idb):
        idb_to_idxs[iid].append(j)

    for i, a_id in enumerate(ida):
        dset = adj_train_direct.get(a_id, set())
        if dset:
            idxs = [j for x in dset for j in idb_to_idxs.get(x, [])]
            if idxs:
                pos_mask[i, idxs] = True
                pos_weight[i, idxs] = 1.0

        # The same image elsewhere in the batch is never a negative.
        same = idb_to_idxs.get(a_id, [])
        for j in same:
            pos_mask[i, j] = True
            if pos_weight[i, j] == 0:
                pos_weight[i, j] = 1.0

    return pos_mask, pos_weight


def build_known_positive_mask(ida: Sequence[str], idb: Sequence[str], adj_known_direct: Mapping[str, set[str]],
                              adj_known_infer: Mapping[str, set[str]], device) -> torch.Tensor:
    """Mask of pairs that must never be negatives: direct or inferred known positives and the same image id."""
    N = len(ida)
    M = len(idb)
    known = torch.zeros((N, M), dtype=torch.bool, device=device)

    idb_to_idxs: dict[str, list[int]] = defaultdict(list)
    for j, iid in enumerate(idb):
        idb_to_idxs[iid].append(j)

    for i, a_id in enumerate(ida):
        kset = set(adj_known_direct.get(a_id, set())) | set(adj_known_infer.get(a_id, set()))
        kset.add(a_id)
        if not kset:
            continue
        idxs = [j for x in kset for j in idb_to_idxs.get(x, [])]
        if idxs:
            known[i, idxs] = True
    return known


def build_candidate_mask(ida: Sequence[str], idb: Sequence[str], train_pos_mask: torch.Tensor,
                         adj_known_direct: Mapping[str, set[str]], adj_known_infer: Mapping[str, set[str]],
                         device) -> torch.Tensor:
    """Candidate mask for the triplet loss: known positives that are not training positives are excluded."""
    candidate_mask = torch.ones_like(train_pos_mask, dtype=torch.bool, device=device)
    known_mask = build_known_positive_mask(ida, idb, adj_known_direct, adj_known_infer, device)
    candidate_mask[known_mask & (~train_pos_mask)] = False
    return candidate_mask


def rowwise_multi_positive_triplet_loss(sims: torch.Tensor, pos_mask: torch.Tensor, pos_weight: torch.Tensor,
                                        margin: float, candidate_mask: torch.Tensor | None,
                                        topk_neg: int) -> torch.Tensor:
    """Row-wise triplet hinge ``relu(margin - s(anchor, positive) + mean of the top-k negative scores)``.

    Positives are the candidate columns in ``pos_mask``, negatives the other candidate
    columns. Each valid row (at least one positive and one negative) averages the hinge
    over its positives, weighted by ``pos_weight``; the loss is the mean over valid rows.
    """
    if candidate_mask is None:
        candidate_mask = torch.ones_like(pos_mask, dtype=torch.bool, device=sims.device)
    else:
        candidate_mask = candidate_mask.to(device=sims.device, dtype=torch.bool)

    pos = pos_mask & candidate_mask
    neg = (~pos_mask) & candidate_mask

    valid_rows = pos.any(dim=1) & neg.any(dim=1)
    if not valid_rows.any():
        return torch.tensor(0.0, device=sims.device)

    sims_v = sims[valid_rows]
    pos_v = pos[valid_rows]
    neg_v = neg[valid_rows]
    w_v = torch.where(
        pos_v,
        pos_weight[valid_rows].clamp_min(1e-6),
        torch.zeros_like(pos_weight[valid_rows]),
    )

    neg_sims = sims_v.masked_fill(~neg_v, float("-inf"))
    k = max(1, min(int(topk_neg), neg_sims.shape[1]))
    topk_vals = torch.topk(neg_sims, k=k, dim=1).values
    finite = torch.isfinite(topk_vals)
    neg_score = topk_vals.masked_fill(~finite, 0.0).sum(dim=1) / finite.sum(dim=1).clamp_min(1)

    triplet = torch.relu(float(margin) - sims_v + neg_score.unsqueeze(1))
    triplet = torch.where(pos_v, triplet * w_v, torch.zeros_like(triplet))

    row_den = w_v.sum(dim=1).clamp_min(1e-12)
    row_loss = triplet.sum(dim=1) / row_den
    return row_loss.mean()


def multi_positive_triplet_loss(sims: torch.Tensor, pos_mask: torch.Tensor, pos_weight: torch.Tensor,
                                margin: float, candidate_mask: torch.Tensor | None, topk_neg: int) -> torch.Tensor:
    """Symmetric triplet loss: the mean of the row-wise loss on ``sims`` and on its transpose."""
    row_loss = rowwise_multi_positive_triplet_loss(
        sims=sims,
        pos_mask=pos_mask,
        pos_weight=pos_weight,
        margin=margin,
        candidate_mask=candidate_mask,
        topk_neg=topk_neg,
    )

    col_loss = rowwise_multi_positive_triplet_loss(
        sims=sims.t(),
        pos_mask=pos_mask.t(),
        pos_weight=pos_weight.t(),
        margin=margin,
        candidate_mask=(candidate_mask.t() if candidate_mask is not None else None),
        topk_neg=topk_neg,
    )
    return 0.5 * (row_loss + col_loss)


def image_image_loss(ea: torch.Tensor, eb: torch.Tensor, ida: list[str], idb: list[str], data: TrainingData,
                     config: Mapping, device) -> tuple[torch.Tensor, tuple[int, int]]:
    """Image-image triplet loss of a batch, called inside the autocast block of the training step.

    Every image (``ea`` and ``eb`` rows) is an anchor against all ``2N`` images of the
    batch, so that the hard negatives the sampler put into the batch take part. Returns the
    loss and, for the batch statistics, the total and the minimum number of valid
    negatives per anchor.
    """
    N = len(ida)
    e_all = torch.cat([ea, eb], dim=0)                 # [2N, D]
    id_all = ida + idb                                 # len 2N

    pos_mask_a, pos_w_a = build_pos_mask_weights(ida, id_all, data.adj_train_direct, device)
    pos_mask_b, pos_w_b = build_pos_mask_weights(idb, id_all, data.adj_train_direct, device)

    sims_a = ea.float() @ e_all.float().t()  # [N, 2N]
    sims_b = eb.float() @ e_all.float().t()  # [N, 2N]

    idx = torch.arange(N, device=device)
    cand_mask_a = build_candidate_mask(ida, id_all, pos_mask_a, data.adj_train_direct, data.adj_train_inferred, device)
    cand_mask_b = build_candidate_mask(idb, id_all, pos_mask_b, data.adj_train_direct, data.adj_train_inferred, device)

    # An anchor's own column (ea[i] in e_all[i], eb[i] in e_all[N + i]) is not a candidate.
    cand_mask_a[idx, idx] = False
    cand_mask_b[idx, N + idx] = False

    neg_a = ((~pos_mask_a) & cand_mask_a).sum(dim=1)
    neg_b = ((~pos_mask_b) & cand_mask_b).sum(dim=1)
    negatives = (int(neg_a.sum().item()) + int(neg_b.sum().item()),
                 min(int(neg_a.min().item()), int(neg_b.min().item())))

    loss_a = rowwise_multi_positive_triplet_loss(
        sims=sims_a,
        pos_mask=pos_mask_a,
        pos_weight=pos_w_a,
        margin=float(config["triplet_margin"]),
        candidate_mask=cand_mask_a,
        topk_neg=int(config["triplet_topk_negatives"]),
    )
    loss_b = rowwise_multi_positive_triplet_loss(
        sims=sims_b,
        pos_mask=pos_mask_b,
        pos_weight=pos_w_b,
        margin=float(config["triplet_margin"]),
        candidate_mask=cand_mask_b,
        topk_neg=int(config["triplet_topk_negatives"]),
    )
    return 0.5 * (loss_a + loss_b), negatives


def image_text_loss(model, tokenizer, ea: torch.Tensor, eb: torch.Tensor, txt_list: Sequence, config: Mapping,
                    device) -> torch.Tensor:
    """Pair-description triplet loss of a batch, called inside the autocast block of the training step.

    The anchor of a pair is the normalised mean of its two image embeddings. The batch's
    descriptions are encoded once each (first-appearance order) by the text tower in
    training mode, and each pair's own description is its positive, in both directions.
    """
    txt_list = [t if isinstance(t, str) else str(t) for t in txt_list]
    keep = [i for i, t in enumerate(txt_list) if t and t.strip()]
    if len(keep) < 2:
        return torch.zeros((), device=device)

    pair_img = 0.5 * (ea[keep].float() + eb[keep].float())
    pair_img = F.normalize(pair_img, dim=1)

    txt_kept = [txt_list[i] for i in keep]
    uniq_texts = []
    t2u = {}
    map_u = []
    for t in txt_kept:
        u = t2u.get(t)
        if u is None:
            u = len(uniq_texts)
            uniq_texts.append(t)
            t2u[t] = u
        map_u.append(u)

    tok = tokenizer(uniq_texts).to(device, non_blocking=True)
    tfeat = model.encode_text(tok)
    tfeat = F.normalize(tfeat.float(), dim=1)
    sims = pair_img @ tfeat.t()  # [M, U]

    M, U = sims.shape
    pos_mask = torch.zeros((M, U), dtype=torch.bool, device=device)
    pos_w = torch.zeros((M, U), dtype=torch.float, device=device)

    rows = torch.arange(M, device=device)
    cols = torch.tensor(map_u, device=device, dtype=torch.long)
    pos_mask[rows, cols] = True
    pos_w[rows, cols] = 1.0

    return multi_positive_triplet_loss(
        sims=sims,
        pos_mask=pos_mask,
        pos_weight=pos_w,
        margin=float(config["triplet_margin"]),
        candidate_mask=None,
        topk_neg=int(config["triplet_topk_negatives"]),
    )


# --------------------------------------------------------------------------------------
# Validation (checkpoint selection)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def embed_gallery(model, gallery_ids: Sequence[str], id2path: Mapping[str, str], config: Mapping, device
                  ) -> tuple[list[str], torch.Tensor]:
    """Unit image embeddings ``[N, D]`` (CPU, fp32) of the gallery from the three deterministic views."""
    model.eval()
    transform = MultiViewEvalTransform(size=config["image_size"])
    loader = DataLoader(ImageDataset(gallery_ids, id2path, transform), batch_size=int(config["eval_batch_size"]),
                        shuffle=False, num_workers=int(config["num_workers"]), pin_memory=torch.cuda.is_available())
    ids: list[str] = []
    embs: list[torch.Tensor] = []
    for iids, imgs in loader:
        imgs = imgs.to(device, non_blocking=True)
        with _autocast(device):
            z = encode_image_views(model, imgs)
        z = F.normalize(z.float(), dim=1)
        ids += list(iids)
        embs.append(z.cpu())
    Z = torch.cat(embs, dim=0).float()
    Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)
    return ids, Z


@torch.no_grad()
def embed_query_texts(model, tokenizer, query_ids: Sequence[str], id_to_gidx: Mapping[str, int],
                      id2texts: Mapping[str, Sequence[str]], batch_size: int, device
                      ) -> tuple[list[str], torch.Tensor, dict[str, int]]:
    """Text vector of every query: the mean of the unit embeddings of its distinct descriptions (not renormalised).

    The distinct descriptions of all queries are encoded once each, in order of first
    appearance over the queries' sorted description lists, in batches of ``batch_size``
    under bf16 autocast; the embeddings are normalised on the device and again on the CPU,
    where the per-query means are taken. Returns the query ids that have a description,
    their text vectors ``[Q, D]`` and the row of each query id.
    """
    cap_ids: list[str] = []
    cap_lists: list[list[str]] = []
    for q in query_ids:
        if q not in id_to_gidx:
            continue
        texts = id2texts.get(q, [])
        if not texts:
            continue
        cap_ids.append(q)
        cap_lists.append(list(texts))
    if not cap_ids:
        return [], torch.empty((0, 0), dtype=torch.float32), {}

    uniq_texts: list[str] = []
    t_to_u: dict[str, int] = {}
    for t in [t for tl in cap_lists for t in tl]:
        u = t_to_u.get(t)
        if u is None:
            u = len(uniq_texts)
            uniq_texts.append(t)
            t_to_u[t] = u

    text_embs: list[torch.Tensor] = []
    for i in range(0, len(uniq_texts), max(1, int(batch_size))):
        sub = uniq_texts[i:i + int(batch_size)]
        tok = tokenizer(sub).to(device, non_blocking=True)
        with _autocast(device):
            tz = model.encode_text(tok)
        tz = F.normalize(tz.float(), dim=1)
        text_embs.append(tz.cpu())

    tz_uniq = torch.cat(text_embs, dim=0).float()
    tz_uniq = tz_uniq / (tz_uniq.norm(dim=1, keepdim=True) + 1e-12)
    txt_Z = torch.stack([tz_uniq[[t_to_u[t] for t in tl]].mean(0) for tl in cap_lists])
    return cap_ids, txt_Z, {img_id: idx for idx, img_id in enumerate(cap_ids)}


def fuse_queries(Q_img: torch.Tensor, Q_txt: torch.Tensor) -> torch.Tensor:
    """Query vectors: the normalised mean of the image embedding and the text vector."""
    q = 0.5 * (Q_img + Q_txt)
    q = q / (q.norm(dim=1, keepdim=True) + 1e-12)
    return q.float()


def score_queries(query_ids: Sequence[str], gallery_ids: Sequence[str], Z: torch.Tensor, txt_Z: torch.Tensor,
                  id_to_txt_idx: Mapping[str, int], data: TrainingData) -> list[dict]:
    """Per-query metrics of the fused queries against the gallery (CPU, fp32 scores, one query at a time).

    A query needs at least one direct partner in the gallery and a text vector. The query
    itself is scored -1e9; its direct partners are the relevant items, and its other known
    positives in the gallery are removed from the ranking.
    """
    id_to_gidx = {img_id: idx for idx, img_id in enumerate(gallery_ids)}
    gallery_set = set(gallery_ids)
    queries: list[str] = []
    for q in query_ids:
        if q not in id_to_gidx:
            continue
        if not data.adj_direct.get(q, set()) & gallery_set:
            continue
        if q not in id_to_txt_idx:
            continue
        queries.append(q)

    Q = fuse_queries(Z[[id_to_gidx[q] for q in queries]], txt_Z[[id_to_txt_idx[q] for q in queries]])
    rows: list[dict] = []
    for i, q in enumerate(queries):
        q_gi = id_to_gidx[q]
        sims = (Q[i:i + 1] @ Z.t()).squeeze(0).numpy()
        sims[q_gi] = -1e9
        order = np.argsort(-sims)

        pos_ids = {j for j in (data.adj_direct.get(q, set()) & gallery_set) if j in id_to_gidx}
        known = (data.adj_direct.get(q, set()) | data.adj_inferred.get(q, set())) & gallery_set
        ignore_idx = {id_to_gidx[x] for x in set(known) - set(pos_ids) - {q} if x in id_to_gidx}
        if ignore_idx:
            order = np.array([idx for idx in order if int(idx) not in ignore_idx], dtype=order.dtype)

        rows.append({"query_id": q, **retrieval_for_query(sims, order, [id_to_gidx[j] for j in pos_ids])})
    return rows


def summarize_queries(rows: Sequence[Mapping], seed: int) -> dict:
    """Mean and 95% bootstrap interval of every metric over the queries."""
    out: dict = {"n_queries": len(rows)}
    for k in SUMMARY_METRICS:
        mean, lo, hi = bootstrap_ci([r[k] for r in rows], n_boot=BOOTSTRAP_RESAMPLES, seed=seed)
        out[k] = {"mean": float(mean), "lo": float(lo), "hi": float(hi)}
    return out


def validate(model, tokenizer, data: TrainingData, config: Mapping, device) -> dict:
    """Validation summary of fused image-plus-text retrieval on the validation split (the selection metric)."""
    model.eval()
    gallery_ids, Z = embed_gallery(model, data.val_ids, data.id2path, config, device)
    id_to_gidx = {img_id: idx for idx, img_id in enumerate(gallery_ids)}
    _, txt_Z, id_to_txt_idx = embed_query_texts(model, tokenizer, data.val_ids, id_to_gidx, data.id2texts,
                                                config["text_batch_size"], device)
    rows = score_queries(data.val_ids, gallery_ids, Z, txt_Z, id_to_txt_idx, data)
    if not rows:
        raise RuntimeError("No validation query has a pair partner in the validation split")
    return {**summarize_queries(rows, config["seed"]), "gallery_size": len(gallery_ids)}


# --------------------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------------------
def _write_json(path: str, obj) -> None:
    """Write JSON atomically (temporary file, then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _append_jsonl(path: str, record: Mapping) -> None:
    """Append one JSON record as a line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def main(config: dict, args: argparse.Namespace) -> None:
    """Train the image_text_to_image configuration (called by ``python -m pmc_pair_retrieval.train``)."""
    check_config(config)
    # The training-pair order depends on fixed string hashing. Check it even when this
    # function is called directly.
    problems = hash_seed_problems()
    if problems:
        raise SystemExit("; ".join(problems) + ". Start the run with 'python -m pmc_pair_retrieval.train'.")
    if not torch.cuda.is_available():
        raise SystemExit("image_text_to_image training needs a CUDA GPU")
    out = args.output_dir
    run_config = config_as_run(args, config)
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)
    print(json.dumps(config, indent=1), flush=True)

    device = "cuda"
    max_steps = args.max_steps
    seed_everything(config["seed"])
    set_backend_flags()
    data = prepare_data(config, args.data_dir, getattr(args, "allow_missing_images", False))

    # Built once, after seeding and before any DataLoader iterator: the random initialisation that
    # the pretrained weights overwrite advances the global torch generator, whose next draw seeds
    # the training workers.
    try:
        model, tokenizer = load_biomedclip(device)
    except ModelCacheError as err:
        raise SystemExit(f"[train] {err}") from None
    print("[env] gpu", torch.cuda.get_device_name(), flush=True)
    n_blocks = freeze_vision_blocks(model, config["frozen_vision_blocks"])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] froze the first {config['frozen_vision_blocks']} of {n_blocks} vision blocks; "
          f"trainable parameters {n_trainable}", flush=True)
    checkpointed = enable_vision_block_checkpointing(model, config["grad_checkpointing"])
    print(f"[model] activation checkpointing of vision blocks {checkpointed}", flush=True)

    train_tf = MultiViewTrainTransform(size=config["image_size"], crop_scale=config["crop_scale"])
    print(f"[views] train {train_tf}; eval_views={config['eval_views']}", flush=True)
    dataset = PairDataset(data.train_pairs, data.id2path, train_tf, data.pair2texts_train)
    sampler = HardNegativeBatchSampler(data.train_pairs, data.hard_negatives, config["batch_size"], seed=config["seed"],
                                       sample_per_img=config["hard_negatives_sampled_per_image"], drop_last=True)
    loader = make_train_loader(dataset, sampler, config["num_workers"])
    steps_per_epoch = len(sampler)
    total_steps = max_steps if max_steps is not None else config["epochs"] * steps_per_epoch
    opt, sched, groups = build_optimizer(model, config, total_steps)
    print(f"[optim] AdamW groups {json.dumps(groups)}; {steps_per_epoch} steps per epoch, schedule horizon "
          f"{total_steps} steps", flush=True)

    stopping = config["early_stopping"]
    metric = stopping["metric"]
    best_path = os.path.join(out, "best_model.safetensors")
    history_path = os.path.join(out, "history.jsonl")
    lam_ii, lam_it = float(config["lambda_image_image"]), float(config["lambda_image_text"])
    best, best_epoch, best_val, bad, epochs_run, global_step, base_seed = None, 0, None, 0, 0, 0, None
    t_start = time.time()
    opt.zero_grad(set_to_none=True)
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        hasher = hashlib.sha256()
        loss_sums, n_losses = [0.0, 0.0, 0.0], 0
        neg_sum, neg_min, neg_anchors, caption_sum, n_batches = 0, 10 ** 9, 0, 0, 0

        batches = start_epoch(loader)
        if base_seed is None:
            base_seed = loader_base_seed(loader)
            print(f"[train] loader base seed {base_seed}", flush=True)
        model.train()
        set_frozen_blocks_eval(model, config["frozen_vision_blocks"])

        for step, batch in enumerate(batches):
            if max_steps is not None and global_step >= max_steps:
                break

            ida, idb, xa, xb, txt = batch
            xa = xa.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            ida_list = list(ida)
            idb_list = list(idb)
            N = len(ida_list)
            texts = [t for t in (str(x) if x is not None else "" for x in txt) if t.strip()]
            caption_sum += len(set(texts))
            n_batches += 1
            hasher.update(batch_sequence_line(ida_list, idb_list))

            with _autocast(device):
                # The two images of each pair go through the vision tower in separate passes.
                ea = encode_image_views(model, xa)
                eb = encode_image_views(model, xb)
                loss_ii, (batch_neg_sum, batch_neg_min) = image_image_loss(ea, eb, ida_list, idb_list, data, config,
                                                                           device)
                neg_sum += batch_neg_sum
                neg_min = min(neg_min, batch_neg_min)
                neg_anchors += 2 * N
                loss_it = torch.zeros((), device=device)
                if lam_it > 0.0:
                    loss_it = image_text_loss(model, tokenizer, ea, eb, list(txt), config, device)
                loss = lam_ii * loss_ii + lam_it * loss_it

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["clip_grad_norm"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            global_step += 1

            loss_sums[0] += float(loss.item())
            loss_sums[1] += float(loss_ii.item())
            loss_sums[2] += float(loss_it.item())
            n_losses += 1
            if max_steps is not None and global_step >= max_steps:
                break
            if (step + 1) % config["log_every"] == 0:
                print(f"[train] epoch {epoch} step {step + 1}/{steps_per_epoch} gstep {global_step} "
                      f"loss={loss_sums[0] / (step + 1):.4f} loss_ii={float(loss_ii.item()):.4f} "
                      f"loss_it={float(loss_it.item()):.4f} lr={opt.param_groups[0]['lr']:.2e} "
                      f"{(time.time() - t0) / (step + 1):.2f}s/step", flush=True)
        t_train = time.time() - t0
        peak_gib = torch.cuda.max_memory_allocated() / 2 ** 30

        val = validate(model, tokenizer, data, config, device)
        score = val[metric]["mean"]
        if not math.isfinite(score):
            raise RuntimeError(f"validation {metric} is not finite at epoch {epoch}")
        if best is None or score > best:
            best, best_epoch, best_val, bad = score, epoch, val, 0
            save_weights(best_path, model.state_dict(),
                         {"configuration": CONFIGURATION, "epoch": epoch, "global_step": global_step, "val": val,
                          "config": run_config})
        else:
            bad += 1
        epochs_run = epoch
        minutes = {"train": t_train / 60, "eval": (time.time() - t0 - t_train) / 60}
        _append_jsonl(history_path, {
            "epoch": epoch, "global_step": global_step,
            "train": {"loss": loss_sums[0] / max(1, n_losses), "loss_image_image": loss_sums[1] / max(1, n_losses),
                      "loss_image_text": loss_sums[2] / max(1, n_losses), "batches": n_batches,
                      "valid_negatives_per_anchor_mean": neg_sum / max(1, neg_anchors),
                      "valid_negatives_per_anchor_min": neg_min if neg_anchors else None,
                      "distinct_descriptions_per_batch_mean": caption_sum / max(1, n_batches)},
            "val": val, "minutes": minutes, "peak_memory_gib": peak_gib, "batch_sequence_sha256": hasher.hexdigest()})
        print(validation_line(epoch, metric, val, minutes, best, best_epoch), flush=True)
        if epoch >= stopping["min_epochs"] and stopping["patience"] > 0 and bad >= stopping["patience"]:
            print(f"[train] early stop at epoch {epoch}: no improvement in {bad} epochs", flush=True)
            break
        if max_steps is not None and global_step >= max_steps:
            print(f"[train] stopped after {global_step} optimizer steps (--max-steps)", flush=True)
            break

    info = load_weights(model, best_path)
    final_val = validate(model, tokenizer, data, config, device)
    print(f"[val] selected epoch {best_epoch} reloaded from {os.path.basename(best_path)}: "
          f"{metric}={final_val[metric]['mean']!r} (at selection {best!r})", flush=True)
    _write_json(os.path.join(out, "summary.json"), {
        "configuration": CONFIGURATION, "best_epoch": best_epoch, "best_val": best_val, "best_model_val": final_val,
        "best_model_state_dict_sha256": info["state_dict_sha256"], "epochs_run": epochs_run,
        "global_step": global_step, "trainable_params": n_trainable, "hours": (time.time() - t_start) / 3600,
        "data": data.counts, "loader_base_seed": base_seed,
        "overrides": {"epochs": args.epochs, "max_steps": max_steps}, "config": run_config})
    print(f"[done] best val {metric}={best:.5f} at epoch {best_epoch}; outputs in {out}", flush=True)
