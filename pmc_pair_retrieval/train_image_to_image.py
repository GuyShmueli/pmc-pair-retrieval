"""Fine-tune the BiomedCLIP vision tower for image-to-image retrieval (configuration ``image_to_image``).

Run through the training command::

    python -m pmc_pair_retrieval.train --config configs/image_to_image.yaml \\
        --data-dir DATA_DIR --output-dir runs/image_to_image

Training. Every training image is drawn once per epoch as a query, in a new random order,
paired with one of its direct pair partners drawn uniformly at random. The incomplete last
batch is dropped, so ``graph.n // batch_size`` full batches are trained (551 of 128 items,
70,528 of the 70,549 items; ``TrainGraph`` leaves out 3 of the 70,552 training images).
A batch holds ``batch_size`` such items: the queries followed
by their positives (256 images), each image seen as its three jittered view crops
(``train_views``), whose embeddings are averaged. Every image of the batch is a candidate
for every query in a per-positive InfoNCE loss. The query's direct partners
are positives. The other members of its connected training component (the query itself,
partners of partners, pixel-identical copies) and repeated copies of an image are
ignored. All other images are negatives. The whole vision tower is trained, while the
text tower and the logit scale stay frozen. Training uses AdamW with layer-wise
learning-rate decay and a warmup-cosine schedule over all epochs. The image encoder runs
under bf16 autocast. The loss is computed on fp32 tensors outside autocast, with TF32
enabled for its matmuls, as in the published run.

Validation and selection. Before training (epoch 0) and after every epoch, the validation
images are encoded from their three deterministic view crops (``eval_views``) and ranked
against each other. The checkpoint of the epoch with the highest validation Recall@10 is
kept: a later epoch must be strictly better, and epoch 0 is never selected.

Outputs in the output directory:

- ``config.yaml``: the configuration as run, i.e. the configuration file as written
  (repository-relative data paths) with the ``--epochs`` override; the weight sidecar and
  ``summary.json`` record the same.
- ``history.jsonl``: one record per epoch, epoch 0 included.
- ``best_model.safetensors`` and ``best_model.json``: vision-tower weights of the selected
  epoch (see ``pmc_pair_retrieval.model.save_weights``).
- ``val_per_query.csv``: per-query validation metrics of the selected epoch.
- ``summary.json``.

Every ``log_every`` optimizer steps a progress line on stdout reports the running mean loss
of the epoch and two learning rates. ``lr_top`` belongs to the top transformer block, the
final norm and the projection head (peak ``lr``). ``lr_embed`` belongs to the patch,
position and class-token embeddings (peak ``lr * layer_decay ** 12``).

Reproducibility. The seed determines the data order and the augmentations through three
random streams, kept exactly as in the published run:

1. A numpy generator draws the items of every epoch.
2. The global torch generator is advanced by building the model. It then gives one base
   seed to every DataLoader iterator: the epoch-0 validation, then the training loader and
   the validation of each epoch, in this order.
3. Each DataLoader worker seeds its own torch generator from its loader's base seed, and
   the augmentations draw from it.

Training on the GPU is not bitwise reproducible (cuDNN autotuning, bf16 kernels); reruns
agree to about 1e-4 in the loss.

``--max-steps N`` is for smoke tests only. Training stops after N optimizer steps in total,
the epoch in which this happens is validated as usual, and the run ends. The learning-rate
schedule keeps its full horizon (``epochs`` x steps per epoch), so the first N steps are
those of a full run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .dataset import load_pairs, load_split
from .metrics import K_LIST, bootstrap_ci
from .model import ModelCacheError, load_biomedclip, save_weights
from .train import config_as_run, validation_line
from .views import N_VIEWS, MultiViewEvalTransform, MultiViewTrainTransform, encode_image_views

CONFIGURATION = "image_to_image"
CONFIG_KEYS = frozenset({
    "configuration", "seed", "hash_seed", "split", "pixel_hashes",
    "image_size", "train_views", "eval_views", "crop_scale", "color_jitter", "rotation_degrees",
    "epochs", "batch_size", "lr", "layer_decay", "weight_decay", "betas", "eps", "warmup_steps",
    "clip_grad_norm", "temperature", "grad_checkpointing",
    "eval_batch_size", "early_stopping", "num_workers", "log_every",
})
# Logit of ignored candidates. Unlike -inf it keeps the log-softmax finite in every row.
NEG = -1e4
# Validation summaries: 200 bootstrap resamples of the queries per metric, with a fresh
# generator seeded 42 for each metric (as in the evaluator).
BOOTSTRAP_RESAMPLES = 200
BOOTSTRAP_SEED = 42
SUMMARY_KEYS = tuple(f"R@{k}" for k in K_LIST) + tuple(f"Hit@{k}" for k in K_LIST) + ("mAP", "nDCG")


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
    if set(config["color_jitter"]) != {"brightness", "contrast", "p"}:
        raise SystemExit("color_jitter needs exactly the keys brightness, contrast and p")
    if set(config["early_stopping"]) != {"metric", "min_epochs", "patience"}:
        raise SystemExit("early_stopping needs exactly the keys metric, min_epochs and patience")
    if config["early_stopping"]["metric"] not in SUMMARY_KEYS:
        raise SystemExit(f"early_stopping.metric must be one of {list(SUMMARY_KEYS)}")


def seed_everything(seed: int) -> None:
    """Seed Python's, numpy's and torch's (CPU and CUDA) global generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_backend_flags() -> None:
    """Backend flags of the published run: TF32 for fp32 matmuls (the loss) and convolutions, cuDNN autotuning.

    The validation scoring turns TF32 off locally (``retrieval_metrics``).
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------------------
# Data: images, direct edges, components, split
# --------------------------------------------------------------------------------------
def check_missing_images(pairs: Sequence[Sequence[str]], id2path: Mapping[str, str], allow_missing: bool) -> None:
    """Refuse to train unless every image of the pairs is present, or warn when that is explicitly allowed.

    Without the missing images, ``PairData`` keeps the pairs whose two images are present.
    """
    used = {str(i).strip() for ab in pairs for i in ab[:2]}
    missing = sorted(used - id2path.keys())
    if not missing:
        return
    if not allow_missing:
        raise SystemExit(f"{len(missing)} of {len(used)} images of image_pairs.json are missing from the data "
                         f"directory (e.g. {missing[0]}); reproducing the published model needs all of them. "
                         "Pass --allow-missing-images to train without them.")
    print(f"[data] WARNING: training without {len(missing)} missing images and their pairs; results will differ "
          "from the published ones.", flush=True)


class PairData:
    """All images of the pairs with their split, the direct-edge graph and its connected components.

    Images are indexed by their position in ``ids`` (sorted image ids). Galleries and
    neighbour lists follow this order, which also breaks ties in the validation ranking.
    """

    def __init__(self, pairs: Sequence[Sequence[str]], id2path: Mapping[str, str], split: Mapping[str, str]):
        all_ids, edges = set(), set()
        for ab in pairs:
            a, b = str(ab[0]).strip(), str(ab[1]).strip()
            all_ids.update([a, b])
            u, v = (a, b) if a < b else (b, a)
            edges.add((u, v))
        present = all_ids & set(id2path)
        edges = sorted((u, v) for u, v in edges if u in present and v in present)
        unassigned = [i for i in present if i not in split]
        if unassigned:
            raise ValueError(f"{len(unassigned)} image ids are missing from the split file")

        self.ids = sorted(present)
        self.index = {iid: i for i, iid in enumerate(self.ids)}
        self.paths = [id2path[i] for i in self.ids]
        self.split = np.array([split[i] for i in self.ids])
        self.edges = np.array([(self.index[u], self.index[v]) for u, v in edges], dtype=np.int64)

        n = len(self.ids)
        nbrs = [[] for _ in range(n)]
        for a, b in self.edges:
            nbrs[a].append(b)
            nbrs[b].append(a)
        self.nbrs = [np.array(sorted(x), dtype=np.int64) for x in nbrs]
        # Connected components over all direct edges (the full transitive closure of the pairs).
        self.comp = components(n, self.edges)

    def split_indices(self, name: str) -> np.ndarray:
        """Global indices of the images of a split, in sorted id order."""
        return np.flatnonzero(self.split == name)

    def check_split(self) -> dict[str, int]:
        """Fail if a direct edge or a component crosses splits; returns the split sizes."""
        sa, sb = self.split[self.edges[:, 0]], self.split[self.edges[:, 1]]
        n_cross = int((sa != sb).sum())
        if n_cross:
            raise ValueError(f"{n_cross} direct edges cross splits")
        comp_splits = defaultdict(set)
        for c, s in zip(self.comp, self.split):
            comp_splits[c].add(s)
        if not all(len(s) == 1 for s in comp_splits.values()):
            raise ValueError("a connected component spans several splits")
        return {s: int((self.split == s).sum()) for s in ("train", "val", "test")}


def components(n: int, links) -> np.ndarray:
    """Connected-component id (0..k-1) of n nodes under undirected ``(a, b)`` links (union-find)."""
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.unique([find(i) for i in range(n)], return_inverse=True)[1]


def load_duplicate_groups(data: PairData, path: str) -> np.ndarray:
    """Group id per image of ``data``; pixel-identical images (same decoded-pixel md5) share one."""
    with open(path, encoding="utf-8") as f:
        pixel_hash = json.load(f)
    groups = np.unique([pixel_hash[i] for i in data.ids], return_inverse=True)[1]
    print(f"[data] pixel-duplicate groups: {int((np.bincount(groups) > 1).sum())}", flush=True)
    return groups


class TrainGraph:
    """Training images, their direct training edges and the masks of the training loss.

    For a query q and a candidate c: c is a positive if (q, c) is a direct training edge;
    c is ignored if it lies in q's training component but is not a positive (this includes
    q itself) or if it repeats an image already among the candidates; otherwise c is a
    negative. Training components join the direct training edges and the pixel-duplicate
    groups, so a pixel copy of the query or of one of its positives is ignored, never a
    negative. Training images that are pixel copies of a validation or test image are not
    used, nor are partners left without a training positive.
    """

    def __init__(self, data: PairData, dup_groups: np.ndarray, device: str | torch.device):
        dup = np.asarray(dup_groups)
        train = data.split == "train"
        leaky_group = np.zeros(dup.max() + 1, dtype=bool)
        leaky_group[dup[~train]] = True
        keep = train & ~leaky_group[dup]
        while True:
            lonely = [g for g in np.flatnonzero(keep) if not keep[data.nbrs[g]].any()]
            if not lonely:
                break
            keep[lonely] = False
        self.dropped = [data.ids[g] for g in np.flatnonzero(train & ~keep)]
        self.idx = np.flatnonzero(keep)  # train-local -> global index
        self.n = len(self.idx)
        g2t = -np.ones(len(data.ids), dtype=np.int64)
        g2t[self.idx] = np.arange(self.n)
        if not all((data.split[data.nbrs[g]] == "train").all() for g in self.idx):
            raise ValueError("a direct edge leaves the training split")
        nbrs = [x[x >= 0] for x in (g2t[data.nbrs[g]] for g in self.idx)]
        self.nbrs = nbrs  # sorted train-local neighbours
        self.indptr, self.indices = csr(nbrs, device)
        self.n_pairs = int(((g2t[data.edges[:, 0]] >= 0) & (g2t[data.edges[:, 1]] >= 0)).sum())
        # Every later member of a pixel-duplicate group is linked to its first member.
        first, dup_links = {}, []
        for t, d in enumerate(dup[self.idx]):
            if d in first:
                dup_links.append((t, first[d]))
            else:
                first[d] = t
        edge_links = [(t, int(u)) for t, x in enumerate(nbrs) for u in x if t < u]
        self.comp_np = components(self.n, edge_links + dup_links)
        self.comp = torch.tensor(self.comp_np, device=device)
        self.ids = [data.ids[g] for g in self.idx]
        self.device = device

    def neighbour_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Dense bool ``[len(q), n]`` of the direct training positives of train-local queries q."""
        q = q.to(self.device)
        start, end = self.indptr[q], self.indptr[q + 1]
        lengths = end - start
        rows = torch.repeat_interleave(torch.arange(len(q), device=self.device), lengths)
        offsets = torch.repeat_interleave(start - torch.cumsum(lengths, 0) + lengths, lengths)
        cols = self.indices[offsets + torch.arange(int(lengths.sum()), device=self.device)]
        out = torch.zeros(len(q), self.n, dtype=torch.bool, device=self.device)
        out[rows, cols] = True
        return out

    def masks(self, q: torch.Tensor, cand: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(positive, ignored)`` bool ``[len(q), len(cand)]`` for train-local indices."""
        q, cand = q.to(self.device), cand.to(self.device)
        pos = self.neighbour_matrix(q)[:, cand]
        ign = (self.comp[q][:, None] == self.comp[cand][None, :]) & ~pos
        ign |= ~first_occurrence(cand)[None, :]  # repeated copies of a candidate
        return pos & ~ign, ign


def csr(rows: Sequence[np.ndarray], device: str | torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``(indptr, indices)`` tensors of a list of int arrays."""
    lengths = np.array([len(x) for x in rows], dtype=np.int64)
    indptr = torch.tensor(np.concatenate([[0], np.cumsum(lengths)]), device=device)
    indices = torch.tensor(np.concatenate(rows).astype(np.int64), device=device)
    return indptr, indices


def first_occurrence(idx: torch.Tensor) -> torch.Tensor:
    """True at the first position of every value of a 1-D tensor."""
    n = idx.numel()
    order = torch.argsort(idx * (n + 1) + torch.arange(n, device=idx.device))
    sorted_idx = idx[order]
    first_sorted = torch.ones(n, dtype=torch.bool, device=idx.device)
    first_sorted[1:] = sorted_idx[1:] != sorted_idx[:-1]
    out = torch.empty_like(first_sorted)
    out[order] = first_sorted
    return out


# --------------------------------------------------------------------------------------
# Images and epochs
# --------------------------------------------------------------------------------------
def load_rgb(path: str) -> Image.Image:
    """The decoded RGB image of a file."""
    with Image.open(path) as im:
        return im.convert("RGB")


class QueryPositiveDataset(Dataset):
    """Training items ``(query, positive)`` (train-local indices) with their augmented view stacks."""

    def __init__(self, items: Sequence[tuple[int, int]], paths: Sequence[str], train_idx: np.ndarray, transform):
        self.items, self.paths, self.train_idx, self.transform = items, paths, train_idx, transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        q, p = self.items[i]
        # Both images draw their augmentations from the worker's torch generator, so the query
        # must be transformed before its positive.
        xq = self.transform(load_rgb(self.paths[self.train_idx[q]]))
        xp = self.transform(load_rgb(self.paths[self.train_idx[p]]))
        return q, p, xq, xp


class ImageViewsDataset(Dataset):
    """Evaluation view stacks of the images at the given global indices."""

    def __init__(self, paths: Sequence[str], indices: np.ndarray, transform):
        self.paths, self.indices, self.transform = paths, indices, transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int) -> torch.Tensor:
        return self.transform(load_rgb(self.paths[self.indices[k]]))


def make_train_transform(config: Mapping) -> MultiViewTrainTransform:
    """Per view: random resized crop inside the view crop, colour jitter with probability p, small rotation."""
    return MultiViewTrainTransform(size=config["image_size"], crop_scale=config["crop_scale"],
                                   color_jitter=config["color_jitter"], rotation_degrees=config["rotation_degrees"])


def uniform_query_epoch(graph: TrainGraph, rng: np.random.Generator) -> list[tuple[int, int]]:
    """The items of one epoch: every training image once as a query, with one direct partner drawn uniformly.

    ``rng`` is the run's only numpy generator and is carried across epochs. The query
    permutation is drawn first, then one partner per query in permuted order from its
    sorted neighbour list (a single neighbour consumes no random number).
    """
    return [(int(q), int(rng.choice(graph.nbrs[q]))) for q in rng.permutation(graph.n)]


def batch_sequence_sha256(items: Sequence[tuple[int, int]], ids: Sequence[str]) -> str:
    """sha256 over the lines ``"{query_id}\\t{positive_id}\\n"`` of all items of an epoch, in order.

    The items that the incomplete last batch leaves out are included.
    """
    digest = hashlib.sha256()
    for q, p in items:
        digest.update(f"{ids[q]}\t{ids[p]}\n".encode())
    return digest.hexdigest()


def make_train_loader(items: Sequence[tuple[int, int]], paths: Sequence[str], train_idx: np.ndarray, transform,
                      config: Mapping) -> DataLoader:
    """The DataLoader of one training epoch.

    The loader is created anew every epoch, with non-persistent workers. Creating its
    iterator draws one base seed from the global torch generator, and worker w seeds its
    torch generator with base_seed + w. Batch b is produced by worker b % num_workers, so
    ``num_workers`` and ``batch_size`` are part of the augmentation stream. Items are taken
    in order and the incomplete last batch is dropped.
    """
    return DataLoader(QueryPositiveDataset(items, paths, train_idx, transform), batch_size=config["batch_size"],
                      shuffle=False, drop_last=True, num_workers=config["num_workers"], pin_memory=True,
                      persistent_workers=False)


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------
def per_positive_infonce(logits: torch.Tensor, pos: torch.Tensor, ign: torch.Tensor) -> torch.Tensor:
    """-mean over a row's positives of log softmax(logits)[p], averaged over the rows with a positive.

    The softmax runs over the non-ignored columns, positives included.
    """
    logits = logits.masked_fill(ign, NEG)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    has_pos = pos.any(1)
    return -(logp.masked_fill(~pos, 0).sum(1)[has_pos] / pos.sum(1)[has_pos]).mean()


def contrastive_loss(graph: TrainGraph, z: torch.Tensor, ids: torch.Tensor, n_query: int,
                     temperature: float) -> torch.Tensor:
    """Per-positive InfoNCE of a batch.

    ``z`` holds the fp32 embeddings of the ``n_query`` queries followed by those of their
    sampled positives, and ``ids`` their train-local indices. The queries are the anchor
    rows; every image of the batch is a candidate.
    """
    q, zq = ids[:n_query], z[:n_query]
    sims = zq @ z.T
    pos, ign = graph.masks(q, ids)
    assert not (pos & ign).any(), "positive and ignored masks overlap"
    return per_positive_infonce(sims / temperature, pos, ign)


# --------------------------------------------------------------------------------------
# Model and optimizer
# --------------------------------------------------------------------------------------
def configure_full_finetuning(model) -> int:
    """Make exactly the vision tower trainable; returns the number of trainable parameters."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.visual.parameters():
        p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(model, config: Mapping, total_steps: int):
    """AdamW with layer-wise learning-rate decay and its warmup-cosine schedule; returns ``(optimizer, scheduler)``.

    Learning rates:

    - Transformer block i trains at ``lr * layer_decay ** (n_blocks - 1 - i)``. The top
      block therefore trains at ``lr``, like the final norm and the projection head.
    - The patch, position and class-token embeddings train at ``lr * layer_decay ** n_blocks``.

    Weight decay applies to weight matrices only, not to biases, norms, the position
    embedding or the class token. Parameters are grouped by (lr, weight decay) in
    ``named_parameters`` order. The schedule warms up linearly over ``warmup_steps`` and then
    decays along a cosine to zero at ``total_steps``.
    """
    n_blocks = len(model.visual.trunk.blocks)
    lr, layer_decay = config["lr"], config["layer_decay"]
    embeddings = ("patch_embed", "pos_embed", "cls_token")

    def lr_of(name):
        block = re.search(r"trunk\.blocks\.(\d+)\.", name)
        if block:
            return lr * layer_decay ** (n_blocks - 1 - int(block.group(1)))
        return lr * (layer_decay ** n_blocks if any(s in name for s in embeddings) else 1.0)

    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = p.ndim <= 1 or name.endswith(".bias") or any(s in name for s in ("norm", "pos_embed", "cls_token"))
        groups.setdefault((lr_of(name), 0.0 if no_decay else config["weight_decay"]), []).append(p)
    opt = torch.optim.AdamW([{"params": ps, "lr": g_lr, "weight_decay": wd} for (g_lr, wd), ps in groups.items()],
                            lr=lr, betas=tuple(config["betas"]), eps=config["eps"])
    print("[optim] param groups (lr, weight_decay, #tensors): "
          + ", ".join(f"({g_lr:.2e}, {wd}, {len(ps)})" for (g_lr, wd), ps in groups.items()), flush=True)
    warmup = config["warmup_steps"]

    def warmcos(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, warmcos)


# --------------------------------------------------------------------------------------
# Validation (image-to-image retrieval on the validation split)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def encode_images(model, paths: Sequence[str], indices: np.ndarray, batch_size: int, num_workers: int,
                  device: str | torch.device = "cuda", size: int = 224) -> torch.Tensor:
    """Unit-norm fp32 embeddings of the images at ``indices``, from their three deterministic view crops.

    The views of a batch go through one bf16-autocast forward pass and are aggregated
    (normalise, average, renormalise). The result is normalised twice more in fp32, as in
    the published validation. The model runs in eval mode and its mode is restored. The
    DataLoader is created once per call, so each call draws exactly one base seed from the
    global torch generator.
    """
    was_training = model.training
    model.eval()
    loader = DataLoader(ImageViewsDataset(paths, indices, MultiViewEvalTransform(size)),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    out = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = encode_image_views(model, x)
        z = F.normalize(z.float(), dim=1)
        out.append(z / (z.norm(dim=1, keepdim=True) + 1e-12))
    model.train(was_training)
    return torch.cat(out)


@torch.no_grad()
def retrieval_metrics(Z: torch.Tensor, gallery: np.ndarray, data: PairData, block: int = 512) -> list[dict]:
    """Per-query image-to-image metrics on a split gallery.

    ``Z`` holds the unit embeddings of ``gallery`` (global indices, sorted ids). The queries
    are the gallery images with at least one direct partner in the gallery. The rank of a
    positive is 1 + the number of non-excluded gallery images scoring higher, with ties
    broken by gallery order. Excluded are the query itself and the members of its connected
    component that are not direct partners. Scores are computed with TF32 off.
    """
    assert torch.isfinite(Z).all(), "non-finite embeddings"  # NaN would rank every positive first
    device = Z.device
    G = len(gallery)
    local = -np.ones(len(data.ids), dtype=np.int64)
    local[gallery] = np.arange(G)
    comp = torch.tensor(data.comp[gallery], device=device)
    pos_rows, pos_cols = [], []
    for r, g in enumerate(gallery):
        cols = local[data.nbrs[g]]
        cols = cols[cols >= 0]
        pos_rows += [r] * len(cols)
        pos_cols += cols.tolist()
    pos_rows = torch.tensor(pos_rows, device=device)
    pos_cols = torch.tensor(pos_cols, device=device)
    queries = torch.unique(pos_rows)
    col_ids = torch.arange(G, device=device)

    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ranks = torch.empty(len(pos_rows), dtype=torch.long, device=device)
        for start in range(0, G, block):
            rows = torch.arange(start, min(start + block, G), device=device)
            S = Z[rows] @ Z.T
            excluded = comp[rows][:, None] == comp[None, :]
            sel = (pos_rows >= start) & (pos_rows < start + len(rows))
            pr, pc = pos_rows[sel] - start, pos_cols[sel]
            excluded[pr, pc] = False  # positives are never excluded
            S = S.masked_fill(excluded, float("-inf"))
            s_pos = S[pr, pc]
            Sp = S[pr]
            higher = (Sp > s_pos[:, None]).sum(1)
            ties_before = ((Sp == s_pos[:, None]) & (col_ids[None, :] < pc[:, None])).sum(1)
            ranks[sel] = 1 + higher + ties_before
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    pos_rows, ranks = pos_rows.cpu().numpy(), ranks.cpu().numpy()  # pos_rows is non-decreasing
    bounds = np.searchsorted(pos_rows, np.arange(G + 1))
    per_query = []
    for q in queries.cpu().numpy():
        r = np.sort(ranks[bounds[q]:bounds[q + 1]])
        n = len(r)
        row = {"query_id": data.ids[gallery[q]], "n_pos": n, "first_rank": int(r[0])}
        for k in K_LIST:
            row[f"R@{k}"] = float((r <= k).sum() / n)
            row[f"Hit@{k}"] = float(r[0] <= k)
        row["AP"] = float(np.mean(np.arange(1, n + 1) / r))
        row["nDCG"] = float(np.sum(1 / np.log2(r + 1)) / np.sum(1 / np.log2(np.arange(2, n + 2))))
        per_query.append(row)
    return per_query


def summarize(per_query: Sequence[dict]) -> dict:
    """Mean and 95% bootstrap interval of every metric over the queries (``AP`` reported as ``mAP``)."""
    out = {"n_queries": len(per_query)}
    for key in [f"R@{k}" for k in K_LIST] + [f"Hit@{k}" for k in K_LIST] + ["AP", "nDCG"]:
        mean, lo, hi = bootstrap_ci([row[key] for row in per_query], n_boot=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED)
        out["mAP" if key == "AP" else key] = {"mean": mean, "lo": lo, "hi": hi}
    return out


def validate(model, data: PairData, config: Mapping, device: str | torch.device = "cuda") -> tuple[dict, list[dict]]:
    """Image-to-image retrieval on the validation split; returns ``(summary, per-query rows)``."""
    gallery = data.split_indices("val")
    Z = encode_images(model, data.paths, gallery, config["eval_batch_size"], config["num_workers"], device,
                      config["image_size"])
    per_query = retrieval_metrics(Z, gallery, data)
    return summarize(per_query), per_query


# --------------------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------------------
def write_per_query(path: str, per_query: Sequence[dict]) -> None:
    """Per-query rows as CSV, in query order."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_query[0].keys()))
        writer.writeheader()
        writer.writerows(per_query)


def write_json(path: str, obj) -> None:
    """Write JSON atomically (temporary file, then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def append_jsonl(path: str, record: Mapping) -> None:
    """Append one JSON record as a line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def main(config: dict, args: argparse.Namespace) -> None:
    """Train the image_to_image configuration (called by ``python -m pmc_pair_retrieval.train``)."""
    check_config(config)
    if not torch.cuda.is_available():
        raise SystemExit("image_to_image training needs a CUDA GPU")
    out = args.output_dir
    run_config = config_as_run(args, config)
    with open(os.path.join(out, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)
    print(json.dumps(config, indent=1), flush=True)

    device = "cuda"
    seed_everything(config["seed"])
    set_backend_flags()

    pairs, _, id2path = load_pairs(args.data_dir)
    check_missing_images(pairs, id2path, getattr(args, "allow_missing_images", False))
    data = PairData(pairs, id2path, load_split(config["split"]))
    print("[data] split sizes", data.check_split(), flush=True)
    graph = TrainGraph(data, load_duplicate_groups(data, config["pixel_hashes"]), device)
    print(f"[data] train images={graph.n} train pairs={graph.n_pairs} "
          f"dropped (pixel copies of val/test images or their partners)={graph.dropped}", flush=True)

    # Built once, after seeding and before any DataLoader: the random initialisation that the
    # pretrained weights overwrite advances the global torch generator, which seeds the workers.
    try:
        model, _ = load_biomedclip(device)
    except ModelCacheError as err:
        raise SystemExit(f"[train] {err}") from None
    print("[env] gpu", torch.cuda.get_device_name(), flush=True)
    n_trainable = configure_full_finetuning(model)
    print(f"[model] vision tower fully trainable, trainable params={n_trainable / 1e6:.2f}M", flush=True)
    if config["grad_checkpointing"]:
        model.visual.trunk.set_grad_checkpointing(True)  # recompute block activations in the backward pass
        print("[model] activation checkpointing on", flush=True)
    train_tf = make_train_transform(config)
    print(f"[views] train_views={config['train_views']} eval_views={config['eval_views']}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    rng = np.random.default_rng(config["seed"])
    steps_per_epoch = graph.n // config["batch_size"]
    total_steps = config["epochs"] * steps_per_epoch
    opt, sched = build_optimizer(model, config, total_steps)
    stopping = config["early_stopping"]
    metric = stopping["metric"]
    max_steps = args.max_steps
    history_path = os.path.join(out, "history.jsonl")

    # Epoch 0 (pretrained model). Its DataLoader iterator draws a base seed from the global
    # generator, which is part of the stream that seeds the training workers of epoch 1.
    t0 = time.time()
    epoch0_val, _ = validate(model, data, config, device)
    epoch0_minutes = {"train": 0.0, "eval": (time.time() - t0) / 60}
    append_jsonl(history_path, {"epoch": 0, "global_step": 0, "train": None, "val": epoch0_val,
                                "minutes": epoch0_minutes, "batch_sequence_sha256": None})
    print(validation_line(0, metric, epoch0_val, epoch0_minutes), flush=True)

    best, best_epoch, best_val, best_per_query, bad, step, epochs_run = -1.0, 0, None, None, 0, 0, 0
    t_start = time.time()
    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        items = uniform_query_epoch(graph, rng)
        sequence_sha256 = batch_sequence_sha256(items, graph.ids)
        loader = make_train_loader(items, data.paths, graph.idx, train_tf, config)
        model.train()
        loss_sum, n_steps = 0.0, 0
        for qi, pi, xq, xp in loader:
            ids = torch.cat([qi, pi]).to(device)
            x = torch.cat([xq, xp]).to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = encode_image_views(model, x)  # [2B, D] fp32, views aggregated
            loss = contrastive_loss(graph, z, ids, len(qi), config["temperature"])  # fp32, outside autocast
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, config["clip_grad_norm"])
            opt.step()
            sched.step()
            step += 1
            n_steps += 1
            loss_sum += float(loss)
            if step % config["log_every"] == 0:
                lrs = sched.get_last_lr()
                print(f"[train] epoch {epoch} step {step}/{total_steps} contrastive={loss_sum / n_steps:.4f} "
                      f"lr_top={max(lrs):.2e} lr_embed={lrs[0]:.2e} {(time.time() - t0) / n_steps:.2f}s/step",
                      flush=True)
            if max_steps is not None and step >= max_steps:
                break
        t_train = time.time() - t0

        val, per_query = validate(model, data, config, device)
        score = val[metric]["mean"]
        if score > best:
            best, best_epoch, best_val, best_per_query, bad = score, epoch, val, per_query, 0
            save_weights(os.path.join(out, "best_model.safetensors"), model.visual.state_dict(),
                         {"configuration": CONFIGURATION, "epoch": epoch, "global_step": step, "val": val,
                          "config": run_config})
        else:
            bad += 1
        epochs_run = epoch
        minutes = {"train": t_train / 60, "eval": (time.time() - t0 - t_train) / 60}
        append_jsonl(history_path, {"epoch": epoch, "global_step": step,
                                    "train": {"contrastive": loss_sum / max(1, n_steps)}, "val": val,
                                    "minutes": minutes, "batch_sequence_sha256": sequence_sha256})
        print(validation_line(epoch, metric, val, minutes, best, best_epoch), flush=True)
        if epoch >= stopping["min_epochs"] and bad >= stopping["patience"]:
            print(f"[train] early stop at epoch {epoch}", flush=True)
            break
        if max_steps is not None and step >= max_steps:
            print(f"[train] stopped after {step} optimizer steps (--max-steps)", flush=True)
            break

    write_per_query(os.path.join(out, "val_per_query.csv"), best_per_query)
    write_json(os.path.join(out, "summary.json"), {
        "configuration": CONFIGURATION, "best_epoch": best_epoch, "best_val": best_val, "epoch0_val": epoch0_val,
        "epochs_run": epochs_run, "global_step": step, "trainable_params": n_trainable,
        "hours": (time.time() - t_start) / 3600, "overrides": {"epochs": args.epochs, "max_steps": max_steps},
        "config": run_config})
    print(f"[done] best val {metric}={best:.5f} at epoch {best_epoch}; outputs in {out}", flush=True)
