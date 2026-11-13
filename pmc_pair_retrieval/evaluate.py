"""Retrieval evaluation of pretrained and fine-tuned BiomedCLIP on the validation or test split.

    python -m pmc_pair_retrieval.evaluate --weights pretrained --retrieval image_to_image \\
        --split test --data-dir DATA_DIR --output-dir eval
    python -m pmc_pair_retrieval.evaluate --weights weights/image_text_to_image.safetensors \\
        --retrieval image_text_to_image --split test --data-dir DATA_DIR --output-dir eval

Settings. The configuration file of the retrieval task, ``configs/<retrieval>.yaml``, is
the only source of the evaluation settings: the split file (``split``), the image size
(``image_size``, 224 px), the number of inference views (``eval_views``, 3), the images
per forward pass (``eval_batch_size``) and, for image_text_to_image, the descriptions per
text batch (``text_batch_size``). The trainer of the configuration validates with the
same values.

Protocol. Queries and gallery are all images of the split. The relevant items of a query
are its direct pair partners in the split. The query itself is scored -1e9 and ranks last;
the other members of its connected component in the pair graph (images linked to it only
through a chain of pairs) are removed from its ranking.

Embeddings. Every image is encoded from its three deterministic 224 px views (see
``views``) with fp32 weights under bf16 autocast; the view embeddings are normalised,
averaged and normalised again. Every distinct description of the split is encoded, in
sorted order, and normalised.

Scores (fp32, TF32 disabled):

    image_to_image       s(q, g) = cos(image_q, image_g)
    image_text_to_image  s(q, g) = cos(image_q, image_g) + t_q . image_g

where ``t_q`` is the mean of the unit embeddings of the query image's distinct
descriptions (all pairs it belongs to), not renormalised.

Metrics per query: Recall@k and Hit@k for k = 1, 5, 10, 100, average precision and nDCG
over the full ranking, and the rank of the first relevant item; the summary adds means
with 95% bootstrap intervals (200 resamples), MRR and the median first relevant rank.

Ranked lists: for every query, the ids of the gallery images at ranks 1 to 10 of the
final ranking from which its metrics are computed, that is, after sorting and after the
removal of the ignored component members, without the query itself.

Outputs in ``<output-dir>/<retrieval>/``, with ``<name>`` = ``pretrained`` or
``fine_tuned`` (``--name``):

    <name>_<split>.jsonl           one row per query, in sorted query-id order
    <name>_<split>.rankings.jsonl  one ranked list per query, in the same order:
                                   {"query_id", "top10": [image ids, rank 1 first]}
    <name>_<split>.summary.json    summary, weights hashes and settings
    <name>_<split>.image_embs.pt   {"ids", "embs" [N, 512] float32, "n_views"}
    <name>_<split>.text_embs.pt    {"texts", "embs"} (image_text_to_image only)

Weights whose sidecar names the other configuration are evaluated only under an output
name other than ``fine_tuned``, the name that ``compare`` reads as the fine-tuned results
of the retrieval task. The pinned model cache is checked (``model.verify_model_cache``)
before the data are read.

The computation is deterministic: on the same GPU model with the same software versions,
the embeddings, the per-query rows and the ranked lists are reproduced bit for bit.
Reference hardware and software: NVIDIA L40S and torch 2.7.1+cu126. The image batch
size (128 images, i.e. 384 view crops per forward pass) and the text batch size are
numerically relevant; the number of loading workers is not.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import dataset, metrics, views
from .model import (MODEL_ID, MODEL_REVISION, PRETRAINED_STATE_DICT_SHA256, WEIGHTS_FILE, WEIGHTS_SHA256,
                    ModelCacheError, describe_weights, load_biomedclip, load_weights, read_sidecar, sha256_file,
                    state_dict_sha256, verify_model_cache)
from .train import load_config, resolve_data_paths

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPOSITORY_ROOT / "configs"
# The split file of both configurations; the default of load_graph.
DEFAULT_SPLIT_FILE = REPOSITORY_ROOT / "data" / "article_disjoint_split.json"
RETRIEVALS = ("image_to_image", "image_text_to_image")
SCORE_DEFINITIONS = {
    "image_to_image": "cos(image_q, image_g)",
    "image_text_to_image": "cos(image_q, image_g) + t_q . image_g",
}
QUERY_TEXT_DEFINITION = ("t_q = mean of the unit embeddings of the query image's distinct descriptions "
                         "(not renormalised)")


# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
def evaluation_settings(retrieval: str) -> dict:
    """Evaluation settings of a retrieval task, read from its configuration file ``configs/<retrieval>.yaml``.

    Returns ``config`` (the repository-relative configuration path), ``split_file`` (the
    absolute path of its ``split``), ``image_size``, ``views`` (``eval_views``),
    ``image_batch_size`` (``eval_batch_size``) and ``text_batch_size`` (``text_batch_size``
    for image_text_to_image, ``None`` for image_to_image, which encodes no text).
    """
    path = CONFIG_DIR / f"{retrieval}.yaml"
    config = load_config(path)
    if config["configuration"] != retrieval:
        raise SystemExit(f"{path}: 'configuration' is {config['configuration']!r}, expected {retrieval!r}")
    if config["eval_views"] != views.N_VIEWS or config["image_size"] != 224:
        raise SystemExit(f"{path}: eval_views must be {views.N_VIEWS} and image_size 224, the inference settings of "
                         "the published evaluation")
    return {"config": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "split_file": resolve_data_paths(config)["split"],
            "image_size": config["image_size"], "views": config["eval_views"],
            "image_batch_size": config["eval_batch_size"],
            "text_batch_size": config["text_batch_size"] if retrieval == "image_text_to_image" else None}


# --------------------------------------------------------------------------------------
# Pair graph
# --------------------------------------------------------------------------------------
class RetrievalGraph:
    """Pair graph of the data with descriptions, image paths and the split assignment."""

    def __init__(self, pairs: Sequence[Sequence[str]], captions: Sequence[Sequence[str]],
                 id2path: dict[str, str], split_map: dict[str, str]):
        all_ids, direct_edges, id2texts, _ = dataset.load_pairs_and_captions(pairs, captions)
        absent = sorted(i for i in all_ids if i not in id2path)
        if absent:
            raise ValueError(f"{len(absent)} images of the pairs have no image file; example {absent[0]}")
        dataset.validate_split(split_map, all_ids, direct_edges)
        self.all_ids = all_ids
        self.id2path = {i: id2path[i] for i in all_ids}
        self.id2texts = {i: list(id2texts.get(i, [])) for i in all_ids}
        self.split = split_map
        self.adj: dict[str, set[str]] = defaultdict(set)
        for u, v in direct_edges:
            self.adj[u].add(v)
            self.adj[v].add(u)
        # Component membership equals the full transitive closure of the pairs.
        self.comp = dataset.connected_components(all_ids, direct_edges)
        self.comp_members: dict[str, set[str]] = defaultdict(set)
        for i, c in self.comp.items():
            self.comp_members[c].add(i)

    def split_ids(self, split: str) -> list[str]:
        """Sorted image ids of a split; they are both the queries and the gallery."""
        return sorted(i for i in self.all_ids if self.split[i] == split)

    def relevance(self, ids: Sequence[str]) -> tuple[list[list[int]], list[list[int]]]:
        """Per query of ``ids`` (queries = gallery): sorted gallery indices of the positives and of the ignored items.

        Positives are the direct pair partners in the gallery; ignored items are the other
        members of the query's connected component in the gallery, except the query.
        """
        index = {i: k for k, i in enumerate(ids)}
        idset = set(ids)
        positives, ignored = [], []
        for q in ids:
            pos = sorted(self.adj[q] & idset)
            comp = (self.comp_members[self.comp[q]] & idset) - set(pos) - {q}
            if not pos:
                raise ValueError(f"query {q} has no pair partner in its gallery")
            positives.append([index[x] for x in pos])
            ignored.append(sorted(index[x] for x in comp))
        return positives, ignored

    def split_texts(self, ids: Sequence[str]) -> list[str]:
        """Sorted distinct descriptions of the images ``ids``."""
        return sorted({t for i in ids for t in self.id2texts[i]})

    def query_text_indices(self, ids: Sequence[str], texts: Sequence[str]) -> list[list[int]]:
        """Per query, the indices into ``texts`` of its distinct descriptions, in sorted order."""
        tindex = {t: k for k, t in enumerate(texts)}
        return [[tindex[t] for t in self.id2texts[q]] for q in ids]


def load_graph(data_dir: str | os.PathLike, split_file: str | os.PathLike = DEFAULT_SPLIT_FILE,
               allow_missing_images: bool = False) -> RetrievalGraph:
    """Build the retrieval graph from ``DATA_DIR/image_pairs.json``, ``DATA_DIR/images/`` and the split file.

    Missing images raise ``ValueError`` unless ``allow_missing_images``; then the pairs
    with a missing image are dropped, which removes positives and gallery images, so the
    results differ from the reported ones.
    """
    pairs, captions, id2path = dataset.load_pairs(data_dir)
    missing = sorted({i for pair in pairs for i in pair if i not in id2path})
    if missing:
        if not allow_missing_images:
            raise ValueError(f"{len(missing)} images of image_pairs.json are missing from {data_dir}/images "
                             f"(example {missing[0]}). The published results use all "
                             f"{dataset.N_PUBLISHED_IMAGES} images; pass --allow-missing-images to evaluate "
                             "without them.")
        keep = [k for k, (a, b) in enumerate(pairs) if a in id2path and b in id2path]
        print(f"[evaluate] WARNING: {len(missing)} images are missing; dropped {len(pairs) - len(keep)} of "
              f"{len(pairs)} pairs. Results will differ from the published ones.", flush=True)
        pairs = [pairs[k] for k in keep]
        captions = [captions[k] for k in keep]
    return RetrievalGraph(pairs, captions, id2path, dataset.load_split(split_file))


# --------------------------------------------------------------------------------------
# Embeddings (CUDA, bf16 autocast)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def encode_images(model: torch.nn.Module, ids: Sequence[str], id2path: dict[str, str], batch_size: int,
                  num_workers: int = 6, size: int = 224) -> torch.Tensor:
    """Unit image embeddings ``[len(ids), D]`` (float32, CPU) in the order of ``ids``, from the three views."""
    ds = dataset.ImageDataset(ids, id2path, views.MultiViewEvalTransform(size))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                    persistent_workers=False)
    embs, order = [], []
    for iids, imgs in dl:
        imgs = imgs.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            z = views.encode_image_views(model, imgs)
        embs.append(z.float().cpu())
        order.extend(iids)
    if order != list(ids):
        raise RuntimeError("image embeddings are out of order")
    return torch.cat(embs)


@torch.no_grad()
def encode_texts(model: torch.nn.Module, tokenizer, texts: Sequence[str], batch_size: int) -> torch.Tensor:
    """Unit text embeddings ``[len(texts), D]`` (float32, CPU) in the order of ``texts``."""
    out = []
    for i in range(0, len(texts), batch_size):
        tok = tokenizer(list(texts[i:i + batch_size])).cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            tz = model.encode_text(tok)
        out.append(F.normalize(tz.float(), dim=1).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------------------
# Scores and per-query metrics
# --------------------------------------------------------------------------------------
@contextlib.contextmanager
def _full_fp32_matmul():
    # Disable TF32 to preserve full-fp32 scores and the order of near ties.
    # Callers may have enabled it.
    saved = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(saved)


@torch.no_grad()
def score_matrix(image_embs: torch.Tensor, text_embs: torch.Tensor | None = None,
                 query_texts: Sequence[Sequence[int]] | None = None, device: str = "cuda") -> torch.Tensor:
    """Query x gallery scores (queries = gallery = the rows of ``image_embs``), float32 on ``device``.

    Without text: ``G G^T``. With ``text_embs`` and ``query_texts`` (per query, the rows of
    ``text_embs`` holding its descriptions): ``G G^T + M G^T``, where row q of ``M`` is
    the mean of the query's description embeddings, not renormalised. The saved scores
    use GPU computation. In these evaluations, CPU computation orders a few near ties
    differently, changing some average precision, nDCG and first-rank values but not
    Recall@k.
    """
    with _full_fp32_matmul():
        G = image_embs.float().to(device)
        scores = G @ G.t()
        if text_embs is not None:
            T = text_embs.float().to(device)
            mean = torch.stack([T[tl].mean(0) for tl in query_texts])
            scores = scores + mean @ G.t()
    return scores


def rank_queries(scores: torch.Tensor, ids: Sequence[str], positives: Sequence[Sequence[int]],
                 ignored: Sequence[Sequence[int]]) -> tuple[list[dict], list[dict]]:
    """Per-query metric rows and ranked lists, both taken from the final ranking of each query.

    The final ranking of a query orders the gallery by ``np.argsort`` of the negated
    float32 scores (numpy's default sort, which decides the order of exact ties), with the
    query itself scored -1e9 so that it ranks last; its ignored items are removed after
    sorting. Returns two lists in the order of ``ids``:

    - rows: ``query_id``, ``n_pos``, R@k and Hit@k, ``mAP``, ``nDCG`` and ``first_rel_rank``;
    - ranked lists: ``query_id`` and ``top10``, the ids of the gallery images at ranks 1
      to 10 of the final ranking, rank 1 first, without the query itself.
    """
    S_np = scores.detach().float().cpu().numpy()
    rows, rankings = [], []
    n = len(ids)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        sims = S_np[i].copy()
        sims[i] = -1e9
        order = np.argsort(-sims)
        ign = ignored[i]
        if ign:
            keep[ign] = False
            order = order[keep[order]]
            keep[ign] = True
        metr = metrics.retrieval_for_query(sims, order, positives[i])
        rel = np.zeros(n, dtype=bool)
        rel[positives[i]] = True
        hit_pos = np.nonzero(rel[order])[0]
        first_rank = int(hit_pos[0]) + 1 if len(hit_pos) else -1
        rows.append({"query_id": ids[i], "n_pos": len(positives[i]), **{k: float(v) for k, v in metr.items()},
                     "first_rel_rank": first_rank})
        rankings.append({"query_id": ids[i], "top10": [ids[j] for j in order[order != i][:10]]})
    return rows, rankings


def per_query_metrics(scores: torch.Tensor, ids: Sequence[str], positives: Sequence[Sequence[int]],
                      ignored: Sequence[Sequence[int]]) -> list[dict]:
    """The per-query metric rows of ``rank_queries``, one per query in the order of ``ids``."""
    return rank_queries(scores, ids, positives, ignored)[0]


def write_jsonl(path: str | os.PathLike, rows: Sequence[dict]) -> None:
    """Write one JSON object per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments of ``python -m pmc_pair_retrieval.evaluate``."""
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.evaluate",
                                     description="Evaluate pretrained or fine-tuned BiomedCLIP on one split, with the "
                                                 "evaluation settings of configs/<retrieval>.yaml.")
    parser.add_argument("--weights", required=True,
                        help="'pretrained' or a fine-tuned .safetensors file (with its .json sidecar)")
    parser.add_argument("--retrieval", required=True, choices=RETRIEVALS,
                        help="image_to_image: image queries; image_text_to_image: image plus description queries")
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument("--data-dir", required=True, help="directory with image_pairs.json and images/<image_id>.jpg")
    parser.add_argument("--output-dir", required=True, help="outputs go to OUTPUT_DIR/<retrieval>/")
    parser.add_argument("--name", default=None,
                        help="output name; default 'pretrained' or 'fine_tuned' according to --weights")
    parser.add_argument("--num-workers", type=int, default=6,
                        help="image loading workers (default: 6; does not change the results)")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="evaluate without missing images (results then differ from the published ones)")
    return parser.parse_args(argv)


def _load_model(weights: str, retrieval: str):
    """Model in eval mode, tokenizer and the description of the weights for the summary."""
    model, tokenizer = load_biomedclip("cuda")
    if weights == "pretrained":
        info = {"model": MODEL_ID, "revision": MODEL_REVISION, "file": WEIGHTS_FILE, "kind": "full_model",
                "sha256": WEIGHTS_SHA256, "state_dict_sha256": PRETRAINED_STATE_DICT_SHA256,
                "model_state_dict_sha256": PRETRAINED_STATE_DICT_SHA256}
    else:
        loaded = load_weights(model, weights)
        if loaded["kind"] == "vision_tower" and retrieval == "image_text_to_image":
            raise SystemExit(f"{weights} holds a vision tower only (the text tower would be the pretrained one); "
                             "it can only be evaluated with --retrieval image_to_image.")
        # The weights file and, from its sidecar, the selected epoch and selection value; compare
        # links the fine-tuned results to these values.
        info = {"file": str(Path(weights).resolve()), "kind": loaded["kind"], "n_tensors": loaded["n_tensors"],
                "sha256": loaded["sha256"], "state_dict_sha256": loaded["state_dict_sha256"],
                "model_state_dict_sha256": state_dict_sha256(model.state_dict()),
                **describe_weights(loaded["metadata"] or {})}
    model.eval()
    return model, tokenizer, info


def main(argv: list[str] | None = None) -> None:
    """Evaluate one model on one split; write the per-query rows, ranked lists, summary and embeddings."""
    args = parse_args(argv)
    settings = evaluation_settings(args.retrieval)
    # Disable TF32 and cuDNN autotuning explicitly instead of inheriting backend settings.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.weights != "pretrained" and not os.path.isfile(args.weights):
        raise SystemExit(f"Weights file not found: {args.weights}")
    name = args.name or ("pretrained" if args.weights == "pretrained" else "fine_tuned")
    out_dir = Path(args.output_dir) / args.retrieval
    suffixes = {"rows": ".jsonl", "rankings": ".rankings.jsonl", "summary": ".summary.json",
                "image_embs": ".image_embs.pt"}
    if args.retrieval == "image_text_to_image":
        suffixes["text_embs"] = ".text_embs.pt"
    outputs = {key: out_dir / f"{name}_{args.split}{suffix}" for key, suffix in suffixes.items()}
    existing = [str(p) for p in outputs.values() if p.exists()]
    if existing:
        raise SystemExit(f"Refusing to overwrite existing outputs: {existing}")
    if args.weights != "pretrained" and name == "fine_tuned":
        # compare takes the default output name for the fine-tuned results of this retrieval task.
        trained_for = (read_sidecar(args.weights) or {}).get("configuration")
        if trained_for not in (None, args.retrieval):
            raise SystemExit(f"{args.weights} holds weights trained for {trained_for}; to evaluate them with "
                             f"--retrieval {args.retrieval}, pass --name with an output name other than 'fine_tuned'.")
    try:
        verify_model_cache()
    except ModelCacheError as err:
        raise SystemExit(f"[evaluate] {err}") from None
    t_start = time.time()

    try:
        graph = load_graph(args.data_dir, settings["split_file"], args.allow_missing_images)
        ids = graph.split_ids(args.split)
        positives, ignored = graph.relevance(ids)
    except ValueError as err:
        raise SystemExit(f"[evaluate] {err}") from None
    n_ignored = sum(len(g) for g in ignored)
    print(f"[evaluate] {args.split}: {len(ids)} queries (= gallery), {sum(len(p) for p in positives)} positive and "
          f"{n_ignored} ignored entries", flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("The evaluation needs a CUDA GPU.")

    try:
        model, tokenizer, weights_info = _load_model(args.weights, args.retrieval)
    except ModelCacheError as err:
        raise SystemExit(f"[evaluate] {err}") from None
    print(f"[evaluate] weights: {args.weights} (state dict sha256 {weights_info['state_dict_sha256']}); settings of "
          f"{settings['config']}: {settings['views']} views, {settings['image_batch_size']} images per batch"
          + (f", {settings['text_batch_size']} descriptions per batch" if settings["text_batch_size"] else ""),
          flush=True)

    t0 = time.time()
    image_embs = encode_images(model, ids, graph.id2path, settings["image_batch_size"], args.num_workers,
                               settings["image_size"])
    print(f"[evaluate] encoded {len(ids)} images x {settings['views']} views in {time.time() - t0:.0f} s", flush=True)
    text_embs = query_texts = texts = None
    if args.retrieval == "image_text_to_image":
        texts = graph.split_texts(ids)
        t0 = time.time()
        text_embs = encode_texts(model, tokenizer, texts, settings["text_batch_size"])
        query_texts = graph.query_text_indices(ids, texts)
        print(f"[evaluate] encoded {len(texts)} descriptions in {time.time() - t0:.0f} s", flush=True)

    t0 = time.time()
    scores = score_matrix(image_embs, text_embs, query_texts, device="cuda")
    rows, rankings = rank_queries(scores, ids, positives, ignored)
    del scores
    summary = metrics.summarize(rows)
    summary.update({
        "retrieval": args.retrieval, "split": args.split, "name": name, "views": settings["views"],
        "gallery_size": len(ids), "n_ignored_entries": n_ignored,
        "score_definition": SCORE_DEFINITIONS[args.retrieval],
        "query_text": QUERY_TEXT_DEFINITION if texts is not None else None,
        "n_texts": len(texts) if texts is not None else None,
        "weights": weights_info,
        "settings": {"config": settings["config"], "image_size": settings["image_size"],
                     "image_batch_size": settings["image_batch_size"], "text_batch_size": settings["text_batch_size"],
                     "precision": "fp32 weights, bf16 autocast encoders, fp32 scores without TF32"},
        "data": {"split_file_sha256": sha256_file(settings["split_file"]),
                 "image_pairs_sha256": sha256_file(Path(args.data_dir) / "image_pairs.json"),
                 "n_images": len(graph.all_ids)},
        "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "numpy": np.__version__, "python": platform.python_version()},
        "seconds": time.time() - t_start,
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "embs": image_embs, "n_views": settings["views"]}, outputs["image_embs"])
    if texts is not None:
        torch.save({"texts": texts, "embs": text_embs}, outputs["text_embs"])
    write_jsonl(outputs["rows"], rows)
    write_jsonl(outputs["rankings"], rankings)
    with open(outputs["summary"], "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"[evaluate] {args.retrieval} {name} {args.split}: R@10 {100 * summary['R@10']['mean']:.2f} "
          f"Hit@10 {100 * summary['Hit@10']['mean']:.2f} mAP {100 * summary['mAP']['mean']:.2f} "
          f"MRR {summary['MRR']:.4f} (scored in {time.time() - t0:.0f} s); wrote {outputs['rows']} and "
          f"{outputs['rankings']}", flush=True)


if __name__ == "__main__":
    main()
