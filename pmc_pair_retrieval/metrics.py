"""Per-query retrieval metrics, bootstrap confidence intervals and paired differences.

Retrieval protocol (shared by the evaluator and the image_text_to_image validation):
the gallery is the split's own images; the relevant items of a query are its direct
pair partners in the gallery; the query itself is scored -1e9 and so ranks last; other
members of the query's connected component that are not direct partners are removed
from the ranking after sorting. Ranks come from ``np.argsort(-sims)`` with numpy's
default (unstable) sort, which decides the order of exactly tied scores.

With float32 scores the relevance vectors are float32, so per-query Recall@k values are
float32 numbers converted to Python floats (for example 4/5 -> 0.800000011920929).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

K_LIST = (1, 5, 10, 100)
SUMMARY_METRICS = ("R@1", "R@5", "R@10", "R@100", "Hit@1", "Hit@5", "Hit@10", "Hit@100", "mAP", "nDCG")
# Generator seed of the paired bootstrap used for the reported paired differences.
PAIRED_BOOTSTRAP_SEED = 1406745469314695659


def average_precision_binary(rel_sorted: np.ndarray) -> float:
    """Average precision of a ranked binary relevance vector (NaN without relevant items)."""
    if rel_sorted.sum() == 0:
        return np.nan
    cum = np.cumsum(rel_sorted)
    ranks = np.arange(1, len(rel_sorted) + 1)
    prec = cum / ranks
    return float((prec * rel_sorted).sum() / rel_sorted.sum())


def dcg_at_k(rel: np.ndarray) -> float:
    """Discounted cumulative gain of a ranked relevance vector."""
    denom = np.log2(np.arange(2, len(rel) + 2))
    return float((rel / denom).sum())


def ndcg(rel_sorted: np.ndarray) -> float:
    """Normalised DCG over the full ranking."""
    dcg = dcg_at_k(rel_sorted)
    ideal = dcg_at_k(np.sort(rel_sorted)[::-1])
    return 0.0 if ideal == 0 else dcg / ideal


def retrieval_for_query(sims: np.ndarray, order: np.ndarray, positives: Sequence[int],
                        k_list: Sequence[int] = K_LIST) -> dict[str, float]:
    """Recall@k, Hit@k, average precision and nDCG of one query.

    ``sims`` are the query's gallery scores (only their length and dtype are used),
    ``order`` the gallery indices in ranked order (ignored items already removed) and
    ``positives`` the gallery indices of the query's direct partners.
    """
    rel_for_ap = np.zeros_like(sims)
    rel_for_ap[positives] = 1.0
    rel_graded = rel_for_ap.copy()

    rel_sorted_bin = rel_for_ap[order]
    rel_sorted_grd = rel_graded[order]

    out: dict[str, float] = {}
    for k in k_list:
        out[f"R@{k}"] = float(rel_sorted_bin[:k].sum() / max(1, rel_sorted_bin.sum()))
        out[f"Hit@{k}"] = float(rel_sorted_bin[:k].max())
    out["mAP"] = average_precision_binary(rel_sorted_bin)
    out["nDCG"] = ndcg(rel_sorted_grd)
    return out


def bootstrap_ci(values: Sequence[float], n_boot: int = 200, alpha: float = 0.05, seed: int = 42
                 ) -> tuple[float, float, float]:
    """``(mean, lo, hi)``: NaN-aware mean and percentile bootstrap interval over queries.

    A fresh ``np.random.default_rng(seed)`` is created on every call, so every metric of
    a cell (and every cell with the same number of queries) uses the same resamples.
    """
    rng = np.random.default_rng(seed)
    vals = np.array(values, dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(vals, size=len(vals), replace=True)
        boots.append(np.nanmean(samp))
    mean = float(np.nanmean(vals))
    lo = float(np.nanpercentile(boots, 100 * alpha / 2))
    hi = float(np.nanpercentile(boots, 100 * (1 - alpha / 2)))
    return (mean, lo, hi)


def summarize(rows: Sequence[dict], n_boot: int = 200, seed: int = 42) -> dict:
    """Summary of per-query rows: mean and 95% bootstrap interval per metric, MRR and median first rank.

    Each row needs the keys of ``SUMMARY_METRICS`` and ``first_rel_rank``.
    """
    out = {"n_queries": len(rows)}
    for k in SUMMARY_METRICS:
        vals = [r[k] for r in rows]
        m, lo, hi = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        out[k] = {"mean": float(m), "lo": float(lo), "hi": float(hi)}
    ranks = [r["first_rel_rank"] for r in rows]
    out["MRR"] = float(np.mean([1.0 / r for r in ranks]))
    out["median_first_relevant_rank"] = float(np.median(ranks))
    return out


def paired_group_ci(diff: np.ndarray, groups: np.ndarray, n_boot: int = 5000,
                    seed: int = PAIRED_BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% interval of the mean paired difference by a query-weighted group bootstrap.

    Groups (sorted by ``np.unique``) are resampled with replacement; each replicate is
    the mean difference over all queries of the drawn groups. Replicates are drawn in
    chunks of 64 as int32 indices: the chunking and the dtype are part of the random
    stream. ``diff`` and ``groups`` must be ordered by sorted query id.
    """
    names, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(names))
    counts = np.bincount(inv, minlength=len(names)).astype(float)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for start in range(0, n_boot, 64):
        m = min(64, n_boot - start)
        idx = rng.integers(0, len(names), size=(m, len(names)), dtype=np.int32)
        values[start:start + m] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)
