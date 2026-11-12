"""Article-disjoint train/validation/test split of PMC-PaIR.

    python -m pmc_pair_retrieval.split --data-dir DATA_DIR --check
    python -m pmc_pair_retrieval.split --data-dir DATA_DIR --output split.json

The split is computed from ``DATA_DIR/image_pairs.json`` alone; the image files are not
read. ``--check`` computes it in memory and compares it byte for byte with the shipped
``data/article_disjoint_split.json``; ``--output`` writes it to a new file.

Procedure. The images are the endpoints of the annotated pairs. Two images are linked
when they form a pair or come from the same article, identified by the PMCID digits
before the first ``_`` of the image id. A connected set of linked images is an article
group. Each group is a sorted list of image ids, and the groups are ordered by their
smallest id in Python string order (``"10000271_1_2"`` precedes ``"9999688_1_1"``). The
groups are shuffled with ``random.Random(42)``; the first ``int(0.8 * N)`` groups form
the training split, the next ``int(0.1 * N)`` the validation split and the remaining
groups the test split. No pair and no article spans two splits, so the positives of
every query lie in the query's own split.

PMC-PaIR has 86,893 images in 86,144 pairs from 56,013 articles. The 18,597 connected
components of the pair graph merge into 13,338 article groups, of which 10,670 / 1,333 /
1,335 hold the 70,552 / 8,245 / 8,096 images of train / val / test.

The split file is a JSON object that maps every image id to ``"train"``, ``"val"`` or
``"test"``, with sorted keys, two-space indentation and a final newline.

``article_groups`` gives every image its article group, named by the smallest image id of
the group, and its article. Both are resampling units of the paired bootstrap intervals
of the results; resampling article groups keeps every query together with its positives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from .dataset import SPLITS, Edge, article_id, validate_split

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = REPOSITORY_ROOT / "data" / "article_disjoint_split.json"
SPLIT_SHA256 = "b0b7be4be1597fa4e557fa4c5a51d0148361c2f253a604cf3df4a95a277c30eb"
SEED = 42
FRACTIONS = (0.8, 0.1, 0.1)


def read_pairs(data_dir: str | os.PathLike) -> list[tuple[str, str]]:
    """The ``pair_id`` of every record of ``DATA_DIR/image_pairs.json``, in file order and as stored."""
    with open(Path(data_dir) / "image_pairs.json", encoding="utf-8") as f:
        records = json.load(f)
    pairs = []
    for record in records:
        a, b = record["pair_id"]
        pairs.append((a, b))
    return pairs


def pair_edges(pairs: Sequence[Sequence[str]]) -> list[Edge]:
    """Sorted undirected pair edges ``(u, v)`` with ``u < v``; a pair that occurs twice is an error."""
    edges = sorted(set(tuple(sorted(pair)) for pair in pairs))
    if len(edges) != len(pairs):
        raise ValueError(f"{len(pairs) - len(edges)} pairs occur more than once in the input")
    return edges


class UnionFind:
    """Disjoint sets of image ids: union by size with ties broken by id, path halving."""

    def __init__(self, ids: Sequence[str]):
        self.parent = {i: i for i in ids}
        self.size = {i: 1 for i in ids}

    def find(self, i: str) -> str:
        """Representative of the set of ``i``; halves the path to it on the way."""
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: str, b: str) -> None:
        """Merge the sets of ``a`` and ``b``; the larger set's representative (on a tie, the larger id) is the root."""
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if (self.size[a], a) < (self.size[b], b):
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _groups(uf: UnionFind, ids: Sequence[str]) -> list[list[str]]:
    """The sets of ``uf`` as lists in the order of ``ids``, ordered by their first member."""
    grouped = defaultdict(list)
    for iid in ids:
        grouped[uf.find(iid)].append(iid)
    return sorted(grouped.values(), key=lambda group: group[0])


def build_groups(image_ids: Iterable[str], direct_edges: Iterable[Edge]) -> tuple[list[list[str]], list[list[str]]]:
    """Connected components of the pair graph and article groups (components merged across shared articles).

    Both are lists of sorted id lists, ordered by their smallest id. Neither the partition
    nor its order depends on the order of ``image_ids`` or ``direct_edges``.
    """
    ids = sorted(set(image_ids))
    uf = UnionFind(ids)
    for a, b in direct_edges:
        if a not in uf.parent or b not in uf.parent:
            raise ValueError(f"Pair endpoint outside image universe: {(a, b)!r}")
        uf.union(a, b)
    components = _groups(uf, ids)
    # Linking every image to the first image of its article merges, transitively, all
    # components that share an article.
    article_representative = {}
    for iid in ids:
        aid = article_id(iid)
        if aid in article_representative:
            uf.union(iid, article_representative[aid])
        else:
            article_representative[aid] = iid
    return components, _groups(uf, ids)


def assign_splits(image_ids: Iterable[str], direct_edges: Iterable[Edge], seed: int = SEED,
                  fractions: Sequence[float] = FRACTIONS) -> dict[str, str]:
    """Image id -> split: the article groups shuffled with ``random.Random(seed)`` and cut by group count.

    The first ``int(fractions[0] * N)`` of the ``N`` shuffled groups are train, the next
    ``int(fractions[1] * N)`` val and the rest test. The result, with keys in sorted
    order, is validated: every image is assigned and no article or pair spans two splits.
    """
    ids = set(image_ids)
    edges = set(map(tuple, direct_edges))
    _, groups = build_groups(ids, edges)
    if (len(fractions) != 3 or any(not math.isfinite(f) or f < 0 for f in fractions)
            or not math.isclose(sum(fractions), 1)):
        raise ValueError("fractions must be three nonnegative numbers summing to one")
    random.Random(seed).shuffle(groups)
    n_train, n_val = int(fractions[0] * len(groups)), int(fractions[1] * len(groups))
    split = {}
    for index, group in enumerate(groups):
        label = "train" if index < n_train else "val" if index < n_train + n_val else "test"
        split.update((iid, label) for iid in group)
    split = dict(sorted(split.items()))
    validate_split(split, ids, edges)
    return split


def make_split(pairs: Sequence[Sequence[str]], seed: int = SEED,
               fractions: Sequence[float] = FRACTIONS) -> dict[str, str]:
    """The article-disjoint split of the images of ``pairs`` (as returned by ``read_pairs``)."""
    edges = pair_edges(pairs)
    return assign_splits({i for edge in edges for i in edge}, edges, seed, fractions)


def serialize_split(split: dict[str, str]) -> bytes:
    """The bytes of a split file: JSON with sorted keys, two-space indentation and a final newline."""
    return (json.dumps(split, indent=2, sort_keys=True) + "\n").encode("ascii")


def article_groups(pairs: Sequence[Sequence[str]]) -> tuple[dict[str, str], dict[str, str]]:
    """Article group and article of every image of ``pairs``.

    Returns ``group`` (image id -> smallest image id of its article group) and
    ``article`` (image id -> the PMCID digits before the first ``_``). These are the
    labels of the published paired bootstrap, whose resampling follows their sorted order.
    """
    edges = pair_edges(pairs)
    _, groups = build_groups({i for edge in edges for i in edge}, edges)
    group = {iid: g[0] for g in groups for iid in g}
    article = {iid: iid.split("_", 1)[0] for iid in group}
    return group, article


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments of ``python -m pmc_pair_retrieval.split``."""
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.split",
                                     description="Compute the article-disjoint split from image_pairs.json.")
    parser.add_argument("--data-dir", required=True, help="directory with image_pairs.json (images are not needed)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="write the split to this new file")
    mode.add_argument("--check", action="store_true",
                      help="compare the computed split byte for byte with the split file")
    parser.add_argument("--split-file", default=str(SPLIT_FILE),
                        help="split file compared by --check (default: data/article_disjoint_split.json)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Compute the split, print its sizes and sha256, then check it or write it."""
    args = parse_args(argv)
    pairs = read_pairs(args.data_dir)
    try:
        split = make_split(pairs)
    except ValueError as err:
        raise SystemExit(f"[split] {err}") from None
    data = serialize_split(split)
    digest = hashlib.sha256(data).hexdigest()

    group, _ = article_groups(pairs)
    group_ids = set(group.values())
    n_groups = Counter(split[g] for g in group_ids)
    n_images = Counter(split.values())
    print(f"[split] {len(split)} images in {len(pairs)} pairs from {len({article_id(i) for i in split})} articles; "
          f"{len(group_ids)} article groups", flush=True)
    for name in SPLITS:
        print(f"[split] {name}: {n_groups[name]} groups, {n_images[name]} images", flush=True)
    print(f"[split] sha256 {digest}", flush=True)

    if args.check:
        with open(args.split_file, "rb") as f:
            reference = f.read()
        problems = []
        if digest != SPLIT_SHA256:
            problems.append(f"the computed split has sha256 {digest}, expected {SPLIT_SHA256}")
        if data != reference:
            problems.append(f"{args.split_file} (sha256 {hashlib.sha256(reference).hexdigest()}) "
                            "differs from the computed split")
        if problems:
            raise SystemExit("[split] check failed: " + "; ".join(problems))
        print(f"[split] OK: identical to {args.split_file} (sha256 {SPLIT_SHA256})", flush=True)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output, "xb") as f:
                f.write(data)
        except FileExistsError:
            raise SystemExit(f"[split] refusing to overwrite {output}") from None
        status = "the published split" if digest == SPLIT_SHA256 else f"NOT the published split ({SPLIT_SHA256})"
        print(f"[split] wrote {output}: {status}", flush=True)


if __name__ == "__main__":
    main()
