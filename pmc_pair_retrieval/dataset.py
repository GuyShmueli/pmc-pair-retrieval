"""Public-data loading, the article-disjoint split and the pair graph.

The data directory holds the public release files::

    DATA_DIR/image_pairs.json        86,144 records {"pair_id": [a, b], "modality", "anatomy", "diagnosis"}
    DATA_DIR/images/<image_id>.jpg   one JPEG per image id

Image ids have the form ``<PMCID digits>_<case>_<figure>``. Each pair carries one
description, ``"{diagnosis} in the {anatomy} on {modality}."``; an image collects the
descriptions of all its pairs.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence

from PIL import Image
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")
# Images used by the published runs; the public image archive withholds 68 of them.
N_PUBLISHED_IMAGES = 86893
# An undirected pair edge (u, v) with u < v.
Edge = tuple[str, str]


# --------------------------------------------------------------------------------------
# Public release files
# --------------------------------------------------------------------------------------
def pair_caption(record: dict) -> str:
    """The description of one pair record."""
    return f"{record['diagnosis']} in the {record['anatomy']} on {record['modality']}."


def load_pairs(data_dir: str | os.PathLike) -> tuple[list[tuple[str, str]], list[list[str]], dict[str, str]]:
    """Read ``image_pairs.json`` and index ``images/``.

    Returns, in file order and unfiltered:

    - ``pairs``: the ``pair_id`` of every record, as a tuple of two image ids;
    - ``captions``: for every record, a single-element list with its description;
    - ``id2path``: image id -> absolute path of every ``images/<id>.jpg`` that is present,
      in sorted id order.

    The strings are returned exactly as stored in the release file. Pairs with a missing
    image are kept, because the image_text_to_image training-pair order depends on the
    complete pair list in file order. Callers compare the pairs with ``id2path``: they
    refuse missing images unless explicitly allowed (``--allow-missing-images``), and each
    caller documents how it then leaves the missing images out. Reproducing the published
    numbers needs all 86,893 images of the pairs.
    """
    data_dir = os.path.abspath(os.fspath(data_dir))
    with open(os.path.join(data_dir, "image_pairs.json"), encoding="utf-8") as f:
        records = json.load(f)
    image_dir = os.path.join(data_dir, "images")
    names = sorted(entry.name for entry in os.scandir(image_dir) if entry.name.endswith(".jpg"))
    id2path = {name[:-len(".jpg")]: os.path.join(image_dir, name) for name in names}

    pairs: list[tuple[str, str]] = []
    captions: list[list[str]] = []
    for record in records:
        a, b = record["pair_id"]
        pairs.append((a, b))
        captions.append([pair_caption(record)])
    return pairs, captions, id2path


def load_pairs_and_captions(pairs: Sequence[Sequence[str]], captions: Sequence[Sequence[str]]
                            ) -> tuple[set[str], set[Edge], dict[str, list[str]], dict[Edge, list[str]]]:
    """Image set, undirected pair edges and descriptions, as the published training code built them.

    ``pairs`` and ``captions`` are the outputs of ``load_pairs`` (same row order). Returns
    ``all_ids`` (set of image ids), ``direct_edges`` (set of ``(u, v)`` with ``u < v``),
    ``id2texts`` (image id -> sorted distinct descriptions of all its pairs; the
    image_text_to_image query text, in the training validation and in the evaluation, is
    the mean of the embeddings of all of them) and ``pair2texts`` (edge -> sorted distinct
    descriptions).

    The sets are filled by the same statements in the same row order as in the published
    runs, so their iteration order under a fixed ``PYTHONHASHSEED`` (and Python 3.11) is
    the same as there; the image_text_to_image training-pair order depends on it.
    """
    if len(pairs) != len(captions):
        raise ValueError(f"pairs ({len(pairs)}) and captions ({len(captions)}) length mismatch")

    all_ids = set()
    direct_edges = set()
    id2texts = defaultdict(list)
    pair2texts = defaultdict(list)

    for (ab, caps) in zip(pairs, captions):
        a, b = str(ab[0]).strip(), str(ab[1]).strip()
        all_ids.update([a, b])
        u, v = (a, b) if a < b else (b, a)
        direct_edges.add((u, v))

        for c in caps:
            c = str(c).strip()
            if not c:
                continue
            id2texts[a].append(c)
            id2texts[b].append(c)
            pair2texts[(u, v)].append(c)

    id2texts = {k: sorted(set(v)) for k, v in id2texts.items()}
    pair2texts = {k: sorted(set(v)) for k, v in pair2texts.items()}
    return all_ids, direct_edges, id2texts, pair2texts


class ImageDataset(Dataset):
    """``(image_id, transform(RGB image))`` for a list of image ids, in list order."""

    def __init__(self, ids: Iterable[str], id2path: dict[str, str], transform: Callable):
        self.ids = list(ids)
        self.id2path = id2path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        iid = self.ids[idx]
        img = Image.open(self.id2path[iid]).convert("RGB")
        return iid, self.transform(img)


# --------------------------------------------------------------------------------------
# Article-disjoint split
# --------------------------------------------------------------------------------------
def article_id(image_id: str) -> str:
    """Canonical PMCID (``'PMC' + digits``) of the source article of an image id."""
    match = re.fullmatch(r"(?:PMC)?([1-9]\d*)_[^/\\]+", str(image_id))
    if not match:
        raise ValueError(f"Cannot identify source article for image {image_id!r}")
    return "PMC" + match.group(1)


def load_split(path: str | os.PathLike) -> dict[str, str]:
    """Image id -> ``'train' | 'val' | 'test'`` from a JSON object that maps image ids to split names."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("The split file must be a JSON object mapping image ids to split names.")
    norm: dict[str, str] = {}
    for iid, sp in obj.items():
        s = str(sp).strip().lower()
        if s in ("train", "tr"):
            norm[str(iid).strip()] = "train"
        elif s in ("val", "valid", "validation", "dev"):
            norm[str(iid).strip()] = "val"
        elif s in ("test", "te"):
            norm[str(iid).strip()] = "test"
        else:
            raise ValueError(f"Invalid split label for id='{iid}': '{sp}'. Expected train/val/test.")
    return norm


def validate_split(split_map: dict[str, str], image_ids: Iterable[str], direct_edges: Iterable[Edge]) -> dict:
    """Fail closed on missing ids, invalid labels, articles spanning splits or pair edges crossing splits.

    Extra ids in ``split_map`` are allowed (subset runs); article integrity is checked on
    all assignments of ``split_map``. Returns counts of the checked items.
    """
    ids = set(image_ids)
    missing = ids - split_map.keys()
    if missing:
        raise ValueError(f"Split missing {len(missing)} IDs; example {min(missing)}")
    articles = defaultdict(set)
    for iid, split in split_map.items():
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r} for {iid}")
        articles[article_id(iid)].add(split)
    leaking = {a: sorted(s) for a, s in articles.items() if len(s) > 1}
    if leaking:
        raise ValueError(f"{len(leaking)} source articles span splits; example {next(iter(leaking.items()))}")
    n_edges = 0
    for a, b in direct_edges:
        if a not in ids or b not in ids:
            raise ValueError(f"Pair endpoint outside image universe: {(a, b)!r}")
        if split_map[a] != split_map[b]:
            raise ValueError(f"Pair-graph component spans splits at edge {(a, b)!r}")
        n_edges += 1
    return {"assigned_images": len(split_map), "active_images": len(ids),
            "source_articles": len(articles), "checked_pair_edges": n_edges,
            "articles_spanning_splits": 0, "pair_edges_spanning_splits": 0}


# --------------------------------------------------------------------------------------
# Connected components of the pair graph
# --------------------------------------------------------------------------------------
class _UnionFind:
    """Disjoint sets of image ids, with path compression and union by rank."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        """Root of the set of ``x`` (a new ``x`` becomes a singleton set); compresses the path to it."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        """Merge the sets of ``a`` and ``b``; the root of higher rank becomes the root of both."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def connected_components(all_ids: Iterable[str], direct_edges: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Image id -> component id (the smallest image id of its connected component).

    Edges and ids are processed in sorted order, so the result does not depend on set
    iteration order. A component equals the set of images linked by the full transitive
    closure of the pair edges.
    """
    uf = _UnionFind()
    for (u, v) in sorted(direct_edges):
        uf.union(u, v)

    root2members = defaultdict(list)
    for i in sorted(all_ids):
        root2members[uf.find(i)].append(i)

    id2comp: dict[str, str] = {}
    for members in root2members.values():
        cid = min(members)
        for i in members:
            id2comp[i] = cid
    return id2comp
