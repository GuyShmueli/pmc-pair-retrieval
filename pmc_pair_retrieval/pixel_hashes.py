"""Decoded-pixel hashes of the images, the source of the pixel-duplicate groups.

    python -m pmc_pair_retrieval.pixel_hashes --data-dir DATA_DIR --check
    python -m pmc_pair_retrieval.pixel_hashes --data-dir DATA_DIR --output pixel_md5.json

The hash of an image is the md5 of ``str(shape)`` followed by the raw bytes of its
decoded RGB pixels (``Image.open(path).convert("RGB")`` as a ``uint8`` array of shape
``(height, width, 3)``), so two images have the same hash exactly when they are
pixel-identical copies. The image_to_image trainer reads ``data/pixel_md5.json`` and
treats pixel-identical images as one image in its loss masks. PMC-PaIR has 49 groups of
pixel-identical images (99 images). With the shipped split, two groups have one image in
train and one in val (6542492_1_2 with 6542490_1_2, and 6542492_1_3 with 6542490_1_3);
no group joins a test image with an image of another split.

The output maps every image id of ``DATA_DIR/image_pairs.json``, in sorted order, to the
hash of ``DATA_DIR/images/<image_id>.jpg``; it is JSON with two-space indentation and no
final newline. ``--check`` hashes the images and compares the hashes with the shipped
``data/pixel_md5.json``, and the whole file byte for byte when all images are present.
The shipped file is reproduced exactly with Pillow 11.3.0; another JPEG decoder may
decode some images to different pixels.

Missing images are an error unless ``--allow-missing-images`` is given; then only the
images that are present are hashed (the public image archive withholds 68 of the 86,893
images).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import multiprocessing
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .split import read_pairs

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
HASH_FILE = REPOSITORY_ROOT / "data" / "pixel_md5.json"
HASH_FILE_SHA256 = "9b6552e6c8a2353e0fd99bc2a6604587d0e4c8b4fc771bb27669543900ad7ab3"
LOG_EVERY = 10000


def pixel_md5(path: str | os.PathLike) -> str:
    """md5 of ``str(shape)`` and the raw bytes of the decoded RGB pixels of an image file."""
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    return hashlib.md5(str(a.shape).encode() + a.tobytes()).hexdigest()


def image_paths(data_dir: str | os.PathLike, ids: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    """Paths of ``DATA_DIR/images/<id>.jpg`` for the ids whose file exists, and the ids whose file is missing."""
    image_dir = Path(data_dir) / "images"
    present = {entry.name[:-len(".jpg")] for entry in os.scandir(image_dir) if entry.name.endswith(".jpg")}
    paths = {i: str(image_dir / f"{i}.jpg") for i in ids if i in present}
    missing = [i for i in ids if i not in present]
    return paths, missing


def compute_hashes(paths: Mapping[str, str], workers: int = 8) -> dict[str, str]:
    """Image id -> ``pixel_md5`` of its path, decoded in ``workers`` processes."""
    ids, files = list(paths), list(paths.values())
    hashes: list[str] = []
    with contextlib.ExitStack() as stack:
        if workers > 1:
            pool = stack.enter_context(multiprocessing.Pool(workers))
            results = pool.imap(pixel_md5, files, chunksize=64)
        else:
            results = map(pixel_md5, files)
        for k, digest in enumerate(results, 1):
            hashes.append(digest)
            if k % LOG_EVERY == 0:
                print(f"[pixel_hashes] {k}/{len(files)} images", flush=True)
    return dict(zip(ids, hashes))


def duplicate_groups(hashes: Mapping[str, str]) -> list[list[str]]:
    """Groups of two or more images with the same hash: sorted id lists, ordered by their first id."""
    by_hash = defaultdict(list)
    for iid in sorted(hashes):
        by_hash[hashes[iid]].append(iid)
    return [group for group in by_hash.values() if len(group) > 1]


def serialize_hashes(hashes: Mapping[str, str]) -> bytes:
    """The bytes of a hash file: image ids in sorted order, two-space indentation, no final newline."""
    return json.dumps({iid: hashes[iid] for iid in sorted(hashes)}, indent=2).encode("ascii")


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments of ``python -m pmc_pair_retrieval.pixel_hashes``."""
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.pixel_hashes",
                                     description="Hash the decoded pixels of the images of image_pairs.json.")
    parser.add_argument("--data-dir", required=True, help="directory with image_pairs.json and images/<image_id>.jpg")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="write the hashes to this new file")
    mode.add_argument("--check", action="store_true", help="compare the hashes with the hash file")
    parser.add_argument("--hash-file", default=str(HASH_FILE),
                        help="hash file compared by --check (default: data/pixel_md5.json)")
    parser.add_argument("--workers", type=int, default=8, help="decoding processes (default: 8)")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="hash only the images that are present")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Hash the images, print the pixel-duplicate groups, then check the hashes or write them."""
    args = parse_args(argv)
    if not (Path(args.data_dir) / "images").is_dir():
        raise SystemExit(f"[pixel_hashes] {Path(args.data_dir) / 'images'} is not a directory")
    ids = sorted({i for pair in read_pairs(args.data_dir) for i in pair})
    paths, missing = image_paths(args.data_dir, ids)
    if missing:
        if not args.allow_missing_images:
            raise SystemExit(f"[pixel_hashes] {len(missing)} of {len(ids)} images of image_pairs.json are missing from "
                             f"{Path(args.data_dir) / 'images'} (example {missing[0]}); pass --allow-missing-images "
                             "to hash only the images that are present.")
        print(f"[pixel_hashes] WARNING: {len(missing)} of {len(ids)} images are missing; only the {len(paths)} "
              "present images are hashed.", flush=True)

    t_start = time.time()
    hashes = compute_hashes(paths, args.workers)
    groups = duplicate_groups(hashes)
    print(f"[pixel_hashes] {len(hashes)} images hashed in {(time.time() - t_start) / 60:.1f} min; {len(groups)} groups "
          f"of pixel-identical images with {sum(map(len, groups))} images", flush=True)
    data = serialize_hashes(hashes)
    digest = hashlib.sha256(data).hexdigest()
    published = ("the published file" if digest == HASH_FILE_SHA256
                 else f"not the published file, whose sha256 is {HASH_FILE_SHA256}")

    if args.check:
        with open(args.hash_file, "rb") as f:
            reference_bytes = f.read()
        reference = json.loads(reference_bytes)
        problems = []
        if sorted(reference) != ids:
            problems.append(f"{args.hash_file} does not cover exactly the {len(ids)} images of image_pairs.json")
        differing = [i for i in hashes if reference.get(i) != hashes[i]]
        if differing:
            problems.append(f"{len(differing)} of {len(hashes)} hashes differ (example {differing[0]})")
        if not missing and data != reference_bytes:
            problems.append(f"{args.hash_file} differs from the computed file")
        if problems:
            raise SystemExit("[pixel_hashes] check failed: " + "; ".join(problems))
        if missing:
            print(f"[pixel_hashes] OK: the {len(hashes)} present images have the hashes of {args.hash_file}; "
                  f"{len(missing)} missing images were not checked", flush=True)
        else:
            print(f"[pixel_hashes] OK: identical to {args.hash_file} (sha256 {digest}, {published})", flush=True)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output, "xb") as f:
                f.write(data)
        except FileExistsError:
            raise SystemExit(f"[pixel_hashes] refusing to overwrite {output}") from None
        print(f"[pixel_hashes] wrote {output} (sha256 {digest}, {published})", flush=True)


if __name__ == "__main__":
    main()
