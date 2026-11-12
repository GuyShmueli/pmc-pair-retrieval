"""Pixel dimensions of the images, the input of the analyses by image geometry.

    python -m pmc_pair_retrieval.image_sizes --data-dir DATA_DIR --check
    python -m pmc_pair_retrieval.image_sizes --data-dir DATA_DIR --output image_sizes.csv

The size of an image is the width and height in pixels of ``DATA_DIR/images/<image_id>.jpg``
as stored, read from the file header (``PIL.Image.open(path).size``). This is the pixel
grid that the evaluation resizes and cuts into its three views; no EXIF orientation is
applied, here or in the evaluation.

``data/image_sizes.csv`` lists every image id of ``DATA_DIR/image_pairs.json`` in sorted
order: UTF-8 text with the header ``image_id,width,height``, one comma-separated line per
image and LF line endings. The file is shipped because the public image archive withholds
68 of the 86,893 images; analyses by image geometry, such as the aspect-ratio subgroups of
the figures, read it (``load_sizes``) and need no image file.

``--check`` reads the sizes from the images and compares them with ``data/image_sizes.csv``,
and the whole file byte for byte when all images are present. Missing images are an error
unless ``--allow-missing-images`` is given; then the sizes of the present images are
compared, and the missing images are reported as not checked.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import time
from collections.abc import Mapping
from pathlib import Path

from PIL import Image

from .pixel_hashes import image_paths
from .split import read_pairs

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SIZES_FILE = REPOSITORY_ROOT / "data" / "image_sizes.csv"
SIZES_FILE_SHA256 = "73c58a2fa0916db5386cfcb7c39cd052ed50b953ab99fead317b835c82f2436d"
HEADER = "image_id,width,height"


def image_size(path: str | os.PathLike) -> tuple[int, int]:
    """``(width, height)`` in pixels of an image file, read from its header."""
    with Image.open(path) as im:
        return im.size


def read_sizes(paths: Mapping[str, str], workers: int = 16) -> dict[str, tuple[int, int]]:
    """Image id -> ``image_size`` of its path, read by ``workers`` threads."""
    with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool:
        sizes = list(pool.map(image_size, paths.values()))
    return dict(zip(paths, sizes))


def serialize_sizes(sizes: Mapping[str, tuple[int, int]]) -> bytes:
    """The bytes of a size file: the header, then ``image_id,width,height`` per image in sorted id order."""
    lines = [HEADER]
    for iid in sorted(sizes):
        if any(c in iid for c in ',"\r\n'):
            raise ValueError(f"image id {iid!r} cannot be written to the size file without quoting")
        width, height = sizes[iid]
        lines.append(f"{iid},{int(width)},{int(height)}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def load_sizes(path: str | os.PathLike = SIZES_FILE) -> dict[str, tuple[int, int]]:
    """Image id -> ``(width, height)`` from a size file in the format of ``serialize_sizes``."""
    with open(path, encoding="utf-8", newline="") as fh:
        lines = fh.read().split("\n")
    if lines[0] != HEADER or lines[-1] != "":
        raise ValueError(f"{path} is not a size file: it must start with the line {HEADER!r} and end with a newline")
    sizes: dict[str, tuple[int, int]] = {}
    for number, line in enumerate(lines[1:-1], start=2):
        fields = line.split(",")
        if len(fields) != 3 or fields[0] in sizes or not all(f.isdigit() and int(f) > 0 for f in fields[1:]):
            raise ValueError(f"{path}, line {number}: expected a new image id and two positive integers, got {line!r}")
        sizes[fields[0]] = (int(fields[1]), int(fields[2]))
    return sizes


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments of ``python -m pmc_pair_retrieval.image_sizes``."""
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.image_sizes",
                                     description="Read the pixel dimensions of the images of image_pairs.json.")
    parser.add_argument("--data-dir", required=True, help="directory with image_pairs.json and images/<image_id>.jpg")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", help="write the sizes to this new file")
    mode.add_argument("--check", action="store_true", help="compare the sizes with the size file")
    parser.add_argument("--sizes-file", default=str(SIZES_FILE),
                        help="size file compared by --check (default: data/image_sizes.csv)")
    parser.add_argument("--workers", type=int, default=16, help="reading threads (default: 16)")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="read only the images that are present")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Read the image sizes, then check them against the size file or write them to a new file."""
    args = parse_args(argv)
    image_dir = Path(args.data_dir) / "images"
    if not image_dir.is_dir():
        raise SystemExit(f"[image_sizes] {image_dir} is not a directory")
    ids = sorted({i for pair in read_pairs(args.data_dir) for i in pair})
    paths, missing = image_paths(args.data_dir, ids)
    if missing:
        if not args.allow_missing_images:
            raise SystemExit(f"[image_sizes] {len(missing)} of {len(ids)} images of image_pairs.json are missing from "
                             f"{image_dir} (example {missing[0]}); pass --allow-missing-images to read only the "
                             "images that are present.")
        print(f"[image_sizes] WARNING: {len(missing)} of {len(ids)} images are missing; only the {len(paths)} "
              "present images are read.", flush=True)

    t_start = time.time()
    sizes = read_sizes(paths, args.workers)
    print(f"[image_sizes] {len(sizes)} images read in {time.time() - t_start:.0f} s", flush=True)
    data = serialize_sizes(sizes)
    digest = hashlib.sha256(data).hexdigest()
    published = ("the published file" if digest == SIZES_FILE_SHA256
                 else f"not the published file, whose sha256 is {SIZES_FILE_SHA256}")

    if args.check:
        with open(args.sizes_file, "rb") as fh:
            reference_bytes = fh.read()
        reference = load_sizes(args.sizes_file)
        problems = []
        if sorted(reference) != ids:
            problems.append(f"{args.sizes_file} does not cover exactly the {len(ids)} images of image_pairs.json")
        differing = [i for i in sizes if reference.get(i) != sizes[i]]
        if differing:
            problems.append(f"{len(differing)} of {len(sizes)} sizes differ (example {differing[0]}: "
                            f"{reference.get(differing[0])} in the file, {sizes[differing[0]]} read)")
        if not missing and data != reference_bytes:
            problems.append(f"{args.sizes_file} differs from the computed file")
        if problems:
            raise SystemExit("[image_sizes] check failed: " + "; ".join(problems))
        if missing:
            print(f"[image_sizes] OK: the {len(sizes)} present images have the sizes of {args.sizes_file}; "
                  f"{len(missing)} missing images were not checked", flush=True)
        else:
            print(f"[image_sizes] OK: identical to {args.sizes_file} (sha256 {digest}, {published})", flush=True)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output, "xb") as fh:
                fh.write(data)
        except FileExistsError:
            raise SystemExit(f"[image_sizes] refusing to overwrite {output}") from None
        print(f"[image_sizes] wrote {output} (sha256 {digest}, {published})", flush=True)


if __name__ == "__main__":
    main()
