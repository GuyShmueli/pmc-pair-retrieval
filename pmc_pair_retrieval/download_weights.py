"""Download the released fine-tuned weights into ``weights/`` and verify them.

    python -m pmc_pair_retrieval.download_weights
    python -m pmc_pair_retrieval.download_weights --configuration image_to_image
    python -m pmc_pair_retrieval.download_weights --check

The two ``safetensors`` files are not stored in Git. Their names and sha256 values are read
from the sidecars ``weights/<configuration>.json`` and must agree with
``WEIGHTS_SHA256SUMS``; if they do not, the command stops. The files come from the Zenodo
record ``WEIGHTS_RECORD_ID`` or from ``--base-url``. While ``WEIGHTS_RECORD_ID`` is
``None`` the record has not been published, and without ``--base-url`` the command exits
with a message that says so.

A file is streamed to ``weights/<name>.part`` and hashed on the way. It replaces
``weights/<name>``, in one atomic step, only if its size and sha256 are the expected ones;
on any failure the ``.part`` file is deleted and the exit status is not zero. A file
already in ``weights/`` is hashed first. With the expected sha256 it is left alone; with
any other it is never overwritten, and the command stops and says what to do. A ``.part``
file that exists before a download starts belongs to another download or to a killed one:
the command leaves it in place and asks to delete it.

``--check`` hashes the local files only and uses no network. The module needs only the
Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import os
import re
import signal
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NamedTuple

from .model import sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = REPOSITORY_ROOT / "weights"
CHECKSUM_FILE = REPOSITORY_ROOT / "WEIGHTS_SHA256SUMS"
CONFIGURATIONS = ("image_to_image", "image_text_to_image")

# Id of the Zenodo record that holds the two weight files; None until the record is published.
WEIGHTS_RECORD_ID = None
ZENODO_FILE_URL = "https://zenodo.org/records/{record}/files/{name}?download=1"
# The sidecars record no file size, so the exact sizes are kept here.
WEIGHT_FILE_BYTES = {"image_to_image.safetensors": 344782408, "image_text_to_image.safetensors": 783652708}

CHUNK_BYTES = 1024 * 1024
TIMEOUT_SECONDS = 60
USER_AGENT = "pmc-pair-retrieval-download-weights"
ABSENT, GOOD = "absent", "good"


class WeightFile(NamedTuple):
    """One weight file: its configuration, target path, expected sha256 and expected size in bytes."""

    configuration: str
    path: Path
    sha256: str
    n_bytes: int


# --------------------------------------------------------------------------------------
# Expected files
# --------------------------------------------------------------------------------------
def _read_checksums() -> dict[str, str]:
    """``{file name: sha256}`` from ``WEIGHTS_SHA256SUMS``, whose lines read ``<sha256>  weights/<name>``."""
    try:
        lines = CHECKSUM_FILE.read_text(encoding="utf-8").splitlines()
    except OSError as err:
        raise SystemExit(f"[download_weights] cannot read {CHECKSUM_FILE.name}: {err}") from None
    checksums: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  weights/([^/\s]+)", line)
        if match is None or match[2] in checksums:
            raise SystemExit(f"[download_weights] {CHECKSUM_FILE.name}: malformed or repeated line {line!r}")
        checksums[match[2]] = match[1]
    return checksums


def _read_sidecar(configuration: str) -> tuple[str, str]:
    """``(file name, sha256)`` of the weight file of ``configuration``, from its sidecar."""
    path = WEIGHTS_DIR / f"{configuration}.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        recorded, name, digest = sidecar["configuration"], sidecar["weights_file"], sidecar["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as err:
        raise SystemExit(f"[download_weights] cannot read weights/{path.name}: {err!r}") from None
    if recorded != configuration:
        raise SystemExit(f"[download_weights] weights/{path.name} describes {recorded!r}, not {configuration!r}")
    return name, digest


def expected_files() -> dict[str, WeightFile]:
    """The weight file of every configuration, from the sidecars.

    Stops unless ``WEIGHTS_SHA256SUMS`` lists exactly these files with the same sha256.
    """
    checksums = _read_checksums()
    files: dict[str, WeightFile] = {}
    for configuration in CONFIGURATIONS:
        name, digest = _read_sidecar(configuration)
        if name not in WEIGHT_FILE_BYTES:
            raise SystemExit(f"[download_weights] weights/{configuration}.json names the unknown file {name!r}")
        if checksums.get(name) != digest:
            raise SystemExit(f"[download_weights] weights/{configuration}.json gives sha256 {digest} for {name}, but "
                             f"{CHECKSUM_FILE.name} gives {checksums.get(name)}. Nothing was downloaded. Verify these "
                             "files with 'sha256sum -c SHA256SUMS'.")
        files[configuration] = WeightFile(configuration, WEIGHTS_DIR / name, digest, WEIGHT_FILE_BYTES[name])
    extra = sorted(set(checksums) - {weight.path.name for weight in files.values()})
    if extra:
        raise SystemExit(f"[download_weights] {CHECKSUM_FILE.name} lists files without a sidecar: {extra}")
    return files


def local_state(weight: WeightFile) -> str:
    """``ABSENT``, ``GOOD`` or what is wrong with the file at the target path of ``weight``."""
    if not os.path.lexists(weight.path):
        return ABSENT
    if not weight.path.is_file():
        return "it is not a regular file"
    digest = sha256_file(weight.path)
    return GOOD if digest == weight.sha256 else f"its sha256 is {digest}, expected {weight.sha256}"


def check_files(files: list[WeightFile]) -> None:
    """Hash the local files, without using the network; stops unless every file has its expected sha256."""
    failed = 0
    for weight in files:
        state = local_state(weight)
        failed += state != GOOD
        status = {GOOD: f"OK (sha256 {weight.sha256})", ABSENT: "missing"}.get(state, f"FAILED: {state}")
        print(f"[download_weights] weights/{weight.path.name}: {status}", flush=True)
    if failed:
        raise SystemExit(f"[download_weights] check failed: {failed} of {len(files)} files are missing or differ. See "
                         '"Download the released weights" in README.md.')
    print(f"[download_weights] OK: {len(files)} of {len(files)} files have the expected sha256", flush=True)


# --------------------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------------------
def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def url_problem(url: str) -> str | None:
    """Why ``url`` may not be fetched, or ``None``: https is accepted, and http only for a loopback address."""
    try:
        parts = urllib.parse.urlsplit(url)
        host, has_credentials = parts.hostname, bool(parts.username or parts.password)
    except ValueError as err:
        return f"it is not a valid URL ({err})"
    if not host:
        return "it names no host"
    if has_credentials:
        return "it must not contain a user name or password"
    if parts.scheme == "https" or (parts.scheme == "http" and _is_loopback(host)):
        return None
    return "only https is accepted, and http only for a loopback address"


def source_url(name: str, base_url: str | None) -> str:
    """The URL of the weight file ``name``: below ``base_url`` if given, otherwise in the Zenodo record."""
    if base_url is not None:
        return f"{base_url.rstrip('/')}/{name}"
    return ZENODO_FILE_URL.format(record=WEIGHTS_RECORD_ID, name=name)


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to a URL that ``url_problem`` accepts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        problem = url_problem(newurl)
        if problem:
            parts = urllib.parse.urlsplit(newurl)
            raise ValueError(f"refused a redirect to {parts.scheme}://{parts.netloc}: {problem}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------
def _stream(url: str, handle, weight: WeightFile) -> None:
    """Write ``url`` to ``handle``; raises ``ValueError`` unless the size and sha256 are the expected ones."""
    name, expected = weight.path.name, weight.n_bytes
    digest = hashlib.sha256()
    received, next_report = 0, expected // 10
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.build_opener(_CheckedRedirects).open(request, timeout=TIMEOUT_SECONDS) as response:
        announced = response.headers.get("Content-Length")
        if announced is not None and announced.strip() != str(expected):
            raise ValueError(f"the server announces {announced.strip()} bytes, expected {expected}")
        # A connection that closes early can end this loop without an error, so the size is checked below.
        for chunk in iter(lambda: response.read(CHUNK_BYTES), b""):
            received += len(chunk)
            if received > expected:
                raise ValueError(f"the server sent more than the expected {expected} bytes")
            digest.update(chunk)
            handle.write(chunk)
            if received >= next_report:
                print(f"[download_weights] {name}: {100 * received // expected}%", flush=True)
                next_report += expected // 10
    handle.flush()
    os.fsync(handle.fileno())
    if received != expected:
        raise ValueError(f"received {received} bytes, expected {expected}")
    if digest.hexdigest() != weight.sha256:
        raise ValueError(f"the sha256 of the received file is {digest.hexdigest()}, expected {weight.sha256}")


def download_file(weight: WeightFile, url: str) -> None:
    """Stream ``url`` to ``<target>.part`` and replace the target only after the file is verified."""
    name = weight.path.name
    part = weight.path.with_name(name + ".part")
    try:
        # Exclusive, so that two downloads cannot write to one file.
        handle = part.open("xb")
    except FileExistsError:
        raise SystemExit(f"[download_weights] weights/{part.name} already exists: another download is running, or an "
                         "earlier one was killed. If none is running, delete that file and run the command "
                         "again.") from None
    except OSError as err:
        raise SystemExit(f"[download_weights] cannot write weights/{part.name}: {err}") from None
    print(f"[download_weights] {name}: downloading {weight.n_bytes} bytes from {url}", flush=True)
    try:
        with handle:
            _stream(url, handle, weight)
        if os.path.lexists(weight.path):
            raise ValueError(f"weights/{name} appeared during the download and was not replaced")
        os.replace(part, weight.path)
    except (OSError, ValueError, http.client.HTTPException) as err:
        raise SystemExit(f"[download_weights] {name}: {err}. The download from {url} was discarded; nothing was "
                         "written to the target file.") from None
    finally:
        # After os.replace, the part file no longer exists.
        part.unlink(missing_ok=True)
    print(f"[download_weights] weights/{name}: downloaded, size and sha256 {weight.sha256} verified", flush=True)


def download_files(files: list[WeightFile], base_url: str | None) -> None:
    """Download the absent files; stops before any download if a file already there has another sha256."""
    states = {weight: local_state(weight) for weight in files}
    wrong = [f"[download_weights] weights/{weight.path.name} already exists, but {state}. It was not changed. Move or "
             "delete it, then run the command again."
             for weight, state in states.items() if state not in (ABSENT, GOOD)]
    if wrong:
        raise SystemExit("\n".join(wrong))
    for weight, state in states.items():
        if state == GOOD:
            print(f"[download_weights] weights/{weight.path.name}: already present with the expected sha256; left "
                  "alone", flush=True)
        else:
            download_file(weight, source_url(weight.path.name, base_url))


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments of ``python -m pmc_pair_retrieval.download_weights``."""
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.download_weights",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--configuration", choices=(*CONFIGURATIONS, "all"), default="all",
                        help="the weight file to handle (default: all)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--base-url", metavar="URL",
                      help="take each file from URL/<file name> instead of the Zenodo record: https, or http for a "
                           "loopback address only; the sha256 values are checked in the same way")
    mode.add_argument("--check", action="store_true",
                      help="only hash the files in weights/, without using the network")
    args = parser.parse_args(argv)
    if args.base_url is not None:
        problem = url_problem(args.base_url)
        if problem is None:
            parts = urllib.parse.urlsplit(args.base_url)
            problem = "it must not contain a query or fragment" if parts.query or parts.fragment else None
        if problem:
            parser.error(f"--base-url: {problem}")
    return args


def _terminate(signum, frame) -> None:
    """Turn SIGTERM into an exit that runs the clean-up of ``download_file``."""
    raise SystemExit(f"[download_weights] stopped by signal {signum}")


def main(argv: list[str] | None = None) -> None:
    """Verify the local weight files, or download the ones that are absent."""
    args = parse_args(argv)
    signal.signal(signal.SIGTERM, _terminate)
    if not args.check and args.base_url is None and WEIGHTS_RECORD_ID is None:
        raise SystemExit('[download_weights] The weight record has not been published, so there is nothing to download '
                         'yet. See "Download the released weights" in README.md. --check verifies files that are '
                         'already in weights/.')
    files = expected_files()
    selected = list(files.values()) if args.configuration == "all" else [files[args.configuration]]
    try:
        if args.check:
            check_files(selected)
        else:
            download_files(selected, args.base_url)
    except KeyboardInterrupt:
        raise SystemExit("[download_weights] interrupted") from None
    except OSError as err:
        raise SystemExit(f"[download_weights] {err}") from None


if __name__ == "__main__":
    main()
