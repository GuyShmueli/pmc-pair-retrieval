"""Pinned pretrained BiomedCLIP (download, loading, verification) and fine-tuned weight files.

The pretrained model is ``microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`` at
revision ``9f341de24bfb00180f1b847274256e9b65a3a32e``; its text tower configuration and
tokenizer come from ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`` at revision
``d673b8835373c6fa116d6d8006b33d48734e305d``. open_clip resolves the ``main`` revision of
a Hugging Face repository and accepts no revision argument, so the pins are enforced
through a dedicated cache that is used offline: ``download`` fetches the pinned files and
points ``refs/main`` of both repositories at the pinned commits, and ``load_biomedclip``
refuses to build the model unless the cache is used offline and still carries these pins.

One-time setup (needs network access)::

    python -m pmc_pair_retrieval.model download --cache-dir CACHE

Environment for every later run::

    export HF_HUB_CACHE=CACHE/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

Fine-tuned weights are stored as a ``safetensors`` file plus a JSON sidecar with the same
stem: image_to_image weights hold the vision tower only (``model.visual`` keys
``trunk.*`` and ``head.proj.weight``, 151 tensors), image_text_to_image weights the full
model (351 tensors).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

MODEL_ID = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
MODEL_REVISION = "9f341de24bfb00180f1b847274256e9b65a3a32e"
TEXT_MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
TEXT_MODEL_REVISION = "d673b8835373c6fa116d6d8006b33d48734e305d"
OPEN_CLIP_NAME = f"hf-hub:{MODEL_ID}"
WEIGHTS_FILE = "open_clip_pytorch_model.bin"
WEIGHTS_SHA256 = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
# state_dict_sha256 of the freshly built pretrained model (351 tensors).
PRETRAINED_STATE_DICT_SHA256 = "28b0886e361ba509b0b4eef721776baa71b5fe99f2755c5dba512c45c3554602"

# Repository -> (pinned revision, {file read while building the model or tokenizer: sha256}).
PINNED_FILES: dict[str, tuple[str, dict[str, str]]] = {
    MODEL_ID: (MODEL_REVISION, {
        "open_clip_config.json": "9a41f334a8c444678772c0ebb9ab854c97ab350bced3a17b803e258d39c23dc0",
        WEIGHTS_FILE: WEIGHTS_SHA256,
    }),
    TEXT_MODEL_ID: (TEXT_MODEL_REVISION, {
        "config.json": "56bab767c02bc792897638feb8addc06d483d9c73769a5f554bbabea1f5507a1",
        "tokenizer_config.json": "53b6f42b8b8daddbdc6d3532c324187a92b46c1602bd6e8d4ad5a413b4fc90e1",
        "vocab.txt": "7b36651908a88bc38bda41b728b2a598191e0d3b553cbacf7b1e5f026d5b5b9f",
    }),
}
# Files the libraries look up that do not exist at the pinned revisions. ``download``
# records their absence in the cache (``.no_exist`` markers), as the Hugging Face client
# does after a failed lookup; in particular open_clip prefers ``open_clip_model.safetensors``
# over the ``.bin`` weights when it exists.
ABSENT_FILES: dict[str, tuple[str, ...]] = {
    MODEL_ID: ("open_clip_model.safetensors",),
    TEXT_MODEL_ID: ("tokenizer.json", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja"),
}


class ModelCacheError(RuntimeError):
    """The environment does not provide the pinned pretrained model: offline mode, cache paths, pins or file hashes.

    The command-line tools report it as a one-line message instead of a traceback.
    """


# --------------------------------------------------------------------------------------
# Hashes
# --------------------------------------------------------------------------------------
def sha256_file(path: str | os.PathLike) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping) -> str:
    """Deterministic hash of a state dict (sorted keys, shapes, dtypes and raw bytes of contiguous CPU tensors)."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# Model cache
# --------------------------------------------------------------------------------------
def _repo_dir(hub_dir: Path, repo_id: str) -> Path:
    return hub_dir / ("models--" + repo_id.replace("/", "--"))


def download(cache_dir: str | os.PathLike) -> Path:
    """Fetch the pinned model files into ``<cache_dir>/hub`` and pin ``refs/main``; returns the hub directory.

    Every file is downloaded at its pinned commit and checked against its sha256. The
    lookup of each file in ``ABSENT_FILES`` must fail at the pinned commit; the client
    then records the absence in the cache. Refuses to repoint an existing ``refs/main``
    that names another commit (use a dedicated cache directory).
    """
    from huggingface_hub import constants, hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    if constants.HF_HUB_OFFLINE:
        raise SystemExit("download needs network access: unset HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE.")
    hub_dir = Path(os.path.abspath(cache_dir)) / "hub"
    for repo_id, (revision, files) in PINNED_FILES.items():
        ref = _repo_dir(hub_dir, repo_id) / "refs" / "main"
        if ref.is_file() and ref.read_text() != revision:
            raise SystemExit(f"{ref} names commit {ref.read_text()!r}, not the pinned {revision}; "
                             "use a dedicated --cache-dir.")
    for repo_id, (revision, files) in PINNED_FILES.items():
        for filename, expected in files.items():
            path = hf_hub_download(repo_id, filename, revision=revision, cache_dir=hub_dir)
            digest = sha256_file(path)
            if digest != expected:
                raise RuntimeError(f"{repo_id}@{revision}/{filename}: sha256 {digest} != pinned {expected}")
            print(f"[model] {repo_id}@{revision[:12]} {filename} sha256 {digest}", flush=True)
        for filename in ABSENT_FILES[repo_id]:
            try:
                hf_hub_download(repo_id, filename, revision=revision, cache_dir=hub_dir)
            except LocalEntryNotFoundError:
                # A failed connection, not an answer from the Hub (this class derives from EntryNotFoundError).
                raise
            except EntryNotFoundError:
                marker = _repo_dir(hub_dir, repo_id) / ".no_exist" / revision / filename
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            else:
                raise RuntimeError(f"{repo_id}@{revision} unexpectedly contains {filename}")
        ref = _repo_dir(hub_dir, repo_id) / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        # The Hugging Face client reads refs verbatim, so no trailing newline.
        ref.write_text(revision)
    print(f"[model] cache ready. Before every run:\n"
          f"  export HF_HUB_CACHE={hub_dir} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1", flush=True)
    return hub_dir


def _offline_hub_dir() -> Path:
    """The cache directory of huggingface_hub; raises ``ModelCacheError`` unless offline mode is on."""
    from huggingface_hub import constants

    if not constants.HF_HUB_OFFLINE:
        raise ModelCacheError("Set HF_HUB_OFFLINE=1 (and TRANSFORMERS_OFFLINE=1): online lookups resolve the latest "
                              "revision instead of the pinned one.")
    return Path(constants.HF_HUB_CACHE)


def verify_model_cache() -> Path:
    """Check that the Hugging Face libraries will read the pinned files offline; returns the hub directory.

    Raises ``ModelCacheError`` unless offline mode is on, both libraries use the same cache,
    ``refs/main`` of both repositories names the pinned commits, and every pinned file is
    present with its pinned sha256. Only files are read; no random number is drawn.
    """
    from transformers.utils import hub as transformers_hub

    hub_dir = _offline_hub_dir()
    if Path(transformers_hub.TRANSFORMERS_CACHE) != hub_dir:
        raise ModelCacheError(f"transformers reads {transformers_hub.TRANSFORMERS_CACHE} but huggingface_hub reads "
                              f"{hub_dir}; unset TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE and "
                              "PYTORCH_PRETRAINED_BERT_CACHE.")
    _check_pinned_files(hub_dir)
    return hub_dir


def precheck_model_cache() -> Path:
    """The checks of ``verify_model_cache`` without importing transformers; returns the hub directory.

    For the training command, which checks the cache before it creates the output directory
    and leaves the import of torch and transformers to the trainer. Instead of asking
    transformers which cache it reads, it requires the variables that transformers reads
    (``TRANSFORMERS_CACHE``, ``PYTORCH_TRANSFORMERS_CACHE``, ``PYTORCH_PRETRAINED_BERT_CACHE``)
    to be unset or equal to ``HF_HUB_CACHE``. The model load repeats the full check.
    """
    hub_dir = _offline_hub_dir()
    for name in ("TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE"):
        value = os.environ.get(name)
        if value is not None and Path(value) != hub_dir:
            raise ModelCacheError(f"{name}={value} but huggingface_hub reads {hub_dir}; unset {name} or set it to "
                                  "HF_HUB_CACHE.")
    _check_pinned_files(hub_dir)
    return hub_dir


def _check_pinned_files(hub_dir: Path) -> None:
    """``refs/main`` of both repositories names the pinned commits and every pinned file has its pinned sha256."""
    for repo_id, (revision, files) in PINNED_FILES.items():
        repo = _repo_dir(hub_dir, repo_id)
        ref = repo / "refs" / "main"
        found = ref.read_text() if ref.is_file() else None
        if found != revision:
            raise ModelCacheError(f"{ref}: expected commit {revision}, found {found!r}. Create the cache with "
                                  "'python -m pmc_pair_retrieval.model download --cache-dir CACHE' and set "
                                  "HF_HUB_CACHE=CACHE/hub.")
        for filename, expected in files.items():
            path = repo / "snapshots" / revision / filename
            if not path.is_file():
                raise ModelCacheError(f"Missing pinned file {path}")
            digest = sha256_file(path)
            if digest != expected:
                raise ModelCacheError(f"{path}: sha256 {digest} != pinned {expected}")


def load_biomedclip(device):
    """Build pretrained BiomedCLIP and return ``(model, tokenizer)``.

    Performs exactly one ``open_clip.create_model_and_transforms`` call for
    ``hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`` (fp32, on the
    CPU), one ``open_clip.get_tokenizer`` call, and then moves the model to ``device``.
    Before building, the cache is checked with ``verify_model_cache`` (pinned commits and
    file hashes) and open_clip must resolve exactly the verified weights file; afterwards
    the model's state dict must hash to ``PRETRAINED_STATE_DICT_SHA256``. A failed check
    raises ``ModelCacheError``. The model is returned in training mode, as open_clip
    builds it.

    Building the model draws from the global torch CPU generator (random initialisation
    that the pretrained weights then overwrite); the number of draws is fixed by the
    pinned library versions. Training data order and augmentation seeds depend on it, so
    trainers call this exactly once, right after seeding and before any DataLoader is
    iterated. Nothing else in this function draws random numbers.
    """
    import open_clip
    from open_clip.pretrained import download_pretrained_from_hf

    hub_dir = verify_model_cache()
    # create_model_and_transforms below resolves its weights with this function and these arguments.
    # It must find the verified file; a safetensors file in the snapshot, for example, would take precedence.
    resolved = Path(download_pretrained_from_hf(MODEL_ID))
    pinned = _repo_dir(hub_dir, MODEL_ID) / "snapshots" / MODEL_REVISION / WEIGHTS_FILE
    if resolved != pinned:
        raise ModelCacheError(f"open_clip resolves {resolved}, not the pinned weights file {pinned}")

    model, _, _ = open_clip.create_model_and_transforms(OPEN_CLIP_NAME)
    state_hash = state_dict_sha256(model.state_dict())
    if state_hash != PRETRAINED_STATE_DICT_SHA256:
        raise ModelCacheError(f"Pretrained state dict hash {state_hash} != pinned {PRETRAINED_STATE_DICT_SHA256}; "
                              "check the library versions.")
    tokenizer = open_clip.get_tokenizer(OPEN_CLIP_NAME)
    return model.to(device), tokenizer


# --------------------------------------------------------------------------------------
# Fine-tuned weights
# --------------------------------------------------------------------------------------
def _weights_kind(keys) -> str:
    keys = list(keys)
    if any(k.startswith(("visual.", "text.")) for k in keys):
        return "full_model"
    if keys and all(k.startswith(("trunk.", "head.")) for k in keys):
        return "vision_tower"
    raise ValueError(f"Not a BiomedCLIP full-model or vision-tower state dict: {keys[:5]}")


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def read_sidecar(path: str | os.PathLike) -> dict | None:
    """The JSON sidecar of a weights file (same stem, ``.json``), or ``None`` if there is none.

    The sidecar is only read; ``load_weights`` checks it against the weights file.
    """
    sidecar_path = _sidecar_path(Path(path))
    if not sidecar_path.is_file():
        return None
    with sidecar_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_weights(path: str | os.PathLike, state_dict: Mapping, metadata: Mapping) -> dict:
    """Write ``state_dict`` to a safetensors file and its JSON sidecar; returns the sidecar content.

    ``state_dict`` is a full-model state dict (``model.state_dict()``) or a vision-tower
    state dict (``model.visual.state_dict()``). The sidecar (same stem, ``.json``) holds
    ``metadata`` followed by ``weights_file``, ``kind`` (``full_model`` or
    ``vision_tower``), ``n_tensors``, ``state_dict_sha256`` and ``sha256`` (of the file).
    """
    from safetensors.torch import save_file

    path = Path(path)
    tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    kind = _weights_kind(tensors)
    reserved = {"weights_file", "kind", "n_tensors", "state_dict_sha256", "sha256"} & set(metadata)
    if reserved:
        raise ValueError(f"metadata uses reserved keys {sorted(reserved)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    sidecar = {**metadata, "weights_file": path.name, "kind": kind, "n_tensors": len(tensors),
               "state_dict_sha256": state_dict_sha256(tensors), "sha256": sha256_file(path)}
    tmp = _sidecar_path(path).with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)
        handle.write("\n")
    tmp.replace(_sidecar_path(path))
    return sidecar


def load_weights(model, path: str | os.PathLike) -> dict:
    """Load fine-tuned weights strictly into ``model`` (full model or ``model.visual``).

    Keys starting with ``visual.`` or ``text.`` mark a full-model file; otherwise the file
    holds the vision tower and the text tower keeps its pretrained weights. If the JSON
    sidecar exists, the file and state-dict hashes must match it. Returns ``kind``,
    ``n_tensors``, ``state_dict_sha256`` (of the loaded tensors), ``sha256`` (of the file)
    and ``metadata`` (the sidecar, or ``None``).
    """
    from safetensors.torch import load_file

    path = Path(path)
    state = load_file(str(path), device="cpu")
    kind = _weights_kind(state)
    info = {"kind": kind, "n_tensors": len(state), "state_dict_sha256": state_dict_sha256(state),
            "sha256": sha256_file(path), "metadata": None}
    sidecar_path = _sidecar_path(path)
    if sidecar_path.is_file():
        with sidecar_path.open("r", encoding="utf-8") as handle:
            sidecar = json.load(handle)
        for key in ("state_dict_sha256", "sha256"):
            if sidecar.get(key) != info[key]:
                raise RuntimeError(f"{path}: {key} {info[key]} does not match {sidecar_path} ({sidecar.get(key)})")
        info["metadata"] = sidecar
    if kind == "full_model":
        model.load_state_dict(state, strict=True)
    else:
        model.visual.load_state_dict(state, strict=True)
    return info


def describe_weights(sidecar: Mapping) -> dict:
    """Configuration, selected epoch, global step and selection value of fine-tuned weights, from their sidecar.

    Returns the sidecar's ``configuration``, ``epoch`` and ``global_step`` and
    ``selection``: the validation value of the selection metric
    (``config['early_stopping']['metric']``) at the selected epoch, as ``{"split": "val",
    "metric", "value"}``. Keys that the sidecar cannot provide are left out.
    """
    out = {key: sidecar[key] for key in ("configuration", "epoch", "global_step") if key in sidecar}
    try:
        metric = sidecar["config"]["early_stopping"]["metric"]
        out["selection"] = {"split": "val", "metric": metric, "value": sidecar["val"][metric]["mean"]}
    except (KeyError, TypeError):
        pass
    return out


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def main(argv=None) -> None:
    """``python -m pmc_pair_retrieval.model download --cache-dir CACHE``."""
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.model", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="fetch the pinned pretrained model into CACHE/hub (needs network access)")
    p.add_argument("--cache-dir", required=True, help="cache root; files go to CACHE/hub")
    args = parser.parse_args(argv)
    if args.command == "download":
        download(args.cache_dir)


if __name__ == "__main__":
    main()
