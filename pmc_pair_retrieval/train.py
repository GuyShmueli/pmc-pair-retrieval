"""Fine-tuning command line.

    python -m pmc_pair_retrieval.train --config configs/image_to_image.yaml \\
        --data-dir DATA_DIR --output-dir runs/image_to_image

The configuration file selects the trainer through its ``configuration`` key
(``image_to_image`` or ``image_text_to_image``).

String hashing. The image_text_to_image training-pair order depends on Python's string
hashing, which is fixed when the interpreter starts. Training requires CPython 3.11
with ``PYTHONHASHSEED=42``. This command tests the hash seed that the interpreter actually
uses, through the hash of a probe string: the environment variable alone is not enough,
because ``python -E`` and ``-I`` leave it visible but make the interpreter ignore it. If the
test fails, the command executes itself once more with ``PYTHONHASHSEED=42`` and without
``-E``/``-I``. It refuses to train if the test still fails after this restart, or under
another Python version.

``--max-steps`` and ``--epochs`` are smoke-test overrides; runs that use them do not
reproduce the published models. ``--allow-missing-images`` trains without the images that
are missing from ``DATA_DIR/images`` (the public image archive withholds 68 of them), so the
training data and the results differ from the published ones; without it, the trainers
refuse to train when images are missing.

Each trainer module provides ``main(config, args)``. It receives the configuration as a
dict, with every top-level value of the form ``data/...`` replaced by the absolute path of
that repository file and ``epochs`` replaced by ``--epochs`` when given, and an
``argparse.Namespace`` with the absolute paths ``config``, ``data_dir`` and
``output_dir``, the overrides ``max_steps`` and ``epochs`` (``None`` when not given) and
the flag ``allow_missing_images``. On entry, string hashing matches the required settings, the
pinned model cache has passed ``model.precheck_model_cache``, the data directory holds
``image_pairs.json`` and ``images/``, and the output directory exists and is empty. The
cache is checked before the output directory is created, so that a wrong environment
stops the command with a message and leaves nothing behind. The trainers record
``config_as_run`` in their outputs and log every validation with ``validation_line``.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from .model import ModelCacheError, precheck_model_cache

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
TRAINERS = {
    "image_to_image": "pmc_pair_retrieval.train_image_to_image",
    "image_text_to_image": "pmc_pair_retrieval.train_image_text_to_image",
}
# Required string hashing: CPython 3.11 with PYTHONHASHSEED=42. The hash of the
# probe string under these conditions identifies the hash seed that an interpreter really uses.
HASH_SEED = 42
PYTHON_VERSION = (3, 11)
HASH_PROBE = "10000271_1_2"
HASH_PROBE_VALUE = 1955230675734984129
# Set for the restarted interpreter, so that a test that still fails stops the run instead of
# restarting it again.
_RESTART_MARKER = "PMC_PAIR_RETRIEVAL_RESTARTED"
# One-letter interpreter options of CPython that take a value; -c and -m also end the options.
_VALUE_OPTIONS = "cmWX"
# PyYAML reads exponent floats without a dot (6e-6) as strings.
_NUMBER_LIKE_STRING = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments of ``python -m pmc_pair_retrieval.train``."""
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.train",
                                     description="Fine-tune BiomedCLIP with one of the configurations in configs/.")
    parser.add_argument("--config", required=True,
                        help="configs/image_to_image.yaml or configs/image_text_to_image.yaml")
    parser.add_argument("--data-dir", required=True, help="directory with image_pairs.json and images/<image_id>.jpg")
    parser.add_argument("--output-dir", required=True, help="new or empty directory for the run outputs")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="smoke test only: stop after this many optimizer steps")
    parser.add_argument("--epochs", type=int, default=None, help="smoke test only: number of epochs")
    parser.add_argument("--allow-missing-images", action="store_true",
                        help="train without the images missing from DATA_DIR/images (the results then differ from "
                             "the published ones)")
    return parser.parse_args(argv)


def _reject_number_strings(value, key: str = "") -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _reject_number_strings(v, f"{key}.{k}" if key else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _reject_number_strings(v, f"{key}[{i}]")
    elif isinstance(value, str) and _NUMBER_LIKE_STRING.fullmatch(value.strip()):
        raise SystemExit(f"Config value {key}={value!r} was read as a string; write floats with a dot (e.g. 6.0e-6).")


def load_config(path: str | os.PathLike) -> dict:
    """Read a configuration YAML and check ``configuration``, ``hash_seed`` and the number formats."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: expected a YAML mapping")
    if config.get("configuration") not in TRAINERS:
        raise SystemExit(f"{path}: 'configuration' must be one of {sorted(TRAINERS)}")
    if type(config.get("hash_seed")) is not int or config["hash_seed"] != HASH_SEED:
        raise SystemExit(f"{path}: 'hash_seed' must be {HASH_SEED}, the required PYTHONHASHSEED")
    _reject_number_strings(config)
    return config


def resolve_data_paths(config: dict, root: Path = REPOSITORY_ROOT) -> dict:
    """Copy of ``config`` with every top-level ``data/...`` value replaced by an absolute path under ``root``."""
    resolved = dict(config)
    for key, value in config.items():
        if isinstance(value, str) and value.startswith("data/"):
            path = root / value
            if not path.is_file():
                raise SystemExit(f"Config '{key}': {path} does not exist")
            resolved[key] = str(path)
    return resolved


def config_as_run(args: argparse.Namespace, config: Mapping) -> dict:
    """The configuration that a trainer records in its outputs (``config.yaml``, weight sidecar, ``summary.json``).

    It is the configuration file as written, so data paths stay repository-relative and the
    outputs carry no machine-specific paths, with the ``--epochs`` override applied.
    ``--max-steps`` is not a configuration value; the trainers record both overrides in
    ``summary.json``.
    """
    return {**load_config(args.config), "epochs": config["epochs"]}


def validation_line(epoch: int, metric: str, val: Mapping, minutes: Mapping, best: float | None = None,
                    best_epoch: int | None = None) -> str:
    """The log line of one validation, in the same format for both trainers.

    The selection metric and the best value so far are printed with full precision, so that
    they can be compared exactly with the recorded summaries. ``best`` is left out for a
    validation that cannot be selected (epoch 0 of image_to_image).
    """
    line = (f"[val] epoch {epoch} {metric}={float(val[metric]['mean'])!r} R@1={val['R@1']['mean']:.5f} "
            f"R@100={val['R@100']['mean']:.5f} Hit@10={val['Hit@10']['mean']:.5f} mAP={val['mAP']['mean']:.5f}")
    if best is not None:
        line += f" (best {float(best)!r} at epoch {best_epoch})"
    return f"{line} minutes train={minutes['train']:.1f} eval={minutes['eval']:.1f}"


def hash_seed_problems() -> list[str]:
    """Return unmet interpreter and string-hashing requirements, or an empty list when all pass.

    The hash seed in use is tested through the hash of a probe string, not through the
    ``PYTHONHASHSEED`` variable, which ``python -E`` and ``-I`` leave in ``os.environ`` while
    the interpreter ignores it.
    """
    problems = []
    if sys.version_info[:2] != PYTHON_VERSION:
        problems.append(f"this is Python {platform.python_version()}, not 3.11")
    if sys.flags.ignore_environment:
        problems.append("Python was started with -E or -I, which ignore PYTHONHASHSEED")
    if hash(HASH_PROBE) != HASH_PROBE_VALUE:
        problems.append(f"hash({HASH_PROBE!r}) differs from its value under PYTHONHASHSEED={HASH_SEED} "
                        f"(PYTHONHASHSEED is {os.environ.get('PYTHONHASHSEED', 'unset')})")
    return problems


def restart_command(orig_argv: Sequence[str]) -> list[str]:
    """The command line of the restart: ``orig_argv`` without the interpreter options ``-E`` and ``-I``.

    ``-I`` stands for ``-E -P -s``; ``-P -s`` are kept, so that only the handling of
    environment variables changes. Interpreter options are read as CPython reads them:
    one-letter options can be grouped (``-BE``), ``-W`` and ``-X`` take a value, and ``-c``
    and ``-m`` take a value and end the interpreter options, as do ``-``, ``--`` and the
    script path. Everything after the interpreter options is kept unchanged.
    """
    args, command = list(orig_argv[1:]), [sys.executable]
    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if arg in ("-", "--") or not arg.startswith("-"):
            return command + [arg] + args[i:]
        if arg.startswith("--"):
            command.append(arg)
            if arg == "--check-hash-based-pycs" and i < len(args):
                command.append(args[i])
                i += 1
            continue
        letters, option, value = arg[1:], "", ""
        for k, letter in enumerate(letters):
            if letter in _VALUE_OPTIONS:
                letters, option, value = letters[:k], letter, letters[k + 1:]
                break
        letters = letters.replace("E", "").replace("I", "Ps")
        if letters or option:
            command.append("-" + letters + option + value)
        if option and not value and i < len(args):
            command.append(args[i])
            i += 1
        if option in ("c", "m"):
            return command + args[i:]
    return command


def ensure_hash_seed(allow_restart: bool) -> None:
    """Enforce the required string hashing, restarting the interpreter once if needed.

    The hash seed is fixed at interpreter start-up, so it can only be corrected by executing
    the same command again (``os.execve``) with ``PYTHONHASHSEED`` set and without
    ``-E``/``-I``. If these options were given, the other ``PYTHON*`` variables, which they
    made the interpreter ignore, are left out of the restarted interpreter's environment.
    Raises ``SystemExit`` when a restart is not allowed or cannot help (another Python
    version), and when the test still fails in the restarted interpreter.
    """
    restarted = os.environ.pop(_RESTART_MARKER, None) is not None
    problems = hash_seed_problems()
    if not problems:
        return
    reasons = "; ".join(problems)
    if sys.version_info[:2] != PYTHON_VERSION:
        raise SystemExit(f"Python 3.11 is required for training: string hashing and the library "
                         f"builds differ between Python versions ({reasons}).")
    if restarted:
        raise SystemExit(f"String hashing still fails the required check after restarting Python with "
                         f"PYTHONHASHSEED={HASH_SEED}: {reasons}.")
    if not allow_restart:
        raise SystemExit(f"String hashing fails the required check ({reasons}). Start Python with "
                         f"PYTHONHASHSEED={HASH_SEED} and without -E or -I.")
    env = dict(os.environ)
    if sys.flags.ignore_environment:
        env = {key: value for key, value in env.items() if not key.startswith("PYTHON")}
    env.update({"PYTHONHASHSEED": str(HASH_SEED), _RESTART_MARKER: "1"})
    without = " and without -E/-I" if sys.flags.ignore_environment else ""
    print(f"[train] restarting with PYTHONHASHSEED={HASH_SEED} (was {os.environ.get('PYTHONHASHSEED', 'unset')})"
          f"{without} to apply the required string-hashing settings", flush=True)
    sys.stderr.flush()
    os.execve(sys.executable, restart_command(sys.orig_argv), env)


def check_data_dir(path: str | os.PathLike) -> str:
    """Absolute data directory; it must contain ``image_pairs.json`` and ``images/``."""
    data_dir = os.path.abspath(path)
    pairs_file, image_dir = os.path.join(data_dir, "image_pairs.json"), os.path.join(data_dir, "images")
    if not os.path.isfile(pairs_file) or not os.path.isdir(image_dir):
        raise SystemExit(f"{data_dir} must contain image_pairs.json and images/")
    return data_dir


def prepare_output_dir(path: str | os.PathLike) -> str:
    """Create the output directory, refusing an existing non-empty one; returns its absolute path."""
    out = os.path.abspath(path)
    if os.path.exists(out):
        if not os.path.isdir(out):
            raise SystemExit(f"{out} exists and is not a directory")
        if os.listdir(out):
            raise SystemExit(f"Output directory {out} is not empty")
    os.makedirs(out, exist_ok=True)
    return out


def main(argv: list[str] | None = None) -> None:
    """Run a trainer. Called without ``argv`` (command line), it may restart the interpreter."""
    args = parse_args(argv)
    args.config = os.path.abspath(args.config)
    config = load_config(args.config)
    ensure_hash_seed(allow_restart=argv is None)
    for name in ("max_steps", "epochs"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be a positive integer")

    config = resolve_data_paths(config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.max_steps is not None or args.epochs is not None:
        print(f"[train] smoke-test overrides: epochs={args.epochs} max_steps={args.max_steps}; "
              "this run does not reproduce the published model", flush=True)
    args.data_dir = check_data_dir(args.data_dir)
    # The trainers load the pretrained model only after writing config.yaml; checking the
    # cache first keeps a wrong environment from leaving a non-empty output directory. The
    # pre-check imports neither torch nor transformers; the model load repeats the full check.
    try:
        precheck_model_cache()
    except ModelCacheError as err:
        raise SystemExit(f"[train] {err}") from None
    args.output_dir = prepare_output_dir(args.output_dir)
    print(f"[train] configuration={config['configuration']} config={args.config} data_dir={args.data_dir} "
          f"output_dir={args.output_dir} allow_missing_images={args.allow_missing_images} "
          f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')} python={platform.python_version()}", flush=True)

    trainer = importlib.import_module(TRAINERS[config["configuration"]])
    trainer.main(config, args)


if __name__ == "__main__":
    main()
