"""Run the four additional training runs used for the three-seed comparison.

Seed 42 is the released model. This driver trains seeds 43 and 44 for both
retrieval configurations, evaluates each selected checkpoint on validation
and test data, and combines their results with the released seed-42 results.
It uses the current Python environment and the package's training,
evaluation and reporting commands with the caller's model-cache settings.
It does not configure the accelerator environment. It needs all 86,893 images: it
does not pass --allow-missing-images, so with the public image archive, which withholds
68 of them, it stops at the first training command, at the latest when that command finds
the images missing. It also needs the two released weight files, which its last step
hashes, and refuses to start without them. This driver has not run to completion: it has
been tested with --dry-run and with mocked subprocesses, not on a GPU. The shipped seed
records come from the train and evaluate commands run as separate jobs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ("image_to_image", "image_text_to_image")
ADDITIONAL_SEEDS = (43, 44)


def build_commands(data_dir: Path, output_dir: Path) -> list[list[str]]:
    """Plan train → validation → test for each run, followed by reporting.

    All paths are absolute so spaces and the caller's working directory do
    not change the paths passed to subprocesses.
    """
    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    python = [sys.executable, "-B", "-m"]
    commands = []
    for seed in ADDITIONAL_SEEDS:
        for configuration in CONFIGURATIONS:
            run_dir = output_dir / "runs" / f"{configuration}_seed{seed}"
            config = PACKAGE_ROOT / "configs" / "multiseed" / f"{configuration}_seed{seed}.yaml"
            commands.append(python + [
                "pmc_pair_retrieval.train", "--config", str(config),
                "--data-dir", str(data_dir), "--output-dir", str(run_dir),
            ])
            for split in ("val", "test"):
                commands.append(python + [
                    "pmc_pair_retrieval.evaluate", "--weights", str(run_dir / "best_model.safetensors"),
                    "--retrieval", configuration, "--split", split,
                    "--data-dir", str(data_dir),
                    "--output-dir", str(output_dir / "eval" / f"seed{seed}"),
                ])
    commands.append(python + [
        "pmc_pair_retrieval.multiseed", "--data-dir", str(data_dir),
        "--runs-dir", str(output_dir / "runs"),
        "--eval-dir", str(output_dir / "eval"), "--verify-weights",
        "--output-dir", str(output_dir / "reports"),
    ])
    return commands


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pmc_pair_retrieval.run_multiseed",
        description="Train and evaluate seeds 43 and 44 for both configurations, "
                    "then report them alongside the released seed-42 results. "
                    "This needs all 86,893 images: with the public image archive, which withholds 68 of them, "
                    "it stops at the first training command, at the latest when that command finds the images "
                    "missing. It also needs the two released weight files and refuses to start without them.",
    )
    parser.add_argument("--data-dir", required=True,
                        help="directory containing image_pairs.json and all images/<image_id>.jpg")
    parser.add_argument("--output-dir", required=True,
                        help="new directory for runs/, eval/ and reports/; must not already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands without reading data, creating files or starting any jobs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    requested_output = Path(args.output_dir).expanduser().absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise SystemExit(f"Refusing to use an existing output path: {requested_output}")
    output_dir = requested_output.resolve()
    commands = build_commands(data_dir, output_dir)
    if args.dry_run:
        print(f"Working directory: {PACKAGE_ROOT}")
        for command in commands:
            print(shlex.join(command))
        return

    if not (data_dir / "image_pairs.json").is_file() or not (data_dir / "images").is_dir():
        raise SystemExit(f"{data_dir} must contain image_pairs.json and images/")
    for command in commands:
        if "--config" in command:
            config = Path(command[command.index("--config") + 1])
            if not config.is_file():
                raise SystemExit(f"Configuration file not found: {config}")
    # The last step hashes the released weight files, which a clone does not have, so find out
    # before hours of training rather than after them.
    missing = [f"weights/{configuration}.safetensors" for configuration in CONFIGURATIONS
               if not (PACKAGE_ROOT / "weights" / f"{configuration}.safetensors").is_file()]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} not found: the last step, multiseed --verify-weights, "
                         'hashes them. See "Download the released weights" in README.md.')
    # Reserve a fresh output directory before any job can write its results.
    output_dir.mkdir(parents=True, exist_ok=False)
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=PACKAGE_ROOT, check=True)


if __name__ == "__main__":
    main()
