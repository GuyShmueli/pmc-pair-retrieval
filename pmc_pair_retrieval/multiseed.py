"""Regenerate the three-seed results from saved per-query evaluations, on the CPU.

    python -m pmc_pair_retrieval.multiseed --data-dir DATA --output-dir OUT
    python -m pmc_pair_retrieval.multiseed --data-dir DATA --check

Seed 42 is the released model. Seeds 43 and 44 describe variability only. The command
computes the metrics, paired bootstrap intervals, sample standard deviations and seed-42
agreement checks. See results/multiseed/PLAN.md for the analysis plan and
results/multiseed/provenance.json for the records of the runs.

Only image_pairs.json is read from DATA, to reconstruct article groups. No images,
model cache, model inference or GPU are needed, so the two commands above work with the
public image archive, which withholds 68 of the 86,893 images. The default inputs are
covered by INPUT_MANIFEST.json. Checkpoint metadata are checked against evaluation
metadata, but the checkpoints of seeds 43 and 44 are not shipped. ``--verify-weights``
additionally requires and hashes the actual checkpoint files.

For retrained seeds, supply both --runs-dir and --eval-dir, optionally with --seeds. Their
evaluations must use all 86,893 images: evaluations made with --allow-missing-images fail the
required checks of the image and query counts, and the command stops at the first such
evaluation with status 1, without writing reports. The base
configurations, data and seed-42 reference remain those of the release. Reports never
overwrite existing files. --check compares freshly computed reports with the shipped reports.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import statistics
import tempfile
from pathlib import Path

from . import compare, metrics, train
from .model import describe_weights
from .split import article_groups, read_pairs

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUTS = ROOT / "results" / "multiseed"
CONFIGURATIONS = ("image_to_image", "image_text_to_image")
SPLITS = ("val", "test")
DEFAULT_SEEDS = (42, 43, 44)
METRICS = ("R@1", "R@5", "R@10", "R@100", "Hit@10", "mAP", "nDCG", "MRR", "median_first_relevant_rank")
PAIRED_METRICS = ("R@10", "Hit@10")
N_QUERIES = {"val": 8245, "test": 8096}
IMAGE_PAIRS_SHA256 = "cda80d74d8ff98f0e6d31dbbf66c9823a109ab37ef51db8c800d16e5da0271d7"
SEED42_EPOCH1_BATCH_SHA256 = {
    "image_to_image": "97293f37db20a33b30f9a7f083083b45e72d15eafe03983a0a83f65999047885",
    "image_text_to_image": "c5f23ecfcec1b3b61f8ec8a550c92e61164c1377182fd7ad88a86cb3f3c4a4ca",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


class Checks:
    """Fresh required checks determine success. Informational checks do not."""

    def __init__(self):
        self.items = []

    def add(self, name, observed, expected, required=True):
        passed = observed == expected
        self.items.append({"check": name, "required": required, "pass": passed,
                           "observed": observed, "expected": expected})
        if not passed:
            print(f"[multiseed] {'FAIL' if required else 'INFO'}: {name}", flush=True)
        return passed

    def counts(self):
        return {f"{kind}_{result}": sum(c["required"] == required and c["pass"] == passed for c in self.items)
                for kind, required in (("required", True), ("informational", False))
                for result, passed in (("passed", True), ("failed", False))}


def describe(values_by_seed: dict[str, float]) -> dict:
    """Mean, sample SD (n - 1), minimum and maximum."""
    values = list(values_by_seed.values())
    return {"values": values_by_seed, "n": len(values), "mean": statistics.mean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else None, "min": min(values), "max": max(values)}


def selected_by_rule(history: list[dict], metric: str) -> int:
    """Highest validation value, requiring strict improvement. Epoch 0 is ineligible."""
    best, best_epoch = None, None
    for record in history:
        if record["epoch"] < 1:
            continue
        value = record["val"][metric]["mean"]
        if best is None or value > best:
            best, best_epoch = value, record["epoch"]
    return best_epoch


def stopping_epoch(history: list[dict], config: dict) -> int:
    """Apply the trainer's stopping rule to the recorded validation history."""
    stopping = config["early_stopping"]
    best, bad = None, 0
    for record in history:
        epoch = record["epoch"]
        if epoch < 1:
            continue
        value = record["val"][stopping["metric"]]["mean"]
        if best is None or value > best:
            best, bad = value, 0
        else:
            bad += 1
        if epoch >= stopping["min_epochs"] and stopping["patience"] > 0 and bad >= stopping["patience"]:
            return epoch
    return config["epochs"]


def verify_manifest(checks: Checks) -> dict[str, str]:
    entries = load_json(DEFAULT_INPUTS / "INPUT_MANIFEST.json")["inputs"]
    digests = {}
    for entry in entries:
        name = entry["path"]
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name in digests:
            raise ValueError(f"invalid or repeated input-manifest path: {name}")
        digests[name] = sha256_file(ROOT / path)
        checks.add(f"input SHA256: {name}", digests[name], entry["sha256"])
    checks.add("input manifest is nonempty", bool(digests), True)
    return digests


def agreement(results, cfg, pt_cells, ft_cells, pds, member):
    """Compare all 232 seed-42 reference values over two configurations."""
    compared, mismatches = 0, []

    def same(name, recomputed, published):
        nonlocal compared
        compared += 1
        if recomputed != published:
            mismatches.append({"value": name, "results_json": published, "recomputed": recomputed})

    cell_keys = ("n_queries", "gallery_size", *metrics.SUMMARY_METRICS, "MRR", "median_first_relevant_rank",
                 "per_query_file", "per_query_sha256", "rankings_file", "rankings_sha256")
    for split in SPLITS:
        for model_name, cell in (("pretrained", pt_cells[split]), ("fine_tuned", ft_cells[split])):
            ref = results["configurations"][cfg][model_name][split]
            for key in cell_keys:
                if key in ref or key in cell:
                    same(f"configurations.{cfg}.{model_name}.{split}.{key}", cell.get(key), ref.get(key))
        ref = results["configurations"][cfg]["fine_tuned"][split]["weights"]
        weights = member["weights"]
        for key, value in (("configuration", cfg), ("epoch", member["selected_epoch"]),
                           ("global_step", member["global_step"]), ("kind", weights["kind"]),
                           ("n_tensors", weights["n_tensors"]), ("sha256", weights["sha256"]),
                           ("state_dict_sha256", weights["state_dict_sha256"]), ("file", weights["file"])):
            same(f"configurations.{cfg}.fine_tuned.{split}.weights.{key}", value, ref.get(key))
        same(f"configurations.{cfg}.fine_tuned.{split}.weights.selection.value", member["selection_value"],
             ref["selection"]["value"])
        for metric in compare.PAIRED_METRICS:
            for key in ("fine_tuned_minus_pretrained", "ci95_article_groups", "ci95_articles"):
                same(f"paired_differences.{cfg}.{split}.{metric}.{key}", pds[split][metric][key],
                     results["paired_differences"][cfg][split][metric][key])
    return compared, mismatches


def member_inputs(cfg, seed, config, args, checks, input_hashes):
    """Read and validate the metadata linking one selected checkpoint to its evaluations."""
    label = f"{cfg} seed {seed}"
    run = args.runs_dir / f"{cfg}_seed{seed}"
    side_path = ROOT / "weights" / f"{cfg}.json" if seed == 42 else run / "best_model.json"
    sidecar = load_json(side_path)
    input_hashes[args.display(side_path)] = sha256_file(side_path)
    expected = {**config, "seed": seed}
    checks.add(f"{label}: sidecar configuration", sidecar["configuration"], cfg)
    checks.add(f"{label}: sidecar config differs only by seed", sidecar["config"], expected)
    weights = ROOT / "weights" / f"{cfg}.safetensors" if seed == 42 else run / "best_model.safetensors"
    selection = describe_weights(sidecar)["selection"]["value"]
    epochs_run, epoch1 = None, SEED42_EPOCH1_BATCH_SHA256[cfg]
    if seed != 42:
        for name in ("summary.json", "config.yaml", "history.jsonl"):
            input_hashes[args.display(run / name)] = sha256_file(run / name)
        summary = load_json(run / "summary.json")
        history = compare._read_jsonl(run / "history.jsonl")
        checks.add(f"{label}: training summary configuration", summary["configuration"], cfg)
        checks.add(f"{label}: summary config differs only by seed", summary["config"], expected)
        checks.add(f"{label}: run config differs only by seed", train.load_config(run / "config.yaml"), expected)
        checks.add(f"{label}: no smoke-test overrides", summary["overrides"], {"epochs": None, "max_steps": None})
        epochs_run = summary["epochs_run"]
        checks.add(f"{label}: complete training history", [r["epoch"] for r in history if r["epoch"] >= 1],
                   list(range(1, epochs_run + 1)))
        checks.add(f"{label}: selected epoch equals sidecar", summary["best_epoch"], sidecar["epoch"])
        checks.add(f"{label}: selected epoch follows validation rule", summary["best_epoch"],
                   selected_by_rule(history, config["early_stopping"]["metric"]))
        checks.add(f"{label}: epochs run follow stopping rule", epochs_run, stopping_epoch(history, config))
        selected = next(r for r in history if r["epoch"] == sidecar["epoch"])
        checks.add(f"{label}: selection value equals training history", selection,
                   selected["val"][config["early_stopping"]["metric"]]["mean"])
        if cfg == "image_text_to_image":
            checks.add(f"{label}: selected model state hash equals training summary", sidecar["state_dict_sha256"],
                       summary["best_model_state_dict_sha256"])
        epoch1 = next(r["batch_sequence_sha256"] for r in history if r["epoch"] == 1)
    verified = False
    if args.verify_weights:
        verified = checks.add(f"{label}: checkpoint bytes match sidecar", sha256_file(weights), sidecar["sha256"])
    paths, eval_summaries = {}, {}
    for split in SPLITS:
        base = ROOT / "results" / "per_query" / cfg if seed == 42 else args.eval_dir / f"seed{seed}" / cfg
        paths[split] = base / f"fine_tuned_{split}.jsonl"
        if seed == 42:
            continue
        ev_path = base / f"fine_tuned_{split}.summary.json"
        ev = load_json(ev_path)
        input_hashes[args.display(ev_path)] = sha256_file(ev_path)
        eval_summaries[split] = ev
        tag = f"{label} {split}"
        checks.add(f"{tag}: evaluation configuration, split, name and views",
                   [ev["retrieval"], ev["split"], ev["name"], ev["views"]], [cfg, split, "fine_tuned", 3])
        checks.add(f"{tag}: evaluation settings", [ev["settings"]["config"], ev["settings"]["image_size"]],
                   [f"configs/{cfg}.yaml", 224])
        for key in ("kind", "n_tensors", "sha256", "state_dict_sha256", "configuration", "epoch", "global_step"):
            checks.add(f"{tag}: evaluated checkpoint {key}", ev["weights"][key], sidecar[key])
        checks.add(f"{tag}: evaluated checkpoint selection value", ev["weights"]["selection"]["value"], selection)
        checks.add(f"{tag}: evaluation data hashes and count", ev["data"],
                   {"split_file_sha256": sha256_file(ROOT / "data" / "article_disjoint_split.json"),
                    "image_pairs_sha256": IMAGE_PAIRS_SHA256, "n_images": 86893})
        checks.add(f"{tag}: query count", ev["n_queries"], N_QUERIES[split])
    return {"source": "released model" if seed == 42 else "additional training seed",
            "selected_epoch": sidecar["epoch"], "epochs": config["epochs"], "epochs_run": epochs_run,
            "global_step": sidecar["global_step"], "selection_value": selection,
            "epoch1_batch_sequence_sha256": epoch1, "checkpoint_bytes_verified": verified,
            "weights": {"file": args.display(weights), **{k: sidecar[k] for k in
                        ("kind", "n_tensors", "sha256", "state_dict_sha256")}},
            "paths": paths, "eval_summaries": eval_summaries}


def build_report(args):
    checks = Checks()
    manifest_hashes = verify_manifest(checks)
    pairs_hash = sha256_file(args.data_dir / "image_pairs.json")
    checks.add("image_pairs.json matches the dataset used in the original experiment", pairs_hash, IMAGE_PAIRS_SHA256)
    groups, articles = article_groups(read_pairs(args.data_dir))
    reference = load_json(ROOT / "results" / "results.json")
    checks.add("reference bootstrap replicates and seed", reference["bootstrap"],
               {**reference["bootstrap"], "replicates": compare.N_PAIRED_RESAMPLES,
                "seed": metrics.PAIRED_BOOTSTRAP_SEED})
    input_hashes = {}
    for name in ("results/results.json", "data/article_disjoint_split.json",
                 "results/multiseed/PLAN.md", "results/multiseed/provenance.json"):
        input_hashes[name] = sha256_file(ROOT / name)
    configurations, compared_total, mismatches_total = {}, 0, []
    for cfg in CONFIGURATIONS:
        print(f"[multiseed] {cfg}", flush=True)
        config_path = ROOT / "configs" / f"{cfg}.yaml"
        config = train.load_config(config_path)
        input_hashes[args.display(config_path)] = sha256_file(config_path)
        checks.add(f"{cfg}: released seed and hash seed", [config["seed"], config["hash_seed"]], [42, 42])
        pt_paths = {s: ROOT / "results" / "per_query" / cfg / f"pretrained_{s}.jsonl" for s in SPLITS}
        pt_rows = {s: compare._read_jsonl(pt_paths[s]) for s in SPLITS}
        pt_cells = {s: compare.cell_entry(pt_rows[s], pt_paths[s], args.output_dir / "multiseed.json") for s in SPLITS}
        for split in SPLITS:
            checks.add(f"{cfg} pretrained {split}: query count", len(pt_rows[split]), N_QUERIES[split])
            input_hashes[args.display(pt_paths[split])] = pt_cells[split]["per_query_sha256"]
            ranked = compare.rankings_path(pt_paths[split])
            if ranked:
                input_hashes[args.display(ranked)] = pt_cells[split]["rankings_sha256"]
        entries = {}
        for seed in args.seeds:
            member = member_inputs(cfg, seed, config, args, checks, input_hashes)
            cells, differences, rows_by_split = {}, {}, {}
            for split in SPLITS:
                path = member["paths"][split]
                rows = compare._read_jsonl(path)
                rows_by_split[split] = rows
                checks.add(f"{cfg} seed {seed} {split}: same queries in the same order as pretrained",
                           [r["query_id"] for r in rows] == [r["query_id"] for r in pt_rows[split]], True)
                cells[split] = compare.cell_entry(rows, path, args.output_dir / "multiseed.json")
                input_hashes[args.display(path)] = cells[split]["per_query_sha256"]
                ranked = compare.rankings_path(path)
                # The released seed-42 archive contains ranked lists for test only.
                # All additional-seed evaluations include both validation and test lists.
                if seed != 42 or split == "test":
                    checks.add(f"{cfg} seed {seed} {split}: ranked lists present", ranked is not None, True)
                if ranked:
                    input_hashes[args.display(ranked)] = cells[split]["rankings_sha256"]
                differences[split] = compare.paired_differences(rows, pt_rows[split], groups, articles)
                if seed != 42:
                    ev = member["eval_summaries"][split]
                    ev_values = {m: ev[m] if m in ("MRR", "median_first_relevant_rank") else 100 * ev[m]["mean"]
                                 for m in METRICS}
                    checks.add(f"{cfg} seed {seed} {split}: per-query metrics equal evaluation summary",
                               {m: cells[split][m] for m in METRICS}, ev_values)
            if seed == 42:
                count, mismatches = agreement(reference, cfg, pt_cells, cells, differences, member)
                compared_total += count
                mismatches_total += mismatches
            val_r10 = metrics.summarize(rows_by_split["val"])["R@10"]["mean"]
            delta = val_r10 - member["selection_value"]
            checks.add(f"{cfg} seed {seed}: evaluator validation R@10 agrees with training selection within 1e-6",
                       abs(delta) <= 1e-6, True, required=False)
            entries[str(seed)] = {
                **{k: v for k, v in member.items() if k not in ("paths", "eval_summaries")},
                "evaluator_val_R@10_minus_selection_value": delta,
                "splits": {s: {"n_queries": cells[s]["n_queries"], "metrics": {m: cells[s][m] for m in METRICS},
                               "per_query_file": args.display(member["paths"][s]),
                               "per_query_sha256": cells[s]["per_query_sha256"],
                               "rankings_sha256": cells[s].get("rankings_sha256")} for s in SPLITS},
                "paired_differences": {s: {m: {k: differences[s][m][k] for k in
                                               ("fine_tuned_minus_pretrained", "ci95_article_groups")}
                                           for m in PAIRED_METRICS} for s in SPLITS}}
        hashes = [entries[str(s)]["epoch1_batch_sequence_sha256"] for s in args.seeds]
        checks.add(f"{cfg}: distinct epoch-one batch orders", len(set(hashes)), len(args.seeds))
        configurations[cfg] = {
            "pretrained": {s: {"n_queries": pt_cells[s]["n_queries"],
                               "metrics": {m: pt_cells[s][m] for m in METRICS},
                               "per_query_file": args.display(pt_paths[s]),
                               "per_query_sha256": pt_cells[s]["per_query_sha256"]} for s in SPLITS},
            "seeds": entries,
            "across_seeds": {
                "n": len(args.seeds), "seeds": args.seeds,
                "selected_epochs": {str(s): entries[str(s)]["selected_epoch"] for s in args.seeds},
                "metrics": {s: {m: describe({str(k): entries[str(k)]["splits"][s]["metrics"][m] for k in args.seeds})
                                for m in METRICS} for s in SPLITS},
                "paired_differences": {s: {m: describe({str(k): entries[str(k)]["paired_differences"][s][m]
                                                        ["fine_tuned_minus_pretrained"] for k in args.seeds})
                                           for m in PAIRED_METRICS} for s in SPLITS}}}
    checks.add("seed 42: differences from results/results.json", len(mismatches_total), 0)
    checks.add("seed 42: number of reference values compared", compared_total, 232)
    base_inputs = {name for name in input_hashes if name.startswith(("weights/", "configs/", "data/",
                                                                   "results/per_query/", "results/rankings/"))}
    covered_inputs = set(input_hashes) if not args.custom else base_inputs | {
        "results/results.json", "results/multiseed/PLAN.md", "results/multiseed/provenance.json"}
    checks.add("consumed release inputs missing from input manifest", sorted(covered_inputs - manifest_hashes.keys()), [])
    counts = checks.counts()
    return {
        "title": "Training-seed variability of the two final retrieval configurations",
        "status": "complete" if counts["required_failed"] == 0 else "failed checks",
        "descriptive_only": "Seed 42 remains the released model. Other seeds describe variability only. "
                            "No seed or metric is selected on test.",
        "seeds": args.seeds, "input_mode": "custom runs" if args.custom else "shipped evaluations",
        "historical_provenance": "results/multiseed/provenance.json",
        "plan": {"file": "results/multiseed/PLAN.md", "sha256": input_hashes["results/multiseed/PLAN.md"]},
        "input_manifest": {"file": "results/multiseed/INPUT_MANIFEST.json", "files_verified": len(manifest_hashes)},
        "inputs": [{"path": name, "sha256": digest} for name, digest in sorted(input_hashes.items())],
        "data": {"image_pairs_sha256": pairs_hash, "split_file_sha256": input_hashes["data/article_disjoint_split.json"]},
        "checkpoint_bytes_verified": all(m["checkpoint_bytes_verified"] for c in configurations.values()
                                         for m in c["seeds"].values()),
        "verification_scope": "Fresh file hashes, configuration and checkpoint metadata, checkpoint selection, "
                              "stopping rules, query alignment, metric recomputation and seed-42 reference agreement. "
                              "Historical training execution is documented separately in provenance.json. "
                              + ("Actual checkpoint bytes were also checked." if args.verify_weights else
                                 "Actual checkpoint bytes were not checked by this analysis."),
        "metrics": list(METRICS), "paired_metrics": list(PAIRED_METRICS),
        "units": "Recall@k, Hit@10, mAP and nDCG in percent. MRR as a fraction. Paired differences and SD of "
                 "percentage metrics in percentage points. Across-seed SD uses denominator n - 1.",
        "bootstrap": {"function": "pmc_pair_retrieval.compare.paired_differences",
                      "replicates": compare.N_PAIRED_RESAMPLES, "seed": metrics.PAIRED_BOOTSTRAP_SEED,
                      "unit": "article groups from all recorded image pairs", "interval": "percentile (2.5, 97.5)"},
        "configurations": configurations,
        "agreement_with_release": {"seed": 42, "reference": "results/results.json", "values_compared": compared_total,
                                   "equal": not mismatches_total, "mismatches": mismatches_total},
        "check_counts": counts, "checks": checks.items}


def render_markdown(report):
    seeds = report["seeds"]
    lines = ["# Three-seed retrieval results" if len(seeds) == 3 else "# Retrieval results across training seeds", "",
             report["descriptive_only"], "", report["verification_scope"], "",
             "The original analysis plan is in `PLAN.md`. Historical audit information is in `provenance.json`. "
             "The counts below refer to the checks performed by this portable report command.", "",
             "## Selected checkpoints", "",
             "| Configuration | Seed | Selected epoch | Epochs run | Validation Recall@10 at selection (%) |",
             "|---|---:|---:|---:|---:|"]
    for cfg, data in report["configurations"].items():
        for seed in seeds:
            m = data["seeds"][str(seed)]
            run = "not in released sidecar" if m["epochs_run"] is None else str(m["epochs_run"])
            label = "42 (released)" if seed == 42 else str(seed)
            lines.append(f"| {cfg} | {label} | {m['selected_epoch']} | {run} | {100 * m['selection_value']:.2f} |")

    def fmt(metric, value):
        return "n/a" if value is None else f"{value:.4f}" if metric == "MRR" else f"{value:.2f}"

    for split, title in (("test", "Test"), ("val", "Validation")):
        lines += ["", f"## {title} metrics", "", report["units"], ""]
        for cfg, data in report["configurations"].items():
            lines += [f"### {cfg}", "", "| Model / seed | " + " | ".join(METRICS) + " |",
                      "|---|" + "---:|" * len(METRICS)]
            rows = [("Pretrained", data["pretrained"][split]["metrics"])]
            rows += [(f"Seed {seed}" + (" (released)" if seed == 42 else ""), data["seeds"][str(seed)]["splits"][split]["metrics"])
                     for seed in seeds]
            for name, values in rows:
                lines.append("| " + name + " | " + " | ".join(fmt(m, values[m]) for m in METRICS) + " |")
            spread = data["across_seeds"]["metrics"][split]
            lines += ["| Mean ± sample SD | " + " | ".join(f"{fmt(m, spread[m]['mean'])} ± {fmt(m, spread[m]['sd'])}"
                                                               for m in METRICS) + " |",
                      "| Minimum to maximum | " + " | ".join(f"{fmt(m, spread[m]['min'])} to {fmt(m, spread[m]['max'])}"
                                                               for m in METRICS) + " |", ""]
    b = report["bootstrap"]
    lines += ["## Paired differences from pretrained", "",
              f"Fine-tuned minus pretrained, in percentage points, with 95% intervals from {b['replicates']:,} "
              f"query-weighted paired bootstrap resamples of article groups (seed {b['seed']}). "
              "Across-seed summaries describe the point differences.", "",
              "| Configuration | Split | Seed | Recall@10 | Hit@10 |", "|---|---|---|---:|---:|"]
    for cfg, data in report["configurations"].items():
        for split in ("test", "val"):
            for seed in seeds:
                values = data["seeds"][str(seed)]["paired_differences"][split]
                cells = []
                for metric in PAIRED_METRICS:
                    d = values[metric]
                    lo, hi = d["ci95_article_groups"]
                    cells.append(f"{d['fine_tuned_minus_pretrained']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
                lines.append(f"| {cfg} | {split} | {seed} | " + " | ".join(cells) + " |")
            spread = data["across_seeds"]["paired_differences"][split]
            for label, keys in (("Mean ± sample SD", ("mean", "sd")), ("Minimum to maximum", ("min", "max"))):
                sep = " ± " if keys[1] == "sd" else " to "
                cells = [sep.join(fmt(m, spread[m][k]) for k in keys) for m in PAIRED_METRICS]
                lines.append(f"| {cfg} | {split} | {label} | " + " | ".join(cells) + " |")
    agreement = report["agreement_with_release"]
    counts = report["check_counts"]
    lines += ["", "## Verification", "",
              f"Seed 42: {agreement['values_compared']} values compared with `results/results.json`, "
              f"{len(agreement['mismatches'])} mismatches.", "",
              f"Fresh required checks: {counts['required_passed']} passed, {counts['required_failed']} failed. "
              f"Informational checks: {counts['informational_passed']} passed, {counts['informational_failed']} "
              "did not. Full observed and expected values are in `multiseed.json`."]
    for check in report["checks"]:
        if not check["pass"]:
            lines.append(f"- {'Required failure' if check['required'] else 'Informational discrepancy'}: {check['check']}.")
    for cfg, data in report["configurations"].items():
        for seed in seeds:
            d = data["seeds"][str(seed)]["evaluator_val_R@10_minus_selection_value"]
            if abs(d) > 1e-6:
                lines += ["", f"For {cfg}, seed {seed}, evaluator validation Recall@10 differs from the trainer's "
                          f"selection value by {100 * d:+.8f} percentage points. Checkpoint selection used the "
                          "recorded training validation history. The tables use the independent evaluation."]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.multiseed", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, metavar="DATA",
                        help="directory containing image_pairs.json; images are not read")
    parser.add_argument("--output-dir", type=Path, metavar="OUT",
                        help="directory for new multiseed.json and multiseed.md; required unless --check is given")
    parser.add_argument("--check", action="store_true",
                        help="recompute from the shipped records and compare with the shipped reports; "
                             "cannot be combined with --output-dir, --runs-dir, --eval-dir or --verify-weights, "
                             "and --seeds, if given, must list the default seeds")
    parser.add_argument("--runs-dir", type=Path, metavar="DIR",
                        help="directory of new training runs, DIR/<configuration>_seed<N>/ (default: the shipped "
                             "results/multiseed/runs); requires --eval-dir")
    parser.add_argument("--eval-dir", type=Path, metavar="DIR",
                        help="directory of their evaluations, DIR/seed<N>/<configuration>/ (default: the shipped "
                             "results/multiseed/eval); requires --runs-dir")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS), metavar="N",
                        help="training seeds to report, for both configurations, in any order (default: 42 43 44). "
                             "The list must include 42, the released model, whose results come from the repository, "
                             "and must not repeat a seed. Without --runs-dir and --eval-dir, the shipped records "
                             "cover seeds 43 and 44. With them, each seed other than 42 needs its run and evaluations "
                             "there: shipped and new records are not merged")
    parser.add_argument("--verify-weights", action="store_true",
                        help="also hash the checkpoint file of every reported seed: those of seed 42 are the two "
                             "files in weights/, and any other seed's is best_model.safetensors in its run folder. "
                             "The checkpoints of seeds 43 and 44 are not shipped, so with the shipped records use "
                             "--seeds 42")
    args = parser.parse_args(argv)
    args.seeds = sorted(args.seeds)
    if 42 not in args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must include 42 and must not repeat a seed")
    if (args.runs_dir is None) != (args.eval_dir is None):
        parser.error("supply both --runs-dir and --eval-dir, or neither")
    args.custom = args.runs_dir is not None
    if args.check and (args.custom or args.output_dir is not None or args.verify_weights or tuple(args.seeds) != DEFAULT_SEEDS):
        parser.error("--check uses shipped inputs, seeds and reports and cannot be combined with other output/input options")
    if not args.check and args.output_dir is None:
        parser.error("--output-dir is required unless --check is used")
    args.runs_dir = (args.runs_dir or DEFAULT_INPUTS / "runs").resolve()
    args.eval_dir = (args.eval_dir or DEFAULT_INPUTS / "eval").resolve()
    args.data_dir = args.data_dir.resolve()

    def display(path):
        path = Path(path).resolve()
        if path.is_relative_to(ROOT):
            return path.relative_to(ROOT).as_posix()
        for root, name in ((args.runs_dir, "runs"), (args.eval_dir, "eval")):
            if path.is_relative_to(root):
                return name + "/" + path.relative_to(root).as_posix()
        raise ValueError(f"unexpected report input outside the repository and supplied run roots: {path}")

    args.display = display
    return args


def write_reports(args):
    paths = [args.output_dir / name for name in ("multiseed.json", "multiseed.md")]
    if args.output_dir.is_symlink() or any(p.exists() or p.is_symlink() for p in paths):
        raise ValueError("refusing to overwrite existing reports or write through an output-directory symlink")
    report = build_report(args)
    texts = [json.dumps(report, indent=2, allow_nan=False) + "\n", render_markdown(report)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, text in zip(paths, texts):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    print(f"[multiseed] {report['check_counts']}", flush=True)
    return report["check_counts"]["required_failed"] == 0


def main(argv=None):
    args = parse_args(argv)
    try:
        if not args.check:
            if args.output_dir.is_symlink():
                raise ValueError("refusing to write through an output-directory symlink")
            args.output_dir = args.output_dir.resolve()
            return 0 if write_reports(args) else 1
        with tempfile.TemporaryDirectory(prefix="pmc-pair-multiseed-") as temporary:
            args.output_dir = Path(temporary)
            ok = write_reports(args)
            for name in ("multiseed.json", "multiseed.md"):
                expected = (DEFAULT_INPUTS / name).read_text(encoding="utf-8")
                actual = (args.output_dir / name).read_text(encoding="utf-8")
                if actual != expected:
                    ok = False
                    print("".join(difflib.unified_diff(expected.splitlines(True), actual.splitlines(True),
                                                     fromfile=f"shipped/{name}", tofile=f"recomputed/{name}")))
            print("[multiseed] shipped reports match" if ok else "[multiseed] verification failed", flush=True)
            return 0 if ok else 1
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"[multiseed] {error}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
