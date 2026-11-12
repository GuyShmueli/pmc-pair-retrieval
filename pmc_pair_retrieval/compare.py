"""Results tables with paired differences from per-query evaluation files.

    python -m pmc_pair_retrieval.compare --data-dir DATA_DIR
    python -m pmc_pair_retrieval.compare --data-dir DATA_DIR --check
    python -m pmc_pair_retrieval.compare --data-dir DATA_DIR --per-query-dir eval

Input: the per-query files written by ``pmc_pair_retrieval.evaluate``,
``<per-query-dir>/<configuration>/<model>_<split>.jsonl`` for the configurations
``image_to_image`` and ``image_text_to_image``, the models ``pretrained`` and
``fine_tuned`` and the splits ``val`` and ``test`` (by default the shipped
``results/per_query``; missing files are skipped, and files with other model names are not
read). Only ``image_pairs.json`` is read from the data directory, to form the article
groups.

Ranked lists. The ranked lists that belong to a per-query file are referenced with their
sha256 when present: the evaluator's ``<model>_<split>.rankings.jsonl`` next to the
per-query file or, for the shipped per-query files, the shipped
``results/rankings/<configuration>/<model>_<split>.jsonl``. Their queries must be those of
the per-query file, in the same order.

Fine-tuned weights. Every fine-tuned cell names the weights it was computed with: the
weights file, the sha256 of the file and of its state dict, the selected epoch, the global
step and the validation value of the selection metric at that epoch. When the evaluator's
summary (``<model>_<split>.summary.json``) lies next to the per-query file, these values
come from the summary; otherwise from the sidecar ``<weights-dir>/<configuration>.json``
(by default the sidecars of the released weights in ``weights/``). A summary or sidecar that
describes another retrieval task or weights of another configuration is refused.

Outputs:

- ``--output`` (default: ``results/results.json`` for the shipped per-query files, otherwise
  ``PER_QUERY_DIR/results.json``): per configuration, model and split the
  number of queries, the mean Recall@k, Hit@k, mAP and nDCG in percent, MRR as a fraction,
  the median rank of the first relevant item, the per-query file with its sha256, the
  ranked lists with their sha256 if present and, for fine-tuned cells, the weights; per
  configuration and split the mean paired difference fine-tuned minus pretrained for R@10,
  Hit@10, R@1, R@100 and MRR with 95% intervals from a query-weighted paired bootstrap
  (5,000 resamples) over article groups and, for comparison, over articles;
- the same tables in Markdown, generated from the results file, next to it with the
  suffix ``.md`` (``results/results.md``);
- when the results file is ``results/results.json`` (the default for the shipped per-query
  files, or given explicitly), the blocks of ``README.md`` between the markers
  ``<!-- BEGIN compare:NAME -->`` and ``<!-- END compare:NAME -->`` (the expected results and
  the released weights), generated from the results file as well.

``--check`` recomputes everything and compares it with the existing files instead of
writing them; it exits with status 1 if anything differs.

Article groups are the units of the article-disjoint split: images are linked when they
come from the same article or form an annotated pair, and a group is a connected set of
linked images, named by its smallest image id (``split.article_groups``, the same function
that builds the split). Pairs never cross groups, so resampling groups keeps every query
together with its positives; resampling articles does not, and gives narrower intervals.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from . import metrics
from .model import MODEL_ID, MODEL_REVISION, WEIGHTS_FILE, WEIGHTS_SHA256, describe_weights, sha256_file
from .split import article_groups, read_pairs

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PER_QUERY_DIR = REPOSITORY_ROOT / "results" / "per_query"
DEFAULT_RANKINGS_DIR = REPOSITORY_ROOT / "results" / "rankings"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "results.json"
DEFAULT_WEIGHTS_DIR = REPOSITORY_ROOT / "weights"
README = REPOSITORY_ROOT / "README.md"
CONFIGURATIONS = ("image_to_image", "image_text_to_image")
MODELS = ("pretrained", "fine_tuned")
SPLITS = ("val", "test")
PAIRED_METRICS = ("R@10", "Hit@10", "R@1", "R@100", "MRR")
N_PAIRED_RESAMPLES = 5000
DESCRIPTION = ("Retrieval results on the article-disjoint PMC-PaIR split. Recall@k, Hit@k, mAP and nDCG are means "
               "over queries in percent; MRR and the selection value of the fine-tuned weights are fractions.")
QUERY_TEXT_DEFINITION = ("image_text_to_image queries: cos(image, g) + mean of the unit embeddings of the query "
                         "image's distinct descriptions (not renormalised) . g")
MODEL_LABELS = {"pretrained": "pretrained", "fine_tuned": "fine-tuned"}
SPLIT_LABELS = {"test": "Test", "val": "Validation"}
KIND_LABELS = {"vision_tower": "vision tower", "full_model": "full model"}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _display_path(path: Path, output: Path) -> str:
    """Repository-relative path, or a path relative to the output file for files outside the repository."""
    path = path.resolve()
    if path.is_relative_to(REPOSITORY_ROOT):
        return path.relative_to(REPOSITORY_ROOT).as_posix()
    return Path(os.path.relpath(path, output.resolve().parent)).as_posix()


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
def rankings_path(per_query_path: Path) -> Path | None:
    """The ranked lists that belong to a per-query file, or ``None`` if there are none.

    The evaluator writes ``<model>_<split>.rankings.jsonl`` next to ``<model>_<split>.jsonl``;
    the ranked lists of the shipped per-query files are in ``results/rankings/``.
    """
    beside = per_query_path.with_name(per_query_path.name.removesuffix(".jsonl") + ".rankings.jsonl")
    if beside.is_file():
        return beside
    if per_query_path.resolve().parent.parent == DEFAULT_PER_QUERY_DIR:
        shipped = DEFAULT_RANKINGS_DIR / per_query_path.parent.name / per_query_path.name
        if shipped.is_file():
            return shipped
    return None


def cell_entry(rows: Sequence[dict], path: Path, output: Path) -> dict:
    """Means of one evaluated cell (metrics in percent) and the references to its per-query file and ranked lists."""
    summ = metrics.summarize(rows)
    entry = {"n_queries": summ["n_queries"],
             # Every image of the split is a query, so the gallery has as many images as there are queries.
             "gallery_size": len(rows),
             **{m: 100 * summ[m]["mean"] for m in metrics.SUMMARY_METRICS},
             "MRR": summ["MRR"], "median_first_relevant_rank": summ["median_first_relevant_rank"],
             "per_query_file": _display_path(path, output), "per_query_sha256": sha256_file(path)}
    ranked_path = rankings_path(path)
    if ranked_path is not None:
        if [r["query_id"] for r in _read_jsonl(ranked_path)] != [r["query_id"] for r in rows]:
            raise ValueError(f"{ranked_path} does not list the queries of {path} in the same order")
        entry["rankings_file"] = _display_path(ranked_path, output)
        entry["rankings_sha256"] = sha256_file(ranked_path)
    return entry


def weights_from_sidecar(configuration: str, weights_dir: Path, output: Path) -> dict | None:
    """The weights of a configuration's fine-tuned cells from ``<weights_dir>/<configuration>.json``, if it exists."""
    sidecar_path = weights_dir / f"{configuration}.json"
    if not sidecar_path.is_file():
        return None
    with open(sidecar_path, encoding="utf-8") as fh:
        sidecar = json.load(fh)
    if sidecar.get("configuration") != configuration:
        raise ValueError(f"{sidecar_path} describes weights of {sidecar.get('configuration')!r}, not {configuration!r}")
    return {"file": _display_path(sidecar_path.parent / sidecar["weights_file"], output), "kind": sidecar["kind"],
            "n_tensors": sidecar["n_tensors"], "sha256": sidecar["sha256"],
            "state_dict_sha256": sidecar["state_dict_sha256"], **describe_weights(sidecar)}


def weights_from_summary(summary: Mapping, configuration: str, output: Path) -> dict:
    """The weights of one fine-tuned cell of ``configuration`` as the evaluator's summary records them.

    The summary must describe the retrieval task ``configuration`` and, when its weights name
    a configuration (from their sidecar), weights of that configuration.
    """
    w = summary["weights"]
    retrieval, trained_for = summary.get("retrieval", configuration), w.get("configuration", configuration)
    if retrieval != configuration or trained_for != configuration:
        raise ValueError(f"the evaluator summary describes retrieval {summary.get('retrieval')!r} with weights of "
                         f"{w.get('configuration')!r}, not {configuration!r}")
    return {"file": _display_path(Path(w["file"]), output),
            **{k: w[k] for k in ("kind", "n_tensors", "sha256", "state_dict_sha256", "configuration", "epoch",
                                 "global_step", "selection") if k in w}}


def paired_differences(fine_tuned: Sequence[dict], pretrained: Sequence[dict], groups: dict[str, str],
                       articles: dict[str, str]) -> dict:
    """Mean difference fine-tuned minus pretrained per metric of ``PAIRED_METRICS`` with paired 95% intervals.

    R@k and Hit@k differences are in percent, MRR differences are fractions. Queries are
    ordered by sorted id, as the bootstrap requires.
    """
    a = {r["query_id"]: r for r in fine_tuned}
    b = {r["query_id"]: r for r in pretrained}
    ids = sorted(a)
    if ids != sorted(b):
        raise ValueError("fine-tuned and pretrained per-query files cover different queries")
    unknown = [i for i in ids if i not in groups]
    if unknown:
        raise ValueError(f"{len(unknown)} query ids are not in image_pairs.json; example {unknown[0]}")
    group_arr = np.array([groups[i] for i in ids])
    article_arr = np.array([articles[i] for i in ids])
    out = {}
    for m in PAIRED_METRICS:
        if m == "MRR":
            x = np.array([1 / a[i]["first_rel_rank"] for i in ids])
            y = np.array([1 / b[i]["first_rel_rank"] for i in ids])
            scale = 1.0
        else:
            x = np.array([a[i][m] for i in ids], float)
            y = np.array([b[i][m] for i in ids], float)
            scale = 100.0
        d = x - y
        out[m] = {"fine_tuned_minus_pretrained": float(scale * d.mean()),
                  "ci95_article_groups": [scale * v for v in metrics.paired_group_ci(d, group_arr, N_PAIRED_RESAMPLES)],
                  "ci95_articles": [scale * v for v in metrics.paired_group_ci(d, article_arr, N_PAIRED_RESAMPLES)]}
    return out


def build_results(per_query_dir: str | os.PathLike, data_dir: str | os.PathLike, output: str | os.PathLike,
                  weights_dir: str | os.PathLike = DEFAULT_WEIGHTS_DIR) -> dict:
    """The content of the results file for all per-query files found in ``per_query_dir``.

    The article groups and articles of the paired bootstrap come from ``split.article_groups``
    over all pairs of ``DATA_DIR/image_pairs.json``; the image files are not needed.
    """
    per_query_dir, output, weights_dir = Path(per_query_dir), Path(output), Path(weights_dir)
    groups, articles = article_groups(read_pairs(data_dir))
    results = {"description": DESCRIPTION, "image_text_query_text": "mean",
               "query_text_definition": QUERY_TEXT_DEFINITION,
               "pretrained_model": {"model": MODEL_ID, "revision": MODEL_REVISION, "weights_file": WEIGHTS_FILE,
                                    "sha256": WEIGHTS_SHA256},
               "configurations": {}, "paired_differences": {}}
    rows = {}
    for conf in CONFIGURATIONS:
        for model in MODELS:
            for split in SPLITS:
                path = per_query_dir / conf / f"{model}_{split}.jsonl"
                if not path.is_file():
                    continue
                rows[(conf, model, split)] = _read_jsonl(path)
                entry = cell_entry(rows[(conf, model, split)], path, output)
                if model == "fine_tuned":
                    summary_path = path.with_name(f"{model}_{split}.summary.json")
                    if summary_path.is_file():
                        with open(summary_path, encoding="utf-8") as fh:
                            summary = json.load(fh)
                        if summary["n_queries"] != entry["n_queries"]:
                            raise ValueError(f"{summary_path} does not belong to {path}")
                        weights = weights_from_summary(summary, conf, output)
                    else:
                        weights = weights_from_sidecar(conf, weights_dir, output)
                    if weights is not None:
                        entry["weights"] = weights
                results["configurations"].setdefault(conf, {}).setdefault(model, {})[split] = entry
    for split in SPLITS:
        for conf in CONFIGURATIONS:
            if (conf, "fine_tuned", split) in rows and (conf, "pretrained", split) in rows:
                results["paired_differences"].setdefault(conf, {})[split] = paired_differences(
                    rows[(conf, "fine_tuned", split)], rows[(conf, "pretrained", split)], groups, articles)
    results["bootstrap"] = {
        "replicates": N_PAIRED_RESAMPLES,
        "unit": "article groups (images linked by a shared article or an annotated pair); articles alongside",
        "seed": metrics.PAIRED_BOOTSTRAP_SEED,
        "method": "query-weighted paired bootstrap: groups resampled with replacement, statistic = mean difference "
                  "over all queries of the drawn groups; percentile interval (2.5, 97.5)"}
    return results


# --------------------------------------------------------------------------------------
# Markdown, generated from the results file
# --------------------------------------------------------------------------------------
def _cells(results: Mapping, split: str):
    """``(configuration, model, entry)`` of every cell of a split, in table order."""
    for conf in CONFIGURATIONS:
        for model in MODELS:
            entry = results["configurations"].get(conf, {}).get(model, {}).get(split)
            if entry is not None:
                yield conf, model, entry


def _n_queries(results: Mapping, split: str) -> int | None:
    return next((entry["n_queries"] for _, _, entry in _cells(results, split)), None)


def _paragraph(text: str) -> list[str]:
    """A paragraph wrapped at 118 characters, never inside a code span or between a number and its unit."""
    keep = "\x00"
    parts = text.split("`")
    for i in range(1, len(parts), 2):
        parts[i] = parts[i].replace(" ", keep)
    text = "`".join(parts).replace(" px", keep + "px")
    return [line.replace(keep, " ") for line in textwrap.wrap(text, width=118, break_long_words=False,
                                                                break_on_hyphens=False)]


def _count(value) -> str:
    return f"{value:,}" if isinstance(value, int) else ""


def _signed(value: float, metric: str) -> str:
    return f"{value:+.4f}" if metric == "MRR" else f"{value:+.2f}"


def _interval(d: Mapping, key: str, metric: str) -> str:
    lo, hi = d[key]
    return f"[{_signed(lo, metric)}, {_signed(hi, metric)}]"


def _weights(results: Mapping) -> list[tuple[str, dict]]:
    """``(configuration, weights)`` of each configuration with fine-tuned cells, from its validation cell if present."""
    out = []
    for conf in CONFIGURATIONS:
        cells = results["configurations"].get(conf, {}).get("fine_tuned", {})
        weights = [cells[s]["weights"] for s in SPLITS if s in cells and "weights" in cells[s]]
        if weights:
            out.append((conf, weights[0]))
    return out


def _selection(weights: Mapping) -> str:
    sel = weights.get("selection")
    return f"{100 * sel['value']:.2f}" if sel else ""


def results_markdown(results: Mapping) -> str:
    """The text of ``results.md``: every table of the results file in Markdown."""
    lines = [
        "# Results",
        "",
        *_paragraph("Retrieval on the article-disjoint PMC-PaIR split. Every image of a split is a query, and the "
                    "images of the same split form the gallery. The relevant images of a query are its direct pair "
                    "partners; images linked to it only through a chain of pairs are left out of its ranking, and the "
                    "query itself ranks last. Images are encoded from 3 views at 224 px. image_to_image queries are "
                    "images; image_text_to_image queries add, to the image similarity, the dot product of the gallery "
                    "image embedding with the mean of the unit embeddings of the query image's distinct descriptions "
                    "(README, \"Query text\")."),
        "",
        *_paragraph("Recall@k (R@k), Hit@k, mAP and nDCG are means over queries in percent, MRR is a fraction, and the "
                    "median rank is the median rank of the first relevant image. This file and `results.json`, which "
                    "holds the unrounded values, are generated by `python -m pmc_pair_retrieval.compare` from the "
                    "per-query files that `results.json` lists."),
    ]
    for split in ("test", "val"):
        n = _n_queries(results, split)
        if n is None:
            continue
        lines += ["", f"## {SPLIT_LABELS[split]} split ({n:,} queries)", "",
                  "| Configuration | Model | R@1 | R@5 | R@10 | R@100 | Hit@10 | mAP | nDCG | MRR | Median rank |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for conf, model, e in _cells(results, split):
            values = " | ".join(f"{e[m]:.2f}" for m in ("R@1", "R@5", "R@10", "R@100", "Hit@10", "mAP", "nDCG"))
            lines.append(f"| {conf} | {MODEL_LABELS[model]} | {values} | {e['MRR']:.4f} | "
                         f"{e['median_first_relevant_rank']:g} |")
    if results["paired_differences"]:
        lines += ["", "## Fine-tuned minus pretrained", "",
                  *_paragraph("Mean paired difference over the same queries. The 95% intervals come from a "
                              "query-weighted paired bootstrap "
                              f"({results['bootstrap']['replicates']:,} resamples) over article groups, the units of "
                              "the split (images linked by a shared article or a pair), and, for comparison, over "
                              "articles. R@k and Hit@k differences are in percentage points, MRR differences are "
                              "fractions.")]
        for split in ("test", "val"):
            entries = [(conf, results["paired_differences"][conf][split]) for conf in CONFIGURATIONS
                       if split in results["paired_differences"].get(conf, {})]
            if not entries:
                continue
            lines += ["", f"### {SPLIT_LABELS[split]} split", "",
                      "| Configuration | Metric | Fine-tuned minus pretrained | 95% CI, article groups "
                      "| 95% CI, articles |",
                      "|---|---|---:|---:|---:|"]
            for conf, per_metric in entries:
                for m in PAIRED_METRICS:
                    d = per_metric[m]
                    lines.append(f"| {conf} | {m} | {_signed(d['fine_tuned_minus_pretrained'], m)} | "
                                 f"{_interval(d, 'ci95_article_groups', m)} | {_interval(d, 'ci95_articles', m)} |")
    weights = _weights(results)
    if weights:
        lines += ["", "## Fine-tuned weights", "",
                  *_paragraph("The fine-tuned results of each configuration were computed with these weights. Each "
                              "trainer keeps the epoch with the highest validation Recall@10 of its own retrieval task "
                              "(image queries for image_to_image, image-plus-description queries for "
                              "image_text_to_image); a later epoch must be strictly better. The last columns give the "
                              "trainer's validation R@10 at the selected epoch and the sha256 of the weights file."),
                  "",
                  "| Configuration | Weights file | Contents | Selected epoch | Global step | Validation R@10 at "
                  "selection (%) | File sha256 |",
                  "|---|---|---|---:|---:|---:|---|"]
        for conf, w in weights:
            lines.append(f"| {conf} | `{w['file']}` | {KIND_LABELS.get(w.get('kind'), w.get('kind', ''))}, "
                         f"{w.get('n_tensors', '')} tensors | {w.get('epoch', '')} | {_count(w.get('global_step'))} | "
                         f"{_selection(w)} | `{w['sha256']}` |")
    return "\n".join(lines) + "\n"


def readme_blocks(results: Mapping) -> dict[str, str]:
    """The generated blocks of README.md: ``expected-results`` and ``released-weights``."""
    lines = _paragraph("Means over queries in percent (MRR as a fraction), from `results/results.json`; "
                       "`results/results.md` has every metric and all paired differences.")
    for split in ("val", "test"):
        n = _n_queries(results, split)
        if n is None:
            continue
        lines += ["", f"{SPLIT_LABELS[split]} split ({n:,} queries):", "",
                  "| Configuration | Model | R@1 | R@10 | R@100 | Hit@10 | mAP | MRR |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for conf, model, e in _cells(results, split):
            values = " | ".join(f"{e[m]:.2f}" for m in ("R@1", "R@10", "R@100", "Hit@10", "mAP"))
            lines.append(f"| {conf} | {MODEL_LABELS[model]} | {values} | {e['MRR']:.4f} |")
    if results["paired_differences"]:
        lines += ["", *_paragraph("Fine-tuned minus pretrained, with 95% intervals from a paired bootstrap over "
                                  f"article groups ({results['bootstrap']['replicates']:,} resamples):"), "",
                  "| Configuration | Split | R@10 | Hit@10 | MRR |",
                  "|---|---|---:|---:|---:|"]
        for split in ("val", "test"):
            for conf in CONFIGURATIONS:
                d = results["paired_differences"].get(conf, {}).get(split)
                if d is None:
                    continue
                cells = [f"{_signed(d[m]['fine_tuned_minus_pretrained'], m)} "
                         f"{_interval(d[m], 'ci95_article_groups', m)}" for m in ("R@10", "Hit@10", "MRR")]
                lines.append(f"| {conf} | {SPLIT_LABELS[split].lower()} | " + " | ".join(cells) + " |")
    weights = ["| Configuration | File | Contents | Selected epoch | Global step | Validation R@10 at selection (%) "
               "| File sha256 | State-dict sha256 |",
               "|---|---|---|---:|---:|---:|---|---|"]
    for conf, w in _weights(results):
        weights.append(f"| {conf} | `{w['file']}` | {KIND_LABELS.get(w.get('kind'), w.get('kind', ''))}, "
                       f"{w.get('n_tensors', '')} tensors | {w.get('epoch', '')} | {_count(w.get('global_step'))} | "
                       f"{_selection(w)} | `{w['sha256']}` | `{w['state_dict_sha256']}` |")
    return {"expected-results": "\n".join(lines), "released-weights": "\n".join(weights)}


def replace_readme_blocks(text: str, blocks: Mapping[str, str]) -> str:
    """``text`` with the content between ``<!-- BEGIN compare:NAME -->`` and ``<!-- END compare:NAME -->`` replaced."""
    for name, content in blocks.items():
        begin, end = re.escape(f"<!-- BEGIN compare:{name} -->"), re.escape(f"<!-- END compare:{name} -->")
        pattern = re.compile(rf"^({begin}\n)(?:.*?\n)?({end})$", flags=re.S | re.M)
        if len(pattern.findall(text)) != 1:
            raise SystemExit(f"README.md must contain the markers of the generated block {name!r} exactly once")
        text = pattern.sub(lambda m: m.group(1) + content + "\n" + m.group(2), text)
    return text


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def _differences(new, old, prefix: str = "") -> list[str]:
    if isinstance(new, dict) and isinstance(old, dict):
        out = [f"{prefix}{k}: missing" for k in old.keys() - new.keys()]
        out += [f"{prefix}{k}: unexpected" for k in new.keys() - old.keys()]
        for k in new.keys() & old.keys():
            out += _differences(new[k], old[k], f"{prefix}{k}.")
        return out
    if isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
        return [d for i, (x, y) in enumerate(zip(new, old)) for d in _differences(x, y, f"{prefix}{i}.")]
    return [] if new == old else [f"{prefix[:-1]}: {old!r} -> {new!r}"]


def main(argv: list[str] | None = None) -> None:
    """Write (or ``--check``) the results file, its Markdown tables and the README blocks."""
    parser = argparse.ArgumentParser(prog="python -m pmc_pair_retrieval.compare",
                                     description="Results tables with paired differences from per-query files.")
    parser.add_argument("--data-dir", required=True, help="directory with image_pairs.json (images are not needed)")
    parser.add_argument("--per-query-dir", default=str(DEFAULT_PER_QUERY_DIR),
                        help="evaluation output directory (default: results/per_query)")
    parser.add_argument("--output", default=None,
                        help="results file; the Markdown tables go next to it, with the suffix .md (default: "
                             "results/results.json for the shipped per-query files, otherwise "
                             "PER_QUERY_DIR/results.json)")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR),
                        help="directory with the <configuration>.json sidecars of the fine-tuned weights, used for "
                             "fine-tuned cells without an evaluator summary (default: weights)")
    parser.add_argument("--check", action="store_true",
                        help="compare with the existing files instead of writing them; exit 1 on differences")
    args = parser.parse_args(argv)

    if args.output is not None:
        output = Path(args.output)
    elif Path(args.per_query_dir).resolve() == DEFAULT_PER_QUERY_DIR:
        output = DEFAULT_OUTPUT
    else:
        output = Path(args.per_query_dir) / "results.json"
    results = build_results(args.per_query_dir, args.data_dir, output, args.weights_dir)
    if not results["configurations"]:
        raise SystemExit(f"No per-query files found in {args.per_query_dir}")
    # Round-trip through JSON so that the comparison sees exactly what would be written.
    text = json.dumps(results, indent=2) + "\n"
    markdown = results_markdown(json.loads(text))
    texts = {output.with_suffix(".md"): markdown}
    if output.resolve() == DEFAULT_OUTPUT:
        texts[README] = replace_readme_blocks(README.read_text(encoding="utf-8"), readme_blocks(json.loads(text)))
    print(markdown, end="")

    if args.check:
        failed = False
        if not output.is_file():
            print(f"[compare] {output} does not exist", file=sys.stderr)
            failed = True
        else:
            with open(output, encoding="utf-8") as fh:
                diffs = _differences(json.loads(text), json.load(fh))
            if diffs:
                print(f"[compare] {output} differs in {len(diffs)} values:", *diffs[:50], sep="\n  ", file=sys.stderr)
                failed = True
        for path, expected in texts.items():
            found = path.read_text(encoding="utf-8") if path.is_file() else ""
            if found != expected:
                diff = list(difflib.unified_diff(found.splitlines(), expected.splitlines(), str(path), "generated",
                                                 lineterm="", n=0))
                print(f"[compare] {path} differs from the generated text:", *diff[:60], sep="\n  ", file=sys.stderr)
                failed = True
        if failed:
            raise SystemExit(1)
        print(f"[compare] reproduced exactly from {args.per_query_dir}: {output}, "
              + ", ".join(str(p) for p in texts))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        fh.write(text)
    for path, content in texts.items():
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    print(f"[compare] wrote {output}, " + ", ".join(str(p) for p in texts))


if __name__ == "__main__":
    main()
