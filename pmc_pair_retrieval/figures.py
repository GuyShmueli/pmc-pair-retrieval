"""Evaluation and error-analysis figures of the two released configurations.

    python -m pmc_pair_retrieval.figures --data-dir DATA_DIR --output-dir figures
    python -m pmc_pair_retrieval.figures --data-dir DATA_DIR --check

The figures compare each released configuration, image_to_image (i2i) and
image_text_to_image ((i+t)2i), with pretrained BiomedCLIP on the 8,096 test queries.
Nothing is re-scored: every number comes from the evaluation shipped in
``results/``.

Inputs:

- package files: the per-query metrics ``results/per_query/<configuration>/<model>_test.jsonl``
  and the top-10 ranked lists ``results/rankings/<configuration>/<model>_test.jsonl`` of
  the same evaluation, whose sha256 must equal the references in ``results/results.json``;
  ``results/results.json`` itself; the weight sidecars ``weights/<configuration>.json``
  (selected epoch and configured number of epochs); ``data/article_disjoint_split.json``,
  ``data/image_sizes.csv`` and ``data/figure_image_exclusions.csv`` (images that the
  example figures do not show, see selection rule 3);
- public dataset files in ``DATA_DIR``: ``image_pairs.json`` (recorded positives,
  descriptions, article groups), ``image_labels.jsonl`` (modality, anatomy and diagnosis
  of every image), ``licenses.csv`` and ``withheld_image_ids.txt`` (which images may be
  shown), and ``images/<image_id>.jpg`` of the displayed images only. None of the 68
  withheld images is needed.

Figures (180 mm wide, Arial 7 pt; see ``figure_drawing``):

- ``retrieval_performance``: mean Recall@1, Recall@10, Recall@100 and Hit@10 of both models
  (a), their paired differences with 95% confidence intervals (b) and the share of queries
  whose Recall@10 or Hit@10 changed (c);
- ``retrieval_subgroups_hit10``: paired Hit@10 difference with 95% interval by modality,
  anatomy and diagnosis (categories with at least 100 queries and 20 article groups),
  for both configurations. Filled green circles mark intervals above zero, and open grey
  circles mark intervals that include zero. Other categories and the aspect-ratio and
  description-count strata remain in the source data, with both Recall@10 and Hit@10;
- ``retrieval_example_<category>_<configuration>`` for the categories improvement,
  deterioration and continued_miss and both configurations: one example each, the query,
  its recorded positive and ranks 1 to 5 of both models.

Statistics. The paired differences of panel b are computed by
``compare.paired_differences`` and equal ``results/results.json``. Subgroup intervals use the
same bootstrap, ``metrics.paired_group_ci`` over the article groups of
``split.article_groups`` (5,000 resamples), restricted to the queries of the category.

Example selection, for each category (token ``improvement``, ``deterioration`` or
``continued_miss``) and configuration (``image_to_image`` or ``image_text_to_image``):

1. the query has exactly one recorded positive, so the rank of the positive is the first
   relevant rank of the per-query file and the (i+t)2i query text is that pair's
   description;
2. category, from the rank r_p (pretrained) and r_f (fine-tuned) of the positive:
   improvement r_p > 10 and r_f <= 5; deterioration r_p <= 5 and r_f > 10; continued miss
   (``continued_miss``) r_p > 10 and r_f > 10;
3. every displayed image (query, positive and the top five of both models) may be shown:
   it is distributed in the public release (``file_redistribution_status`` included, not
   in ``withheld_image_ids.txt``), its image licence and every licence component are CC BY
   (any version) or a public-domain statement, commercial use is permitted and there is no
   share-alike condition; its longer side has at least ``figure_drawing.MIN_LONG_PX``
   pixels; and it is not listed in ``data/figure_image_exclusions.csv``, the images found
   by visual review of the candidate panels, in draw order up to each selected example, to
   show burned-in dates or times (of the examination or of birth), identification or
   accession numbers, ages or sexes, or names of persons, institutions or departments, or
   listed because such text could not be ruled out; images listed for an earlier state of
   the figures stay in the list;
4. the eligible queries are sorted in ascending order of the lower-case SHA-256 hex digest
   of the UTF-8 string ``"<configuration>/<category>/<query_id>"`` (for example
   ``image_to_image/continued_miss/9019962_1_2``), and the first query not already
   selected as the example query of an earlier figure is taken; the figures are processed
   in the order improvement, deterioration, continued_miss, and within each category
   image_to_image before image_text_to_image. The string and its digest are the
   ``selection_string`` and ``selection_key`` columns of ``source_data/examples.csv``.

The pool sizes in ``examples.csv`` count the single-positive test queries of the category
(rules 1 and 2, ``category_size``) and those that also pass rule 3 with the exclusion list
as it stands (``eligible_pool``). Burned-in text was reviewed only up to each selected
query, so ``eligible_pool`` is an upper bound on the queries whose displayed images show no
burned-in text.

Outputs in ``--output-dir`` (which must not contain any of them): ``<figure>.pdf``,
``<figure>.svg`` and ``<figure>.png`` for the eight figures; ``captions.tex`` (one LaTeX
figure environment per figure, legends of at most 350 words, numbers read from the source
data); ``image_references.tex`` (the reference list cited by the example legends);
``figure_pages.tex`` (a document that sets every figure with its legend on an A4 page, the
reference layout); ``source_data/*.csv`` (overall metrics, paired differences, per-query
change, subgroups, per-query table, examples with both top-10 lists, image credits); and
``manifest.json`` with the sha256 of every input and output, the library versions and the
font files.

``--check`` regenerates everything into a temporary directory (or ``--output-dir``) and
compares it with the shipped ``results/figures/``: each shipped file other than
``manifest.json`` byte for byte, every input and every other output with the sha256
recorded in the shipped manifest, and the font files by content (family, style, version
and sha256; their file names may differ). The software versions recorded in the manifest
are not compared; each one that differs is printed as a note. The PNG previews are
compared only when the FreeType versions agree, because FreeType rasterises their text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import re
import sys
import tempfile
import unicodedata
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import fontTools
import matplotlib
import numpy as np
import PIL
from matplotlib import ft2font
from PIL import features

from . import compare, metrics
from . import figure_drawing as drawing
from .dataset import connected_components, pair_caption
from .image_sizes import SIZES_FILE, load_sizes
from .split import SPLIT_FILE, article_groups, read_pairs

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PER_QUERY_DIR = REPOSITORY_ROOT / "results" / "per_query"
RANKINGS_DIR = REPOSITORY_ROOT / "results" / "rankings"
RESULTS_FILE = REPOSITORY_ROOT / "results" / "results.json"
WEIGHTS_DIR = REPOSITORY_ROOT / "weights"
EXCLUSIONS_FILE = REPOSITORY_ROOT / "data" / "figure_image_exclusions.csv"
SHIPPED_DIR = REPOSITORY_ROOT / "results" / "figures"
SPLIT = "test"
CONFIGURATIONS = compare.CONFIGURATIONS
MODELS = compare.MODELS
DATASET_FILES = ("image_pairs.json", "image_labels.jsonl", "licenses.csv", "withheld_image_ids.txt")
LEVEL_METRICS = ("R@1", "R@10", "R@100", "Hit@10")
CHANGE_METRICS = ("R@10", "Hit@10")
TOP_EVALUATED = 10
TOP_SHOWN = drawing.RANKS_SHOWN
MIN_SUBGROUP_QUERIES = 100
MIN_SUBGROUP_GROUPS = 20

EXAMPLE_CATEGORIES = ("improvement", "deterioration", "continued_miss")
EXAMPLE_FIGURES = tuple((category, conf) for category in EXAMPLE_CATEGORIES for conf in CONFIGURATIONS)


def example_stem(category: str, configuration: str) -> str:
    """File stem of the example figure of a category and configuration."""
    return f"retrieval_example_{category}_{configuration}"


FIGURES = ("retrieval_performance", "retrieval_subgroups_hit10") + tuple(example_stem(k, c) for k, c in EXAMPLE_FIGURES)
FIGURE_FORMATS = ("pdf", "svg", "png")
SOURCE_DATA = ("overall_metrics.csv", "paired_differences.csv", "per_query_change.csv", "subgroups.csv",
               "per_query.csv", "examples.csv", "image_credits.csv")
CAPTIONS = "captions.tex"
REFERENCES = "image_references.tex"
PAGES = "figure_pages.tex"
MANIFEST = "manifest.json"
OUTPUTS = tuple([f"{stem}.{ext}" for stem in FIGURES for ext in FIGURE_FORMATS] + [CAPTIONS, REFERENCES, PAGES]
                + [f"source_data/{name}" for name in SOURCE_DATA] + [MANIFEST])

CATEGORY_RULES: dict[str, Callable[[int, int], bool]] = {
    "improvement": lambda r_p, r_f: r_p > TOP_EVALUATED and r_f <= TOP_SHOWN,
    "deterioration": lambda r_p, r_f: r_p <= TOP_SHOWN and r_f > TOP_EVALUATED,
    "continued_miss": lambda r_p, r_f: r_p > TOP_EVALUATED and r_f > TOP_EVALUATED,
}
DISPLAY_LICENCE = re.compile(r"CC BY(?: \d\.\d| \(version not stated\))|Public domain \(publisher statement\)"
                             r"|Public Domain Mark 1\.0")
EXCLUSION_REASONS = ("date or time", "identification or accession number", "age or sex",
                     "name of a person or institution")

LABEL_FAMILIES = ("modality", "anatomy", "diagnosis")
STRATA_FAMILIES = ("aspect_ratio", "descriptions")
FAMILY_HEADINGS = {"modality": "Modality", "anatomy": "Anatomy", "diagnosis": "Diagnosis",
                   "aspect_ratio": "Aspect ratio", "descriptions": "Descriptions per query"}
ASPECT_STRATA = ("square", "up_to_2", "2_to_3", "over_3")
DESCRIPTION_STRATA = ("1", "2", "3_or_more")
CATEGORY_LABELS = {
    "modality": {
        "ct": "CT", "mri": "MRI", "radiography": "Radiography", "ultrasound": "Ultrasound",
        "angiography_fluoroscopy": "Angiography/fluoroscopy", "nuclear_medicine": "Nuclear medicine",
        "microscopy": "Microscopy", "optical_imaging": "Optical imaging", "photography": "Photography",
        "endoscopy": "Endoscopy", "electrophysiology": "Electrophysiology", "other": "Other"},
    "anatomy": {
        "vasculature": "Vasculature", "musculoskeletal": "Musculoskeletal", "thorax_respiratory": "Thorax/respiratory",
        "brain_intracranial": "Brain/intracranial", "abdomen_retroperitoneum": "Abdomen/retroperitoneum",
        "hepatobiliary_pancreatic": "Hepatobiliary/pancreatic", "reproductive_pelvic": "Reproductive/pelvic",
        "spine_spinal_cord": "Spine/spinal cord", "gastrointestinal": "Gastrointestinal", "heart": "Heart",
        "dental_oral": "Dental/oral", "lymphatic_splenic": "Lymphatic/splenic", "renal_urinary": "Renal/urinary",
        "ear_nose_throat": "Ear, nose, throat", "head_neck": "Head/neck", "eye_orbit": "Eye/orbit",
        "skin_soft_tissue": "Skin/soft tissue", "endocrine": "Endocrine", "breast": "Breast", "other": "Other"},
    "diagnosis": {
        "mass_or_lesion": "Mass/lesion", "aneurysm_or_dissection": "Aneurysm/dissection",
        "infection_or_abscess": "Infection/abscess", "inflammatory_or_autoimmune_disorder": "Inflammatory/autoimmune",
        "congenital_or_developmental_disorder": "Congenital/developmental",
        "fracture_or_traumatic_injury": "Fracture/trauma", "perforation_or_rupture": "Perforation/rupture",
        "thrombosis_or_embolism": "Thrombosis/embolism", "hemorrhage_or_hematoma": "Hemorrhage/hematoma",
        "cystic_disorder": "Cystic disorder", "enlargement_or_atrophy": "Enlargement/atrophy",
        "obstruction_or_volvulus": "Obstruction/volvulus", "stone_or_calcification": "Stone/calcification",
        "ischemia_or_infarction": "Ischemia/infarction", "other": "Other",
        "dislocation_or_instability": "Dislocation/instability", "edema_or_effusion": "Edema/effusion",
        "vascular_stenosis_or_vasospasm": "Vascular stenosis/vasospasm", "neurologic_disorder": "Neurologic disorder",
        "device_or_treatment_complication": "Device/treatment complication",
        "necrosis_or_tissue_damage": "Necrosis/tissue damage",
        "vascular_malformation_or_fistula": "Vascular malformation/fistula",
        "respiratory_disorder": "Respiratory disorder",
        "pregnancy_related_disorder": "Pregnancy-related", "musculoskeletal_disorder": "Musculoskeletal disorder",
        "abnormal_gas": "Abnormal gas", "fibrosis_or_scarring": "Fibrosis/scarring",
        "dental_or_oral_disorder": "Dental/oral disorder", "cardiac_disorder": "Cardiac disorder",
        "herniation_or_prolapse": "Herniation/prolapse", "eye_or_orbit_disorder": "Eye/orbit disorder",
        "postoperative_or_resolved_state": "Postoperative/resolved", "endocrine_or_metabolic_disorder":
        "Endocrine/metabolic", "normal_or_negative_report": "Normal/negative report",
        "degenerative_disorder": "Degenerative disorder",
        "gastrointestinal_or_hepatobiliary_disorder": "Gastrointestinal/hepatobiliary",
        "ear_nose_throat_disorder": "Ear, nose, throat disorder", "hematologic_disorder": "Hematologic disorder",
        "renal_or_urinary_disorder": "Renal/urinary disorder"},
    "aspect_ratio": {"square": "Square", "up_to_2": "Above 1:1 to 2:1", "2_to_3": "Above 2:1 to 3:1",
                     "over_3": "Above 3:1"},
    "descriptions": {"1": "1", "2": "2", "3_or_more": "3 or more"},
}
# Abbreviations that may occur in the displayed pair descriptions, defined in the example legends.
ABBREVIATIONS = {
    "CNS": "central nervous system",
    "CT": "computed tomography",
    "CTA": "computed tomography angiography",
    "DSA": "digital subtraction angiography",
    "MRA": "magnetic resonance angiography",
    "MRI": "magnetic resonance imaging",
    "NEHI": "neuroendocrine cell hyperplasia of infancy",
    "PET": "positron emission tomography",
    "SPECT": "single-photon emission computed tomography",
    "TRUSG": "transrectal ultrasonography",
}
ABBREVIATION_TOKEN = re.compile(r"\b[A-Z][A-Z0-9]+\b")


def category_label(family: str, category: str) -> str:
    """Short display label of a subgroup category."""
    label = CATEGORY_LABELS.get(family, {}).get(category)
    if label is None:
        label = category.replace("_or_", "/").replace("_", " ")
        label = label[:1].upper() + label[1:]
    return label


def sha256_file(path: str | Path) -> str:
    """sha256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
@dataclass
class Inputs:
    """Everything the figures read, loaded and cross-checked by ``load_inputs``."""

    data_dir: Path
    query_ids: list[str]
    rows: dict[tuple[str, str], list[dict]]
    rankings: dict[tuple[str, str], list[list[str]]]
    results: dict
    weights: dict[str, dict]
    partners: dict[str, set[str]]
    descriptions: dict[str, list[str]]
    groups: dict[str, str]
    articles: dict[str, str]
    ignored: dict[str, int]
    labels: dict[str, dict[str, list[str]]]
    sizes: dict[str, tuple[int, int]]
    licences: dict[str, dict[str, str]]
    withheld: set[str]
    exclusions: dict[str, str]
    files: dict[str, Path] = field(default_factory=dict)

    def licensed(self, image_id: str) -> bool:
        """Whether an image is distributed under a licence that allows it in a CC BY figure (rule 3)."""
        row = self.licences.get(image_id)
        if row is None or image_id in self.withheld or row["file_redistribution_status"] != "included":
            return False
        if row["image_commercial_use_under_recorded_license"] != "permitted" or row["image_share_alike"] != "no":
            return False
        components = json.loads(row["image_license_components"] or "[]")
        licences = [row["image_license"]] + [str(c.get("license", "")) for c in components]
        return all(DISPLAY_LICENCE.fullmatch(lic) for lic in licences)

    def displayable(self, image_id: str) -> bool:
        """Whether an example figure may show an image (selection rule 3)."""
        return (self.licensed(image_id) and max(self.sizes[image_id]) >= drawing.MIN_LONG_PX
                and image_id not in self.exclusions)

    def ranked_gallery_size(self, query_id: str) -> int:
        """Number of gallery images in the ranking of a query: the other test images minus its ignored ones."""
        return len(self.query_ids) - 1 - self.ignored[query_id]


LICENCE_COLUMNS = ("image_id", "pmcid", "source_article_url", "figure_xml_id", "file_redistribution_status",
                   "image_license", "image_license_url", "image_license_components",
                   "image_commercial_use_under_recorded_license", "image_share_alike", "article_authors",
                   "article_title", "journal", "year", "doi", "copyright_and_credit_notices", "attribution")


def read_exclusions(path: Path) -> dict[str, str]:
    """Image id -> reason, from ``data/figure_image_exclusions.csv`` (columns ``image_id`` and ``reason``)."""
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["image_id", "reason"]:
            raise SystemExit(f"[figures] {path} must have the columns image_id and reason")
        exclusions = {}
        for number, row in enumerate(reader, start=2):
            reasons = (row.get("reason") or "").split("; ")
            if (None in row or not row.get("image_id") or row["image_id"] in exclusions
                    or any(r not in EXCLUSION_REASONS for r in reasons)):
                raise SystemExit(f"[figures] {path}, line {number}: invalid or repeated entry")
            exclusions[row["image_id"]] = row["reason"]
    if list(exclusions) != sorted(exclusions):
        raise SystemExit(f"[figures] {path} must list the image ids in sorted order")
    return exclusions


def load_inputs(data_dir: str | Path) -> Inputs:
    """Read every input, check the evaluation files against ``results/results.json`` and each other."""
    data_dir = Path(data_dir)
    missing = [name for name in DATASET_FILES if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit(f"[figures] {data_dir} lacks the public dataset files {', '.join(missing)}")
    files: dict[str, Path] = {}

    def package_file(path: Path) -> Path:
        files[path.relative_to(REPOSITORY_ROOT).as_posix()] = path
        return path

    def data_file(name: str) -> Path:
        files[f"DATA_DIR/{name}"] = data_dir / name
        return data_dir / name

    with open(package_file(RESULTS_FILE), encoding="utf-8") as fh:
        results = json.load(fh)
    with open(package_file(SPLIT_FILE), encoding="utf-8") as fh:
        split = json.load(fh)
    query_ids = sorted(i for i, s in split.items() if s == SPLIT)
    rows, rankings, weights = {}, {}, {}
    for conf in CONFIGURATIONS:
        for model in MODELS:
            cell = results["configurations"][conf][model][SPLIT]
            per_query = package_file(PER_QUERY_DIR / conf / f"{model}_{SPLIT}.jsonl")
            ranked = package_file(RANKINGS_DIR / conf / f"{model}_{SPLIT}.jsonl")
            for path, key in ((per_query, "per_query_sha256"), (ranked, "rankings_sha256")):
                if sha256_file(path) != cell.get(key):
                    raise SystemExit(f"[figures] {path} does not have the sha256 that results/results.json records")
            rows[(conf, model)] = _read_jsonl(per_query)
            lists = _read_jsonl(ranked)
            for name, items in (("per-query file", rows[(conf, model)]), ("ranked lists", lists)):
                if [r["query_id"] for r in items] != query_ids:
                    raise SystemExit(f"[figures] the {name} of {conf}/{model} do not list the {SPLIT} queries in "
                                     "sorted order")
            rankings[(conf, model)] = [r["top10"] for r in lists]
        with open(package_file(WEIGHTS_DIR / f"{conf}.json"), encoding="utf-8") as fh:
            sidecar = json.load(fh)
        link = results["configurations"][conf]["fine_tuned"][SPLIT]["weights"]
        selection = link["selection"]
        if (sidecar["configuration"] != conf or sidecar["epoch"] != link["epoch"]
                or sidecar["global_step"] != link["global_step"] or sidecar["sha256"] != link["sha256"]
                or link["file"] != f"weights/{sidecar['weights_file']}"
                or sidecar[selection["split"]][selection["metric"]]["mean"] != selection["value"]
                or not 1 <= link["epoch"] <= sidecar["config"]["epochs"]):
            raise SystemExit(f"[figures] weights/{conf}.json disagrees with the weights recorded in "
                             "results/results.json")
        weights[conf] = {"file": link["file"], "epoch": link["epoch"], "epochs": sidecar["config"]["epochs"],
                         "global_step": link["global_step"], "selection_split": selection["split"],
                         "selection_metric": selection["metric"], "selection_value": selection["value"]}

    with open(data_file("image_pairs.json"), encoding="utf-8") as fh:
        records = json.load(fh)
    partners: dict[str, set[str]] = defaultdict(set)
    descriptions: dict[str, set[str]] = defaultdict(set)
    edges = set()
    for record in records:
        a, b = record["pair_id"]
        partners[a].add(b)
        partners[b].add(a)
        edges.add((a, b) if a < b else (b, a))
        text = pair_caption(record)
        descriptions[a].add(text)
        descriptions[b].add(text)
    groups, articles = article_groups(read_pairs(data_dir))
    components = connected_components(partners.keys(), edges)
    members: dict[str, set[str]] = defaultdict(set)
    for image_id in query_ids:
        members[components[image_id]].add(image_id)
    # Ignored items of a query (as in evaluate): the other test images of its connected component that are not
    # its direct partners.
    ignored = {q: len(members[components[q]] - partners[q] - {q}) for q in query_ids}

    labels = {}
    with open(data_file("image_labels.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                labels[record["image_id"]] = record["labels"]
    sizes = load_sizes(package_file(SIZES_FILE))
    licences = {}
    with open(data_file("licenses.csv"), encoding="utf-8", newline="") as fh:
        for record in csv.DictReader(fh):
            licences[record["image_id"]] = {k: record[k] for k in LICENCE_COLUMNS}
    with open(data_file("withheld_image_ids.txt"), encoding="utf-8") as fh:
        withheld = set(fh.read().split())
    exclusions = read_exclusions(package_file(EXCLUSIONS_FILE))
    unknown = [i for i in exclusions if i not in licences]
    if unknown:
        raise SystemExit(f"[figures] {EXCLUSIONS_FILE.name} lists images without a licence record ({unknown[0]})")

    absent = [q for q in query_ids if q not in labels or q not in sizes or q not in licences]
    if absent:
        raise SystemExit(f"[figures] {len(absent)} test queries lack labels, sizes or licence records "
                         f"(example {absent[0]})")
    inputs = Inputs(data_dir, query_ids, rows, rankings, results, weights, dict(partners),
                    {i: sorted(t) for i, t in descriptions.items()}, groups, articles, ignored, labels, sizes,
                    licences, withheld, exclusions, files)
    check_rankings(inputs, split)
    return inputs


def check_rankings(inputs: Inputs, split: Mapping[str, str]) -> None:
    """Check that every ranked list agrees with its per-query row and the recorded positives.

    A list has ten distinct test images other than the query; its recorded positives give
    the row's Hit@10, R@10 (computed in float32, as by the evaluator) and, when the first
    relevant rank is at most 10, that rank.
    """
    for key, rows in inputs.rows.items():
        for row, top in zip(rows, inputs.rankings[key]):
            q = row["query_id"]
            positives = inputs.partners[q]
            problems = []
            if len(top) != TOP_EVALUATED or len(set(top)) != TOP_EVALUATED or q in top:
                problems.append("not ten distinct images other than the query")
            if any(split.get(c) != SPLIT for c in top):
                problems.append("an image outside the split")
            if row["n_pos"] != len(positives):
                problems.append("n_pos differs from image_pairs.json")
            flags = [c in positives for c in top]
            first = flags.index(True) + 1 if any(flags) else None
            rank = row["first_rel_rank"]
            if (first is None) != (rank > TOP_EVALUATED) or (first is not None and first != rank):
                problems.append(f"first relevant rank {rank} disagrees with the list")
            if row["Hit@10"] != float(any(flags)):
                problems.append("Hit@10 disagrees with the list")
            if row["R@10"] != float(np.float32(sum(flags)) / np.float32(len(positives))):
                problems.append("R@10 disagrees with the list")
            if problems:
                raise SystemExit(f"[figures] ranked list of {q} ({key[0]}/{key[1]}): " + "; ".join(problems))


# --------------------------------------------------------------------------------------
# Source data
# --------------------------------------------------------------------------------------
# Units of the source-data columns: R@k, Hit@k, mAP and nDCG in percent, MRR as a fraction and
# differences in percentage points (MRR: fraction); the per-query table keeps the per-query
# values as fractions between 0 and 1, as in the per-query files. Column names carry the unit.
def _percent(metric: str) -> str:
    return f"{metric}_percent"


def overall_table(inputs: Inputs) -> list[dict]:
    """Means of the per-query metrics per configuration and model (equal to ``results/results.json``), and weights."""
    table = []
    for conf in CONFIGURATIONS:
        for model in MODELS:
            summary = metrics.summarize(inputs.rows[(conf, model)])
            values = {m: 100 * summary[m]["mean"] for m in metrics.SUMMARY_METRICS}
            values["MRR"] = summary["MRR"]
            values["median_first_relevant_rank"] = summary["median_first_relevant_rank"]
            cell = inputs.results["configurations"][conf][model][SPLIT]
            differing = [k for k, v in values.items() if cell.get(k) != v]
            if differing or cell["n_queries"] != summary["n_queries"]:
                raise SystemExit(f"[figures] {conf}/{model}: {', '.join(differing) or 'n_queries'} differ from "
                                 "results/results.json")
            row = {"configuration": conf, "model": model, "n_queries": summary["n_queries"]}
            row.update({_percent(m): values[m] for m in metrics.SUMMARY_METRICS})
            row["MRR_fraction"] = values["MRR"]
            row["median_first_relevant_rank"] = values["median_first_relevant_rank"]
            w = inputs.weights[conf] if model == "fine_tuned" else None
            row["weights"] = w["file"] if w else "pretrained"
            for key in ("epoch", "epochs", "global_step", "selection_split", "selection_metric"):
                row[key] = w[key] if w else ""
            row["selection_value_fraction"] = w["selection_value"] if w else ""
            table.append(row)
    return table


def differences_table(inputs: Inputs, overall: Sequence[Mapping]) -> list[dict]:
    """Paired differences fine-tuned minus pretrained from ``compare.paired_differences``."""
    means = {(r["configuration"], r["model"]): r for r in overall}
    n_groups = len({inputs.groups[q] for q in inputs.query_ids})
    n_articles = len({inputs.articles[q] for q in inputs.query_ids})
    table = []
    for conf in CONFIGURATIONS:
        paired = compare.paired_differences(inputs.rows[(conf, "fine_tuned")], inputs.rows[(conf, "pretrained")],
                                            inputs.groups, inputs.articles)
        if paired != inputs.results["paired_differences"][conf][SPLIT]:
            raise SystemExit(f"[figures] the paired differences of {conf} differ from results/results.json")
        for metric in compare.PAIRED_METRICS:
            p = paired[metric]
            column = "MRR_fraction" if metric == "MRR" else _percent(metric)
            table.append({
                "configuration": conf, "metric": metric,
                "level_unit": "fraction" if metric == "MRR" else "percent",
                "difference_unit": "fraction" if metric == "MRR" else "percentage points",
                "pretrained": means[(conf, "pretrained")][column], "fine_tuned": means[(conf, "fine_tuned")][column],
                "difference": p["fine_tuned_minus_pretrained"],
                "ci95_low": p["ci95_article_groups"][0], "ci95_high": p["ci95_article_groups"][1],
                "ci95_low_articles": p["ci95_articles"][0], "ci95_high_articles": p["ci95_articles"][1],
                "n_queries": len(inputs.query_ids), "n_article_groups": n_groups, "n_articles": n_articles,
                "resamples": compare.N_PAIRED_RESAMPLES, "seed": metrics.PAIRED_BOOTSTRAP_SEED})
    return table


def change_table(inputs: Inputs) -> list[dict]:
    """Numbers and shares of queries whose Recall@10 or Hit@10 is higher, lower or equal with the fine-tuned model."""
    table = []
    n = len(inputs.query_ids)
    for conf in CONFIGURATIONS:
        for metric in CHANGE_METRICS:
            pt = np.array([r[metric] for r in inputs.rows[(conf, "pretrained")]], dtype=float)
            ft = np.array([r[metric] for r in inputs.rows[(conf, "fine_tuned")]], dtype=float)
            d = ft - pt
            if np.any((d != 0) & (np.abs(d) < 1e-6)):
                raise SystemExit(f"[figures] {conf} {metric}: per-query values differ by rounding only")
            better, worse = int((d > 0).sum()), int((d < 0).sum())
            zero, full = int(((d == 0) & (ft == 0)).sum()), int(((d == 0) & (ft == 1)).sum())
            unchanged = n - better - worse
            table.append({
                "configuration": conf, "metric": metric, "n_queries": n, "better": better, "worse": worse,
                "unchanged": unchanged, "unchanged_zero": zero, "unchanged_one": full,
                "unchanged_other": unchanged - zero - full, "better_percent": 100 * better / n,
                "worse_percent": 100 * worse / n, "unchanged_percent": 100 * unchanged / n,
                "unchanged_zero_percent": 100 * zero / n})
    return table


def aspect_stratum(width: int, height: int) -> str:
    """Aspect-ratio stratum of an image, longer over shorter side: 1, above 1 to 2, above 2 to 3, above 3."""
    long_side, short_side = max(width, height), min(width, height)
    if long_side == short_side:
        return "square"
    if long_side <= 2 * short_side:
        return "up_to_2"
    if long_side <= 3 * short_side:
        return "2_to_3"
    return "over_3"


def description_stratum(n: int) -> str:
    """Stratum of the number of distinct descriptions of a query image."""
    return str(n) if n < 3 else "3_or_more"


def subgroup_members(inputs: Inputs) -> dict[str, dict[str, list[int]]]:
    """Family -> category -> indices of the queries (in sorted query order) that belong to it."""
    members: dict[str, dict[str, list[int]]] = {f: defaultdict(list) for f in LABEL_FAMILIES + STRATA_FAMILIES}
    for k, q in enumerate(inputs.query_ids):
        for family in LABEL_FAMILIES:
            for category in inputs.labels[q][family]:
                members[family][category].append(k)
        members["aspect_ratio"][aspect_stratum(*inputs.sizes[q])].append(k)
        members["descriptions"][description_stratum(len(inputs.descriptions[q]))].append(k)
    return members


def drawn_subgroup(n_queries: int, n_groups: int) -> bool:
    """Whether counts meet the display thresholds: at least 100 queries from 20 article groups."""
    return n_queries >= MIN_SUBGROUP_QUERIES and n_groups >= MIN_SUBGROUP_GROUPS


def subgroup_table(inputs: Inputs) -> list[dict]:
    """Recall@10 and Hit@10 per subgroup category, with paired differences, 95% intervals and group concentration."""
    members = subgroup_members(inputs)
    group_array = np.array([inputs.groups[q] for q in inputs.query_ids])
    table = []
    for conf in CONFIGURATIONS:
        values = {(m, metric): np.array([r[metric] for r in inputs.rows[(conf, m)]], dtype=float)
                  for m in MODELS for metric in CHANGE_METRICS}
        for family in LABEL_FAMILIES + STRATA_FAMILIES:
            categories = members[family]
            counts = {c: Counter(group_array[np.array(idx)].tolist()) for c, idx in categories.items()}
            if family in STRATA_FAMILIES:
                order = [c for c in (ASPECT_STRATA if family == "aspect_ratio" else DESCRIPTION_STRATA)
                         if c in categories]
            else:
                ranked = sorted(categories, key=lambda c: (-len(categories[c]), c))
                order = ([c for c in ranked if drawn_subgroup(len(categories[c]), len(counts[c]))]
                         + [c for c in ranked if not drawn_subgroup(len(categories[c]), len(counts[c]))])
            for category in order:
                idx = np.array(categories[category])
                sizes = np.array(sorted(counts[category].values()), dtype=float)
                row = {"configuration": conf, "family": family, "category": category,
                       "label": category_label(family, category),
                       "shown": "yes" if family in LABEL_FAMILIES and drawn_subgroup(len(idx), len(sizes)) else "no",
                       "n_queries": len(idx), "n_article_groups": len(sizes),
                       "largest_group_share_percent": 100 * sizes.max() / len(idx),
                       "effective_article_groups": float(sizes.sum() ** 2 / (sizes ** 2).sum())}
                for metric in CHANGE_METRICS:
                    pt, ft = values[("pretrained", metric)][idx], values[("fine_tuned", metric)][idx]
                    d = ft - pt
                    lo, hi = metrics.paired_group_ci(d, group_array[idx], compare.N_PAIRED_RESAMPLES)
                    row.update({f"pretrained_{metric}_percent": 100 * pt.mean(),
                                f"fine_tuned_{metric}_percent": 100 * ft.mean(),
                                f"difference_{metric}_pp": 100 * d.mean(), f"ci95_low_{metric}_pp": 100 * lo,
                                f"ci95_high_{metric}_pp": 100 * hi})
                table.append(row)
    return table


def per_query_table(inputs: Inputs) -> list[dict]:
    """One row per test query: article, positives, geometry, labels, displayability and the metrics of the figures."""
    table = []
    for k, q in enumerate(inputs.query_ids):
        width, height = inputs.sizes[q]
        row = {"query_id": q, "pmcid": f"PMC{inputs.articles[q]}", "article_group": inputs.groups[q],
               "n_recorded_positives": len(inputs.partners[q]), "n_descriptions": len(inputs.descriptions[q]),
               "ranked_gallery_size": inputs.ranked_gallery_size(q),
               "width": width, "height": height, "aspect_ratio": aspect_stratum(width, height)}
        for family in LABEL_FAMILIES:
            row[family] = "|".join(inputs.labels[q][family])
        row["displayable"] = "yes" if inputs.displayable(q) else "no"
        row["example_exclusion"] = inputs.exclusions.get(q, "")
        for conf in CONFIGURATIONS:
            for model in MODELS:
                r = inputs.rows[(conf, model)][k]
                row[f"{conf}_{model}_R@10_fraction"] = r["R@10"]
                row[f"{conf}_{model}_Hit@10_fraction"] = r["Hit@10"]
                row[f"{conf}_{model}_first_rel_rank"] = r["first_rel_rank"]
        table.append(row)
    return table


def selection_string(configuration: str, category: str, query_id: str) -> str:
    """The string whose SHA-256 digest orders the example draw: ``"<configuration>/<category>/<query_id>"``."""
    return f"{configuration}/{category}/{query_id}"


def selection_key(configuration: str, category: str, query_id: str) -> str:
    """Sort key of the example draw: lower-case SHA-256 hex digest of ``selection_string``."""
    return hashlib.sha256(selection_string(configuration, category, query_id).encode("utf-8")).hexdigest()


def candidate_kind(inputs: Inputs, query_id: str, image_id: str) -> str:
    """``positive`` (recorded pair partner), ``same_article`` (unannotated, query's article) or ``unannotated``."""
    if image_id in inputs.partners[query_id]:
        return "positive"
    if inputs.articles[image_id] == inputs.articles[query_id]:
        return "same_article"
    return "unannotated"


def shown_images(inputs: Inputs, configuration: str, k: int) -> list[str]:
    """The images an example of the ``k``-th test query displays: query, positive, top five of both models."""
    q = inputs.query_ids[k]
    (positive,) = inputs.partners[q]
    return ([q, positive] + inputs.rankings[(configuration, "pretrained")][k][:TOP_SHOWN]
            + inputs.rankings[(configuration, "fine_tuned")][k][:TOP_SHOWN])


def example_pool(inputs: Inputs, category: str, configuration: str) -> tuple[int, list[tuple[str, int, str]]]:
    """Number of single-positive test queries in a category (rules 1 and 2) and the eligible ones (rules 1 to 3).

    The eligible queries are returned as ``(selection key, index, query id)`` in draw order.
    Rule 3 uses the exclusion list as it stands; burned-in text was reviewed only for the
    candidate panels up to each selected example, so the number of eligible queries is an
    upper bound on those whose displayed images show no burned-in text.
    """
    pretrained = inputs.rows[(configuration, "pretrained")]
    fine_tuned = inputs.rows[(configuration, "fine_tuned")]
    size, eligible = 0, []
    for k, q in enumerate(inputs.query_ids):
        if pretrained[k]["n_pos"] != 1:
            continue
        if not CATEGORY_RULES[category](pretrained[k]["first_rel_rank"], fine_tuned[k]["first_rel_rank"]):
            continue
        size += 1
        if all(inputs.displayable(i) for i in shown_images(inputs, configuration, k)):
            eligible.append((selection_key(configuration, category, q), k, q))
    return size, sorted(eligible)


def select_examples(inputs: Inputs) -> list[dict]:
    """One example per category and configuration, by the selection rule of the module docstring."""
    examples, taken = [], set()
    for category, conf in EXAMPLE_FIGURES:
        pretrained, fine_tuned = inputs.rows[(conf, "pretrained")], inputs.rows[(conf, "fine_tuned")]
        size, eligible = example_pool(inputs, category, conf)
        choice = next(((key, k, q) for key, k, q in eligible if q not in taken), None)
        if choice is None:
            raise SystemExit(f"[figures] no eligible {category} example for {conf}")
        key, k, q = choice
        taken.add(q)
        (positive,) = inputs.partners[q]
        if len(inputs.descriptions[q]) != 1:
            raise SystemExit(f"[figures] example query {q} has more than one description")
        example = {"figure": example_stem(category, conf), "configuration": conf, "category": category,
                   "query_id": q, "pmcid": f"PMC{inputs.articles[q]}", "recorded_positive": positive,
                   "description": inputs.descriptions[q][0],
                   "query_text_used": "yes" if conf == "image_text_to_image" else "no",
                   "pretrained_rank": pretrained[k]["first_rel_rank"],
                   "fine_tuned_rank": fine_tuned[k]["first_rel_rank"],
                   "ranked_gallery_size": inputs.ranked_gallery_size(q),
                   "chain_linked_images": inputs.ignored[q],
                   "category_size": size, "eligible_pool": len(eligible),
                   "selection_string": selection_string(conf, category, q), "selection_key": key}
        for model in MODELS:
            top = inputs.rankings[(conf, model)][k]
            kinds = [candidate_kind(inputs, q, c) for c in top]
            example[f"{model}_top10"] = "|".join(top)
            example[f"{model}_top10_recorded_positive"] = "".join("1" if t == "positive" else "0" for t in kinds)
            example[f"{model}_top10_same_article"] = "".join(
                "1" if inputs.articles[c] == inputs.articles[q] else "0" for c in top)
        examples.append(example)
    return examples


def example_positions(example: Mapping) -> list[tuple[str, str, str]]:
    """``(position, image_id, role)`` of every image an example figure displays, in drawing order.

    The role (``query``, ``positive`` or ``candidate``) selects the image's box in the figure's
    ``figure_drawing.example_layout``.
    """
    positions = [("query", example["query_id"], "query"),
                 ("recorded positive", example["recorded_positive"], "positive")]
    for model, name in (("pretrained", "pretrained"), ("fine_tuned", "fine-tuned")):
        top = str(example[f"{model}_top10"]).split("|")[:TOP_SHOWN]
        positions += [(f"{name}, rank {k + 1}", image_id, "candidate") for k, image_id in enumerate(top)]
    return positions


def prepare_images(inputs: Inputs, examples: Sequence[Mapping]
                   ) -> tuple[dict[tuple[str, tuple[float, float]], drawing.PreparedImage], list[dict]]:
    """Load the displayed images, sized for their boxes, and list their credits, one row per displayed position.

    The boxes of each figure follow from the pixel sizes of its query and recorded positive
    (``figure_drawing.example_layout``); the images are keyed by ``(image_id, box)``.
    """
    images: dict[tuple[str, tuple[float, float]], drawing.PreparedImage] = {}
    digests: dict[str, str] = {}
    credits = []
    for example in examples:
        layout = drawing.example_layout(inputs.sizes[example["query_id"]], inputs.sizes[example["recorded_positive"]])
        for position, image_id, role in example_positions(example):
            box = layout.box(role)
            if not inputs.displayable(image_id):
                raise SystemExit(f"[figures] {image_id} may not be displayed")
            path = inputs.data_dir / "images" / f"{image_id}.jpg"
            if image_id not in digests:
                if not path.is_file():
                    raise SystemExit(f"[figures] the displayed image {path} is missing")
                inputs.files[f"DATA_DIR/images/{image_id}.jpg"] = path
                digests[image_id] = sha256_file(path)
            if (image_id, box) not in images:
                images[(image_id, box)] = drawing.prepare_image(path, box)
                if images[(image_id, box)].source_size != inputs.sizes[image_id]:
                    raise SystemExit(f"[figures] {path} is not {inputs.sizes[image_id]} px as in data/image_sizes.csv")
            image, lic = images[(image_id, box)], inputs.licences[image_id]
            placed_w, placed_h = image.placed_mm
            credits.append({
                "figure": example["figure"], "position": position, "image_id": image_id,
                "pmcid": lic["pmcid"], "figure_xml_id": lic["figure_xml_id"], "image_license": lic["image_license"],
                "image_license_url": lic["image_license_url"], "article_authors": lic["article_authors"],
                "article_title": lic["article_title"], "journal": lic["journal"], "year": lic["year"],
                "doi": lic["doi"], "source_article_url": lic["source_article_url"],
                "attribution": _attribution(lic["attribution"], lic["article_title"]),
                "copyright_notice": _copyright_notice(lic["copyright_and_credit_notices"]),
                "source_width": image.source_size[0], "source_height": image.source_size[1],
                "embedded_width": image.size[0], "embedded_height": image.size[1],
                "box_width_mm": box[0], "box_height_mm": box[1],
                "placed_width_mm": round(placed_w, 3), "placed_height_mm": round(placed_h, 3),
                "effective_ppi": round(image.ppi, 1), "downscaled": "yes" if image.resampled else "no",
                "icc_profile": image.icc_profile, "converted_to_srgb": "yes" if image.converted else "no",
                "source_sha256": digests[image_id]})
    return images, credits


# --------------------------------------------------------------------------------------
# CSV files
# --------------------------------------------------------------------------------------
def _cell(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("booleans are written as yes/no strings")
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def csv_text(rows: Sequence[Mapping]) -> str:
    """CSV text (UTF-8, LF line ends) of rows with identical keys; floats at full precision."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(rows[0]))
    for row in rows:
        if list(row) != list(rows[0]):
            raise ValueError("CSV rows must have identical keys")
        writer.writerow([_cell(v) for v in row.values()])
    return buffer.getvalue()


def parse_csv(text: str) -> list[dict[str, str]]:
    """Rows of a CSV text as dictionaries of strings."""
    return list(csv.DictReader(io.StringIO(text)))


# --------------------------------------------------------------------------------------
# Captions, references and page layout
# --------------------------------------------------------------------------------------
MAX_LEGEND_WORDS = 350
# Reference layout of the figure pages (figure_pages.tex): A4 with 15 mm margins and legends in 8 pt.
PAGE_TEXT_HEIGHT_MM = 267.0
TEX_POINTS_PER_INCH = 72.27
LEGEND_LINE_MM = 9.5 / TEX_POINTS_PER_INCH * 25.4  # 8 pt type on 9.5 pt line spacing
# Typical words per legend line, not a bound: the shipped legends set 19 to 22 words per line in
# figure_pages.tex, the example legends fewest, because a citation or URL counts as one word but takes the
# space of several. The margin of the page check lies in LEGEND_SKIP_MM; the pdflatex compile of
# figure_pages.tex (no float too large for the page) is the authority.
LEGEND_WORDS_PER_LINE = 20
LEGEND_SKIP_MM = 6.0  # the 4 pt caption skip plus a margin of about 1.4 legend lines (4.6 mm)
# How a licence name of licenses.csv reads in a sentence ("reproduced under ..."); CC BY names read as they are.
LICENCE_PHRASES = {"Public domain (publisher statement)": "the publisher's public-domain statement",
                   "Public Domain Mark 1.0": "the Public Domain Mark 1.0"}
CODE_REPOSITORY = "of the code repository, see Code availability"


# Characters outside the Latin script that occur in article titles, set in LaTeX text mode.
TEX_SYMBOLS = {"\u03b1": "$\\alpha$", "\u03b2": "$\\beta$", "\u03b3": "$\\gamma$", "\u03b4": "$\\delta$",
               "\u03ba": "$\\kappa$", "\u03bc": "$\\mu$", "\u2010": "-", "\u2011": "-", "\u2212": "$-$",
               "\u00a0": "~"}
# Further characters outside ASCII that pdflatex sets with its default UTF-8 input encoding.
TEX_UNICODE = set("\u2013\u2014\u2018\u2019\u201c\u201d\u2026\u00a9\u00ae\u00b0\u00b1\u00d7")
# Combining accents (Unicode decomposition) and the LaTeX accent commands that set them.
TEX_ACCENTS = {"\u0300": "\\`", "\u0301": "\\'", "\u0302": "\\^", "\u0303": "\\~", "\u0304": "\\=",
               "\u0306": "\\u", "\u0307": "\\.", "\u0308": '\\"', "\u030c": "\\v", "\u0327": "\\c",
               "\u0328": "\\k"}


def tex(text: str) -> str:
    """Text with the LaTeX special characters escaped; characters that pdflatex cannot set are an error."""
    out = []
    for ch in text:
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "<":
            out.append(r"\textless{}")
        elif ch == ">":
            out.append(r"\textgreater{}")
        elif ch in TEX_SYMBOLS:
            out.append(TEX_SYMBOLS[ch])
        elif ord(ch) <= 0x17F or ch in TEX_UNICODE:
            out.append(ch)
        elif len(unicodedata.normalize("NFD", ch)) == 2 and unicodedata.normalize("NFD", ch)[1] in TEX_ACCENTS:
            base, accent = unicodedata.normalize("NFD", ch)
            out.append(f"{TEX_ACCENTS[accent]}{{{base}}}")
        else:
            raise ValueError(f"the character {ch!r} (U+{ord(ch):04X}) of {text!r} has no LaTeX text form here")
    return "".join(out)


def legend_words(latex: str) -> int:
    """Number of words of a legend as it renders: LaTeX commands not counted, a citation counted as one word."""
    text = re.sub(r"\\cite\{[^}]*\}", " 1 ", latex)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    text = re.sub(r"[{}$\\]", " ", text)
    return sum(1 for word in text.split() if re.search(r"[A-Za-z0-9]", word))


def legend_height_mm(latex: str) -> float:
    """Estimated height of a legend set in 8 pt across 180 mm (reference layout)."""
    return math.ceil(legend_words(latex) / LEGEND_WORDS_PER_LINE) * LEGEND_LINE_MM


def _diff(row: Mapping[str, str]) -> str:
    d, lo, hi = float(row["difference"]), float(row["ci95_low"]), float(row["ci95_high"])
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]".replace("-", "$-$")


def _title(legend: str) -> str:
    return legend.split(". ", 1)[0] + "."


def figure_environment(stem: str, label: str, legend: str, height_mm: float) -> str:
    """A LaTeX figure environment: the figure at its natural size, centred, and a legend of at most 350 words.

    The figure must fit on a page of the reference layout with its legend directly below.
    """
    words = legend_words(legend)
    if words > MAX_LEGEND_WORDS:
        raise ValueError(f"the legend of {stem} has {words} words, more than {MAX_LEGEND_WORDS}")
    if height_mm + LEGEND_SKIP_MM + legend_height_mm(legend) > PAGE_TEXT_HEIGHT_MM:
        raise ValueError(f"{stem} ({height_mm:.1f} mm) and its legend do not fit a {PAGE_TEXT_HEIGHT_MM:g} mm "
                         "text block")
    return ("\\begin{figure}[p]\n\\centering\n"
            f"\\makebox[\\linewidth][c]{{\\includegraphics{{{stem}.pdf}}}}\n"
            f"\\caption[{_title(legend)}]{{{legend}}}\n\\label{{{label}}}\n\\end{{figure}}\n")


def performance_legend(tables: Mapping[str, list[dict[str, str]]]) -> str:
    """Legend of ``retrieval_performance``."""
    levels = {(r["configuration"], r["model"]): r for r in tables["overall_metrics.csv"]}
    diffs = {(r["configuration"], r["metric"]): r for r in tables["paired_differences.csv"]}
    changes = {(r["configuration"], r["metric"]): r for r in tables["per_query_change.csv"]}
    first = diffs[(CONFIGURATIONS[0], "R@10")]
    n, groups, resamples = int(first["n_queries"]), int(first["n_article_groups"]), int(first["resamples"])
    i2i, it2i = CONFIGURATIONS

    def epoch(conf: str) -> str:
        row = levels[(conf, "fine_tuned")]
        return f"epoch {int(row['epoch'])} of {int(row['epochs'])}"

    def unchanged(conf: str) -> str:
        row = changes[(conf, "R@10")]
        return f"{int(row['unchanged_zero']):,} of {int(row['unchanged']):,}"

    return (
        f"Retrieval performance of pretrained and fine-tuned BiomedCLIP on the {n:,} test queries. "
        f"Every test image is a query, the other {n - 1:,} test images form the gallery, and the images recorded as "
        "pair partners of the query (recorded positives) are the relevant images; images linked to the query only "
        "through a chain of pairs are left out of its ranking. Both models encode every image from three "
        "$224 \\times 224$ pixel views. i2i, image-only queries (configuration image\\_to\\_image), scored by the "
        "cosine similarity of the query and gallery image embeddings; (i+t)2i, image-plus-text queries "
        "(configuration image\\_text\\_to\\_image), scored by that cosine similarity plus the dot product of the "
        "gallery image embedding with the mean (not renormalised) of the unit text embeddings of the query image's "
        "distinct descriptions. Fine-tuned, the released weights of each configuration, from the epoch with the "
        f"highest validation Recall@10 (i2i, {epoch(i2i)}; (i+t)2i, {epoch(it2i)}, with the mean-description query "
        "text). \\textbf{a}, Mean Recall@k (share of a query's recorded positives among its "
        "k highest-ranked images) for k = 1, 10 and 100, and Hit@10 (share of queries with a recorded positive in the "
        "top 10), in percent; symbols as in the key. \\textbf{b}, Mean paired difference, fine-tuned minus "
        f"pretrained, in percentage points, with 95\\% confidence intervals (CI) from a query-weighted bootstrap over "
        f"the {groups:,} article groups of the test split (articles connected by recorded pairs; {resamples:,} "
        f"resamples; percentile intervals); Recall@10: {_diff(diffs[(i2i, 'R@10')])} for i2i and "
        f"{_diff(diffs[(it2i, 'R@10')])} for (i+t)2i. \\textbf{{c}}, Share of queries whose Recall@10 or Hit@10 is "
        "higher (better, bars to the right) or lower (worse, bars to the left) with the fine-tuned than with the "
        "pretrained model; the column at the right gives the share with equal values under both models "
        "(unchanged). Shares are rounded to one decimal and may not sum to 100\\%. Most unchanged queries have no "
        f"recorded positive in the top 10 of either model ({unchanged(i2i)} for i2i and {unchanged(it2i)} for "
        "(i+t)2i, Recall@10).")


def subgroups_legend(tables: Mapping[str, list[dict[str, str]]]) -> str:
    """Legend of the Hit@10 figure, ``retrieval_subgroups_hit10``."""
    diffs = {(r["configuration"], r["metric"]): r for r in tables["paired_differences.csv"]}
    i2i, it2i = CONFIGURATIONS
    n = int(diffs[(i2i, "Hit@10")]["n_queries"])
    resamples = int(diffs[(i2i, "Hit@10")]["resamples"])
    return (
        f"Change in Hit@10 with fine-tuning across label categories of the {n:,} test queries. "
        "Mean Hit@10 is the share of queries with at least one recorded positive partner among their ten "
        "highest-ranked images. Each row shows the mean paired difference, fine-tuned minus pretrained, in "
        "percentage points. Panel \\textbf{a} shows i2i (image-only queries), and panel \\textbf{b} shows "
        "(i+t)2i (image-plus-text queries). Circles mark mean differences. Lines show 95\\% confidence "
        f"intervals from {resamples:,} paired bootstrap resamples of article groups, with each query weighted "
        "equally. Filled green circles indicate intervals entirely above zero. Open grey circles indicate "
        "intervals that include zero. The dashed lines mark the differences over all test queries "
        f"({float(diffs[(i2i, 'Hit@10')]['difference']):+.2f} and "
        f"{float(diffs[(it2i, 'Hit@10')]['difference']):+.2f} percentage points, respectively). "
        "The columns to the left of panel \\textbf{a} give the number of queries ($n$) and article groups "
        "represented in each category. These are the groups used to partition the dataset. Numbers to the "
        "right of each panel give mean Hit@10 (\\%) for the pretrained $\\rightarrow$ fine-tuned model. "
        f"Only categories with at least {MIN_SUBGROUP_QUERIES} queries from at least {MIN_SUBGROUP_GROUPS} "
        "article groups are shown. All categories are listed in "
        f"\\path{{results/figures/source_data/subgroups.csv}} {CODE_REPOSITORY}. "
        "Categories overlap, and the intervals are not adjusted for multiple comparisons. The results are "
        "descriptive. Intervals are approximate when few groups are represented or one group contributes "
        "many queries. CI denotes confidence interval, CT denotes computed tomography, and MRI denotes "
        "magnetic resonance imaging.")


EXAMPLE_TITLES = {
    "improvement": "Retrieval example in which fine-tuning moved the recorded positive into the top five",
    "deterioration": "Retrieval example in which fine-tuning moved the recorded positive out of the top ten",
    "continued_miss": "Retrieval example in which neither model ranks the recorded positive in the top ten",
}
EXAMPLE_RULES = {
    "improvement": "beyond 10 with the pretrained and within the top five with the fine-tuned model",
    "deterioration": "within the top five with the pretrained and beyond 10 with the fine-tuned model",
    "continued_miss": "beyond 10 with both models",
}
CONFIGURATION_PHRASES = {
    "image_to_image": "image-only query, i2i",
    "image_text_to_image": "image-plus-text query, (i+t)2i (query image and pair description)",
}


def abbreviations(text: str) -> list[str]:
    """Abbreviations of ``ABBREVIATIONS`` in ``text``, in order of first appearance; an unknown one is an error."""
    found: list[str] = []
    for token in ABBREVIATION_TOKEN.findall(text):
        if token not in ABBREVIATIONS:
            raise ValueError(f"the displayed description {text!r} contains the undefined abbreviation {token!r}")
        if token not in found:
            found.append(token)
    return found


def figure_references(tables: Mapping[str, list[dict[str, str]]]) -> dict[str, list[dict[str, str]]]:
    """Figure stem -> its source articles in order of first display, each with its licence and licence URL."""
    out: dict[str, list[dict[str, str]]] = {}
    for row in tables["image_credits.csv"]:
        articles = out.setdefault(row["figure"], [])
        entry = next((a for a in articles if a["pmcid"] == row["pmcid"]), None)
        if entry is None:
            articles.append({"pmcid": row["pmcid"], "image_license": row["image_license"],
                             "image_license_url": row["image_license_url"]})
        elif entry["image_license"] != row["image_license"]:
            raise SystemExit(f"[figures] {row['pmcid']} has images under different licences in {row['figure']}")
    return out


def licence_text(licence: str, url: str) -> str:
    """A licence as it reads in a legend or reference, with its URL when the licence record gives one."""
    return tex(LICENCE_PHRASES.get(licence, licence)) + (f" (\\url{{{url}}})" if url else "")


def credit_sentence(articles: Sequence[Mapping[str, str]]) -> str:
    """Credit sentence of an example legend: the cited source articles grouped by licence, with licence URLs.

    The copyright notices and full attributions are in the cited references and in
    ``source_data/image_credits.csv``, to which the legend points.
    """
    by_licence: dict[tuple[str, str], list[str]] = {}
    for article in articles:
        by_licence.setdefault((article["image_license"], article["image_license_url"]), []).append(article["pmcid"])
    parts = []
    for (licence, url), pmcids in by_licence.items():
        reference = f"ref.~\\cite{{{pmcids[0]}}}" if len(pmcids) == 1 else f"refs.~\\cite{{{','.join(pmcids)}}}"
        parts.append(f"{reference} under {licence_text(licence, url)}")
    listed = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"Images reproduced from {listed}."


def example_legend(stem: str, tables: Mapping[str, list[dict[str, str]]],
                   references: Mapping[str, list[dict[str, str]]]) -> str:
    """Legend of one example figure, with its image credits."""
    (row,) = [r for r in tables["examples.csv"] if r["figure"] == stem]
    category, conf = row["category"], row["configuration"]
    weights = next(r for r in tables["overall_metrics.csv"]
                   if r["configuration"] == conf and r["model"] == "fine_tuned")
    short = "i2i" if conf == "image_to_image" else "(i+t)2i"
    chained = int(row["chain_linked_images"])
    gallery = ("all other test images" if not chained else
               f"the other {int(row['ranked_gallery_size']) + chained:,} test images except {chained:,} linked to the "
               "query only through a chain of pairs")
    found = abbreviations(row["description"])
    definitions = "; ".join(f"{a}, {ABBREVIATIONS[a]}" for a in found)
    token = tex(f"{conf}/{category}/")
    converted = any(r["converted_to_srgb"] == "yes" for r in tables["image_credits.csv"] if r["figure"] == stem)
    integrity = ("Images are shown whole, with their embedded colour profiles applied, and otherwise unaltered except "
                 "for uniform downscaling" if converted else
                 "Images are shown whole, unaltered except for uniform downscaling")
    return (
        f"{EXAMPLE_TITLES[category]}: {CONFIGURATION_PHRASES[conf]}. Shown are the test query, its single recorded "
        "positive (annotated pair) and the pair description, above the five highest-ranked gallery images of "
        f"pretrained BiomedCLIP (left) and the fine-tuned model (right; released {short} weights, epoch "
        f"{int(weights['epoch'])}, selected by validation Recall@10); frames and tags as in the key. The recorded "
        f"positive ranks {int(row['pretrained_rank']):,} (pretrained) and {int(row['fine_tuned_rank']):,} "
        f"(fine-tuned) among the {int(row['ranked_gallery_size']):,} ranked gallery images ({gallery}). Only ranks "
        "1--5 are shown, whereas Recall@10 (share of the query's recorded positives in the top 10) and Hit@10 "
        "(any recorded positive in the top 10) count ranks 1--10. The description is composed from the "
        "dataset's diagnosis, anatomy and modality fields (\\emph{diagnosis} in the \\emph{anatomy} on "
        "\\emph{modality})"
        + (f"; {definitions}." if definitions else ".")
        + " Selection: among test queries whose single recorded positive ranks "
        f"{EXAMPLE_RULES[category]} and whose displayed images are all openly licensed (Creative Commons "
        "Attribution, CC BY, or public domain; commercial reuse, no share-alike), have at least "
        f"{drawing.MIN_LONG_PX} pixels on the longer side and show no burned-in dates, identification numbers, ages "
        "or names, the first in ascending order of the SHA-256 hexadecimal digest of "
        f"\\texttt{{{token}}}\\emph{{query ID}} not selected for an earlier example figure (order: improvement, "
        f"deterioration, continued miss; image-only first). {integrity}; letters, arrows and other marks inside "
        "them belong to the original figures. "
        f"{credit_sentence(references[stem])} The complete selection rule and pool sizes "
        "(pmc\\_pair\\_retrieval/figures.py, results/figures/source\\_data/examples.csv) and the images' "
        "copyright notices and attributions (results/figures/source\\_data/image\\_credits.csv) are in the code "
        "repository, see Code availability.")


def _copyright(notices: str) -> str:
    """The copyright statement of a ``copyright_and_credit_notices`` field (a JSON list), or an empty string."""
    entries = [str(e).strip() for e in json.loads(notices or "[]")]
    return next((e for e in entries if "\u00a9" in e or e.lower().startswith("copyright")), "")


def _copyright_notice(notices: str) -> str:
    """The copyright notice of a ``copyright_and_credit_notices`` field; stops if a notice would be lost.

    The notice is the copyright statement of the field. An entry that the notice contains, such as the year, is
    covered by it. A further entry is appended unchanged, after a space, if the notice so far ends with a full
    stop and the entry is one line of text: it contains a letter, has no full stop before its last character
    and is neither a JSON structure nor a second copyright statement. At any other entry the command stops.
    """
    notice = _copyright(notices)
    for entry in (str(e).strip() for e in json.loads(notices or "[]")):
        if entry in ("", "[]") or entry in notice:
            continue
        line = (re.search(r"[^\W\d_]", entry) and entry[0] not in "[{" and "." not in entry[:-1]
                and "\u00a9" not in entry and not entry.lower().startswith("copyright"))
        if not (notice.endswith(".") and line):
            raise SystemExit(f"[figures] a copyright or credit notice would be dropped: {entry[:80]!r}")
        notice = f"{notice} {entry}"
    return notice


def _attribution(text: str, title: str) -> str:
    """The attribution text of licenses.csv with its doubled full stops corrected.

    The text sets a full stop after the authors and after the article title (``title``, the ``article_title``
    field of the same record). The doubled full stop after "et al." is corrected, and so is the one directly
    after a title that itself ends with a full stop. At any other doubled full stop, and at a full stop after a
    question or exclamation mark, the command stops.
    """
    fixed = text.replace("et al..", "et al.")
    if title.endswith(".") and f"{title}. " in fixed:
        fixed = fixed.replace(f"{title}. ", f"{title} ", 1)
    if re.search(r"\.\.|[?!]\.", fixed):
        raise SystemExit(f"[figures] unexpected punctuation in the attribution {fixed[:80]!r}")
    return fixed


def _authors(field_value: str, limit: int = 5) -> str:
    names = [a.strip() for a in field_value.split(";") if a.strip()]
    if len(names) > limit:
        return ", ".join(names[:limit]) + " et al."
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " & " + names[-1]


SAFE_DOI = re.compile(r"10\.\d{4,9}/[A-Za-z0-9._;()/:-]+")
# Footnote markers that some publishers append to article titles.
TITLE_MARKERS = re.compile(r"[\u2605\u2606\u2020\u2021*]+$")


def references_text(tables: Mapping[str, list[dict[str, str]]], order: Sequence[str]) -> str:
    """The text of ``image_references.tex``: one ``\\bibitem`` per cited article, in order of first citation."""
    articles: dict[str, dict[str, str]] = {}
    for row in tables["image_credits.csv"]:
        articles.setdefault(row["pmcid"], row)
    lines = ["% Reference list of the source articles of the images in the example figures, generated by",
             "% python -m pmc_pair_retrieval.figures from source_data/image_credits.csv. The keys are the PMCIDs",
             "% cited in the example legends of captions.tex; the entries are in order of first citation, so that",
             "% the numbering starts at 1 in a document that sets the example figures (for example the",
             "% Supplementary Information). Each entry gives the authors, title, journal, year and DOI of the",
             "% article, the licence of the reproduced images and the article's copyright notice.",
             f"\\begin{{thebibliography}}{{{len(order)}}}"]
    for pmcid in order:
        a = articles[pmcid]
        if not SAFE_DOI.fullmatch(a["doi"]):
            raise SystemExit(f"[figures] the DOI {a['doi']!r} of {pmcid} needs escaping in LaTeX")
        notice = a["copyright_notice"]
        authors = _authors(a["article_authors"])
        title = TITLE_MARKERS.sub("", a["article_title"].strip()).rstrip(".")
        entry = (f"\\bibitem{{{pmcid}}} {tex(authors if authors.endswith('.') else authors + '.')} {tex(title)}. "
                 f"\\textit{{{tex(a['journal'])}}} ({tex(a['year'])}). \\url{{https://doi.org/{a['doi']}}}. "
                 f"{tex(pmcid)}. Images reproduced under {licence_text(a['image_license'], a['image_license_url'])}."
                 + (f" {tex(notice.rstrip('.'))}." if notice else ""))
        lines.append(entry)
    lines.append("\\end{thebibliography}")
    return "\n".join(lines) + "\n"


def pages_text() -> str:
    """The text of ``figure_pages.tex``: every figure with its legend on an A4 page of the reference layout."""
    return "\n".join([
        "% Reference page layout of the figures: every figure at its natural size (180 mm wide, all text 7 pt)",
        "% with its legend directly below it, one figure per page. A4 with 15 mm margins gives a text block of",
        "% 180 x 267 mm; legends are set in 8 pt Helvetica. Compile with pdflatex (twice, for the citations) in",
        "% the directory that holds the figure PDFs, captions.tex and image_references.tex.",
        "\\documentclass[10pt]{article}",
        "\\usepackage[a4paper,margin=15mm]{geometry}",
        "\\usepackage[T1]{fontenc}",
        "\\usepackage{helvet}",
        "\\renewcommand{\\familydefault}{\\sfdefault}",
        "\\usepackage{graphicx}",
        "\\usepackage{url}",
        "\\usepackage[font=footnotesize,labelfont=bf,skip=4pt]{caption}",
        "\\begin{document}",
        "\\input{captions.tex}",
        "\\clearpage",
        "\\input{image_references.tex}",
        "\\end{document}",
        ""])


def captions_text(tables: Mapping[str, list[dict[str, str]]], heights: Mapping[str, float]) -> tuple[str, list[str]]:
    """The text of ``captions.tex`` and the cited articles in order of first citation."""
    low, high = min(heights.values()), max(heights.values())
    parts = ["% Figure environments and legends of the retrieval figures, generated by",
             "% python -m pmc_pair_retrieval.figures from the source data in source_data/.",
             f"% Every figure is 180 mm wide ({low:.1f} to {high:.1f} mm high) and must be included at 100%, so",
             "% that all of its text stays at 7 pt: the text block must be at least 180 mm wide (on A4, side",
             "% margins of 15 mm or less). Any reduction must be applied equally to every figure, and the",
             "% resulting text size stated. Each figure fits on one page with its legend directly below it in the",
             "% reference layout of figure_pages.tex: A4 with 15 mm margins (text block 180 x 267 mm), legends",
             "% in 8 pt.",
             "% Requires \\usepackage{graphicx} and \\usepackage{url}; the citations of the example legends",
             "% refer to the entries of image_references.tex.", ""]
    parts.append(figure_environment(FIGURES[0], "fig:retrieval-performance", performance_legend(tables),
                                    heights[FIGURES[0]]))
    parts.append(figure_environment(FIGURES[1], "fig:retrieval-subgroups", subgroups_legend(tables),
                                    heights[FIGURES[1]]))
    references = figure_references(tables)
    order: list[str] = []
    for category, conf in EXAMPLE_FIGURES:
        stem = example_stem(category, conf)
        legend = example_legend(stem, tables, references)
        parts.append(figure_environment(stem, "fig:" + stem.replace("_", "-"), legend, heights[stem]))
        # References are numbered in order of first citation in the legends.
        for keys in re.findall(r"\\cite\{([^}]*)\}", legend):
            order += [key for key in keys.split(",") if key not in order]
    return "\n".join(parts), order


# --------------------------------------------------------------------------------------
# Drawing from the source data
# --------------------------------------------------------------------------------------
def draw_figures(tables: Mapping[str, list[dict[str, str]]], images: Mapping[tuple[str, str], drawing.PreparedImage],
                 fonts: Mapping[str, drawing.FontFile]) -> dict[str, drawing.Canvas]:
    """Draw the eight figures from the parsed source data."""
    levels = {(r["configuration"], r["model"]): {m: float(r[_percent(m)]) for m in LEVEL_METRICS}
              for r in tables["overall_metrics.csv"]}
    differences = {(r["configuration"], r["metric"]): (float(r["difference"]), float(r["ci95_low"]),
                                                        float(r["ci95_high"]))
                   for r in tables["paired_differences.csv"] if r["metric"] in LEVEL_METRICS}
    changes = {(r["configuration"], r["metric"]): {"better": float(r["better_percent"]),
                                                    "worse": float(r["worse_percent"]),
                                                    "unchanged": float(r["unchanged_percent"])}
               for r in tables["per_query_change.csv"]}
    drawn = [{"configuration": r["configuration"], "family": r["family"], "category": r["category"],
              "label": r["label"], "n_queries": r["n_queries"], "n_article_groups": r["n_article_groups"],
              "pretrained": r["pretrained_Hit@10_percent"], "fine_tuned": r["fine_tuned_Hit@10_percent"],
              "difference": r["difference_Hit@10_pp"], "ci_low": r["ci95_low_Hit@10_pp"],
              "ci_high": r["ci95_high_Hit@10_pp"]}
             for r in tables["subgroups.csv"] if r["shown"] == "yes" and r["family"] in LABEL_FAMILIES]
    overall = {conf: differences[(conf, "Hit@10")][0] for conf in CONFIGURATIONS}
    families = [(f, FAMILY_HEADINGS[f]) for f in LABEL_FAMILIES]
    canvases = {
        FIGURES[0]: drawing.draw_performance(levels, differences, changes, CONFIGURATIONS, LEVEL_METRICS, fonts),
        FIGURES[1]: drawing.draw_subgroups(drawn, overall, CONFIGURATIONS, families, fonts)}
    credits = {(r["figure"], r["position"]): r for r in tables["image_credits.csv"]}
    for row in tables["examples.csv"]:
        panel = {k: row[k] for k in ("configuration", "query_id", "description", "pretrained_rank",
                                     "fine_tuned_rank", "ranked_gallery_size")}
        panel["positive_id"] = row["recorded_positive"]
        for model in MODELS:
            top = row[f"{model}_top10"].split("|")[:TOP_SHOWN]
            positive = row[f"{model}_top10_recorded_positive"]
            same = row[f"{model}_top10_same_article"]
            panel[f"{model}_top"] = [(image_id, "positive" if positive[k] == "1" else
                                      "same_article" if same[k] == "1" else "unannotated")
                                     for k, image_id in enumerate(top)]
        # The boxes follow from the pixel sizes of the query and its positive; image_credits.csv records them.
        sizes = [(int(credits[(row["figure"], p)]["source_width"]), int(credits[(row["figure"], p)]["source_height"]))
                 for p in ("query", "recorded positive")]
        layout = drawing.example_layout(*sizes)
        for position, _, role in example_positions(row):
            credit = credits[(row["figure"], position)]
            if (float(credit["box_width_mm"]), float(credit["box_height_mm"])) != layout.box(role):
                raise SystemExit(f"[figures] the box of {row['figure']}, {position} differs from image_credits.csv")
        canvases[row["figure"]] = drawing.draw_example(panel, images, layout, fonts, row["figure"])
    return canvases


# --------------------------------------------------------------------------------------
# Build, manifest and check
# --------------------------------------------------------------------------------------
def environment() -> dict:
    """Versions of the software that determines the bytes of the outputs."""
    return {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__,
            "freetype": ft2font.__freetype_version__, "fonttools": fontTools.version, "pillow": PIL.__version__,
            "pillow_libjpeg_turbo": features.version("libjpeg_turbo"), "pillow_zlib": features.version("zlib"),
            "pillow_littlecms2": features.version("littlecms2"), "python_zlib": zlib.ZLIB_RUNTIME_VERSION}


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def build(data_dir: str | Path, output_dir: str | Path) -> dict:
    """Write all outputs to ``output_dir`` and return the manifest."""
    output_dir = Path(output_dir)
    existing = [name for name in OUTPUTS if (output_dir / name).exists()]
    if existing:
        raise SystemExit(f"[figures] refusing to overwrite {len(existing)} existing output file"
                         f"{'s' if len(existing) > 1 else ''} in {output_dir} (for example {existing[0]})")
    fonts, published_fonts = drawing.find_arial()
    print(f"[figures] fonts: {', '.join(f'{Path(f.path).name} ({f.version})' for f in fonts.values())}"
          + ("" if published_fonts else "; not the fonts of the published figures"), flush=True)
    inputs = load_inputs(data_dir)
    print(f"[figures] {len(inputs.query_ids):,} test queries; evaluation files match results/results.json",
          flush=True)
    overall = overall_table(inputs)
    examples = select_examples(inputs)
    images, credits = prepare_images(inputs, examples)
    texts = {"overall_metrics.csv": csv_text(overall),
             "paired_differences.csv": csv_text(differences_table(inputs, overall)),
             "per_query_change.csv": csv_text(change_table(inputs)),
             "subgroups.csv": csv_text(subgroup_table(inputs)),
             "per_query.csv": csv_text(per_query_table(inputs)),
             "examples.csv": csv_text(examples),
             "image_credits.csv": csv_text(credits)}
    tables = {name: parse_csv(text) for name, text in texts.items()}
    for example in examples:
        print(f"[figures] {example['figure']}: query {example['query_id']} (ranks {example['pretrained_rank']} -> "
              f"{example['fine_tuned_rank']}; {example['eligible_pool']} eligible, with the exclusion list as it "
              f"stands, of {example['category_size']} single-positive queries in the category)", flush=True)
    with drawing.arial(fonts):
        canvases = draw_figures(tables, images, fonts)
        heights = {stem: canvas.height for stem, canvas in canvases.items()}
        captions, cited = captions_text(tables, heights)
        (output_dir / "source_data").mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            _write(output_dir / "source_data" / name, text)
        _write(output_dir / CAPTIONS, captions)
        _write(output_dir / REFERENCES, references_text(tables, cited))
        _write(output_dir / PAGES, pages_text())
        for stem in FIGURES:
            drawing.save(canvases[stem], output_dir, stem)
    manifest = {
        "generator": "python -m pmc_pair_retrieval.figures",
        "split": SPLIT,
        "environment": environment(),
        "fonts": [f.describe() for f in fonts.values()],
        "inputs": [{"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                   for name, path in sorted(inputs.files.items())],
        "outputs": [{"path": name, "bytes": (output_dir / name).stat().st_size,
                     "sha256": sha256_file(output_dir / name)} for name in OUTPUTS if name != MANIFEST],
    }
    _write(output_dir / MANIFEST, json.dumps(manifest, indent=2) + "\n")
    print(f"[figures] wrote {len(OUTPUTS)} files to {output_dir}", flush=True)
    return manifest


def compare_with_shipped(output_dir: Path, shipped_dir: Path = SHIPPED_DIR) -> tuple[list[str], list[str]]:
    """Differences between regenerated outputs and the shipped figure directory: ``(problems, previews)``.

    Every input and output of the shipped manifest must have the same sha256, every shipped
    file other than the manifest itself must equal its regenerated version, and the font
    files must have the same content (family, style, version and sha256; installations name
    the files differently). The software versions recorded in the manifest are printed
    where they differ and are not compared. The PNG previews are the exception when
    matplotlib was built with another FreeType version than the shipped figures: FreeType
    rasterises their text, so differing previews are then returned in ``previews`` instead
    of ``problems``.
    """
    reference_path = shipped_dir / MANIFEST
    if not reference_path.is_file():
        return [f"{reference_path} does not exist"], []
    with open(reference_path, encoding="utf-8") as fh:
        reference = json.load(fh)
    with open(output_dir / MANIFEST, encoding="utf-8") as fh:
        regenerated = json.load(fh)
    env_old, env_new = reference.get("environment", {}), regenerated.get("environment", {})
    for key in sorted(env_old.keys() | env_new.keys()):
        if env_old.get(key) != env_new.get(key):
            print(f"[figures] note: {key} {env_new.get(key)} here, {env_old.get(key)} for the shipped figures",
                  flush=True)
    other_freetype = env_old.get("freetype") != env_new.get("freetype")
    problems, previews = [], []
    for section in ("inputs", "outputs"):
        old = {e["path"]: e["sha256"] for e in reference[section]}
        new = {e["path"]: e["sha256"] for e in regenerated[section]}
        problems += [f"{section[:-1]} {p}: missing" for p in sorted(old.keys() - new.keys())]
        problems += [f"{section[:-1]} {p}: unexpected" for p in sorted(new.keys() - old.keys())]
        for p in sorted(old.keys() & new.keys()):
            if old[p] == new[p]:
                continue
            if section == "outputs" and other_freetype and p.endswith(".png"):
                previews.append(p)
            else:
                problems.append(f"{section[:-1]} {p}: sha256 {new[p][:12]}... instead of {old[p][:12]}...")
    for key in ("generator", "split"):
        if reference.get(key) != regenerated.get(key):
            problems.append(f"manifest {key}: {regenerated.get(key)!r} instead of {reference.get(key)!r}")
    old_fonts = sorted((drawing.font_identity(e) for e in reference.get("fonts", [])), key=lambda e: str(e["style"]))
    new_fonts = sorted((drawing.font_identity(e) for e in regenerated.get("fonts", [])), key=lambda e: str(e["style"]))
    if old_fonts != new_fonts:
        problems.append(f"font files: {new_fonts!r} instead of {old_fonts!r} (family, style, version, sha256)")
    shipped_sha256 = {e["path"]: e["sha256"] for e in reference["outputs"]}
    for path in sorted(p for p in shipped_dir.rglob("*") if p.is_file()):
        name = path.relative_to(shipped_dir).as_posix()
        if name == MANIFEST:
            continue
        candidate = output_dir / name
        if not candidate.is_file():
            problems.append(f"shipped file {name}: not regenerated")
        elif sha256_file(path) != shipped_sha256.get(name):
            problems.append(f"shipped file {name}: does not match the shipped manifest")
        elif candidate.read_bytes() != path.read_bytes() and not (other_freetype and name.endswith(".png")):
            problems.append(f"shipped file {name}: differs from the regenerated file")
    return problems, previews


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments of ``python -m pmc_pair_retrieval.figures``."""
    parser = argparse.ArgumentParser(
        prog="python -m pmc_pair_retrieval.figures",
        description="Draw the evaluation and error-analysis figures from the shipped evaluation outputs.")
    parser.add_argument("--data-dir", required=True,
                        help="public dataset directory with image_pairs.json, image_labels.jsonl, licenses.csv, "
                             "withheld_image_ids.txt and images/")
    parser.add_argument("--output-dir",
                        help="directory for the figures, captions, source data and manifest; it must not contain "
                             "any of these files. Required unless --check is given; with --check, the regenerated "
                             "files are kept there")
    parser.add_argument("--check", action="store_true",
                        help="regenerate into a temporary directory and compare with the shipped results/figures; "
                             "exit 1 on differences")
    args = parser.parse_args(argv)
    if not args.check and not args.output_dir:
        parser.error("--output-dir is required unless --check is given")
    return args


def main(argv: list[str] | None = None) -> None:
    """Generate the figures, or regenerate and check them against the shipped ones."""
    args = parse_args(argv)
    if not args.check:
        build(args.data_dir, args.output_dir)
        return
    with tempfile.TemporaryDirectory(prefix="pmc_pair_figures_") as tmp:
        output_dir = Path(args.output_dir) if args.output_dir else Path(tmp)
        build(args.data_dir, output_dir)
        problems, previews = compare_with_shipped(output_dir)
    if problems:
        print(f"[figures] check failed: {len(problems)} differences from {SHIPPED_DIR}:", *problems, sep="\n  ",
              file=sys.stderr)
        raise SystemExit(1)
    if previews:
        print(f"[figures] OK: the regenerated outputs equal the shipped files in {SHIPPED_DIR} other than "
              f"manifest.json, and the sha256 values and fonts of its manifest, except the {len(previews)} PNG "
              "previews, which matplotlib rasterised with another FreeType version", flush=True)
    else:
        print(f"[figures] OK: the regenerated outputs equal the shipped files in {SHIPPED_DIR} other than "
              "manifest.json, and every sha256 and the fonts of its manifest", flush=True)


if __name__ == "__main__":
    main()
