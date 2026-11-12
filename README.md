# BiomedCLIP image retrieval on PMC-PaIR

Train and evaluate BiomedCLIP for retrieval of annotated image pairs from PubMed Central case reports.
This repository includes an article-disjoint split, two training configurations, per-query results,
ranked lists, three-seed comparisons, and commands to generate results tables and figures.

Dataset construction is documented in the companion
[PMC-PaIR pipeline repository](https://github.com/GuyShmueli/pmc-pair-pipeline).

<!-- BEGIN release-status -->
## Data and weights

The dataset is not yet available for download. Its reserved Zenodo DOI is
[`10.5281/zenodo.23439169`](https://doi.org/10.5281/zenodo.23439169).

Fine-tuned checkpoint downloads are also not yet available. Their metadata and checksums are included
here. See [Released weights](#released-weights) for availability and terms.
<!-- END release-status -->

## Installation

```bash
conda env create -f environment.yml
conda activate pmc-pair-retrieval
```

Run commands from the repository root. The environment uses Python 3.11.10 and the versions in
[`requirements.txt`](requirements.txt), including PyTorch 2.7.1 with CUDA 12.6. Training and evaluation
require a CUDA GPU and a compatible NVIDIA driver. The reported runs used an NVIDIA L40S with 48 GB
of memory. Split generation, saved-results reports and figures run on a CPU.

Keep the pinned versions and configuration settings when reproducing results. Transitive dependencies
are resolved at installation time and are not fully pinned.

## Data

Download the dataset files into a new directory when the record is available, verify the checksums,
and extract the four image archives:

```bash
export DATA_DIR=/path/to/pmc-pair
(
  cd "$DATA_DIR"
  sha256sum -c SHA256SUMS.txt
  for part in 001 002 003 004; do
    unzip -q -n "images_captions_${part}.zip" -d .
  done
)
```

Use a new directory because extraction does not remove existing files. The retrieval workflow reads:

```text
DATA_DIR/
  image_pairs.json          pair IDs and modality, anatomy and diagnosis descriptions
  images/<image_id>.jpg     image files
  image_labels.jsonl        image-level labels, used for figures
  licenses.csv              image licences and attribution, used for figures
  withheld_image_ids.txt    withheld image IDs, used for figures
```

The pair file contains 86,144 pairs linking 86,893 images from 56,013 articles. Each image ID begins
with the PMCID digits of its source article. Images and captions retain the terms recorded in
`licenses.csv`. See the dataset's `LICENSE.txt` for the scope of its licence.

### Withheld images

The public archives contain 86,825 images and withhold 68, listed in `withheld_image_ids.txt`.
`withheld.csv` records the reasons. Commands that read the image collection stop on missing files
unless `--allow-missing-images` is supplied.

The training and evaluation examples below use this flag. It filters unavailable images and their
116 incident pairs, leaving 8,239 validation queries and 8,088 test queries for `evaluate`.
The reference metrics in [Expected results](#expected-results) cover the complete split, so those
metrics cannot be reproduced from the public image subset. On the missing-image path, trainer
validation also retains some images and descriptions that `evaluate` excludes, so their validation
metrics can differ. This path has not been tested on a GPU.

The split, saved-results reports and figure checks work with the public dataset. Their inputs do not
require any withheld image. The three-seed training driver requires the complete image collection.

### Bundled data files

| File | Purpose |
|---|---|
| [`data/article_disjoint_split.json`](data/article_disjoint_split.json) | 70,552 training, 8,245 validation and 8,096 test images. No pair or source article spans partitions. |
| [`data/pixel_md5.json`](data/pixel_md5.json) | Decoded-pixel hashes used by image-only training to handle duplicate images. |
| [`data/hard_negatives.json`](data/hard_negatives.json) | Fifteen cached hard negatives per training image for image-plus-text training. Mining is not provided by this package. |
| [`data/image_sizes.csv`](data/image_sizes.csv) | Image dimensions used for figure analyses. |
| [`data/figure_image_exclusions.csv`](data/figure_image_exclusions.csv) | Display exclusions used only by the figure command. |

Always use the bundled split for the commands below. The pipeline repository supplies the same
assignments as `image_splits.json`.

Two groups of pixel-identical images span training and validation. None spans test and another
partition. Duplicate filtering excludes three training images from `image_to_image`, leaving
70,549 eligible images. `image_text_to_image` does not apply this pixel-hash filter.

## Pretrained model

The model is `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` at revision
`9f341de24bfb00180f1b847274256e9b65a3a32e`. The tokenizer is
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` at revision
`d673b8835373c6fa116d6d8006b33d48734e305d`.

Download the pinned files into a dedicated cache:

```bash
python -m pmc_pair_retrieval.model download --cache-dir /path/to/biomedclip_cache
```

Then set the environment for training and evaluation:

```bash
export HF_HUB_CACHE=/path/to/biomedclip_cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Model loading verifies the pinned revisions and file hashes. Legacy Transformers cache variables
must be unset or point to `HF_HUB_CACHE`. Training sets and checks `PYTHONHASHSEED=42`.

BiomedCLIP's model card limits its intended use to research and reproduction of the associated
experiments, with deployed uses outside scope. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
for the upstream model notices and links.

## Commands

### 1. Split

```bash
python -m pmc_pair_retrieval.split --data-dir "$DATA_DIR" --check
```

This reconstructs the split from `image_pairs.json` and checks it against the bundled file.
Use `--output FILE` instead of `--check` to write a split file.

Optional checks of the available image files:

```bash
python -m pmc_pair_retrieval.pixel_hashes --data-dir "$DATA_DIR" \
    --check --workers 16 --allow-missing-images
python -m pmc_pair_retrieval.image_sizes --data-dir "$DATA_DIR" \
    --check --allow-missing-images
```

### 2. Training

```bash
python -m pmc_pair_retrieval.train --config configs/image_to_image.yaml \
    --data-dir "$DATA_DIR" --output-dir runs/image_to_image --allow-missing-images
python -m pmc_pair_retrieval.train --config configs/image_text_to_image.yaml \
    --data-dir "$DATA_DIR" --output-dir runs/image_text_to_image --allow-missing-images
```

Use a new or empty output directory. Each run writes its configuration, epoch history, selected
checkpoint (`best_model.safetensors`), checkpoint metadata (`best_model.json`) and summary.
Checkpoint selection uses validation Recall@10. With all images available, omit
`--allow-missing-images` to enforce completeness.

For a short smoke test, add `--max-steps 200` to the image-only command or
`--epochs 1 --max-steps 40` to the image-plus-text command, using separate output directories.
These overrides stop training early and do not reproduce the fine-tuned checkpoints.

### 3. Evaluation

Evaluate pretrained BiomedCLIP on both partitions:

```bash
for split in val test; do
  for retrieval in image_to_image image_text_to_image; do
    python -m pmc_pair_retrieval.evaluate --weights pretrained --retrieval "$retrieval" \
        --split "$split" --data-dir "$DATA_DIR" --output-dir eval --allow-missing-images
  done
done
```

Once the fine-tuned weights are available, evaluate them with the corresponding configuration:

```bash
for split in val test; do
  for retrieval in image_to_image image_text_to_image; do
    python -m pmc_pair_retrieval.evaluate --weights "weights/$retrieval.safetensors" \
        --retrieval "$retrieval" --split "$split" --data-dir "$DATA_DIR" \
        --output-dir eval --allow-missing-images
  done
done
```

For your own run, pass `runs/<configuration>/best_model.safetensors` and use a separate output
directory. Keep its JSON sidecar beside the weight file. Evaluate the pretrained model into the
same directory so that paired comparisons use matching queries.

Outputs under `eval/<configuration>/` include per-query metrics (`<name>_<split>.jsonl`), top-10
rankings (`.rankings.jsonl`), summary statistics and model metadata (`.summary.json`), and cached
embeddings. Existing outputs are not overwritten. `compare` reads the default names `pretrained`
and `fine_tuned`.

### 4. Results tables

```bash
python -m pmc_pair_retrieval.compare --data-dir "$DATA_DIR" --per-query-dir eval
```

This writes `eval/results.json` and `eval/results.md`. Paired differences use a query-weighted
bootstrap over article groups with 5,000 resamples and 95% intervals. Article-level intervals are
also included in the reports.

To check the bundled results and the generated tables in this README without loading a model:

```bash
python -m pmc_pair_retrieval.compare --data-dir "$DATA_DIR" --check
```

## Released weights

The seed-42 checkpoint metadata and checksums are included in `weights/`. Weight files are
downloaded separately and belong beside their JSON sidecars. Image-only weights contain the
vision tower. Image-plus-text weights contain the full model.

<!-- BEGIN compare:released-weights -->
| Configuration | File | Contents | Selected epoch | Global step | Validation R@10 at selection (%) | File sha256 | State-dict sha256 |
|---|---|---|---:|---:|---:|---|---|
| image_to_image | `weights/image_to_image.safetensors` | vision tower, 151 tensors | 11 | 6,061 | 19.55 | `cb24963bcb0ddba73a1d7e4a1a4539933b05ee9b8bf8d8f3454ca5c7c9949622` | `23825766247fe2b64d218f9500330e376811105ba6bc670e1ab872025b78794a` |
| image_text_to_image | `weights/image_text_to_image.safetensors` | full model, 351 tensors | 21 | 11,508 | 30.87 | `4d159e28b2ab1621e463bc3be8aab907356b83536d0ac9698620f32dabd54292` | `e7ebac89af320be1d57c7a51388623501d62b94353d84259be0c7f2d50bf56a9` |
<!-- END compare:released-weights -->

The terms for the fine-tuned weights and their sidecars are not yet set. The repository's MIT
licence does not cover them. See [`LICENSING.md`](LICENSING.md).

### Download the released weights

The downloader has no public weight record configured and currently reports that downloads are
unavailable. Once a record is configured, download both files with:

```bash
python -m pmc_pair_retrieval.download_weights
```

Use `--configuration image_to_image` to select one configuration. Files are checked against the
sidecars and `WEIGHTS_SHA256SUMS` before installation. To verify existing local files:

```bash
python -m pmc_pair_retrieval.download_weights --check
sha256sum -c WEIGHTS_SHA256SUMS
```

Training and pretrained evaluation do not require the fine-tuned weight files. Checks of the
saved reports and figures need only their sidecars. `run_multiseed` and `multiseed --verify-weights`
require the checkpoint files.

## Configurations

Full hyperparameters are documented in [`configs/image_to_image.yaml`](configs/image_to_image.yaml)
and [`configs/image_text_to_image.yaml`](configs/image_text_to_image.yaml).

| Setting | image_to_image | image_text_to_image |
|---|---|---|
| Query | Image | Image and its pair descriptions |
| Trainable parameters | Vision tower | Both towers except vision blocks 0–7 |
| Loss | Per-positive InfoNCE, temperature 0.10 | Weighted image-image and image-text triplet losses, margin 0.09 |
| Training batch | 128 image queries and their positives | 128 pairs sampled with cached hard negatives |
| Epoch limit | 12 | 30, with early stopping |
| Selected checkpoint | Epoch 11 | Epoch 21, from a run stopped after epoch 24 |
| Selection criterion | Validation Recall@10 | Validation Recall@10 |

Both configurations use three 224 × 224 crops along each image's long axis. View embeddings are
L2-normalised, averaged and normalised again. Training adds crop augmentation. Evaluation uses
fixed crops. Keep the configured batch sizes and worker counts when reproducing runs.

## Query text

Each pair supplies a description: `"{diagnosis} in the {anatomy} on {modality}."`
An image's text query averages the unit embeddings of its distinct pair descriptions. This mean,
`t_q`, is not renormalised. Image-plus-text retrieval scores gallery image `g` as:

```text
s(q, g) = cos(image_q, image_g) + t_q · image_g
```

This configuration assumes the pair descriptions are available at query time. The same text
definition is used for checkpoint selection and evaluation.

Every image in a partition is a query against the same partition's gallery. Its direct pair
partners are relevant. Other members of its connected pair-graph component are excluded from
the ranking, and the query itself ranks last. Reported metrics include Recall@k, Hit@k, mAP,
nDCG, MRR and median first-relevant rank.

## Expected results

<!-- BEGIN compare:expected-results -->
Means over queries in percent (MRR as a fraction), from `results/results.json`; `results/results.md` has every metric
and all paired differences.

Validation split (8,245 queries):

| Configuration | Model | R@1 | R@10 | R@100 | Hit@10 | mAP | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| image_to_image | pretrained | 3.24 | 14.50 | 47.35 | 20.99 | 8.32 | 0.1090 |
| image_to_image | fine-tuned | 4.02 | 19.55 | 58.56 | 27.42 | 10.58 | 0.1362 |
| image_text_to_image | pretrained | 4.41 | 20.44 | 59.42 | 29.00 | 11.30 | 0.1481 |
| image_text_to_image | fine-tuned | 7.13 | 30.87 | 74.70 | 42.00 | 16.78 | 0.2144 |

Test split (8,096 queries):

| Configuration | Model | R@1 | R@10 | R@100 | Hit@10 | mAP | MRR |
|---|---|---:|---:|---:|---:|---:|---:|
| image_to_image | pretrained | 3.32 | 14.62 | 47.71 | 21.47 | 8.22 | 0.1105 |
| image_to_image | fine-tuned | 4.31 | 20.15 | 60.03 | 28.31 | 10.89 | 0.1434 |
| image_text_to_image | pretrained | 4.87 | 20.53 | 60.08 | 29.68 | 11.59 | 0.1541 |
| image_text_to_image | fine-tuned | 7.46 | 31.62 | 76.33 | 42.98 | 17.13 | 0.2196 |

Fine-tuned minus pretrained, with 95% intervals from a paired bootstrap over article groups (5,000 resamples):

| Configuration | Split | R@10 | Hit@10 | MRR |
|---|---|---:|---:|---:|
| image_to_image | validation | +5.05 [+4.18, +6.01] | +6.43 [+5.28, +7.58] | +0.0272 [+0.0213, +0.0332] |
| image_text_to_image | validation | +10.44 [+9.15, +11.86] | +13.00 [+11.51, +14.61] | +0.0663 [+0.0543, +0.0807] |
| image_to_image | test | +5.53 [+4.61, +6.55] | +6.84 [+5.73, +8.09] | +0.0328 [+0.0271, +0.0387] |
| image_text_to_image | test | +11.09 [+9.89, +12.38] | +13.30 [+12.07, +14.66] | +0.0654 [+0.0568, +0.0747] |
<!-- END compare:expected-results -->

## Three-seed results

[`results/multiseed/multiseed.md`](results/multiseed/multiseed.md) compares seeds 42, 43 and 44.
It includes means, sample standard deviations, ranges and paired confidence intervals. Test
Recall@10 is 20.11% ± 0.07 percentage points for image-only retrieval and 31.57% ± 0.04 for
image-plus-text retrieval (mean ± sample SD across seeds).

Regenerate or check the reports on a CPU using `image_pairs.json` and the saved evaluation files:

```bash
python -m pmc_pair_retrieval.multiseed --data-dir "$DATA_DIR" --output-dir seed_reports
python -m pmc_pair_retrieval.multiseed --data-dir "$DATA_DIR" --check
```

To inspect or run training and evaluation for seeds 43 and 44:

```bash
python -m pmc_pair_retrieval.run_multiseed --data-dir "$DATA_DIR" --output-dir seed_runs --dry-run
python -m pmc_pair_retrieval.run_multiseed --data-dir "$DATA_DIR" --output-dir seed_runs
```

The driver requires all 86,893 images, both seed-42 weight files in `weights/`, and a new output
directory. It has not been tested through a complete GPU run. The saved-results report also
requires the complete validation/test query sets and cannot combine subset evaluations with
the bundled full-query baselines. Seeds 43 and 44 have saved training and evaluation records,
but their checkpoint files are not bundled. No seed is selected using test results.

## Figures

Generate figures, LaTeX captions, source-data tables and an input/output manifest from the saved
results, or check the bundled outputs:

```bash
python -m pmc_pair_retrieval.figures --data-dir "$DATA_DIR" --output-dir figures
python -m pmc_pair_retrieval.figures --data-dir "$DATA_DIR" --check
```

These commands run on a CPU and need the extracted images plus the four metadata files listed
under [Data](#data). They do not load a model or recompute retrieval scores. Use a new output
directory. Generation writes PDF, SVG and PNG files. The bundled PDFs, captions and source data
are in [`results/figures/`](results/figures/).

Arial regular and bold are required. Exact reproduction uses Arial 2.82, the pinned Python
packages and matching dataset inputs. The check compares PNG previews only when FreeType
versions match. PDF, SVG, CSV and LaTeX outputs must still match.

Subgroup intervals use 5,000 paired bootstrap resamples of article groups. Plotted label categories
have at least 100 queries from at least 20 article groups. A query can contribute to multiple
categories. Image credits and third-party terms are recorded in the figure source data and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Reproducibility notes

The following checks run on a CPU without model weights. Each command is
`python -m pmc_pair_retrieval.<module> --data-dir "$DATA_DIR"` plus the listed options.

| Module and options | Checks |
|---|---|
| `split --check` | Article-disjoint partition assignments |
| `compare --check` | Results tables and generated README tables |
| `multiseed --check` | Three-seed reports and their input records |
| `figures --check` | Figures, captions and source data, subject to the font requirements above |
| `pixel_hashes --check --allow-missing-images` | Decoded pixels of available images |
| `image_sizes --check --allow-missing-images` | Dimensions of available images |

These checks verify the saved artifacts against their inputs. Bitwise reproducibility of a
complete training run has not been established. Hardware, dependency versions, worker counts
and GPU arithmetic can affect weights, checkpoint selection and rankings of near-tied images.

Verify repository files with `sha256sum -c SHA256SUMS`. Fine-tuned weight files are checked
separately with `sha256sum -c WEIGHTS_SHA256SUMS`.

## Licence

The code, documentation and configuration are MIT-licensed under [`LICENSE`](LICENSE).
[`LICENSING.md`](LICENSING.md) defines the covered paths. The licence does not cover the weights,
`data/` or `results/`, whose terms are not yet set. Third-party material retains the terms recorded
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

Please cite this software, the PMC-PaIR dataset and Data Descriptor, and BiomedCLIP. Cite the
dataset construction pipeline if you use it. Bibliographic details are in [`CITATION.cff`](CITATION.cff).

## Repository layout

| Path | Contents |
|---|---|
| `pmc_pair_retrieval/` | Training, evaluation, reporting and figure commands |
| `configs/` | Training and evaluation settings |
| `data/` | Split, cached negatives, pixel hashes and figure inputs |
| `weights/` | Checkpoint sidecars and separately downloaded weight files |
| `results/per_query/`, `results/rankings/` | Per-query metrics and top-10 retrieval lists |
| `results/multiseed/` | Three-seed reports and training/evaluation records |
| `results/figures/` | Figure PDFs, captions and source-data tables |
| `results/results.json`, `results/results.md` | Aggregate results and paired comparisons |

Each command documents its options with `--help`.
