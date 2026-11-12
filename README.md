# BiomedCLIP image retrieval on PMC-PaIR

This repository holds the BiomedCLIP retrieval experiments on PMC-PaIR, a dataset of annotated image pairs from
PubMed Central articles. It contains

1. the code to recompute the article-disjoint train/validation/test split, byte for byte;
2. the code to fine-tune BiomedCLIP in two configurations, image_to_image and image_text_to_image, with a fixed data
   order, augmentation, losses, optimizer, learning-rate schedule and checkpoint selection;
3. the sidecars and sha256 values of the fine-tuned weights of both configurations, in `weights/` and
   `WEIGHTS_SHA256SUMS`, and a command to download the two weight files, which are not stored in Git (see
   "Released weights");
4. the code to evaluate pretrained BiomedCLIP and the fine-tuned weights on the validation and test splits, the
   per-query results of that evaluation, in `results/per_query/`, and the top-10 ranked lists of its test
   queries, in `results/rankings/`;
5. the code to build the results tables, with paired bootstrap confidence intervals;
6. the code to draw the evaluation and error-analysis figures from these results, and the figures with their legends
   and source data, in `results/figures/`;
7. the code and saved records of the comparison of training seeds 42, 43 and 44, in `results/multiseed/`.

<!-- BEGIN release-status -->
## Data and weights

The dataset is not yet available for download. Its reserved Zenodo DOI is `10.5281/zenodo.23439169`.

The fine-tuned seed-42 checkpoints are also not yet available for download. Their metadata and checksums are
included here; see [Released weights](#released-weights) for details.

For dataset construction and annotation, see the companion `pmc-pair-pipeline` repository.
<!-- END release-status -->

## Overview

Every query is built from an image, and the relevant images are that image's pair partners: the images it forms a
pair with in `image_pairs.json`. There are two configurations:

- **image_to_image**: the query is the image alone. The BiomedCLIP vision tower is fine-tuned with a per-positive
  InfoNCE loss over the images of each batch. The released weights are those of epoch 11 of 12.
- **image_text_to_image**: the query is the image together with its descriptions; the query text is the mean of the
  unit embeddings of the image's distinct descriptions (see "Query text"). Both towers are fine-tuned, except the
  first 8 of the 12 vision transformer blocks. The loss combines an image-image and an image-text triplet loss, in
  batches built around cached hard negatives. The released weights are those of epoch 21 of at most 30; training
  stopped early after epoch 24.

Both configurations use **3 training views and 3 inference views at 224 px**. Each image is resized to a short side of
224 px (bicubic) and cut into three 224 x 224 crops along its long axis: start, centre and end. Each view embedding
is L2-normalised; the view embeddings of an image are averaged and normalised again. Training jitters every crop with
a random resized crop; for image_to_image, it also applies colour jitter and a small rotation. Evaluation uses the
crops as they are.

**Evaluation protocol.** Every image of a split is a query, and the images of the same split form the gallery. The
relevant images of a query are its direct pair partners. Images linked to the query only through a chain of pairs
(the other members of its connected component in the pair graph) are removed from its ranking, and the query itself
ranks last. Reported metrics are Recall@k (R@k: the fraction of the query's partners in the top k) and Hit@k (at
least one partner in the top k) for k = 1, 5, 10 and 100, mean average precision (mAP) and nDCG over the full
ranking, the mean reciprocal rank of the first partner (MRR) and the median rank of the first partner.

## Installation

```bash
conda env create -f environment.yml
conda activate pmc-pair-retrieval
```

`environment.yml` installs Python 3.11.10 and the packages pinned in `requirements.txt`. These are torch 2.7.1 and
torchvision 0.22.1 built for CUDA 12.6 (from the PyTorch wheel index), open_clip_torch 3.1.0, timm 1.0.19,
transformers 4.55.2, tokenizers 0.21.1, huggingface_hub 0.34.4, safetensors 0.4.5, numpy 2.1.2, Pillow 11.3.0,
PyYAML 6.0.2, ftfy 6.3.1, regex 2024.11.6 and tqdm 4.67.1, plus matplotlib 3.9.2 and fontTools 4.51.0 for the
figures. torch fixes the versions of its CUDA libraries and of Triton. The other dependencies are not pinned: pip
resolves them at install time, to current versions (for example filelock, fsspec, jinja2, networkx, sympy, requests,
hf-xet, and matplotlib's contourpy, kiwisolver and pyparsing). The runs behind the released weights, their evaluations
and the CPU checks of the shipped files used an environment built from `environment.yml` on 2026-10-24, with the
versions that pip resolved then. Later resolutions were not tested.
`environment.yml` also sets minimum versions of pip, certifi and ca-certificates, which do not affect any computation.
In another Python 3.11.10 environment, run `pip install -r requirements.txt`; the checks under "Reproducibility notes"
did not use this route.

Keep the pinned versions. Model construction consumes random numbers, and the seeds of the
augmentation streams are drawn after it; augmentation, JPEG decoding and tokenisation also depend on the library
versions.

The package is not installed: run every command from the repository root, as `python -m pmc_pair_retrieval.<module>`.

Training and evaluation need a CUDA GPU. The CUDA 12.6 wheels include the CUDA runtime, so no CUDA toolkit is needed,
but they need an NVIDIA driver that supports CUDA 12: on Linux, version 560.28.03 or newer (the driver of CUDA 12.6),
or 525.60.13 or newer through CUDA minor-version compatibility. The training runs behind the released weights and the
three-seed results used one NVIDIA L40S (48 GB) per run, and the shipped results were computed on an L40S; the driver
version of these runs was not recorded. Typical durations on one L40S:

| Task | Duration |
|---|---|
| image_to_image training | 551 steps per epoch at 1.3 to 1.9 s per step, i.e. about 12 to 16 min per epoch, plus 0.2 to 1.2 min of validation; the 12 epochs of the run behind the released weights took 2.4 h, those of seeds 43 and 44 2.4 h and 2.5 h |
| image_text_to_image training | 548 steps per epoch at about 1.5 s per step, i.e. about 14 min per epoch including validation (27 min for the first epoch of the run behind the released weights); the 24 epochs of that run took 5.8 h, the 24 epochs of seed 43 5.6 h and the 29 epochs of seed 44 6.8 h |
| Evaluation of one model on one split | about 0.5 to 2.5 min; the first evaluation in a job takes 1.5 to 6 min, including the start of the job |
| Split check | about 5 s, CPU only |
| Results tables | about 15 s, CPU only |
| Pixel-hash check | about 4 min with 16 workers when the images are not cached (under 1 min when they are), about 10 min with 2 workers; reading the images dominates, decoding takes about 8 CPU-minutes |
| Image-size check | about 2 to 3 min with 16 threads when the images are not cached (under 30 s when they are); only the image headers are read |
| Figures | about 1 min, CPU only |

## Data

The dataset is *PMC-PaIR: Clinically similar medical image pairs from PubMed Central case reports*, DOI
10.5281/zenodo.23439169 (see [Data and weights](#data-and-weights)). Once available, download all files of the record
into one new directory and set `DATA_DIR` to it:

```bash
export DATA_DIR=/path/to/pmc-pair
```

The separate `pmc-pair-pipeline` repository (PMC-PaIR dataset construction pipeline) provides a full check of the
deposited files, including the contents of all four ZIPs. Optionally, run that check before extraction, from the
pipeline repository with its dependencies installed:

```bash
python -m pmc_pair verify-deposit --data-dir "$DATA_DIR" --split-file image_splits.json
```

The split in that command belongs to the pipeline code repository, not the Zenodo download. See its
`docs/DEPOSIT.md` for details, and its `docs/ANNOTATION_PROVENANCE.md` for the code, prompts, and settings used to
generate the deposited annotations. The retrieval workflow itself needs only the files listed below and its bundled
split. Check the download checksums and extract the four image archives in place:

```bash
cd "$DATA_DIR"
sha256sum -c SHA256SUMS.txt
for z in images_captions_00*.zip; do unzip -q -n "$z" -d .; done
```

The archives `images_captions_001.zip` to `images_captions_004.zip` hold `images/<image_id>.jpg` and
`captions/<image_id>.txt`. This code reads only `images/` and the following files at the top level of `DATA_DIR`:

```
DATA_DIR/
  image_pairs.json          86,144 pair records {"pair_id": [image_a, image_b], "modality", "anatomy", "diagnosis"}
  images/<image_id>.jpg     the images
  image_labels.jsonl        modality, anatomy and diagnosis labels of every image (figures only)
  licenses.csv              licence and attribution of every image (figures only)
  withheld_image_ids.txt    the ids of the withheld images (figures only)
```

Use a new `DATA_DIR`, because extraction does not delete files that are already there.
The release also contains `withheld.csv`, which gives the reason why each withheld image is not distributed; this
code does not read it.

Images and captions keep the terms recorded for each image in `licenses.csv`. `LICENSE.txt` of the dataset release
states the scope of the authors' own grant. [`LICENSING.md`](LICENSING.md) gives the terms of the files of this
repository.

The pairs link 86,893 images from 56,013 articles. Each image id starts with the PMCID digits of its source article,
followed by `_`. Each pair has one description, `"{diagnosis} in the {anatomy} on {modality}."`. The descriptions of
an image are the distinct descriptions of all the pairs it belongs to. `image_pairs.json` of the release has sha256
`cda80d74d8ff98f0e6d31dbbf66c9823a109ab37ef51db8c800d16e5da0271d7`.

**Withheld images.** The public image archive withholds 68 of the 86,893 images. The shipped results were
computed with all of them. The commands that read images (`train`, `evaluate`, `pixel_hashes` and `image_sizes`) stop
with an error that gives the number of missing images, unless `--allow-missing-images` is given. With this flag,
`train` trains and validates without the missing images and the pairs that contain them, `evaluate` drops those
pairs, so the query sets, positives and galleries change, and `pixel_hashes` and `image_sizes` check the images that
are present. Training and evaluation results then differ from the shipped ones. On this path the trainers'
validation also differs slightly from `evaluate`: the trainers keep, as gallery images, the present images whose pairs
were all dropped (3 validation images with the public archive), and the image_text_to_image validation still averages
the descriptions of the dropped pairs into the query text of an image (4 validation queries); `evaluate` drops both.
The validation numbers that a trainer logs then differ slightly from those of `evaluate` for the same weights. With
all images, the two agree as described in "Query text" and "Reproducibility notes". The split check and `compare` read
only `image_pairs.json`, so they work exactly with the public release. So does `figures`: it reads the three other
dataset files listed above and only the images it displays, none of which is withheld.

**Split.** `data/article_disjoint_split.json` assigns each image to train (70,552 images), val (8,245) or test
(8,096). Images linked by a pair or by a shared article always share a split. The 18,597 connected components of the
pair graph merge into 13,338 article groups. These are shuffled with seed 42 and divided 80/10/10 by group count
(10,670 / 1,333 / 1,335 groups), so no pair and no article spans two splits. The file is shipped, and `split`
recomputes it from `image_pairs.json` alone (see "Commands"). The split is distributed with this code, not in
the Zenodo dataset. The `pmc-pair-pipeline` repository supplies the same assignments as `image_splits.json`.
Always use this repository's bundled `data/article_disjoint_split.json` for the retrieval commands below.

**Pixel hashes.** `data/pixel_md5.json` holds the md5 of the array shape and the decoded RGB pixels of every image.
PMC-PaIR has 49 groups of pixel-identical images (99 images). With the shipped split, two of them join a training image
with a validation image (6542492_1_2 with 6542490_1_2, and 6542492_1_3 with 6542490_1_3); no group joins a test image
with an image of another split. The image_to_image trainer uses these groups so that a pixel copy of a query or of one
of its positives is never treated as a negative, and it does not use training images that are pixel copies of a
validation or test image, nor partners left without a training positive: 3 of the 70,552 training images (6542492_1_2,
6542492_1_3 and 5831235_1_1). The image_text_to_image trainer does not read the pixel hashes, so its training pairs
include those of the two training images in these groups. `pixel_hashes` regenerates or checks the file; the hashes
are reproduced exactly with Pillow 11.3.0.

**Image sizes.** `data/image_sizes.csv` holds the width and height in pixels of every image, as stored in its JPEG
header (`image_id,width,height`, one line per image in sorted id order). Analyses by image geometry, such as the
aspect-ratio subgroups of the figures, read this file and therefore need none of the withheld images. `image_sizes`
regenerates or checks the file.

**Figure image exclusions.** `data/figure_image_exclusions.csv` (`image_id,reason`) lists images that `figures` never
displays, with the reason for each. Only the figures read it.

**Hard negatives.** `data/hard_negatives.json` lists, for every training image, the 15 training images that the
pretrained model ranks most similar to it, using one 224 px centre crop per image and cosine similarity. Excluded are
the image itself, its known positives, and the images of the same connected training component or the same article.
The image_text_to_image trainer builds its batches around these lists. The file was mined once on a GPU, with bf16 and
TF32 arithmetic, and is loaded unchanged. Mining is not part of this package and has not been repeated, so this code
can check the file only against the sha256 below. Whether a repeat gives the same bytes has not been tested; GPU
arithmetic and ties between pixel-identical images may change it.

```
b0b7be4be1597fa4e557fa4c5a51d0148361c2f253a604cf3df4a95a277c30eb  data/article_disjoint_split.json
b965e1cf723b7c802388cf9148bb852a227e6573fb121176396182b703c06231  data/hard_negatives.json
91f4db1497140e392a77f0d611c14f27bd3b29e5e87f7bc0b7d349d50ca54443  data/figure_image_exclusions.csv
73c58a2fa0916db5386cfcb7c39cd052ed50b953ab99fead317b835c82f2436d  data/image_sizes.csv
9b6552e6c8a2353e0fd99bc2a6604587d0e4c8b4fc771bb27669543900ad7ab3  data/pixel_md5.json
```

## Pretrained model

The pretrained model is BiomedCLIP, `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` at revision
`9f341de24bfb00180f1b847274256e9b65a3a32e`. Its weights are `open_clip_pytorch_model.bin`, sha256
`52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be`. The text configuration and tokenizer come from
`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` at revision `d673b8835373c6fa116d6d8006b33d48734e305d`.
open_clip loads the `main` revision of a Hugging Face repository and accepts no revision argument, so the revisions
are pinned through a dedicated cache that is used offline. Create the cache once (this needs network access):

```bash
python -m pmc_pair_retrieval.model download --cache-dir /path/to/biomedclip_cache
```

`download` fetches the pinned files and checks their sha256. It records the files that do not exist at the pinned
revisions, as the Hugging Face client does after a failed lookup, and points `refs/main` of both repositories at the
pinned commits. It refuses a cache whose `refs/main` names another commit, so use a dedicated directory. Depending on
the installed version of hf-xet (the Hugging Face transfer backend, a dependency of huggingface_hub), the download may
also write a chunk cache under `HF_HOME` (about 850 MB with hf-xet 1.1.5); setting `HF_HUB_DISABLE_XET=1` for the
download avoids it.

Before every other command, set:

```bash
export HF_HUB_CACHE=/path/to/biomedclip_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Every model load first checks the cache. Offline mode must be on, `refs/main` of both repositories must name the
pinned commits, and every pinned file must have its pinned sha256. `TRANSFORMERS_CACHE`, `PYTORCH_TRANSFORMERS_CACHE`
and `PYTORCH_PRETRAINED_BERT_CACHE` must be unset or equal to `HF_HUB_CACHE`. After the model is built, its state dict
must hash to `28b0886e361ba509b0b4eef721776baa71b5fe99f2755c5dba512c45c3554602`. `train` and `evaluate` also check
the cache before they load any data or write anything, and stop with a message that names the problem. Each check
reads the 784 MB weights file, which takes a few seconds, or a few minutes on a slow file system.

**Licence and intended use.** Both model repositories give the MIT licence, with the notice `Copyright (c) Microsoft
Corporation`. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) holds its full text, the statements of the model card
and links to the model cards and licence files at the pinned revisions. The BiomedCLIP model card states: "This model
is intended to be used solely for (I) future research on visual-language processing and (II) reproducibility of the
experimental results reported in the reference paper." It also states: "**Any** deployed use case of the model ---
commercial or otherwise --- is currently out of scope."

## Commands

Environment for all commands below (the settings of the runs behind the shipped results), from the repository root:

```bash
export DATA_DIR=/path/to/pmc-pair                     # the dataset release, arranged as in "Data" (the public
                                                      # archive lacks 68 images; see "Withheld images")
export HF_HUB_CACHE=/path/to/biomedclip_cache/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4            # CPU threads of the image_text_to_image validation scoring
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**With the public image archive.** Step 2 trained the released weights and step 3 computed the shipped results, both
with all 86,893 images.
The public image archive withholds 68 of them (listed in `withheld_image_ids.txt`; `withheld.csv` of the
dataset release gives the reasons). With it, add `--allow-missing-images` to `train` and `evaluate`. This drops the
116 pairs that contain a withheld image (103 training, 8 validation and 5 test pairs), so the evaluation has 8,239
validation and 8,088 test queries instead of 8,245 and 8,096. The resulting per-query files, ranked lists and
smoke-test values differ from the shipped ones and cannot be compared with them byte for byte. This path has not been
run on a GPU. The split check of
step 1, `compare --check` of step 4 and the figures check work with the public archive unchanged, because they read
no withheld image. For the three-seed commands, see the table under "Three-seed results".

The numbers depend on the following settings:

- the library versions and Python 3.11;
- `PYTHONHASHSEED=42`, which fixes the image_text_to_image training-pair order. The training command takes care of
  it: it checks the hash seed that Python actually uses and, if it differs, restarts itself once with
  `PYTHONHASHSEED=42` and without `-E` or `-I` (which make Python ignore the variable). It refuses to train if the
  check still fails, and under Python versions other than 3.11;
- the four CPU threads of the image_text_to_image validation;
- the GPU model, for exact evaluation;
- the configuration values, including `num_workers` and the batch sizes.

### 1. Split

```bash
python -m pmc_pair_retrieval.split --data-dir $DATA_DIR --check
```

This recomputes the split from `image_pairs.json` and compares it byte for byte with
`data/article_disjoint_split.json`. It prints the numbers of images, pairs, articles and article groups, the groups
and images of each split and the sha256, and ends with
`[split] OK: identical to .../data/article_disjoint_split.json (sha256 b0b7be4b...)`. With `--output FILE` instead
of `--check`, it writes the computed split to a new file.

Optionally, check the pixel hashes. This decodes every image:

```bash
python -m pmc_pair_retrieval.pixel_hashes --data-dir $DATA_DIR --check --workers 16
```

With the public image archive, add `--allow-missing-images`; the hashes of all present images are then compared with
`data/pixel_md5.json`.

The image sizes are checked in the same way; this reads only the image headers:

```bash
python -m pmc_pair_retrieval.image_sizes --data-dir $DATA_DIR --check
```

It ends with `[image_sizes] OK: identical to .../data/image_sizes.csv (sha256 73c58a2f..., the published file)`. With
the public image archive, add `--allow-missing-images`; the sizes of all present images are then compared with
`data/image_sizes.csv`, and the entries of the 68 withheld images remain unchecked.

### 2. Training

```bash
python -m pmc_pair_retrieval.train --config configs/image_to_image.yaml \
    --data-dir $DATA_DIR --output-dir runs/image_to_image
python -m pmc_pair_retrieval.train --config configs/image_text_to_image.yaml \
    --data-dir $DATA_DIR --output-dir runs/image_text_to_image
```

The `configuration` key of the configuration file selects the trainer. The output directory must be new or empty.
Each run writes:

| File | Content |
|---|---|
| `config.yaml` | the configuration as run: the configuration file as written, with repository-relative data paths and the `--epochs` override |
| `history.jsonl` | one record per epoch: training losses, the validation summary (mean and 95% bootstrap interval of every metric), minutes and `batch_sequence_sha256`, a hash of the epoch's data order; image_to_image also records epoch 0, the pretrained model |
| `best_model.safetensors`, `best_model.json` | the weights of the selected epoch and their sidecar, in the format of the released weights (see "Released weights"); image_to_image saves the vision tower, image_text_to_image the full model |
| `summary.json` | the selected epoch and its validation summary, epochs run, trainable parameters and hours |
| `val_per_query.csv` | image_to_image only: per-query validation metrics of the selected epoch |

Progress lines go to standard output. image_to_image reports, every 100 optimizer steps, the running mean loss of the
epoch and two learning rates: `lr_top` for the top block, final norm and projection head, and `lr_embed` for the
patch, position and class-token embeddings. image_text_to_image reports, every 20 steps, the running mean loss of the
epoch, the image-image and image-text losses of the current step and the learning rate. After every validation, both
trainers print a `[val]` line with the selection metric (validation R@10) at full precision, a few other metrics, the
best value so far and the minutes spent.

**Smoke tests.** These runs take a few minutes and stop early, so they do not produce the released weights:

```bash
python -m pmc_pair_retrieval.train --config configs/image_to_image.yaml \
    --data-dir $DATA_DIR --output-dir runs/smoke_image_to_image --max-steps 200
python -m pmc_pair_retrieval.train --config configs/image_text_to_image.yaml \
    --data-dir $DATA_DIR --output-dir runs/smoke_image_text_to_image --epochs 1 --max-steps 40
```

`--max-steps N` stops training after N optimizer steps, validates that epoch and ends the run.

- For image_to_image, the learning-rate schedule keeps its full horizon, so the first N steps are those of a full run.
- For image_text_to_image, N also becomes the schedule horizon. This equals a full run for
  N up to 200, where every step is a warmup step.

The smoke commands have not been run with the shipped split. By the two rules above, they should give the values
that the seed-42 runs behind the released weights logged, with all images on an L40S and the pinned versions:

- **image_to_image.**
  - `[val] epoch 0 R@10=0.1449659091507984`, the value that `history.jsonl` records for epoch 0.
  - `contrastive=3.4220` at step 100 and `contrastive=3.1855` at step 200.
  - The epoch-1 `batch_sequence_sha256` is `97293f37db20a33b30f9a7f083083b45e72d15eafe03983a0a83f65999047885`.
- **image_text_to_image.**
  - `[data] training pairs: 70262 (order sha256 9adbb8dd16264428dddb4a280ce65027bba155688b3383a12540582ed7ac0aa9)`.
  - `[train] loader base seed 8845398357506713019`.
  - `loss=0.1225 loss_ii=0.2181 loss_it=0.0840 lr=1.05e-07` at step 20.
  - The epoch-1 `batch_sequence_sha256` of a full run is
    `c5f23ecfcec1b3b61f8ec8a550c92e61164c1377182fd7ad88a86cb3f3c4a4ca`.

### 3. Evaluation

The fine-tuned evaluations below read `weights/<retrieval>.safetensors`, which must be in place first (see
"Download the released weights"). The pretrained evaluations do not read them.

```bash
for split in val test; do
  for retrieval in image_to_image image_text_to_image; do
    python -m pmc_pair_retrieval.evaluate --weights pretrained --retrieval $retrieval \
        --split $split --data-dir $DATA_DIR --output-dir eval
    python -m pmc_pair_retrieval.evaluate --weights weights/$retrieval.safetensors --retrieval $retrieval \
        --split $split --data-dir $DATA_DIR --output-dir eval
  done
done
```

This evaluates pretrained BiomedCLIP and the released fine-tuned weights on both splits: the eight evaluations that
`results/per_query/` records (see "Reproducibility notes"). The evaluation settings come from the configuration file
of the retrieval task,
`configs/<retrieval>.yaml`, the same values that the trainer's validation uses: the split file (`split`), 3 inference
views at 224 px (`eval_views`, `image_size`), 128 images per forward pass (`eval_batch_size`) and, for
image_text_to_image, 256 descriptions per text batch (`text_batch_size`). To evaluate your own run, pass its
`runs/<configuration>/best_model.safetensors` as `--weights` with a separate `--output-dir` (for example
`eval_myrun`). Evaluate pretrained BiomedCLIP into the same directory, or copy `eval/<retrieval>/pretrained_<split>.*`
there, so that `compare --per-query-dir eval_myrun` can build the tables and the paired differences. image_to_image
weights hold the vision tower only, so they can only be evaluated with `--retrieval image_to_image`. Weights trained
for the other configuration, as their sidecar records, are evaluated only with an output name other than
`fine_tuned` (`--name`), and `compare` refuses a fine-tuned cell whose weights belong to another configuration. The
outputs go to `eval/<retrieval>/`. The output name is `pretrained` or `fine_tuned`, according to `--weights`, unless
`--name` gives another one; `compare` reads only outputs named `pretrained` and `fine_tuned` and skips files with any
other name:

- `<name>_<split>.jsonl`: one row per query, in sorted query-id order, with `query_id`, `n_pos`, R@k, Hit@k, `mAP`,
  `nDCG` and `first_rel_rank`;
- `<name>_<split>.rankings.jsonl`: the ranked list of every query, in the same order, as
  `{"query_id": ..., "top10": [...]}`: the ids of the gallery images at ranks 1 to 10 (rank 1 first) of the ranking
  from which the metrics of the query are computed. The query itself and the images left out of its ranking (see
  "Evaluation protocol") never appear, so a query with `first_rel_rank` r of at most 10 has its first relevant image
  at position r of `top10`;
- `<name>_<split>.summary.json`: means with 95% bootstrap intervals, MRR, the median first rank, the weights (file,
  sha256 of the file and of the state dict and, from the sidecar, the selected epoch, the global step and the
  validation R@10 at selection), the settings and the environment;
- `<name>_<split>.image_embs.pt`: the image embeddings;
- `<name>_<split>.text_embs.pt`: the description embeddings (image_text_to_image only).

Existing outputs are never overwritten. The shipped per-query files are in `results/per_query/`, under the same
names. With all 86,893 images on an L40S, compare an evaluation with them byte for byte (see "Reproducibility
notes"), for example:

```bash
cmp eval/image_to_image/pretrained_test.jsonl results/per_query/image_to_image/pretrained_test.jsonl
```

The ranked lists of the shipped test evaluation are in `results/rankings/<configuration>/<model>_test.jsonl`; they
are compared in the same way:

```bash
cmp eval/image_to_image/pretrained_test.rankings.jsonl results/rankings/image_to_image/pretrained_test.jsonl
```

### 4. Results tables

```bash
python -m pmc_pair_retrieval.compare --data-dir $DATA_DIR --per-query-dir eval
```

`compare` reads `eval/<configuration>/<model>_<split>.jsonl`, where `<model>` is `pretrained` or `fine_tuned`
(files with other names are not read), skips missing files, and writes `eval/results.json` and the same tables in
Markdown, `eval/results.md` (`--output` sets another results file; the Markdown file goes next to it). For each
configuration, model and split, it writes the means of all metrics, the sha256 of the per-query file
and, if present, the sha256 of its ranked lists (`<model>_<split>.rankings.jsonl` next to the per-query file; for the
shipped per-query files, `results/rankings/<configuration>/<model>_<split>.jsonl`). For each fine-tuned cell, it
records the weights the cell was computed with: the file, the sha256 of the file and of its state dict, the selected
epoch, the global step and the validation R@10 at selection, from the evaluator's summary next to the per-query file
or, without one, from the sidecar in `weights/`. It also writes the paired differences fine-tuned minus pretrained for
R@10, Hit@10, R@1, R@100 and MRR. Their 95% intervals come from a query-weighted paired bootstrap over article groups
(5,000 resamples) and, for comparison, over articles. From `DATA_DIR` it reads only `image_pairs.json`, to form the
article groups.

Without `--per-query-dir`, `compare` uses the shipped `results/per_query/` and `results/rankings/` and writes
`results/results.json`, `results/results.md` and the generated tables of this file ("Released weights" and "Expected
results"). `--check` recomputes all three and compares them with the files instead of writing them, so it also
detects a changed per-query file or ranked list through its sha256:

```bash
python -m pmc_pair_retrieval.compare --data-dir $DATA_DIR --check
```

## Released weights

The two fine-tuned weight files are not stored in this Git repository; see "Download the released weights" below. The
commands expect them in `weights/`, the paths named by the table below and by `results/results.json`. They are the
selected epochs of two training runs with seed 42, made with `pmc_pair_retrieval.train` of this repository (see
"Reproducibility notes"):

<!-- BEGIN compare:released-weights -->
| Configuration | File | Contents | Selected epoch | Global step | Validation R@10 at selection (%) | File sha256 | State-dict sha256 |
|---|---|---|---:|---:|---:|---|---|
| image_to_image | `weights/image_to_image.safetensors` | vision tower, 151 tensors | 11 | 6,061 | 19.55 | `cb24963bcb0ddba73a1d7e4a1a4539933b05ee9b8bf8d8f3454ca5c7c9949622` | `23825766247fe2b64d218f9500330e376811105ba6bc670e1ab872025b78794a` |
| image_text_to_image | `weights/image_text_to_image.safetensors` | full model, 351 tensors | 21 | 11,508 | 30.87 | `4d159e28b2ab1621e463bc3be8aab907356b83536d0ac9698620f32dabd54292` | `e7ebac89af320be1d57c7a51388623501d62b94353d84259be0c7f2d50bf56a9` |
<!-- END compare:released-weights -->

Each weights file comes with a JSON sidecar of the same name, `weights/image_to_image.json` and
`weights/image_text_to_image.json`. The sidecar records the configuration name, the selected epoch and global step,
the trainer's validation summary of that epoch, the hyperparameters (the values of `configs/<configuration>.yaml`),
and the sha256 of the file and of its state dict. The sidecars are stored in this repository, unlike the weight
files. Keep each sidecar next to its weights file: `evaluate` checks the file against it and prints the state-dict
sha256. The image_to_image file holds the vision tower only; its text tower is
the pretrained one. The image_text_to_image file holds the full model.

**Terms.** The terms for the two weight files and their sidecars are not yet set, and [`LICENSE`](LICENSE) grants no
licence for them (see [`LICENSING.md`](LICENSING.md)). Both weight files are fine-tuned from the pretrained BiomedCLIP
weights. "Pretrained model" and `THIRD_PARTY_NOTICES.md` give the licence and the intended use that Microsoft states
for the pretrained weights. These statements do not set terms for the fine-tuned files.

### Download the released weights

**Weight downloads are not yet available:** the separate Zenodo record is unpublished and the weight terms are
unresolved (see [`LICENSING.md`](LICENSING.md)). The default download command below reports that the record is unavailable
and downloads nothing. The JSON sidecars and expected sha256 values are included in this repository.

From the repository root:

```bash
python -m pmc_pair_retrieval.download_weights                                 # both files
python -m pmc_pair_retrieval.download_weights --configuration image_to_image  # one file
python -m pmc_pair_retrieval.download_weights --check                         # hash the local files, no network
```

The command needs only the Python standard library. It reads the file names and sha256 values from the sidecars, and it
stops if they disagree with `WEIGHTS_SHA256SUMS`. Each file is streamed to `weights/<name>.part` and replaces
`weights/<name>` only after its size and sha256 are verified. On any failure the `.part` file is deleted and the exit
status is not zero. A file already in `weights/` with the expected sha256 is left alone. A file with another sha256 is
never overwritten: the command stops and says to move or delete it. A `.part` file left by a killed download is not
touched either; the command asks you to delete it. `--base-url URL` takes each file from `URL/<name>` instead, over
https, or over http for a loopback address only, with the same checks.

To download by hand once the record is published, set `RECORD_ID` to the id of the record and run, from the repository
root:

```bash
for name in image_to_image image_text_to_image; do
  curl -fL -o "weights/$name.safetensors" "https://zenodo.org/records/$RECORD_ID/files/$name.safetensors?download=1"
done
sha256sum -c WEIGHTS_SHA256SUMS
```

Delete a file that `sha256sum` reports as `FAILED`, and download it again.

**Verification.** `sha256sum -c SHA256SUMS` checks every file that Git tracks, including the two sidecars and
`WEIGHTS_SHA256SUMS`. It does not read the weight files, so it also passes on a clone without them.
`sha256sum -c WEIGHTS_SHA256SUMS` checks the weight files once they are in `weights/`; it holds the same sha256 values
as the sidecars and the table under "Released weights". `results/multiseed/INPUT_MANIFEST.json` also pins the sidecars,
and `multiseed --check` verifies them.

**Which commands need the weight files.** In the table, `evaluate`, `split` and so on stand for
`python -m pmc_pair_retrieval.evaluate`, `python -m pmc_pair_retrieval.split` and so on. The checks of the shipped
results read at most the sidecars.

| Command | Reads the two weight files | Without them |
|---|---|---|
| `evaluate --weights weights/<configuration>.safetensors` | yes | stops with `Weights file not found: weights/<configuration>.safetensors` |
| `multiseed --verify-weights` | yes, as the checkpoints of seed 42 | prints `No such file or directory` and the path, and exits with status 1 |
| `run_multiseed` | yes, in its last step, which uses `--verify-weights` | refuses to start, and names the missing files |
| `split --check`, `compare --check`, `multiseed --check`, `figures --check` | no | not affected |
| `run_multiseed --dry-run` | no | not affected |
| `train`, `evaluate --weights pretrained` | no | not affected |

**Weight files kept outside `weights/`.** `results/results.json`, the table under "Released weights" and
`multiseed --verify-weights` all refer to `weights/`, so that is where the files belong. `evaluate --weights FILE`
accepts another path, but it compares the file with a sidecar only if that sidecar is beside the file, with the same
stem. Without one, `evaluate` loads the file without comparing its hashes with anything, cannot check which
configuration the weights were trained for, and records no configuration, selected epoch or selection value in its
summary. If you keep the files elsewhere, copy the sidecar next to each file. A symbolic link in `weights/` to a file
elsewhere passes `sha256sum -c WEIGHTS_SHA256SUMS` and `download_weights --check`, but `multiseed --verify-weights`
refuses a path outside the repository, so copy the files into `weights/` instead.

## Configurations

`configs/image_to_image.yaml` and `configs/image_text_to_image.yaml` list the configurable hyperparameters, with
comments; the table below also gives the values fixed in the trainers.

| | image_to_image | image_text_to_image |
|---|---|---|
| Query | image | image and its descriptions (see "Query text") |
| Trained parameters | the vision tower (86,191,872 parameters); the text tower and the logit scale stay frozen | both towers except vision blocks 0 to 7 (139,199,745 parameters); the patch, position and class-token embeddings are trained |
| Data files | split `data/article_disjoint_split.json`; pixel hashes `data/pixel_md5.json` | split `data/article_disjoint_split.json`; hard negatives `data/hard_negatives.json` |
| Views | 3 for training, 3 for inference; crops of 224 x 224 px | 3 for training, 3 for inference; crops of 224 x 224 px |
| Training augmentation, per view | random resized crop inside the view crop (scale 0.9 to 1.0, aspect ratio 1); colour jitter (brightness 0.1, contrast 0.1) with probability 0.5; rotation by up to 5 degrees | random resized crop inside the view crop (scale 0.9 to 1.0, aspect ratio 1) |
| Training items per epoch | every training image in use once as a query, in a new random order each epoch, with one of its pair partners drawn uniformly at random (70,549 items: 3 of the 70,552 training images are left out, see "Pixel hashes"); the 551 full batches of 128 train on 70,528 of them, and the incomplete last batch (21 items, a different random set each epoch) is dropped | each training pair at most once: 548 full batches of 128 use 70,144 of the 70,262 pairs, and the 118 pairs left when fewer than 128 remain are skipped (the sampler reshuffles all pairs every epoch); the order of a pair's two images is drawn at random |
| Batch | 128 queries and their positives: 256 images, 768 crops | 128 pairs built around cached hard negatives (starting from the next unused pair in the epoch's shuffled order, the images of the batch are visited breadth-first in rounds; a visited image draws up to 3 of its at most 15 hard negatives and tries them in turn, making up to 8 random draws among the pairs that contain each one, until one unused pair is added; it therefore adds at most one pair, and once a pair has been added in the current round, later images of that round try only their first drawn negative; a round that adds nothing adds the next unused pair in shuffled order instead, which adds about 18% of the pairs of a pass with these settings): 256 images, 768 crops |
| Steps per epoch | 551 | 548 |
| Loss | per-positive InfoNCE over the 256 images of the batch, temperature 0.10; images of the query's training component that are not its partners, and repeated copies, are ignored | 0.3 x image-image + 0.6 x image-text triplet loss, margin 0.09; each positive is compared with the mean of the 4 highest-scoring negatives |
| Optimizer | AdamW, betas (0.9, 0.98), eps 1e-6, weight decay 0.05 on weight matrices | AdamW, betas (0.9, 0.98), eps 1e-6, weight decay 0.2 on weight matrices and embeddings |
| Learning rate | 6e-6 for the top block, final norm and head; block i (0 to 11) at 6e-6 x 0.75^(11 - i); embeddings at 6e-6 x 0.75^12 | 1e-6 for all trained parameters |
| Schedule | 200 linear warmup steps, then cosine decay to zero at 12 x 551 = 6,612 steps | 200 linear warmup steps, then cosine decay to zero at 30 x 548 = 16,440 steps |
| Gradient clipping | global norm 1.5 | global norm 1.5 |
| Precision | fp32 weights; bf16 autocast for the image encoder; loss on fp32 tensors outside autocast, with TF32 enabled for fp32 matmuls | fp32 weights; bf16 autocast for the encoders and the losses; TF32 enabled for fp32 matmuls |
| Activation checkpointing | all vision blocks | all vision blocks |
| Epochs | 12 | at most 30 |
| Validation | before training (epoch 0) and after every epoch: image queries on the validation split | after every epoch: image-plus-description queries on the validation split, with the mean-description query text |
| Checkpoint selection | highest validation R@10; a later epoch must be strictly better; epoch 0 is not eligible | highest validation R@10; a later epoch must be strictly better |
| Early stopping | after 12 epochs without improvement, not before epoch 3, so it never triggers in 12 epochs | after 3 epochs without improvement, not before epoch 3 |
| Selected epoch (released weights) | epoch 11 of 12, global step 6,061, validation R@10 19.55% | epoch 21 of at most 30, global step 11,508, validation R@10 30.87%; training stopped early after epoch 24 |
| Seeds | seed 42 (Python, numpy and torch generators, epoch items); `PYTHONHASHSEED=42` | seed 42 (Python, numpy and torch generators, pair shuffle, batch sampler, validation bootstrap); `PYTHONHASHSEED=42` |
| DataLoader workers | 8, started anew every epoch | 4, persistent, prefetch factor 2 |
| Validation and evaluation batch | 128 images | 128 images, 256 descriptions |
| Progress lines | every 100 optimizer steps | every 20 steps |

Both trainers report validation intervals from 200 bootstrap resamples of the queries.

## Query text

The image_text_to_image query for an image q combines q's image embedding with a text vector `t_q`. `t_q` is the mean
of the unit embeddings of q's distinct descriptions (from all the pairs q belongs to), and it is not renormalised.
Gallery image g is scored by

```
s(q, g) = cos(image_q, image_g) + t_q . image_g
```

The same query text is used for checkpoint selection, in the trainer's validation after every epoch, and for the
reported evaluation. A query with one description scores exactly as with that description. In the validation split,
2,859 of the 8,245 queries have two or more distinct descriptions (up to 16); in the test split, 2,819 of the 8,096
queries do (up to 13).

The trainer computes the same ranking in a different way. Its query vector is the normalised mean of the image
embedding and `t_q`, and it scores on the CPU. For the released weights, the trainer's validation means of every
Recall@k and Hit@k equal those of `evaluate` (validation R@10 0.3087449292140053 in both); the mAP and nDCG means
differ by less than 1e-6, because near-tied images can swap places in the CPU ranking.

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

`pmc_pair_retrieval.multiseed` regenerates the comparison of training seeds 42, 43 and 44 from saved per-query
evaluation files. Seed 42 supplies the released checkpoints (see [Reproducibility notes](#reproducibility-notes) for
their training provenance). All six runs (seeds 42, 43 and 44 of both configurations) were trained to completion
with `pmc_pair_retrieval.train` and evaluated with `pmc_pair_retrieval.evaluate` of this repository. The
configurations of seeds 43 and 44, in `configs/multiseed/`, differ from `configs/<configuration>.yaml` only in
`seed`; `hash_seed` stays 42.

**Regenerate the report on a CPU.** Use the package environment and a data directory containing the released
`image_pairs.json`, which supplies the article groups for the paired confidence intervals. Images, model downloads
and a GPU are not needed for this step.

```bash
python -m pmc_pair_retrieval.multiseed --data-dir "$DATA_DIR" --output-dir seed_reports
python -m pmc_pair_retrieval.multiseed --data-dir "$DATA_DIR" --check
```

The first command writes `multiseed.json` and `multiseed.md` to a new output location. The second recomputes and
compares them with the reports in `results/multiseed/`. Existing report files are never overwritten.
Metrics are computed with `compare.cell_entry` and paired differences with `compare.paired_differences`.
Across-seed summaries give the mean, sample standard deviation (denominator n - 1), minimum
and maximum. The test Recall@10 mean and sample SD are 20.11% and 0.07 percentage points for image-only retrieval,
and 31.57% and 0.04 percentage points for image-plus-text retrieval.

`results/multiseed/` includes the additional seeds' per-query metrics, ranked lists, evaluation summaries, training
histories, configurations and checkpoint sidecars. Seed-42 records are read from the existing release files.
`INPUT_MANIFEST.json` records the input hashes, including those of the training and evaluation modules.
[`provenance.json`](results/multiseed/provenance.json) identifies the initial analysis script and report, which are not
shipped, and records the agreement of the reports shipped here with that report, the hashes of the copied records and
how the files listed in `INPUT_MANIFEST.json` differ from the files that the runs used. In these records, "historical"
and "original" refer to the runs and that initial analysis; "portable" refers to the `multiseed` command provided here.
[`PLAN.md`](results/multiseed/PLAN.md) is the analysis plan dated 2026-10-23 of an earlier three-seed comparison, made
with an earlier version of the dataset and split; it is kept unchanged. The runs reported here were made later and have
no dated plan of their own; the sentence of `multiseed.md` that calls `PLAN.md` the original analysis plan is fixed
text of the report command.
The additional seeds' model weights and cached embeddings are not bundled. Their recorded
checkpoint hashes identify the evaluated models, but verifying those checkpoint bytes requires the weight files.

For all six runs, the independently evaluated validation Recall@10 agrees with the trainer's checkpoint-selection
value within the tolerance of the report's informational checks (1e-6, Recall@10 as a fraction): the two values are
equal for image-plus-text retrieval and differ by less than 1e-9 for image-only retrieval. Both values are retained.
The selected epoch is the one chosen from the training history by validation Recall@10.

**Repeat training and evaluation.** After following "Installation", "Pretrained model" and the environment settings
under "Commands", use a new output directory. The driver needs all 86,893 images (see the table below), and its last
step hashes `weights/<configuration>.safetensors`, so it refuses to start without those two files (see
"Download the released weights"). Inspect the commands first with `--dry-run`:

```bash
python -m pmc_pair_retrieval.run_multiseed --data-dir "$DATA_DIR" --output-dir seed_runs --dry-run
python -m pmc_pair_retrieval.run_multiseed --data-dir "$DATA_DIR" --output-dir seed_runs
```

The driver runs four training jobs sequentially, each followed by validation and then test evaluation of its
selected checkpoint. It stops if a command fails and refuses an existing output directory. It then generates
`seed_runs/reports/` using the new runs and the released seed-42 results, with `--verify-weights` to check the new
checkpoint files and the released weight files in `weights/`. The runs in `results/multiseed/` used NVIDIA L40S GPUs
and the pinned environment, with the
`train` and `evaluate` commands run as separate jobs. The driver itself has not run to completion: it has been tested
with `--dry-run` and with mocked subprocesses, not on a GPU. Other hardware or software can produce different values.
No seed is selected using test results.

**Which commands work with the public image archive.** The public image archive withholds 68 of the 86,893 images
(see "Withheld images"). `multiseed` and `run_multiseed` below stand for `python -m pmc_pair_retrieval.multiseed` and
`python -m pmc_pair_retrieval.run_multiseed`.

| Command | Images needed | With the public image archive |
|---|---|---|
| `multiseed --data-dir "$DATA_DIR" --check` | none | Works: reads `image_pairs.json` and the shipped records |
| `multiseed --data-dir "$DATA_DIR" --output-dir seed_reports` | none | Works: same inputs |
| `run_multiseed --data-dir "$DATA_DIR" --output-dir seed_runs --dry-run` | none | Works: prints the commands and reads no data |
| `run_multiseed --data-dir "$DATA_DIR" --output-dir seed_runs` | all 86,893 | Stops at the first training command, which runs without `--allow-missing-images`. On a machine with a GPU and the model cache, that command reports that 68 of the 86,893 images are missing, with one example, and exits; without them it stops earlier with the cache or GPU message. The driver then ends with a traceback and leaves `seed_runs/`, which must be removed before a rerun |
| `multiseed --data-dir "$DATA_DIR" --runs-dir RUNS --eval-dir EVALS --output-dir OUT`, on new runs and evaluations | none, but the evaluations must have used all 86,893 | Fails on evaluations made with `--allow-missing-images`: the required checks of the image count (86,893) and the query counts (8,245 validation, 8,096 test) print `FAIL`, and the first such evaluation then stops the command with `fine-tuned and pretrained per-query files cover different queries`, status 1 and no report |

On the public archive, `train` with a configuration from `configs/multiseed/`, and `evaluate`, can still be run by hand
with `--allow-missing-images`. As described under "With the public image archive", the results then differ from the shipped
ones, and this path has not been run on a GPU. Whatever the images, `--verify-weights` needs the checkpoint file of every
reported seed: those of seed 42 are the two files in `weights/` (see "Download the released weights"), and those of
seeds 43 and 44 are not shipped.

## Figures

`python -m pmc_pair_retrieval.figures` draws the evaluation and error-analysis figures of the two configurations from
the shipped results; it scores nothing. Each figure compares the released fine-tuned weights of a configuration
(image_to_image epoch 11 of 12, image_text_to_image epoch 21 of at most 30, both selected by validation Recall@10;
image_text_to_image training stopped early after epoch 24) with pretrained BiomedCLIP on the 8,096 test queries. The
figures are built at their printed size for a Scientific Data article, 180 mm wide (double column), with all text in
Arial at 7 pt and a 2 mm margin on every side.

| Figure | Panels |
|---|---|
| `retrieval_performance` (180 x 85.5 mm) | a: mean Recall@1, Recall@10, Recall@100 and Hit@10 of both models; b: paired differences fine-tuned minus pretrained with 95% intervals, as in `results/results.json`; c: shares of the queries whose Recall@10 or Hit@10 is higher, lower or equal with the fine-tuned model |
| `retrieval_subgroups_hit10` (180 x 179.89 mm) | paired Hit@10 difference with 95% interval for 50 label categories (9 modality, 18 anatomy and 23 diagnosis) with at least 100 test queries from at least 20 article groups, with the numbers of queries and article groups; a: image_to_image, b: image_text_to_image |

In `retrieval_subgroups_hit10`, filled bluish-green circles and lines mark intervals entirely above zero, and open
grey circles and lines mark intervals that include zero. The marker fill preserves this distinction in greyscale.
The dashed lines show the Hit@10 gains over all test queries: +6.84 and +13.30 percentage points for image_to_image
and image_text_to_image, respectively.

**Inputs.** From the repository: the per-query files and ranked lists of the test split
(`results/per_query/<configuration>/<model>_test.jsonl` and `results/rankings/<configuration>/<model>_test.jsonl`,
whose sha256 must equal the references in `results/results.json`), `results/results.json`, the weight sidecars
`weights/<configuration>.json` (selected epoch and configured number of epochs, checked against `results/results.json`),
`data/article_disjoint_split.json`, `data/image_sizes.csv` and `data/figure_image_exclusions.csv`. From `DATA_DIR`:
`image_pairs.json`, `image_labels.jsonl`, `licenses.csv`, `withheld_image_ids.txt` and the images that the example
figures display. All of them are part of the public release; none of the withheld images is needed.

```bash
python -m pmc_pair_retrieval.figures --data-dir $DATA_DIR --output-dir figures
```

The command takes about a minute on a CPU and refuses to overwrite existing outputs. It writes:

- `<figure>.pdf` (editable vector file: TrueType fonts embedded, text kept as text), `<figure>.svg` (text kept as
  text) and `<figure>.png` (preview at 300 dpi) for each of the eight figures: the two in the table above and the six
  example figures, `retrieval_example_<category>_<configuration>`;
- `captions.tex`: one LaTeX figure environment per figure, with the figure at its natural size
  (`\makebox[\linewidth][c]{\includegraphics{<figure>.pdf}}`) and a legend of at most 350 words whose numbers are read
  from the source data;
- `image_references.tex`: the reference list of the source articles of the images in the example figures;
- `figure_pages.tex`: a document that sets every figure with its legend on an A4 page of the reference layout below;
  compile it with pdflatex (twice) in the output directory;
- `source_data/overall_metrics.csv`: the means of all metrics, equal to `results/results.json`, and for the fine-tuned
  rows the weights file, selected epoch, configured number of epochs (`epochs`, the number after "of" in the legend of
  `retrieval_performance`; image_text_to_image training stopped early after epoch 24), global step and selection
  split, metric and value;
- `source_data/paired_differences.csv`: the paired differences and 95% intervals of `results/results.json`;
- `source_data/per_query_change.csv`: for each configuration and metric (Recall@10, Hit@10), the number and share
  (`_percent`) of test queries whose value is higher (`better`), lower (`worse`) or equal (`unchanged`) with the
  fine-tuned model, with the equal cases split into both 0 (`unchanged_zero`), both 1 (`unchanged_one`) and other
  equal values (`unchanged_other`);
- `source_data/subgroups.csv`: every category, drawn or not, with the numbers of queries and article groups, the share
  of the queries in the largest article group, the effective number of article groups ((sum of n_g)^2 / sum of
  n_g^2, with n_g the queries of group g), and the means, paired differences and 95% intervals of Recall@10 and Hit@10.
  The `shown` column identifies the label categories drawn in `retrieval_subgroups_hit10`. Aspect-ratio and
  description-count strata remain in the file with `shown=no`;
- `source_data/per_query.csv`: one row per test query with its article, article group, number of recorded positives
  and descriptions, number of ranked gallery images, image size, labels (modality, anatomy and diagnosis; several
  labels of one family are joined with `|`), two columns of the example selection (`displayable` and
  `example_exclusion`; `yes` in `displayable` does not mean that an image is free of burned-in text), and the
  Recall@10, Hit@10 and first relevant rank with each model;
- `source_data/examples.csv`: the selected queries, their ranks and top-10 lists of the example figures;
- `source_data/image_credits.csv`: the licence, attribution and copyright notice of every image shown in the
  example figures;
- `manifest.json`: the sha256 and size of every input and of every output, the software versions and the font files.

Units of the source data, also given by the column names: Recall@k, Hit@k, mAP and nDCG are in percent (`_percent`),
MRR is a fraction (`_fraction`), differences and interval bounds of `subgroups.csv` are in percentage points (`_pp`);
in `paired_differences.csv`, `level_unit` is the unit of the `pretrained` and `fine_tuned` columns and
`difference_unit` that of the difference and interval columns (percentage points, MRR: fraction). `per_query.csv`
keeps the per-query Recall@10 and Hit@10 as fractions between 0 and 1 (`_fraction`), as in the per-query files. Ranks
are 1-based.

**Page layout.** Every figure must be included at 100%, so that all of its text stays at 7 pt: the text block must be
at least 180 mm wide (on A4, side margins of 15 mm or less), and any reduction must be applied equally to every figure,
with the resulting text size stated. In the reference layout of `figure_pages.tex` (A4, 15 mm margins, text block
180 x 267 mm, legends in 8 pt Helvetica), every figure fits on one page with its legend directly below it.
The command checks an estimate of figure plus
legend height against the text block, and pdflatex reports no float that is too large for the page.

**Statistics.** The 95% intervals come from the bootstrap of `compare` (`metrics.paired_group_ci` over the article
groups, 5,000 resamples); for a subgroup, the same bootstrap is restricted to the category: the article groups of its
queries are resampled, each contributing only its queries in the category. A query counts in every
modality, anatomy and diagnosis category it carries. A label category is drawn when it has at least 100 queries from
at least 20 article groups; its interval rests on the article groups, so it is approximate where there are few groups
or one group contributes a large share of the queries (`subgroups.csv` gives both). Aspect-ratio and description-count
strata are reported only in the source data. The aspect ratio is the longer over
the shorter image side, in four strata that follow the three-view geometry: the three views of a square (1:1) image
coincide, above 1:1 to 2:1 the first and last views overlap or meet, above 2:1 to 3:1 the three views together cover
the image, and above 3:1 parts of the longer side lie outside every view for 444 of the 449 test queries of that
stratum. In general, the resized longer side is int(224 x ratio) pixels and the views cover it only when it is at most
672 pixels, so parts of it lie outside every view from a ratio of 673/224 (about 3.004:1); the other five test queries
of the stratum have smaller ratios.

**Fonts.** All text is Arial. Identical output requires Arial version 2.82, the version of the Microsoft core fonts
for the web (on Debian and Ubuntu, the package `ttf-mscorefonts-installer`, which installs `Arial.ttf` and
`Arial_Bold.ttf` with the lower-case links `arial.ttf` and `arialbd.ttf`). The command recognises the files by their
content, whatever their names, among the fonts known to fontconfig, the usual font directories and matplotlib's font
cache; after installing fonts, matplotlib's font cache (in `~/.cache/matplotlib` or `$MPLCONFIGDIR`) may have to be
deleted. With another Arial version, the figures are drawn with that version and a warning; they then differ from the
shipped ones, and `--check` reports it. Without Arial regular and bold, the command stops with an error. The subset
tag of each font embedded in a PDF (the six letters before `+` in its name) is derived from the PostScript name and
the embedded characters: matplotlib's default tag is the same for Arial regular and bold, and tools that merge PDF
files by font name, such as Ghostscript, then take one subset for the other and drop glyphs.

**Check.**

```bash
python -m pmc_pair_retrieval.figures --data-dir $DATA_DIR --check
```

This regenerates all outputs into a temporary directory (or into `--output-dir`, if given) and compares them with
`results/figures/`: each shipped file other than `manifest.json` byte for byte, every input and every other output with
the sha256 values of the shipped `manifest.json`, and the font files by content (family, style, version and sha256;
their file names may differ). The software versions that `manifest.json` records are not compared: the check prints a
note for each one that differs. It ends with `[figures] OK: ...`, or lists the differences and exits with status 1.
The dataset files among the inputs must be those of the dataset release, unchanged: a `licenses.csv` that differs in any
column, even one that the figures do not read, is reported as `input DATA_DIR/licenses.csv: sha256 ... instead of ...`.
The figures, LaTeX files and source data contain no timestamps, so regenerations with the pinned versions and the same
fonts are identical byte for byte. The PNG previews are the exception: FreeType rasterises their text, and the FreeType
version depends on how matplotlib was built. The shipped `manifest.json` records the FreeType and zlib versions used
to generate the reference outputs. The check compares PNG previews only when the FreeType versions agree; otherwise
it notes the difference and ends with `[figures] OK: ... except the N PNG previews ...`. PDF, SVG, CSV and LaTeX files
must still match. The regenerated `manifest.json` can differ in recorded software versions, font-file names, and the
sha256 and sizes of PNG previews drawn with another FreeType version. `environment.yml` does not pin zlib, so a new
installation can resolve a different version; any resulting change to a checked output is reported by `--check`.

`results/figures/` holds the outputs that the repository ships: the eight PDF files, `captions.tex`,
`image_references.tex`, `figure_pages.tex`, `source_data/` and `manifest.json`. The SVG files and PNG previews are left
out to keep the repository small; the check verifies them through the manifest.

**Third-party images and fonts.** The six `retrieval_example_*` PDF files reproduce images from articles. Each image
keeps the licence given for it in `source_data/image_credits.csv`. The other two PDF files contain no images from
articles. All eight embed subsets of Arial. `THIRD_PARTY_NOTICES.md` lists this material. The terms for the rest of
`results/` are not yet set (see `LICENSING.md`).

## Reproducibility notes

The notes below describe how the shipped evaluation and training records were made and what was checked. They
identify the hardware, software and reference files; the underlying job logs and check reports are not shipped. The
table lists the checks that can be repeated with the public image archive.

- **Evaluating the released weights.** The eight evaluations that `results/per_query/` records (both models of both
  configurations, on the validation and the test split) were made with `evaluate`, each once, with all 86,893 images
  on one NVIDIA L40S, with the pinned versions, in an environment built from `environment.yml`. The eight per-query
  files of `results/per_query/` and the four test ranked lists of `results/rankings/` are their outputs, unchanged,
  and `compare` on these files gives `results/results.md` and `results/results.json` byte for byte (see
  `compare --check` below). The evaluator encodes fixed crops with the batch sizes of the configuration file, turns
  off TF32 and cuDNN autotuning, and ranks with numpy's default sort. The evaluations were not repeated, and other GPU
  models were not tested; repeating them needs all 86,893 images, so the public image archive does not allow it. When
  the scores are computed on the CPU, and possibly on other hardware, near-tied images can swap places. This can
  change some average precision, nDCG and first-rank values and the order within some ranked lists.
- **Training the released weights.** The released weights are the selected checkpoints of two training runs with
  seed 42, made with `pmc_pair_retrieval.train` and `configs/<configuration>.yaml`, each once, with all 86,893 images
  on one NVIDIA L40S, with the pinned versions, in an environment built from `environment.yml`, in the same way as
  seeds 43 and 44 (see "Three-seed results"). image_to_image ran all 12 epochs and kept epoch 11; image_text_to_image
  stopped early after epoch 24 and kept epoch 21. The `val` entries of the sidecars in `weights/` are the trainers'
  validation values at these epochs. The logs, per-epoch histories and run summaries of these two runs are not
  shipped.
- **Retraining to completion.** All six runs, those of the released weights and seeds 43 and 44 of both
  configurations, were trained to completion with the trainers of this repository, each once, and show the
  variation between training seeds (see "Three-seed results"). The records of seeds 43 and 44 are shipped in
  `results/multiseed/`: `runs/<configuration>_seed<N>/summary.json` gives the selected epoch, the epochs run, the
  hours and the overrides (none), and `eval/seed<N>/<configuration>/fine_tuned_<split>.summary.json` the GPU and the
  versions. The training and evaluation modules listed in `INPUT_MANIFEST.json` differ from those that ran them only
  as described under "Preserved experiment inputs". The weights of seeds 43 and 44 are not shipped. Bitwise
  reproducibility of a full training is not established: cuDNN autotuning is on in both trainers, and no full run of
  these trainers was repeated. A retraining can end with different weights and, with small validation margins, can
  select another epoch. The reported numbers are those of the released weights.

**Checks that a reader can repeat with the public image archive.** These run on a CPU and load no model. They recompute
files from the dataset and from shipped records, so they show that the shipped files follow from those inputs, not that
a training or an evaluation gives them. Each command stands for `python -m pmc_pair_retrieval.<command> --data-dir
"$DATA_DIR"` with the options shown.

| Command | What it recomputes | Result |
|---|---|---|
| `split --check` | the split, from `image_pairs.json` | byte-identical to `data/article_disjoint_split.json` |
| `compare --check` | the results tables and the generated blocks of this file, from the shipped per-query files and ranked lists | byte-identical |
| `multiseed --check` | the three-seed reports, from the saved records in `results/multiseed/` | byte-identical; 268 required and 6 informational checks pass |
| `figures --check`, with the images extracted and Arial 2.82 installed | the figures, legends and source data, from the shipped results | equal, except the PNG previews when the FreeType versions differ |
| `pixel_hashes --check --allow-missing-images` | the hashes of the 86,825 present images | equal to `data/pixel_md5.json` |
| `image_sizes --check --allow-missing-images` | the sizes of the 86,825 present images | equal to `data/image_sizes.csv` |

Further notes:

- **The trainers validate with their own code.** The image_text_to_image validation scores on the CPU, with four
  threads, and the image_to_image validation counts ranks on the GPU. Checkpoint selection depends on these values; the
  reported numbers come from `evaluate`. At the selected epochs of the released weights, the trainers' validation R@10
  and the evaluator's agree: they are identical for image_text_to_image (see "Query text") and agree to 1e-9 for
  image_to_image (0.19548507289054146 and 0.19548507347305375; the evaluator stores per-query values in float32). In
  all six runs of the three-seed comparison, the two values are equal for image_text_to_image and differ by less than
  1e-9 for image_to_image (see "Three-seed results").
- **The image_text_to_image training-pair order depends on string hashing.** The pairs are listed by iterating Python
  sets of image ids and then shuffled with seed 42, so the order depends on `PYTHONHASHSEED=42` and on the Python version;
  the training command requires 3.11. It sets the hash seed, the trainer checks it,
  and the log and `summary.json` record the sha256 of the pair order.
- **The random streams depend on the data loaders.** When a DataLoader starts its workers, it draws a base seed from
  the global torch generator, which model construction has advanced. Worker w draws its augmentations from a
  generator seeded with base seed + w, so the number of workers and the batch sizes change the numbers.
  - image_to_image starts new training workers every epoch, and every validation also starts its own workers.
  - image_text_to_image starts its persistent training workers once. It creates one extra iterator per epoch, because
    the batch order depends on it.
- **`--max-steps` and `--epochs` are smoke-test overrides.** Runs that use them stop early and are not full trainings.
- **Historical source comments.** In the modules, `configs/` and `requirements.txt`, "published" has two meanings.
  In messages and comments about what this repository ships ("the published results", "the published file", "the
  published model", "published run (seed 42)" in `configs/`), it refers to the shipped results and the seed-42 runs
  behind the released weights. In comments that explain why the code is written as it is ("as in the published
  run", "the published code", "as the published runs did"), it refers to the authors' earlier seed-42 scripts and
  runs, made with an earlier version of the dataset and split; those scripts are not included. Statements about
  matching those runs describe the trainers' intended behaviour.
- **Preserved experiment inputs.** [`INPUT_MANIFEST.json`](results/multiseed/INPUT_MANIFEST.json) lists 78 files. The
  split, the hard negatives, `environment.yml`, `requirements.txt` and six of the ten modules among them are
  byte-identical to the files of the package copy that ran the six runs. The six configuration files and the other
  four modules (`dataset.py`, `split.py` and the two trainers) differ from those files only in comment and docstring
  lines and, in `dataset.py` and `split.py`, in one constant each: the image count that an error message prints and
  the sha256 that `split --check` compares with. The image_to_image runs read a `data/pixel_md5.json` with 212 more
  entries, for images that are in no pair of `image_pairs.json`. The other 57 listed files are the shipped records,
  `PLAN.md` and `provenance.json`, which records these differences. Two source comments need clarification:
  - The approximately 1e-4 loss difference mentioned in `train_image_to_image.py` is the first-epoch difference
    between these trainers and the authors' earlier seed-42 run, measured with an earlier version of the dataset and
    split; it is not a measured difference between repeated runs of these trainers. Bitwise reproducibility of
    complete training remains unestablished.
  - The early-stopping comment in the seed-43/44 copies of `configs/image_text_to_image.yaml` describes seed 42, which
    stopped after epoch 24 and kept epoch 21. Seed 43 also stopped after epoch 24 and kept epoch 21; seed 44 stopped
    after epoch 29 and kept epoch 26.

## Licence

The code, its documentation and its configuration are released under the MIT licence ([`LICENSE`](LICENSE)).
[`LICENSING.md`](LICENSING.md) lists the paths that this covers. The licence does not cover the weights, `data/` or
`results/`, whose terms are not yet set, or third-party material, which keeps the terms recorded for it
([`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

## Citation

Please cite this code (Guy Shmueli and Tom Hope, *BiomedCLIP image retrieval on PMC-PaIR*), the dataset (*PMC-PaIR:
Clinically similar medical image pairs from PubMed Central case reports*, DOI 10.5281/zenodo.23439169, see
[Data and weights](#data-and-weights)), the PMC-PaIR Data Descriptor, BiomedCLIP (Zhang et al., *NEJM AI* 2(1), 2025)
and, if you use it, the PMC-PaIR dataset construction pipeline. Citation metadata: [`CITATION.cff`](CITATION.cff).

## Repository layout

```
README.md                       this file
LICENSE                         MIT licence; LICENSING.md lists the paths it covers
LICENSING.md                    the terms for each path of the repository
THIRD_PARTY_NOTICES.md          licence notice of the pretrained models; third-party images and fonts in the figures
SHA256SUMS                      sha256 of every other file that Git tracks; the two weight files are not among them
WEIGHTS_SHA256SUMS              sha256 of the two weight files, for use once they are downloaded
.gitattributes                  stops Git from converting line endings, so a checkout keeps the checksummed bytes
.gitignore                      the outputs of the documented commands, the two weight files, their partial
                                downloads, and Python and editor files
CITATION.cff                    citation metadata
requirements.txt                pinned Python packages (PyTorch from the CUDA 12.6 wheel index)
environment.yml                 conda environment: Python 3.11.10 and requirements.txt
configs/
  image_to_image.yaml           image_to_image hyperparameters and evaluation settings
  image_text_to_image.yaml      image_text_to_image hyperparameters and evaluation settings
  multiseed/                    copies of both configurations for training seeds 43 and 44
data/
  article_disjoint_split.json   image id -> train / val / test
  figure_image_exclusions.csv   images that the example figures do not show, with the reason
  hard_negatives.json           15 hard negatives per training image (image_text_to_image training)
  image_sizes.csv               width and height in pixels per image (analyses by image geometry)
  pixel_md5.json                decoded-pixel md5 per image (image_to_image training)
weights/
  image_to_image.safetensors    fine-tuned vision tower of image_to_image (epoch 11); downloaded, not in Git
  image_to_image.json           its sidecar
  image_text_to_image.safetensors  fine-tuned full model of image_text_to_image (epoch 21); downloaded, not in Git
  image_text_to_image.json      its sidecar
pmc_pair_retrieval/
  dataset.py                    public data loading, split loading and validation, pair-graph components
  views.py                      three-view crops, training augmentation, view aggregation, activation checkpointing
  model.py                      pinned BiomedCLIP download, loading and verification; weight files
  download_weights.py           download and verification of the two weight files
  metrics.py                    per-query metrics, bootstrap intervals, paired group bootstrap
  train.py                      training command; runs the trainer named by the configuration
  train_image_to_image.py       image_to_image trainer
  train_image_text_to_image.py  image_text_to_image trainer
  evaluate.py                   evaluation of pretrained and fine-tuned models; per-query metrics and ranked lists
  compare.py                    results tables with paired differences
  multiseed.py                  three-seed reports and input/metadata checks
  run_multiseed.py              driver for the four additional training runs and their evaluation
  split.py                      article-disjoint split
  pixel_hashes.py               decoded-pixel hashes
  image_sizes.py                image sizes from the JPEG headers
  figures.py                    figures: inputs, statistics, example selection, source data, legends, check
  figure_drawing.py             figure drawing: fonts, millimetre layout, figure types, export
results/
  results.md                    results tables, generated by compare
  results.json                  all values, written by compare
  multiseed/                    three-seed reports, saved evaluation and training records, and provenance
  per_query/<configuration>/<model>_<split>.jsonl   per-query metrics of the pretrained and released models
  rankings/<configuration>/<model>_test.jsonl       top-10 ranked lists of the same test queries
  figures/                      figure PDFs, legends (captions.tex), image references, reference page layout,
                                source data and manifest, written by figures
```

Verify the files that Git tracks with `sha256sum -c SHA256SUMS` and, once they are downloaded, the two weight files
with `sha256sum -c WEIGHTS_SHA256SUMS`, both run from the repository root.

Command-line tools: `python -m pmc_pair_retrieval.train`, `.evaluate`, `.compare`, `.split`, `.pixel_hashes`,
`.image_sizes`, `.figures`, `.multiseed`, `.run_multiseed`, `.download_weights` and `.model download`; each one
describes its options with `--help`.
