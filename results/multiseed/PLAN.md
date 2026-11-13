# Three-seed retrieval analysis procedure

This document describes the configurations, evaluation and reporting procedure for the
bundled three-seed results.

## Purpose

Measure how much the two configurations vary with the training seed. Seed 42 is the released
model. Seeds 43 and 44 describe variability.

## Configurations

Use the configuration files in `configs/` and the environment specified in `environment.yml`.
Both configurations use three training and three inference views at 224 pixels.

| Configuration | Seed-42 checkpoint | Selection rule |
|---|---|---|
| image_to_image | Epoch 11 | Best validation Recall@10 over all 12 epochs |
| image_text_to_image | Epoch 21 | Best validation Recall@10 with the mean-of-descriptions query text. Early stopping with patience 3 after at least 3 epochs, with a maximum of 30 epochs. |

## Seeds

- Seeds 42, 43 and 44 for each configuration.
- Only the `seed` value changes. It controls the random generators, model-construction randomness, the order of training data, batch composition, augmentations and dropout.
- `hash_seed` is 42 to fix Python's internal set order.
- Additional seeds can be analyzed with the same procedure.

## Code and environment

- The code and configurations in this repository.
- The environment created from `environment.yml`.
- The pinned model cache.
- Training: `python -m pmc_pair_retrieval.train` with a copy of the configuration file in which only `seed` differs.
- Evaluation: `python -m pmc_pair_retrieval.evaluate`.

## Hardware

- One CUDA GPU per training run. See the README for the hardware and software requirements.
- Environment variables as documented in the README.

## Evaluation

For each selected checkpoint:

1. Validation, both configurations.
2. Test, read once, after the selected epoch has been fixed by validation.

The report uses the bundled pretrained per-query results as its baselines.

## Analysis

- **Per seed:**
  - selected epoch
  - validation and test R@1, R@5, R@10, R@100, Hit@10, mAP, nDCG, MRR and median first relevant rank
  - the paired fine-tuned minus pretrained difference for R@10 and Hit@10, with the release's 95% interval (5,000 resamples over article groups, same seed and groups as compare.py).
- **Across seeds (n = 3 per configuration):** mean, sample standard deviation, minimum and maximum of each metric and of each paired difference.
- **Agreement with the released result:** the seed-42 values must equal `results/results.json` exactly. Any mismatch is reported.
- **Descriptive only:** no metric is selected or tuned on test, and no seed result replaces the seed-42 release.

## Outputs

Bundled inputs are in `results/multiseed/runs/` and `results/multiseed/eval/` for seeds 43
and 44. Seed-42 evaluations and checkpoint metadata are in `results/` and `weights/`.

`python -m pmc_pair_retrieval.multiseed --data-dir DATA --output-dir OUT` writes
`multiseed.json` and `multiseed.md` to the chosen output directory. The command refuses
to overwrite existing reports. Use `--check` to compare freshly computed reports with
the bundled reports.
