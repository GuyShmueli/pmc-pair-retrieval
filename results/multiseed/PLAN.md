# Multi-seed robustness analysis of the two final configurations

Written 2026-10-23, before any multi-seed training. This analysis is separate from the code release.

## Purpose

Measure how much the two final configurations vary with the training seed. Seed 42 stays the released model and the reported result; the additional seeds are used only to describe variability.

## Configurations

Both configurations are taken exactly from the verified release archive (configuration files, code and environment). They use 3 training and 3 inference views at 224 px.

| Configuration | Seed-42 member of the analysis | Selection rule (unchanged) |
|---|---|---|
| image_to_image | Released epoch-8 checkpoint, from the existing run | Best validation Recall@10 over all 12 epochs |
| image_text_to_image | Released epoch-28 checkpoint, from the existing run | Best validation Recall@10 with the mean-of-descriptions query text; early stopping with patience 3 after at least 3 epochs; at most 30 epochs |

## Seeds

- New seeds 43 and 44 for each configuration: 4 training runs. With seed 42 this gives 3 seeds per configuration.
- Only the `seed` value changes. It controls the random generators, model-construction randomness, the order of training data, batch composition, augmentations and dropout.
- `hash_seed` stays 42, as the release requires; it only fixes Python's internal set order.
- More seeds (45, 46, ...) can be added later under this same plan.

## Code and environment

- The release package as extracted from the final archive after the clean-room verification passes.
- The fresh environment created from the release's environment.yml.
- The pinned model cache.
- Training: `python -m pmc_pair_retrieval.train` with a copy of the configuration file in which only `seed` differs.
- Evaluation: `python -m pmc_pair_retrieval.evaluate`. Nothing else is modified.

## Hardware

- One L40S GPU per run, SLURM partition short, constraint firefoot.
- Environment variables as documented in the release README.

## Evaluation

For every new seed's selected checkpoint:

1. Validation, both configurations.
2. Test, read once, after the selected epoch has been fixed by validation.

The pretrained baselines are deterministic. Their shipped per-query files are reused unchanged.

## Analysis

- **Per seed:**
  - selected epoch;
  - validation and test R@1, R@5, R@10, R@100, Hit@10, mAP, nDCG, MRR and median first relevant rank;
  - the paired fine-tuned minus pretrained difference for R@10 and Hit@10, with the release's 95% interval (5,000 resamples over article groups, same seed and groups as compare.py).
- **Across seeds (n = 3 per configuration):** mean, sample standard deviation, minimum and maximum of each metric and of each paired difference.
- **Agreement with the released result:** the seed-42 values must equal the release's results.json exactly; any mismatch is reported.
- **Descriptive only:** no metric is selected or tuned on test, and no seed result replaces the seed-42 release.

## Outputs

All written inside this directory:

- runs/: training outputs per configuration and seed;
- eval/: per-query files and summaries;
- reports/multiseed.json and reports/multiseed.md: tables and the plain-language summary;
- logs/: job logs.

Existing files outside this directory are not modified.
