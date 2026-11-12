# Three-seed retrieval results

Seed 42 remains the released model. Other seeds describe variability only. No seed or metric is selected on test.

Fresh file hashes, configuration and checkpoint metadata, checkpoint selection, stopping rules, query alignment, metric recomputation and seed-42 reference agreement. Historical training execution is documented separately in provenance.json. Actual checkpoint bytes were not checked by this analysis.

The original analysis plan is in `PLAN.md`. Historical audit information is in `provenance.json`. The counts below refer to the checks performed by this portable report command.

## Selected checkpoints

| Configuration | Seed | Selected epoch | Epochs run | Validation Recall@10 at selection (%) |
|---|---:|---:|---:|---:|
| image_to_image | 42 (released) | 11 | not in released sidecar | 19.55 |
| image_to_image | 43 | 9 | 12 | 19.49 |
| image_to_image | 44 | 7 | 12 | 19.58 |
| image_text_to_image | 42 (released) | 21 | not in released sidecar | 30.87 |
| image_text_to_image | 43 | 21 | 24 | 30.89 |
| image_text_to_image | 44 | 26 | 29 | 30.94 |

## Test metrics

Recall@k, Hit@10, mAP and nDCG in percent. MRR as a fraction. Paired differences and SD of percentage metrics in percentage points. Across-seed SD uses denominator n - 1.

### image_to_image

| Model / seed | R@1 | R@5 | R@10 | R@100 | Hit@10 | mAP | nDCG | MRR | median_first_relevant_rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained | 3.32 | 9.57 | 14.62 | 47.71 | 21.47 | 8.22 | 24.11 | 0.1105 | 65.00 |
| Seed 42 (released) | 4.31 | 12.91 | 20.15 | 60.03 | 28.31 | 10.89 | 27.65 | 0.1434 | 37.00 |
| Seed 43 | 4.23 | 12.99 | 20.15 | 59.98 | 28.43 | 10.85 | 27.62 | 0.1425 | 37.00 |
| Seed 44 | 4.35 | 12.88 | 20.03 | 59.96 | 28.37 | 10.91 | 27.67 | 0.1440 | 37.00 |
| Mean ± sample SD | 4.30 ± 0.06 | 12.93 ± 0.06 | 20.11 ± 0.07 | 59.99 ± 0.03 | 28.37 ± 0.06 | 10.89 ± 0.03 | 27.65 ± 0.03 | 0.1433 ± 0.0007 | 37.00 ± 0.00 |
| Minimum to maximum | 4.23 to 4.35 | 12.88 to 12.99 | 20.03 to 20.15 | 59.96 to 60.03 | 28.31 to 28.43 | 10.85 to 10.91 | 27.62 to 27.67 | 0.1425 to 0.1440 | 37.00 to 37.00 |

### image_text_to_image

| Model / seed | R@1 | R@5 | R@10 | R@100 | Hit@10 | mAP | nDCG | MRR | median_first_relevant_rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained | 4.87 | 13.99 | 20.53 | 60.08 | 29.68 | 11.59 | 28.30 | 0.1541 | 34.00 |
| Seed 42 (released) | 7.46 | 21.19 | 31.62 | 76.33 | 42.98 | 17.13 | 34.59 | 0.2196 | 15.00 |
| Seed 43 | 7.42 | 21.15 | 31.55 | 76.30 | 42.91 | 17.08 | 34.55 | 0.2188 | 15.00 |
| Seed 44 | 7.54 | 21.02 | 31.55 | 76.36 | 42.92 | 17.17 | 34.64 | 0.2200 | 15.00 |
| Mean ± sample SD | 7.48 ± 0.06 | 21.12 ± 0.09 | 31.57 ± 0.04 | 76.33 ± 0.03 | 42.94 ± 0.04 | 17.13 ± 0.05 | 34.59 ± 0.04 | 0.2195 ± 0.0006 | 15.00 ± 0.00 |
| Minimum to maximum | 7.42 to 7.54 | 21.02 to 21.19 | 31.55 to 31.62 | 76.30 to 76.36 | 42.91 to 42.98 | 17.08 to 17.17 | 34.55 to 34.64 | 0.2188 to 0.2200 | 15.00 to 15.00 |


## Validation metrics

Recall@k, Hit@10, mAP and nDCG in percent. MRR as a fraction. Paired differences and SD of percentage metrics in percentage points. Across-seed SD uses denominator n - 1.

### image_to_image

| Model / seed | R@1 | R@5 | R@10 | R@100 | Hit@10 | mAP | nDCG | MRR | median_first_relevant_rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained | 3.24 | 9.73 | 14.50 | 47.35 | 20.99 | 8.32 | 24.06 | 0.1090 | 67.00 |
| Seed 42 (released) | 4.02 | 12.56 | 19.55 | 58.56 | 27.42 | 10.58 | 27.16 | 0.1362 | 40.00 |
| Seed 43 | 4.04 | 12.48 | 19.49 | 58.56 | 27.37 | 10.56 | 27.15 | 0.1363 | 40.00 |
| Seed 44 | 4.02 | 12.51 | 19.58 | 58.47 | 27.54 | 10.56 | 27.15 | 0.1362 | 40.00 |
| Mean ± sample SD | 4.03 ± 0.01 | 12.52 ± 0.04 | 19.54 ± 0.05 | 58.53 ± 0.05 | 27.45 ± 0.09 | 10.57 ± 0.01 | 27.16 ± 0.01 | 0.1362 ± 0.0000 | 40.00 ± 0.00 |
| Minimum to maximum | 4.02 to 4.04 | 12.48 to 12.56 | 19.49 to 19.58 | 58.47 to 58.56 | 27.37 to 27.54 | 10.56 to 10.58 | 27.15 to 27.16 | 0.1362 to 0.1363 | 40.00 to 40.00 |

### image_text_to_image

| Model / seed | R@1 | R@5 | R@10 | R@100 | Hit@10 | mAP | nDCG | MRR | median_first_relevant_rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pretrained | 4.41 | 13.49 | 20.44 | 59.42 | 29.00 | 11.30 | 27.93 | 0.1481 | 34.00 |
| Seed 42 (released) | 7.13 | 20.84 | 30.87 | 74.70 | 42.00 | 16.78 | 34.11 | 0.2144 | 16.00 |
| Seed 43 | 7.19 | 20.63 | 30.89 | 74.48 | 42.11 | 16.79 | 34.11 | 0.2143 | 16.00 |
| Seed 44 | 7.18 | 20.85 | 30.94 | 74.62 | 42.17 | 16.83 | 34.17 | 0.2152 | 16.00 |
| Mean ± sample SD | 7.17 ± 0.03 | 20.77 ± 0.12 | 30.90 ± 0.03 | 74.60 ± 0.11 | 42.09 ± 0.09 | 16.80 ± 0.03 | 34.13 ± 0.03 | 0.2146 ± 0.0005 | 16.00 ± 0.00 |
| Minimum to maximum | 7.13 to 7.19 | 20.63 to 20.85 | 30.87 to 30.94 | 74.48 to 74.70 | 42.00 to 42.17 | 16.78 to 16.83 | 34.11 to 34.17 | 0.2143 to 0.2152 | 16.00 to 16.00 |

## Paired differences from pretrained

Fine-tuned minus pretrained, in percentage points, with 95% intervals from 5,000 query-weighted paired bootstrap resamples of article groups (seed 1406745469314695659). Across-seed summaries describe the point differences.

| Configuration | Split | Seed | Recall@10 | Hit@10 |
|---|---|---|---:|---:|
| image_to_image | test | 42 | +5.53 [+4.61, +6.55] | +6.84 [+5.73, +8.09] |
| image_to_image | test | 43 | +5.53 [+4.62, +6.55] | +6.97 [+5.84, +8.18] |
| image_to_image | test | 44 | +5.41 [+4.52, +6.41] | +6.90 [+5.80, +8.13] |
| image_to_image | test | Mean ± sample SD | 5.49 ± 0.07 | 6.90 ± 0.06 |
| image_to_image | test | Minimum to maximum | 5.41 to 5.53 | 6.84 to 6.97 |
| image_to_image | val | 42 | +5.05 [+4.18, +6.01] | +6.43 [+5.28, +7.58] |
| image_to_image | val | 43 | +4.99 [+4.12, +5.90] | +6.38 [+5.25, +7.48] |
| image_to_image | val | 44 | +5.08 [+4.19, +6.03] | +6.55 [+5.36, +7.72] |
| image_to_image | val | Mean ± sample SD | 5.04 ± 0.05 | 6.45 ± 0.09 |
| image_to_image | val | Minimum to maximum | 4.99 to 5.08 | 6.38 to 6.55 |
| image_text_to_image | test | 42 | +11.09 [+9.89, +12.38] | +13.30 [+12.07, +14.66] |
| image_text_to_image | test | 43 | +11.02 [+9.85, +12.29] | +13.23 [+12.01, +14.55] |
| image_text_to_image | test | 44 | +11.01 [+9.85, +12.27] | +13.24 [+12.03, +14.57] |
| image_text_to_image | test | Mean ± sample SD | 11.04 ± 0.04 | 13.26 ± 0.04 |
| image_text_to_image | test | Minimum to maximum | 11.01 to 11.09 | 13.23 to 13.30 |
| image_text_to_image | val | 42 | +10.44 [+9.15, +11.86] | +13.00 [+11.51, +14.61] |
| image_text_to_image | val | 43 | +10.45 [+9.17, +11.85] | +13.11 [+11.63, +14.72] |
| image_text_to_image | val | 44 | +10.50 [+9.22, +11.89] | +13.17 [+11.71, +14.73] |
| image_text_to_image | val | Mean ± sample SD | 10.46 ± 0.03 | 13.09 ± 0.09 |
| image_text_to_image | val | Minimum to maximum | 10.44 to 10.50 | 13.00 to 13.17 |

## Verification

Seed 42: 232 values compared with `results/results.json`, 0 mismatches.

Fresh required checks: 268 passed, 0 failed. Informational checks: 6 passed, 0 did not. Full observed and expected values are in `multiseed.json`.
