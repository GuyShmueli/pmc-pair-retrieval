# Results

Retrieval on the article-disjoint PMC-PaIR split. Every image of a split is a query, and the images of the same split
form the gallery. The relevant images of a query are its direct pair partners; images linked to it only through a
chain of pairs are left out of its ranking, and the query itself ranks last. Images are encoded from 3 views at
224 px. image_to_image queries are images; image_text_to_image queries add, to the image similarity, the dot product
of the gallery image embedding with the mean of the unit embeddings of the query image's distinct descriptions
(README, "Query text").

Recall@k (R@k), Hit@k, mAP and nDCG are means over queries in percent, MRR is a fraction, and the median rank is the
median rank of the first relevant image. This file and `results.json`, which holds the unrounded values, are generated
by `python -m pmc_pair_retrieval.compare` from the per-query files that `results.json` lists.

## Test split (8,096 queries)

| Configuration | Model | R@1 | R@5 | R@10 | R@100 | Hit@10 | mAP | nDCG | MRR | Median rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_to_image | pretrained | 3.32 | 9.57 | 14.62 | 47.71 | 21.47 | 8.22 | 24.11 | 0.1105 | 65 |
| image_to_image | fine-tuned | 4.31 | 12.91 | 20.15 | 60.03 | 28.31 | 10.89 | 27.65 | 0.1434 | 37 |
| image_text_to_image | pretrained | 4.87 | 13.99 | 20.53 | 60.08 | 29.68 | 11.59 | 28.30 | 0.1541 | 34 |
| image_text_to_image | fine-tuned | 7.46 | 21.19 | 31.62 | 76.33 | 42.98 | 17.13 | 34.59 | 0.2196 | 15 |

## Validation split (8,245 queries)

| Configuration | Model | R@1 | R@5 | R@10 | R@100 | Hit@10 | mAP | nDCG | MRR | Median rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_to_image | pretrained | 3.24 | 9.73 | 14.50 | 47.35 | 20.99 | 8.32 | 24.06 | 0.1090 | 67 |
| image_to_image | fine-tuned | 4.02 | 12.56 | 19.55 | 58.56 | 27.42 | 10.58 | 27.16 | 0.1362 | 40 |
| image_text_to_image | pretrained | 4.41 | 13.49 | 20.44 | 59.42 | 29.00 | 11.30 | 27.93 | 0.1481 | 34 |
| image_text_to_image | fine-tuned | 7.13 | 20.84 | 30.87 | 74.70 | 42.00 | 16.78 | 34.11 | 0.2144 | 16 |

## Fine-tuned minus pretrained

Mean paired difference over the same queries. The 95% intervals come from a query-weighted paired bootstrap (5,000
resamples) over article groups, the units of the split (images linked by a shared article or a pair), and, for
comparison, over articles. R@k and Hit@k differences are in percentage points, MRR differences are fractions.

### Test split

| Configuration | Metric | Fine-tuned minus pretrained | 95% CI, article groups | 95% CI, articles |
|---|---|---:|---:|---:|
| image_to_image | R@10 | +5.53 | [+4.61, +6.55] | [+4.80, +6.23] |
| image_to_image | Hit@10 | +6.84 | [+5.73, +8.09] | [+5.91, +7.75] |
| image_to_image | R@1 | +0.99 | [+0.59, +1.38] | [+0.63, +1.35] |
| image_to_image | R@100 | +12.31 | [+11.06, +13.66] | [+11.37, +13.23] |
| image_to_image | MRR | +0.0328 | [+0.0271, +0.0387] | [+0.0281, +0.0377] |
| image_text_to_image | R@10 | +11.09 | [+9.89, +12.38] | [+10.10, +12.09] |
| image_text_to_image | Hit@10 | +13.30 | [+12.07, +14.66] | [+12.12, +14.47] |
| image_text_to_image | R@1 | +2.59 | [+1.90, +3.37] | [+2.02, +3.15] |
| image_text_to_image | R@100 | +16.25 | [+15.05, +17.49] | [+15.27, +17.27] |
| image_text_to_image | MRR | +0.0654 | [+0.0568, +0.0747] | [+0.0584, +0.0727] |

### Validation split

| Configuration | Metric | Fine-tuned minus pretrained | 95% CI, article groups | 95% CI, articles |
|---|---|---:|---:|---:|
| image_to_image | R@10 | +5.05 | [+4.18, +6.01] | [+4.39, +5.73] |
| image_to_image | Hit@10 | +6.43 | [+5.28, +7.58] | [+5.56, +7.29] |
| image_to_image | R@1 | +0.78 | [+0.39, +1.18] | [+0.42, +1.14] |
| image_to_image | R@100 | +11.21 | [+10.12, +12.38] | [+10.27, +12.12] |
| image_to_image | MRR | +0.0272 | [+0.0213, +0.0332] | [+0.0226, +0.0316] |
| image_text_to_image | R@10 | +10.44 | [+9.15, +11.86] | [+9.56, +11.34] |
| image_text_to_image | Hit@10 | +13.00 | [+11.51, +14.61] | [+11.87, +14.14] |
| image_text_to_image | R@1 | +2.73 | [+1.93, +3.65] | [+2.17, +3.29] |
| image_text_to_image | R@100 | +15.28 | [+14.02, +16.53] | [+14.30, +16.26] |
| image_text_to_image | MRR | +0.0663 | [+0.0543, +0.0807] | [+0.0595, +0.0731] |

## Fine-tuned weights

The fine-tuned results of each configuration were computed with these weights. Each trainer keeps the epoch with the
highest validation Recall@10 of its own retrieval task (image queries for image_to_image, image-plus-description
queries for image_text_to_image); a later epoch must be strictly better. The last columns give the trainer's
validation R@10 at the selected epoch and the sha256 of the weights file.

| Configuration | Weights file | Contents | Selected epoch | Global step | Validation R@10 at selection (%) | File sha256 |
|---|---|---|---:|---:|---:|---|
| image_to_image | `weights/image_to_image.safetensors` | vision tower, 151 tensors | 11 | 6,061 | 19.55 | `cb24963bcb0ddba73a1d7e4a1a4539933b05ee9b8bf8d8f3454ca5c7c9949622` |
| image_text_to_image | `weights/image_text_to_image.safetensors` | full model, 351 tensors | 21 | 11,508 | 30.87 | `4d159e28b2ab1621e463bc3be8aab907356b83536d0ac9698620f32dabd54292` |
