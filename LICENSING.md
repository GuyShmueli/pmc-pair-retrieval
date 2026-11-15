# Licensing

Different terms apply to different files of this repository. Every file belongs to one class below. The MIT license in
[`LICENSE`](LICENSE) applies to the first class only.

| Class | Paths | Terms |
|---|---|---|
| Code, documentation and configuration | `pmc_pair_retrieval/**`, `configs/**`, `README.md`, `LICENSE`, `LICENSING.md`, `CITATION.cff`, `SHA256SUMS`, `WEIGHTS_SHA256SUMS`, `requirements.txt`, `environment.yml`, `.gitattributes`, `.gitignore` | MIT license in `LICENSE` |
| Weights | `weights/**` | Not yet set |
| Data files | `data/**` | Not yet set |
| Results | `results/results.json`, `results/results.md`, `results/multiseed/**`, `results/per_query/**`, `results/rankings/**`, `results/figures/captions.tex`, `results/figures/figure_pages.tex`, `results/figures/manifest.json`, `results/figures/source_data/examples.csv`, `results/figures/source_data/overall_metrics.csv`, `results/figures/source_data/paired_differences.csv`, `results/figures/source_data/per_query.csv`, `results/figures/source_data/per_query_change.csv`, `results/figures/source_data/subgroups.csv` | Not yet set |
| Third-party material | `results/figures/*.pdf`, `results/figures/image_references.tex`, `results/figures/source_data/image_credits.csv`, `THIRD_PARTY_NOTICES.md` | The terms recorded for each item; see "Third-party material" |

A path that is not in this table is not covered by `LICENSE`.

## What `LICENSE` covers

`LICENSE` holds the standard MIT text. In this repository, its words "this software and associated documentation files"
mean the files of the first class and no others. It grants no license for the weights, the data files, the results or
the third-party material.

## Weights

`weights/image_to_image.safetensors` and `weights/image_text_to_image.safetensors` are the two weight files.
`weights/*.json` are their sidecars. The weight files are not stored in Git, so a clone does not contain them (see
"Released weights" in `README.md`).

The terms for these four files are not yet set. `LICENSE` grants no license for them, whether or not the weight files
are in your copy. The weight files are fine-tuned from BiomedCLIP. `THIRD_PARTY_NOTICES.md` quotes the license and the
statements on use that Microsoft gives for pretrained BiomedCLIP. These do not set terms for the fine-tuned files.

## Data files and results

The terms for `data/` and for the results are not yet set. `LICENSE` grants no license for them.

## Third-party material

The images and captions of the PMC-PaIR dataset are not in this repository. Each keeps the terms recorded for it in
`licenses.csv` of the dataset.

The six `results/figures/retrieval_example_*.pdf` files reproduce 55 of these images. Each image keeps the license
recorded for it in `results/figures/source_data/image_credits.csv`. All eight PDF files in `results/figures/` embed
subsets of Arial. In these PDF files, the terms for everything other than the reproduced images and the embedded fonts
are the terms for the results.

`image_credits.csv` and `image_references.tex` quote the copyright notices of the source articles. Quoted text stays with
its source, and the rest of these two files has the terms for the results. `THIRD_PARTY_NOTICES.md` quotes Microsoft's
license notice and model cards, which stay with Microsoft.

The files of the pretrained models and the Python packages in `requirements.txt` are not in this repository. They are
downloaded or installed separately, under their own terms.

## Quotations and copied values

Quoted text keeps the terms of its source wherever it appears, including the quotations from the BiomedCLIP model card
in `README.md`. The blocks of `README.md` between `BEGIN compare:` and `END compare:` markers copy values from the
results; those values have the terms for the results.
