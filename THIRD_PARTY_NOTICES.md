# Third-party notices

This file lists third-party material that this repository holds or builds on, and the terms that its sources state. It
sets no terms for any other file. [`LICENSING.md`](LICENSING.md) gives the terms for each path of this repository.

## BiomedCLIP and BiomedBERT

The two weight files, `weights/image_to_image.safetensors` and `weights/image_text_to_image.safetensors`, are fine-tuned
from BiomedCLIP, `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` at revision
`9f341de24bfb00180f1b847274256e9b65a3a32e`. The first holds the fine-tuned vision tower, and the second the fine-tuned
full model. They are not stored in Git (see "Released weights" in `README.md`). The text configuration and tokenizer
that `pmc_pair_retrieval.model` downloads come from `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` at revision
`d673b8835373c6fa116d6d8006b33d48734e305d`. The files of these two model repositories are not in this repository.

This section quotes what Microsoft states for its pretrained models. It does not set terms for the fine-tuned weight
files, whose terms are not yet set (see `LICENSING.md`).

- BiomedCLIP model card:
  <https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/9f341de24bfb00180f1b847274256e9b65a3a32e/README.md>
- BiomedCLIP licence file:
  <https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/9f341de24bfb00180f1b847274256e9b65a3a32e/LICENSE.md>
- BiomedBERT model card:
  <https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract/blob/d673b8835373c6fa116d6d8006b33d48734e305d/README.md>
- BiomedBERT licence file:
  <https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract/blob/d673b8835373c6fa116d6d8006b33d48734e305d/LICENSE.md>

Both licence files hold this text, apart from line breaks:

```text
MIT License

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
```

The BiomedCLIP model card states, under "Intended Use":

> This model is intended to be used solely for (I) future research on visual-language processing and (II)
> reproducibility of the experimental results reported in the reference paper.

under "Out-of-Scope Use":

> **Any** deployed use case of the model --- commercial or otherwise --- is currently out of scope. Although we
> evaluated the models using a broad set of publicly-available research benchmarks, the models and evaluations are not
> intended for deployed use cases. Please refer to [the associated paper](https://aka.ms/biomedclip-paper) for more
> details.

and under "Limitations":

> This model was developed using English corpora, and thus can be considered English-only.

References given by the model cards:

- BiomedCLIP: Sheng Zhang et al. A Multimodal Biomedical Foundation Model Trained from Fifteen Million Image–Text Pairs.
  *NEJM AI* 2(1). <https://doi.org/10.1056/AIoa2400640>
- BiomedBERT: Yu Gu et al. Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing.
  arXiv:2007.15779. <https://arxiv.org/abs/2007.15779>

## Images in the example figures

The six `results/figures/retrieval_example_*.pdf` files reproduce 55 images from 53 articles in PubMed Central, the
full-text archive of the U.S. National Library of Medicine. Each image keeps the licence given for it in
`results/figures/source_data/image_credits.csv`, which also gives its attribution and the article's copyright notice:

| Licence | Images | Articles |
|---|---:|---:|
| [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | 35 | 34 |
| [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | 20 | 19 |

The licences are the ones recorded in `licenses.csv` of the PMC-PaIR dataset. For each of these images, that file also
records: "Automatic caption/XML screening and article metadata extraction; no individual visual or legal clearance is
implied."

The figures show each image whole, in a framed box, with labels and tags outside the image. `image_credits.csv` records
for each placement whether the image was downscaled (`downscaled`) and whether its colours were converted to sRGB
(`converted_to_srgb`). The legends in `results/figures/captions.tex` and the entries of
`results/figures/image_references.tex` credit the same sources. The other two PDF files in `results/figures/` contain no
images from articles.

## Fonts

All eight PDF files in `results/figures/` embed subsets of Arial, version 2.82: Arial regular in all of them, and also
Arial bold in `retrieval_performance.pdf` and `retrieval_subgroups_hit10.pdf`. The embedded font data state:

> Typeface © The Monotype Corporation plc. Data © The Monotype Corporation plc/Type Solutions Inc. 1990-1992. All Rights
> Reserved

The font files themselves are not in this repository.

## Python packages

The packages in `requirements.txt` are not in this repository. They are installed separately, and each has its own
licence.
