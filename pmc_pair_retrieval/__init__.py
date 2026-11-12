"""BiomedCLIP image retrieval on PMC-PaIR.

Provides the article-disjoint data split, the fine-tuning of the image_to_image and
image_text_to_image configurations, the evaluation of pretrained and fine-tuned models
on the validation and test splits, the results tables with paired bootstrap confidence
intervals, and the evaluation and error-analysis figures. Further tools regenerate or
check the shipped pixel hashes and image sizes. Command-line tools are modules, for
example ``python -m pmc_pair_retrieval.train``, ``python -m pmc_pair_retrieval.evaluate``
and ``python -m pmc_pair_retrieval.compare``. ``multiseed`` regenerates the three-seed
reports from saved evaluation records, and ``run_multiseed`` repeats the additional
training runs and their evaluation.

Importing the package loads no third-party library, so that ``train`` can restart the
interpreter with the required hash seed before anything heavy is imported.
"""

__version__ = "1.1.0rc1"
