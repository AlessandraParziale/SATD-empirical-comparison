# PILOT-inspired feed-forward classifier

This is a reconstruction of Di Salle et al., *PILOT: Synergy between Text
Processing and Neural Networks to Detect Self-Admitted Technical Debt*
(TechDebt 2022), based on the [paper](3524843.3528093.pdf). The original
implementation and all training settings have not been verified, so it is
not an exact reproduction.

The reusable `NLPProcessor` applies lowercasing and Unicode punctuation
replacement with spaces, NLTK word-punctuation tokenization, NLTK English
stopword removal, Porter stemming, then WordNet lemmatization with its default
noun part of speech. These tokenizer, punctuation and part-of-speech choices
are reconstruction assumptions. The `--no-standardize`, `--no-tokenize`,
`--no-stopwords`, `--no-stemming` and `--no-lemmatize` flags disable individual
steps; without tokenization, whitespace still separates input units for the
remaining word-level steps.

TF-IDF uses `min_df=4`: a term must appear in at least four **documents**, not
four total occurrences. The paper describes a four-occurrence cutoff; document
frequency is the explicit approximation used here. Both vocabulary and IDF
are fitted only on each fold's training subset, not validation or test text.
Consequently vocabulary sizes differ by fold and need not match the paper.
The `feature_factory` interface in `run_cv` permits another sparse extractor
with `fit_transform` and `transform` methods. No Word2Vec/GloVe extractor is
implemented.

The FFNN has 10 units per hidden layer and two or five output units. Choose
`--hidden-layers 1`, `2`, or `3` for C1, C2, or C3. Defaults are ReLU,
softmax/categorical cross-entropy, Adam at 0.001, batch size 32, at most 50
epochs, and early stopping after five epochs without validation-loss
improvement. Cross-entropy is calculated from logits with PyTorch's numerically
stable softmax; predicted classes use argmax. Other configurable options are
`--activation`, `--output-activation`, `--loss`, `--optimizer`,
`--learning-rate`, `--batch-size`, `--epochs`, `--patience` and `--min-df`;
cross-entropy requires softmax. MSE uses one-hot targets and the selected
softmax or sigmoid output. Alternative settings are experiments, not verified
paper parameters.

Evaluation uses shuffled stratified 10-fold CV with seed 6666. For early
stopping, a stratified 10% of each outer training portion is reserved for
validation (`--validation-fraction`); only the remaining subset trains the
network and fits TF-IDF. The best validation-loss weights are evaluated once
on the untouched test fold. This validation split and the optimizer settings
are explicit reconstruction choices. Duplicate comments are retained as in
the other DL models and can therefore occur across splits. Dataset mappings
and class filters are shared with GGSATD.

## Setup and execution

Install [shared dependencies](../requirements.txt), then download NLTK's
`stopwords` and `wordnet` corpora once. For the shared Ubuntu workstation,
use a project-specific directory:

```bash
cd /work/satd-empirical-comparison
conda activate satd-experiments
python -m pip install -r DL/requirements.txt
export NLTK_DATA=/work/satd-empirical-comparison/nltk_data
python -m nltk.downloader -d "$NLTK_DATA" stopwords wordnet
python DL/FF/ffnn.py --task binary --dataset maldonado \
  --dataset-dir /data/satd-empirical-comparison/Dataset \
  --hidden-layers 1 --device cuda
```

Set `NLTK_DATA` in each shell that runs experiments, or configure it in the
environment. Missing required corpora produce an explicit error. On other
systems, use `python -m nltk.downloader stopwords wordnet` to install to the
usual NLTK data directory. Binary datasets are `maldonado`,
`maldonado_undersampled`, `yu`, and `yu_undersampled`; multiclass datasets are
`maldonado` and `obrien` (`--task multi`).

Use `--dataset-dir` and `--output-dir` to override defaults. By default the
runner saves `FFNN_fold_metrics.csv`, `FFNN_aggregated_metrics.csv` and
CodeCarbon's `emissions.csv` in the relevant `DL/Binary` or `DL/Multi-Class`
dataset directory; metric files are overwritten. CodeCarbon measures NLP
setup, loading and preprocessing, feature extraction, training and evaluation;
use `--no-codecarbon` only when measurement is not needed.

The [experiment orchestrator](../README_EXPERIMENTS.md) example configurations
include FFNN C1. The workstation configuration uses this entry:

```json
{
  "id": "ffnn-c1-maldonado",
  "model": "ffnn",
  "args": ["--task", "binary", "--dataset", "maldonado", "--dataset-dir", "/data/satd-empirical-comparison/Dataset", "--hidden-layers", "1", "--device", "cuda"]
}
```

Use separate IDs and `--hidden-layers` values to compare C1, C2 and C3. The
orchestrator assigns seeds and output directories.

Run focused tests with:

```bash
python -m unittest discover -s DL/FF -p test_ffnn.py
```
