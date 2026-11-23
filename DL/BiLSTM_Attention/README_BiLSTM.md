# BiLSTM with attention

This is a reconstruction of Yu et al., *Using BiLSTM with attention mechanism
to automatically detect self-admitted technical debt* (2021),
[DOI: 10.1007/s11704-020-9281-z](https://doi.org/10.1007/s11704-020-9281-z).
The [article](s11704-020-9281-z.pdf) is in this directory.
It supports this repository's binary and multiclass datasets and reports the
same metrics as the other approaches.

## What follows the article

- Train 300-dimensional GloVe embeddings on the training comments; fine-tune
  embeddings during classifier training. The optional `--glove` uses a general
  pretrained file instead, as in the paper's embedding comparison.
- BiLSTM, learned context-vector attention and a sigmoid binary output.
- Binary loss `-alpha*y*log(p) - (1-alpha)*(1-y)*log(1-p)`, with `alpha`
  searched in [0.5, 1] using a genetic algorithm. Its fitness is validation
  F1 for SATD. Population size is 6, crossover probability 0.6, and mutation
  probability 0.2.
- Hidden size 256, batch size 64, Adam learning rate 0.0001, L2 0.01,
  dropout 0.4. The hidden size is interpreted as **per direction**.
- Lowercase, remove URLs, date-prefixed revision lines, markup, numbers and
  punctuation, filter stopwords and words outside 3–20 letters, retaining
  `todo`. Empty comments use a special token.

## Adaptations and assumptions

The original source code and exact stopword list are not specified in the
article. We use scikit-learn's English stopwords except `todo`; source-code
fragments are not generically removed because the paper does not define how to
identify them. Duplicate comments are retained to match the samples evaluated
by the other project models. Consequently, identical comments can occur across
folds; this is a project-comparison protocol, not the paper's preprocessing.

The paper does not specify the GloVe training settings. This reconstruction
uses a symmetric window of 10, inverse-distance counts, `x_max=100`, exponent
0.75, AdaGrad learning rate 0.05 and 25 epochs. Vocabulary is fitted only on
the current fold's training subset, and unseen words map to `<unk>`. Vocabulary,
co-occurrence counts and embeddings never use validation or test text. General
pretrained embeddings are an external resource, also indexed by training
vocabulary only.

The paper does not specify the chromosome size, selection algorithm, number of
generations or stopping rule. We use 16-bit binary chromosomes, rank-weighted
parent selection, single-point crossover, one-bit mutation per selected
individual, elitism, five generations, and cache repeated individuals.
Each distinct candidate is trained independently from the same fold seed;
the candidate checkpoint with the best validation F1 is selected. Training
uses at most 30 epochs and stops after five epochs without validation F1
improvement. L2 is implemented as Adam's weight decay. These are explicit
reconstruction choices, not verified original settings.

For multiclass, the binary formula has no literal equivalent. One genetic
factor interpolates between equal class weights (`alpha=0.5`) and normalized
inverse training-class frequencies (`alpha=1`); the fitness is macro F1.
The output uses softmax/cross-entropy for the five classes. Thus this is a
multiclass extension, not a result reported by the paper.

As in GGSATD and the ML scripts, evaluation uses the full selected dataset
with shuffled stratified 10-fold CV and seed 6666. Each candidate training
fold reserves 10% for stratified validation. This differs from the paper's
leave-one-project-out evaluation and 10 repetitions per project. Multiclass
filters and labels are identical to the existing ML experiments. Model
selection and GloVe training never use test-fold labels or text.

## Running

Install the shared dependencies in [requirements.txt](../requirements.txt).
From the repository root, domain-trained GloVe is the default:

```powershell
python DL\BiLSTM_Attention\bilstm_attention.py --task binary --dataset maldonado
python DL\BiLSTM_Attention\bilstm_attention.py --task multi --dataset obrien
```

Binary datasets are `maldonado`, `maldonado_undersampled`, `yu`, and
`yu_undersampled`; multiclass datasets are `maldonado` and `obrien`.
To use the previously downloaded general GloVe instead:

```powershell
python DL\BiLSTM_Attention\bilstm_attention.py --task binary --dataset maldonado --glove C:\path\to\glove.6B.300d.txt
```

`--dataset-dir` and `--output-dir` override the default paths. `--device cuda`
enables CUDA; `--no-codecarbon` disables emissions measurement. The
`--folds`, `--seed`, `--epochs`, `--patience`, `--ga-generations` and
`--glove-epochs` options can reduce runtime for testing or change the
experiment, but changes must be reported. Full genetic searches can require
substantial time and memory, especially for GloVe co-occurrences.
CodeCarbon starts after argument and path validation but before dataset
loading, so loading and tokenization are included with fold training and
evaluation in the measured run.

Results are placed in the corresponding `DL\Binary` or `DL\Multi-Class`
dataset directory. `BiLSTM_Attention_aggregated_metrics.csv` and
`BiLSTM_Attention_fold_metrics.csv` match the project's metric format;
`BiLSTM_Attention_genetic_search.csv` and
`BiLSTM_Attention_selected_alpha.csv` record validation choices. CodeCarbon
appends to `emissions.csv` by default. Result files are overwritten.

Run focused tests with `python -m unittest discover -s DL\BiLSTM_Attention -p test_bilstm_attention.py`.
