# GGSATD reconstruction

For the separate BiLSTM with attention reconstruction, see
[README_BiLSTM.md](../BiLSTM_Attention/README_BiLSTM.md).

This is an approximate reconstruction of Yu et al., *Exploiting gated graph neural
network for detecting and explaining self-admitted technical debts* (2022), based
on the [article](1-s2.0-S0164121222000036-main.pdf) in this directory. The authors' [Figshare record](https://figshare.com/articles/dataset/JSS-Graph/16869737)
currently contains no files. The original implementation, its exact preprocessing,
optimizer, training schedule, GloVe corpus, and all architecture details could not
be verified; this is **not an exact reproduction**.

The published parameters are 300-dimensional GloVe embeddings, a six-token
co-occurrence window, two gated graph layers, hidden size 96, learning rate 0.001,
batch size 32, dropout 0.5, and random OOV initialization in [-0.01, 0.01].
The model uses undirected, unweighted word graphs, soft node attention and
average/max pooling. The output layer is extended from binary to five classes
for the multiclass task.

Unspecified choices in this implementation include the tokenizer, trainable
embeddings and input projection, attention/feature MLPs, Adam optimizer,
cross-entropy loss, maximum of 30 epochs and early stopping after five epochs
without validation-loss improvement. A graph node represents a distinct word;
consecutive occurrences in a window create an edge between distinct nodes.
Each word absent from GloVe receives its own random vector. The vocabulary is
initialized from the training and validation partitions; test-only word vectors
are added after training and remain fixed during testing. Empty comments use a
special token. These choices may affect results.

For comparison with the other approaches, evaluation uses shuffled stratified
10-fold cross-validation over the whole selected dataset with seed 6666, not
the paper's within-project/cross-project evaluation. Within each fold, 10% of
the candidate training data is held out, stratified, for model selection. No
test labels or test-fold text are used for vocabulary initialization or training
before the final evaluation.
Multiclass filters and class names match the ML scripts; notably, O'Brien's
`Requirement` and unlabeled examples and Maldonado's unclassified examples
are excluded.

## Running

Install dependencies into a Python environment, and download a 300-dimensional
GloVe text file separately (for example `glove.6B.300d.txt` from the
[Stanford GloVe page](https://nlp.stanford.edu/projects/glove/)). The GloVe
corpus is not bundled or downloaded automatically. From the repository root:

```powershell
python -m pip install -r DL\requirements.txt
python DL\GGSATD\ggsatd.py --task binary --dataset maldonado --glove C:\path\to\glove.6B.300d.txt
python DL\GGSATD\ggsatd.py --task multi --dataset obrien --glove C:\path\to\glove.6B.300d.txt
```

Binary datasets are `maldonado`, `maldonado_undersampled`, `yu`, and
`yu_undersampled`; multiclass datasets are `maldonado` and `obrien`.
All paths default to the repository's `Dataset` directory and the corresponding
`DL\Binary` or `DL\Multi-Class` results directory, regardless of the current
working directory. Use `--dataset-dir` or `--output-dir` to override them.
`--device cuda` enables CUDA when available. `--epochs`, `--patience`, `--folds`,
and `--seed` are adjustable; changing them changes the experimental protocol.
Use `--no-codecarbon` only when emissions measurement is not needed.
CodeCarbon starts after argument and path validation but before dataset loading,
so dataset loading and tokenization are included with fold training and
evaluation in the measured run.

Each run saves `GGSATD_aggregated_metrics.csv` with the same model/class,
precision, recall, F1 and support summary columns as the ML scripts, plus
`GGSATD_fold_metrics.csv` and, by default, CodeCarbon's `emissions.csv`.
Existing result files in the chosen output directory are overwritten.
GloVe is read up to twice per fold; full experiments can be computationally expensive.

Run the focused checks with `python -m unittest discover -s DL\GGSATD -p test_ggsatd.py`.
