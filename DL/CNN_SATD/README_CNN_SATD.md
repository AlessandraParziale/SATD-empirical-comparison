# CNN SATD classifier

This is a reconstruction of Ren et al., *Neural Network-based Detection of
Self-Admitted Technical Debt: From Performance to Explainability*, ACM TOSEM
28(3), 2019, [DOI: 10.1145/3324916](https://doi.org/10.1145/3324916).
It adapts the paper's binary CNN classification pipeline to the repository's
binary and five-class datasets. The [article](3324916-1.pdf) is in this directory.

## Paper settings reproduced

- 300-dimensional domain-trained continuous skip-gram Word2Vec embeddings.
- CNN filter widths 1, 2, 3, 4, 5, and 6; 128 filters per width.
- ReLU convolutions, one-max pooling, concatenated pooled features, and a
  linear softmax classifier.
- Weighted cross-entropy, where the class weight is its fraction in the
  training fold and its loss multiplier is `1 - class_fraction`.
- Adam with learning rate `0.0001`.
- The paper's evaluation is project-based; this implementation instead uses
  the repository's shuffled stratified 10-fold CV and seed 6666.

## Explicit reconstruction choices

The paper does not report Word2Vec's window, minimum count, negative-sampling
count or epochs. We use a symmetric skip-gram window of 5, `min_count=1`,
five negative samples per positive pair, five Word2Vec epochs and deterministic
seeded sampling. The implementation uses PyTorch skip-gram with negative
sampling, so it does not depend on Gensim. Embeddings are trained on the
training fold only and fine-tuned by the CNN; test-fold text never contributes
to the vocabulary or embeddings. Unknown test words map to a zero-initialized
trainable `<unk>` embedding.

The paper does not fully specify the text tokenizer or fixed input length.
We use the existing project's comment cleaning/tokenization. The paper
zero-pads comments to the maximum comment length, applies each filter to every
window in the resulting sequence, and uses 1-max pooling over each complete
feature map. Accordingly, each example here is zero-padded on both sides to
the selected dataset's maximum token length (at least the largest filter
width), and pooling includes windows that overlap padding. If the required
padding is odd, the extra zero is placed at the end. This makes sequence
length independent of batch composition while keeping tensors batch-local.
When explanations are saved, padded tokens are omitted from the displayed
phrase; all-padding winning windows are omitted.

The article specifies a learning rate and optimized CNN sizes but does not
state all final training settings. We use batch size 64 and dropout 0.5 from
the referenced Kim CNN implementation, with 25 epochs as an explicit runtime
reconstruction choice. The original paper's preprocessing removes exact
duplicate comments; duplicates are retained here to match the other repository
models and their input samples.

The original paper is binary. For multiclass, the same class-proportion
formula is applied to each class: each per-example cross-entropy is multiplied
by `1 - (training_count_of_class / total_training_count)`. This is a direct
extension of the published formula, not a result evaluated in the paper.

The paper also describes deconvolution-based key phrase and pattern extraction.
When `--save-explanations` is enabled, this implementation exports the
max-pooling n-grams with positive classifier-weight contrast for the predicted
class. This is a useful, transparent max-pooling attribution approximation,
not the paper's full deconvolution algorithm or global pattern-identification
analysis.

## Running

Install dependencies from the repository root:

```powershell
python -m pip install -r DL\requirements.txt
python DL\CNN_SATD\cnn_satd.py --task binary --dataset maldonado
python DL\CNN_SATD\cnn_satd.py --task multi --dataset obrien --save-explanations
```

Binary datasets are `maldonado`, `maldonado_undersampled`, `yu`, and
`yu_undersampled`; multiclass datasets are `maldonado` and `obrien`.
`--dataset-dir` and `--output-dir` override default paths. `--device cuda`
requires an available CUDA-enabled PyTorch. `--no-codecarbon` disables emission
measurement. `--epochs`, `--embedding-epochs`, `--embedding-window`,
`--filters`, `--batch-size` and `--dropout` are exposed for controlled tests;
changing paper-reported values should be documented.

Results are saved beneath the corresponding `DL\Binary` or `DL\Multi-Class`
dataset directory. `CNN_SATD_fold_metrics.csv` and
`CNN_SATD_aggregated_metrics.csv` use the project's standard metric columns.
`--save-explanations` additionally writes `CNN_SATD_explanations.csv`.
CodeCarbon starts after argument validation but before dataset loading, so
loading and tokenization are included with fold training and evaluation in
the measured run. It appends `emissions.csv` by default.

Run focused tests with `python -m unittest discover -s DL\CNN_SATD -p test_cnn_satd.py`.
