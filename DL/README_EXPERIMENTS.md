# Repeated DL experiments

`run_experiments.py` runs GGSATD, BiLSTM, CNN, FFNN and the adapted MT-BERT
sequentially in separate processes. It schedules complete experiments, not
individual folds or training epochs. Each repetition therefore runs the
selected experiment's full cross-validation; for example, 10 repetitions of
a 10-fold configuration run 100 outer folds per experiment. BiLSTM also
trains multiple candidates per fold. Runs may take considerable time.

The implementations, papers and tests are organized under
[GGSATD](GGSATD/README_GGSATD.md), [BiLSTM](BiLSTM_Attention/README_BiLSTM.md),
[CNN](CNN_SATD/README_CNN_SATD.md), and [FFNN](FF/README_FFNN.md). The adapted
[MT-BERT](2023-MT-BERT-SATD/README_PROJECT_ADAPTATION.md) remains in its
original directory. The other models reuse the dataset and seed helpers in
GGSATD; shared dependencies and GloVe remain in `DL`.

## Configuration and execution

Copy and edit [experiments.example.json](experiments.example.json), then run:

```powershell
Copy-Item DL\experiments.example.json DL\experiments.json
python DL\run_experiments.py DL\experiments.json
```

The example runs one full repetition of the same 17 experiments and model
settings as the workstation profile, using seed 6666 and repository-local
dataset and model paths. It covers the four dataset/task combinations for
GGSATD, BiLSTM, CNN and FFNN, plus one joint MT-BERT run over all four tasks.
Each run still uses 10-fold cross-validation, so this is not a quick smoke test.
The example uses CUDA, a fixed experiment order and a 60-second pause between
runs. Before running, verify CUDA, datasets, GloVe, MT-BERT model files, and
FFNN's NLTK resources are available. See the individual model READMEs for
dependencies and options. In particular, install NLTK's `stopwords` and
`wordnet` corpora separately as described in the FFNN README.

Use the results for exploratory parameter assessment, not an independent final
test after selecting parameters based on them. Energy and timing results from a
different computer are not directly comparable to workstation measurements.
The arguments in `args` are passed as separate strings to the corresponding
runner. For instance, add `"--folds", "5"` to one experiment's `args` to
change its folds, or remove `"--device", "cuda"` to use its CPU default.
MT-BERT's `--datasets` selects tasks trained jointly, not independent runs.

Each entry must have a unique `id` (letters, digits, `_` or `-`), a `model`
(`ggsatd`, `bilstm`, `cnn`, `ffnn`, or `mt-bert`), and an `args` list. Paths in
arguments and the configured `output_dir` are resolved from the repository
root, regardless of where the orchestrator is invoked. In JSON, write Windows
backslashes as `\\`. Do not pass `--seed` or `--output-dir` in `args`: the
orchestrator sets them for every run. Model-specific argument validation
occurs in the individual runner when that run starts.

The remaining configuration fields are:

| Field | Default | Meaning |
| --- | --- | --- |
| `repetitions` | `10` | Number of complete runs of each experiment. |
| `delay_seconds` | `0` | Nonnegative pause between consecutive runs, not between folds. |
| `randomize_order` | `true` | Shuffle experiments within each repetition; when `false`, run all repetitions of each experiment consecutively in configuration order. |
| `order_seed` | `6666` | Seed for reproducible scheduling; independent of model seeds. |
| `model_seed` | `6666` | Seed passed to every model in the first repetition. |
| `seed_mode` | `"fixed"` | `"fixed"` reuses the seed; `"increment"` adds the repetition index minus one. |
| `output_dir` | `DL\experiment_runs` | Parent directory for uniquely named experiment sessions. |

With `"fixed"`, runs use the same seed across repetitions to assess variation
under the same seed. With `"increment"`, all experiments in a repetition
receive the same new seed. A runner's seed controls both its splits and model
randomness, so these two effects cannot currently be varied independently.
Using the same seed does not guarantee identical splits across different
models; MT-BERT uses grouped splits.

## Ubuntu workstation (CS-WS-AI-2)

Docker is not required. Keep the Git checkout and experiment outputs on `/work`,
and place the dataset and large model resources on `/data`. The workstation
profile assumes the checkout is `/work/satd-empirical-comparison` and uses:

- `/data/satd-empirical-comparison/Dataset` for the CSV files from the
  repository's `Dataset` directory.
- `/data/satd-empirical-comparison/resources/glove.6B.300d.txt` for the
  300-dimensional GloVe file.
- `/data/satd-empirical-comparison/resources/bert-base-uncased` for
  `config.json`, `vocab.txt`, and `pytorch_model.bin`.

Create the directories and a project-specific environment, then install the
project dependencies:

```bash
cd /work/satd-empirical-comparison
mkdir -p /data/satd-empirical-comparison/Dataset
mkdir -p /data/satd-empirical-comparison/resources
conda env create -f DL/environment.yml
conda activate satd-experiments
python -m pip install -r DL/requirements.txt
```

Put the dataset files and the two model resources at the paths above. Verify
the NVIDIA driver and that the installed PyTorch can use the GPU:

```bash
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
```

If PyTorch reports that CUDA is unavailable, install the Linux/Pip/Python CUDA
build recommended by the official [PyTorch installation selector](https://pytorch.org/get-started/locally/),
then rerun the check. In VS Code, select the `satd-experiments` Conda
interpreter.

Copy [experiments.cs-ws-ai-2.example.json](experiments.cs-ws-ai-2.example.json)
to a working config if you need to change its paths or schedule, then launch it
from the repository root:

```bash
python DL/run_experiments.py DL/experiments.cs-ws-ai-2.example.json
```

This profile covers Maldonado and Yu for binary classification, and Maldonado
and O'Brien for multiclass classification, without undersampling. GGSATD,
BiLSTM, CNN and FFNN each run the four dataset/task combinations; MT-BERT runs
all four tasks jointly. That is 17 experiments per repetition, or 170 full runs
with the default 10 repetitions; at 10 folds per run this amounts to 1,700 outer
folds, and MT-BERT fine-tunes BERT across all selected tasks in each fold. It
can occupy the workstation for a very long time. MT-BERT's joint run is not
directly equivalent to the four models' independent dataset/task runs.

For CNN on multiclass Maldonado and O'Brien, the profile uses 100 CNN epochs
and 25 Word2Vec epochs instead of the defaults of 25 and 5. These settings are
exploratory: the separate epoch increases for O'Brien were assessed on test
folds, and the settings have not been independently validated for either dataset.
Select hyperparameters using validation data before reporting final test
comparisons.

The profile runs the 10 repetitions of each experiment consecutively in the
configured order, with a 60-second pause between runs. It uses
`seed_mode: "increment"`: repetitions use seeds 6666 through 6675, changing
both the splits and model randomness. For a smoke test, make a
separate config with one repetition and fewer folds; do not use those test
results in the paper comparison. Before collecting results, verify all
configured paths and record the workstation/GPU and software versions with the
experiment conditions.

## Measurement and outputs

Each runner controls CodeCarbon, which measures loading and preprocessing,
training and evaluation. The orchestrator's pause is outside CodeCarbon's
measured interval. Avoid `--no-codecarbon` when collecting energy results.
The pause is not a guarantee of thermal stability; use a dedicated machine
and document hardware, software and execution conditions.

Each invocation creates a new timestamped session under `output_dir`. Its
`manifest.json` records the effective configuration, run order, seeds,
commands, timestamps, status and exit codes. Each repetition/experiment
has its own directory containing the runner's CSV files, CodeCarbon output
and `run.log`; previous runs are never intentionally overwritten.
An experiment failure stops the schedule, records the failing exit code and
leaves subsequent experiments pending. Interruptions are recorded where
possible. There is no automatic resume: a new invocation creates a new session.
An interrupted or failed run can still leave partial files or emissions output;
only completed runs should be included in the analysis after verifying them.

Run the runner tests with
`python -m unittest discover -s DL -p test_run_experiments.py`.
