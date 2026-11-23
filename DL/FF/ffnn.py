"""PILOT-inspired feed-forward SATD classifier."""

import argparse
import copy
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from DL.GGSATD.ggsatd import DATASETS, ROOT, load_dataset, seed_everything


MODEL_NAME = "FFNN"


class NLPProcessor:
    def __init__(self, standardize=True, tokenize=True, remove_stopwords=True,
                 stemming=True, lemmatize=True):
        self.standardize = standardize
        self.tokenize = tokenize
        self.stemming = stemming
        self.lemmatize = lemmatize
        self.tokenizer = WordPunctTokenizer()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()
            if lemmatize:
                self.lemmatizer.lemmatize("tests")
        except LookupError as exc:
            raise RuntimeError(
                "Missing NLTK data; install it with: python -m nltk.downloader stopwords wordnet"
            ) from exc

    def __call__(self, text):
        if self.standardize:
            text = "".join(" " if unicodedata.category(char).startswith("P") else char
                           for char in text.lower())
        tokens = self.tokenizer.tokenize(text) if self.tokenize else text.split()
        tokens = [token for token in tokens if token not in self.stop_words]
        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return tokens


def identity(tokens):
    return tokens


def make_tfidf(min_df=4):
    return TfidfVectorizer(analyzer=identity, min_df=min_df, dtype=np.float32)


class FeedForward(nn.Module):
    def __init__(self, inputs, classes, hidden_layers=1, activation="relu"):
        super().__init__()
        activation_class = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[activation]
        layers = []
        width = inputs
        for _ in range(hidden_layers):
            layers.extend((nn.Linear(width, 10), activation_class()))
            width = 10
        layers.append(nn.Linear(width, classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)


def make_loader(labels, batch_size, shuffle, seed):
    indices = torch.arange(len(labels))
    dataset = TensorDataset(indices, torch.as_tensor(labels, dtype=torch.long))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def loss_for_logits(logits, labels, loss_name, output_activation):
    if loss_name == "cross-entropy":
        return nn.functional.cross_entropy(logits, labels)
    if output_activation == "softmax":
        probabilities = torch.softmax(logits, dim=1)
    else:
        probabilities = torch.sigmoid(logits)
    targets = nn.functional.one_hot(labels, num_classes=logits.shape[1]).to(probabilities.dtype)
    return nn.functional.mse_loss(probabilities, targets)


def evaluate(model, matrix, loader, device, loss_name, output_activation):
    model.eval()
    loss_sum, truth, predictions = 0.0, [], []
    with torch.no_grad():
        for indices, labels in loader:
            features = torch.from_numpy(matrix[indices.numpy()].toarray()).to(device)
            labels = labels.to(device)
            logits = model(features)
            loss_sum += loss_for_logits(logits, labels, loss_name, output_activation).item() * len(labels)
            truth.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return loss_sum / len(truth), truth, predictions


def run_cv(texts, labels, names, output_dir, folds=10, seed=6666, epochs=50,
           patience=5, validation_fraction=0.1, batch_size=32, hidden_layers=1,
           activation="relu", optimizer_name="adam", learning_rate=0.001,
           output_activation="softmax", loss_name="cross-entropy", device="cpu",
           feature_factory=make_tfidf):
    if loss_name == "cross-entropy" and output_activation != "softmax":
        raise ValueError("cross-entropy requires softmax output")
    counts = np.bincount(labels, minlength=len(names))
    if len(counts) != len(names) or np.any(counts == 0) or int(counts.min()) < folds:
        raise ValueError(f"Each class must have at least {folds} examples; counts: {counts.tolist()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (candidate, test_indices) in enumerate(splitter.split(texts, labels), 1):
        fold_seed = seed + fold
        seed_everything(fold_seed)
        train_indices, val_indices = train_test_split(
            candidate, test_size=validation_fraction, stratify=labels[candidate],
            random_state=fold_seed)
        extractor = feature_factory()
        train_matrix = extractor.fit_transform([texts[i] for i in train_indices])
        val_matrix = extractor.transform([texts[i] for i in val_indices])
        test_matrix = extractor.transform([texts[i] for i in test_indices])
        train_loader = make_loader(labels[train_indices], batch_size, True, fold_seed)
        val_loader = make_loader(labels[val_indices], batch_size, False, fold_seed)
        test_loader = make_loader(labels[test_indices], batch_size, False, fold_seed)
        model = FeedForward(train_matrix.shape[1], len(names), hidden_layers, activation).to(device)
        optimizer_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer_name]
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)
        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for indices, batch_labels in train_loader:
                features = torch.from_numpy(train_matrix[indices.numpy()].toarray()).to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                loss = loss_for_logits(model(features), batch_labels, loss_name, output_activation)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss in fold {fold}, epoch {epoch}")
                loss.backward()
                optimizer.step()
            val_loss, _, _ = evaluate(model, val_matrix, val_loader, device, loss_name, output_activation)
            print(f"Fold {fold}/{folds}, epoch {epoch}/{epochs}: "
                  f"validation loss {val_loss:.6f}", flush=True)
            if not np.isfinite(val_loss):
                raise RuntimeError(f"Non-finite validation loss in fold {fold}, epoch {epoch}")
            if val_loss < best_loss:
                best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        _, truth, predictions = evaluate(model, test_matrix, test_loader, device, loss_name, output_activation)
        report = classification_report(truth, predictions, labels=list(range(len(names))),
                                       target_names=names, output_dict=True, zero_division=0)
        for name in names + ["macro avg", "weighted avg"]:
            metrics = report[name]
            rows.append({"Fold": fold, "Model": MODEL_NAME, "Class": name,
                         "Precision": metrics["precision"], "Recall": metrics["recall"],
                         "F1": metrics["f1-score"], "Support": metrics["support"]})

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(output_dir / f"{MODEL_NAME}_fold_metrics.csv", index=False)
    aggregate = fold_df.groupby(["Model", "Class"], as_index=False).agg(
        Precision_Mean=("Precision", "mean"), Precision_Std=("Precision", "std"),
        Recall_Mean=("Recall", "mean"), Recall_Std=("Recall", "std"),
        F1_Mean=("F1", "mean"), F1_Std=("F1", "std"), Support_Mean=("Support", "mean"))
    aggregate.to_csv(output_dir / f"{MODEL_NAME}_aggregated_metrics.csv", index=False)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "multi"], required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-layers", type=int, default=1)
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--output-activation", choices=["softmax", "sigmoid"], default="softmax")
    parser.add_argument("--loss", choices=["cross-entropy", "mse"], default="cross-entropy")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--min-df", type=int, default=4)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--no-tokenize", action="store_true")
    parser.add_argument("--no-stopwords", action="store_true")
    parser.add_argument("--no-stemming", action="store_true")
    parser.add_argument("--no-lemmatize", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-codecarbon", action="store_true")
    args = parser.parse_args()
    if args.folds < 2 or min(args.epochs, args.patience, args.batch_size,
                             args.hidden_layers, args.min_df) < 1:
        parser.error("folds must be at least 2; epochs, patience, batch size, "
                     "hidden layers and min-df must be positive")
    if not 0 < args.validation_fraction < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("validation fraction must be in (0, 1) and learning rate must be finite and positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.loss == "cross-entropy" and args.output_activation != "softmax":
        parser.error("cross-entropy requires softmax output")
    output_dir = args.output_dir or (ROOT / "DL" / ("Binary" if args.task == "binary" else "Multi-Class")
                                     / DATASETS[args.dataset][1])
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = None
    if not args.no_codecarbon:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="ITA", project_name="DL",
                                          experiment_id=f"{args.task}_{MODEL_NAME}_{args.dataset}",
                                          output_dir=str(output_dir), on_csv_write="append")
        tracker.start()
    try:
        processor = NLPProcessor(not args.no_standardize, not args.no_tokenize, not args.no_stopwords,
                                 not args.no_stemming, not args.no_lemmatize)
        texts, labels, names = load_dataset(args.task, args.dataset, args.dataset_dir, tokenizer=processor)
        run_cv(texts, labels, names, output_dir, folds=args.folds, seed=args.seed,
               epochs=args.epochs, patience=args.patience, validation_fraction=args.validation_fraction,
               batch_size=args.batch_size, hidden_layers=args.hidden_layers, activation=args.activation,
               optimizer_name=args.optimizer, learning_rate=args.learning_rate, device=args.device,
               output_activation=args.output_activation, loss_name=args.loss,
               feature_factory=lambda: make_tfidf(args.min_df))
    finally:
        if tracker is not None:
            tracker.stop()
    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main()
