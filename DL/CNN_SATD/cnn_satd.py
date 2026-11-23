"""Reconstruction of Ren et al.'s CNN for SATD classification."""

import argparse
import sys
from array import array
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from DL.GGSATD.ggsatd import DATASETS, ROOT, load_dataset, seed_everything


MODEL_NAME = "CNN_SATD"
PAD, UNK = "<pad>", "<unk>"
FILTER_SIZES = (1, 2, 3, 4, 5, 6)


def train_word2vec(texts, seed, dimension=300, window=5, epochs=5, negatives=5,
                   batch_size=8192, device="cpu"):
    frequencies = Counter(word for text in texts for word in text)
    vocabulary = sorted(frequencies)
    if not vocabulary:
        raise ValueError("Cannot train Word2Vec on an empty training corpus")
    word_ids = {word: index + 2 for index, word in enumerate(vocabulary)}
    word_ids.update({PAD: 0, UNK: 1})
    centers, contexts = array("i"), array("i")
    for text in texts:
        ids = [word_ids[word] for word in text]
        for i, center in enumerate(ids):
            for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                if i != j:
                    centers.append(center)
                    contexts.append(ids[j])
    if not centers:
        raise ValueError("Word2Vec needs at least one context pair in the training corpus")

    seed_everything(seed)
    input_vectors = nn.Embedding(len(word_ids), dimension, device=device, sparse=True)
    output_vectors = nn.Embedding(len(word_ids), dimension, device=device, sparse=True)
    nn.init.uniform_(input_vectors.weight, -0.5 / dimension, 0.5 / dimension)
    nn.init.zeros_(output_vectors.weight)
    optimizer = torch.optim.SparseAdam(
        list(input_vectors.parameters()) + list(output_vectors.parameters()), lr=0.025)
    center_ids = torch.from_numpy(np.frombuffer(centers, dtype=np.int32).copy().astype(np.int64))
    context_ids = torch.from_numpy(np.frombuffer(contexts, dtype=np.int32).copy().astype(np.int64))
    noise = torch.tensor([frequencies[word] ** 0.75 for word in vocabulary], dtype=torch.float32)
    noise = noise / noise.sum()
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(centers), generator=generator)
        total_loss = 0.0
        for indices in order.split(batch_size):
            centers_batch = center_ids[indices].to(device)
            contexts_batch = context_ids[indices].to(device)
            negative_ids = torch.multinomial(
                noise, len(indices) * negatives, replacement=True, generator=generator
            ).reshape(len(indices), negatives).to(device) + 2
            center_vectors = input_vectors(centers_batch)
            positive_scores = (center_vectors * output_vectors(contexts_batch)).sum(dim=1)
            negative_scores = torch.bmm(
                output_vectors(negative_ids), center_vectors.unsqueeze(2)
            ).squeeze(2)
            loss = (F.softplus(-positive_scores) + F.softplus(negative_scores).sum(dim=1)).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite Word2Vec loss in epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(f"Word2Vec epoch {epoch}/{epochs}: loss {total_loss / len(centers):.6f}", flush=True)
    embeddings = torch.zeros((len(word_ids), dimension), dtype=torch.float32)
    embeddings[2:] = input_vectors.weight[2:].detach().cpu()
    return word_ids, embeddings


class CommentDataset(Dataset):
    def __init__(self, texts, labels, indices, word_ids):
        self.samples = [
            (torch.tensor([word_ids.get(word, word_ids[UNK]) for word in texts[index]],
                          dtype=torch.long), int(labels[index]), int(index))
            for index in indices
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_comments(batch, max_length=None):
    sequences, labels, sample_ids = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded_length = max(max(lengths.tolist()), max(FILTER_SIZES), max_length or 0)
    padded_sequences = []
    for sequence in sequences:
        padding = padded_length - len(sequence)
        left_padding = padding // 2
        right_padding = padding - left_padding
        padded_sequences.append(F.pad(sequence, (left_padding, right_padding), value=0))
    tokens = torch.stack(padded_sequences)
    return tokens, torch.tensor(labels, dtype=torch.long), torch.tensor(sample_ids)


def make_loader(texts, labels, indices, word_ids, batch_size, shuffle, seed, max_length):
    dataset = CommentDataset(texts, labels, indices, word_ids)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=partial(collate_comments, max_length=max_length),
                      generator=torch.Generator().manual_seed(seed))


class SATDCNN(nn.Module):
    def __init__(self, embeddings, num_classes, filters=128, dropout=0.5,
                 filter_sizes=FILTER_SIZES, trainable_embeddings=True):
        super().__init__()
        self.filter_sizes = tuple(filter_sizes)
        self.filters_per_size = filters
        self.embedding = nn.Embedding.from_pretrained(
            embeddings, freeze=not trainable_embeddings, padding_idx=0)
        self.convolutions = nn.ModuleList([
            nn.Conv1d(embeddings.shape[1], filters, kernel_size=size)
            for size in self.filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(filters * len(self.filter_sizes), num_classes)

    def forward(self, tokens, return_features=False):
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled_features, winning_positions = [], []
        for convolution in self.convolutions:
            activations = F.relu(convolution(embedded))
            values, indices = activations.max(dim=2)
            pooled_features.append(values)
            winning_positions.append(indices)
        features = torch.cat(pooled_features, dim=1)
        logits = self.classifier(self.dropout(features))
        if return_features:
            return logits, features, winning_positions
        return logits


def weighted_cross_entropy(logits, labels, class_counts):
    proportions = class_counts / class_counts.sum()
    class_weights = 1.0 - proportions
    losses = F.cross_entropy(logits, labels, reduction="none")
    return (losses * class_weights[labels]).mean()


def explain_batch(model, sample_ids, texts, labels, predictions,
                  features, winning_positions, names, padding_offsets=None, top_k=10):
    explanations = []
    feature_index = 0
    for size, positions in zip(model.filter_sizes, winning_positions):
        for sample_index, starts in enumerate(positions):
            predicted_class = int(predictions[sample_index])
            for filter_index in range(model.filters_per_size):
                feature = feature_index + filter_index
                activation = float(features[sample_index, feature].item())
                if len(names) == 2:
                    other_class = 1 - predicted_class
                    contribution = float(
                        (model.classifier.weight[predicted_class, feature]
                         - model.classifier.weight[other_class, feature]).item() * activation)
                else:
                    others = [index for index in range(len(names)) if index != predicted_class]
                    contrast = model.classifier.weight[predicted_class, feature] - \
                        model.classifier.weight[others, feature].mean()
                    contribution = float(contrast.item() * activation)
                if contribution <= 0:
                    continue
                start_index = int(starts[filter_index].item())
                offset = 0 if padding_offsets is None else int(padding_offsets[sample_index])
                first_word = min(max(start_index - offset, 0), len(texts[sample_index]))
                last_word = max(0, min(start_index + size - offset, len(texts[sample_index])))
                phrase = " ".join(texts[sample_index][first_word:last_word])
                if phrase:
                    explanations.append({
                        "Sample_Index": int(sample_ids[sample_index]),
                        "True_Label": names[int(labels[sample_index])],
                        "Predicted_Label": names[predicted_class],
                        "Filter_Size": size,
                        "Filter": filter_index,
                        "Phrase": phrase,
                        "Activation": activation,
                        "Class_Contribution": contribution,
                    })
        feature_index += model.filters_per_size
    explanations.sort(key=lambda item: (item["Sample_Index"], -item["Class_Contribution"]))
    selected, per_sample = [], {}
    for item in explanations:
        sample_id = item["Sample_Index"]
        if per_sample.get(sample_id, 0) < top_k:
            selected.append(item)
            per_sample[sample_id] = per_sample.get(sample_id, 0) + 1
    return selected


def evaluate(model, loader, device, names, texts, keep_explanations=False):
    model.eval()
    truth, predictions, explanations = [], [], []
    with torch.no_grad():
        for tokens, labels, sample_ids in loader:
            tokens = tokens.to(device)
            features, positions = None, None
            if keep_explanations:
                logits, features, positions = model(tokens, return_features=True)
            else:
                logits = model(tokens)
            predicted = logits.argmax(dim=1)
            truth.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
            if keep_explanations:
                if features is None or positions is None:
                    raise RuntimeError("CNN feature maps were not returned for explanation")
                padding_offsets = (tokens == 0).long().cumprod(dim=1).sum(dim=1)
                explanations.extend(explain_batch(
                    model, sample_ids, [texts[index] for index in sample_ids.tolist()],
                    labels, predicted.cpu().tolist(), features.cpu(), positions, names,
                    padding_offsets=padding_offsets.cpu().tolist()))
    return truth, predictions, explanations


def run_cv(texts, labels, names, output_dir, folds=10, seed=6666, epochs=25,
           batch_size=64, embedding_epochs=5, embedding_window=5, filters=128,
           dropout=0.5, device="cpu", save_explanations=False):
    counts = np.bincount(labels, minlength=len(names))
    if np.any(counts == 0) or int(counts.min()) < folds:
        raise ValueError(f"Each class must have at least {folds} examples; counts: {counts.tolist()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, explanation_rows = [], []
    max_length = max(max(len(text) for text in texts), max(FILTER_SIZES))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_indices, test_indices) in enumerate(splitter.split(texts, labels), 1):
        fold_seed = seed + fold
        seed_everything(fold_seed)
        training_texts = [texts[index] for index in train_indices]
        word_ids, embeddings = train_word2vec(
            training_texts, fold_seed, window=embedding_window, epochs=embedding_epochs, device=device)
        train_loader = make_loader(
            texts, labels, train_indices, word_ids, batch_size, True, fold_seed,
            max_length=max_length)
        test_loader = make_loader(
            texts, labels, test_indices, word_ids, batch_size, False, fold_seed,
            max_length=max_length)
        model = SATDCNN(embeddings, len(names), filters=filters, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        class_counts = torch.tensor(
            np.bincount(labels[train_indices], minlength=len(names)),
            dtype=torch.float32, device=device)
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss, total_samples = 0.0, 0
            for tokens, batch_labels, _ in train_loader:
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(tokens.to(device))
                loss = weighted_cross_entropy(logits, batch_labels, class_counts)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite CNN loss in fold {fold}, epoch {epoch}")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_labels)
                total_samples += len(batch_labels)
            print(f"Fold {fold}/{folds}, epoch {epoch}/{epochs}: "
                  f"training loss {total_loss / total_samples:.6f}", flush=True)
        truth, predictions, explanations = evaluate(
            model, test_loader, device, names, texts, keep_explanations=save_explanations)
        report = classification_report(truth, predictions, labels=list(range(len(names))),
                                       target_names=names, output_dict=True, zero_division=0)
        if not isinstance(report, dict):
            raise TypeError("Expected a classification report dictionary")
        for name in names + ["macro avg", "weighted avg"]:
            metrics = report[name]
            if not isinstance(metrics, dict):
                raise TypeError(f"Expected classification metrics for {name}")
            rows.append({"Fold": fold, "Model": MODEL_NAME, "Class": name,
                         "Precision": metrics["precision"], "Recall": metrics["recall"],
                         "F1": metrics["f1-score"], "Support": metrics["support"]})
        explanation_rows.extend({"Fold": fold, **item} for item in explanations)

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(output_dir / f"{MODEL_NAME}_fold_metrics.csv", index=False)
    aggregate = fold_df.groupby(["Model", "Class"], as_index=False).agg(
        Precision_Mean=("Precision", "mean"), Precision_Std=("Precision", "std"),
        Recall_Mean=("Recall", "mean"), Recall_Std=("Recall", "std"),
        F1_Mean=("F1", "mean"), F1_Std=("F1", "std"), Support_Mean=("Support", "mean"))
    aggregate.to_csv(output_dir / f"{MODEL_NAME}_aggregated_metrics.csv", index=False)
    if save_explanations:
        pd.DataFrame(explanation_rows, columns=[
            "Fold", "Sample_Index", "True_Label", "Predicted_Label", "Filter_Size",
            "Filter", "Phrase", "Activation", "Class_Contribution",
        ]).to_csv(output_dir / f"{MODEL_NAME}_explanations.csv", index=False)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "multi"], required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--embedding-epochs", type=int, default=5)
    parser.add_argument("--embedding-window", type=int, default=5)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save-explanations", action="store_true")
    parser.add_argument("--no-codecarbon", action="store_true")
    args = parser.parse_args()
    if args.folds < 2 or min(args.epochs, args.embedding_epochs, args.embedding_window,
                             args.filters, args.batch_size) < 1:
        parser.error("folds must be at least 2; epoch, window, filter and batch values must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("dropout must be in [0, 1)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
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
        texts, labels, names = load_dataset(args.task, args.dataset, args.dataset_dir)
        run_cv(texts, labels, names, output_dir, folds=args.folds, seed=args.seed,
               epochs=args.epochs, batch_size=args.batch_size,
               embedding_epochs=args.embedding_epochs, embedding_window=args.embedding_window,
               filters=args.filters, dropout=args.dropout, device=args.device,
               save_explanations=args.save_explanations)
    finally:
        if tracker is not None:
            tracker.stop()
    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main()
