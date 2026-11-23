"""BiLSTM with attention and genetic class balancing for SATD detection."""

import argparse
import copy
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

REPO_DIR = Path(__file__).resolve().parents[2]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from DL.GGSATD.ggsatd import DATASETS, ROOT, load_dataset, load_embeddings, seed_everything


MODEL_NAME = "BiLSTM_Attention"
PAD, UNK = "<pad>", "<unk>"
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
REVISION_RE = re.compile(r"\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s*:[^\n]*")
MARKUP_RE = re.compile(r"\[\[SEP\]\]|\{@link\s+[^}]+\}|<[^>]+>|\{[^}]+\}")
WORD_RE = re.compile(r"[a-z]+")
STOP_WORDS = ENGLISH_STOP_WORDS - {"todo"}
GENE_MAX = (1 << 16) - 1


def tokenize_comment(text):
    text = URL_RE.sub(" ", str(text))
    text = REVISION_RE.sub(" ", text)
    text = MARKUP_RE.sub(" ", text).lower()
    words = [word for word in WORD_RE.findall(text)
             if 3 <= len(word) <= 20 and word not in STOP_WORDS]
    return words or ["<empty>"]


def train_glove(texts, seed, device, dimension=300, window=10, epochs=25,
                batch_size=4096, learning_rate=0.05, x_max=100.0, exponent=0.75):
    """Train GloVe on training comments only, with inverse-distance co-occurrences."""
    vocabulary = sorted({word for tokens in texts for word in tokens} - {PAD, UNK})
    word_ids = {word: index + 2 for index, word in enumerate(vocabulary)}
    word_ids.update({PAD: 0, UNK: 1})
    counts = defaultdict(float)
    for tokens in texts:
        ids = [word_ids[word] for word in tokens]
        for i, center in enumerate(ids):
            for distance in range(1, min(window, len(ids) - i - 1) + 1):
                context = ids[i + distance]
                counts[(center, context)] += 1.0 / distance
                counts[(context, center)] += 1.0 / distance
    if not counts:
        raise ValueError("GloVe needs at least one pair of words in a training comment")

    seed_everything(seed)
    target = nn.Embedding(len(word_ids), dimension, device=device, sparse=True)
    context = nn.Embedding(len(word_ids), dimension, device=device, sparse=True)
    target_bias = nn.Embedding(len(word_ids), 1, device=device, sparse=True)
    context_bias = nn.Embedding(len(word_ids), 1, device=device, sparse=True)
    nn.init.uniform_(target.weight, -0.5 / dimension, 0.5 / dimension)
    nn.init.uniform_(context.weight, -0.5 / dimension, 0.5 / dimension)
    nn.init.zeros_(target_bias.weight)
    nn.init.zeros_(context_bias.weight)
    optimizer = torch.optim.Adagrad(
        list(target.parameters()) + list(context.parameters())
        + list(target_bias.parameters()) + list(context_bias.parameters()), lr=learning_rate)
    pairs = list(counts.items())
    centers = torch.tensor([pair[0][0] for pair in pairs], dtype=torch.long)
    contexts = torch.tensor([pair[0][1] for pair in pairs], dtype=torch.long)
    frequencies = torch.tensor([pair[1] for pair in pairs], dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(pairs), generator=generator)
        total_loss = 0.0
        for indices in order.split(batch_size):
            i = centers[indices].to(device)
            j = contexts[indices].to(device)
            x = frequencies[indices].to(device)
            prediction = (target(i) * context(j)).sum(dim=1)
            prediction = prediction + target_bias(i).squeeze(1) + context_bias(j).squeeze(1)
            weights = torch.clamp(x / x_max, max=1.0).pow(exponent)
            loss = (weights * (prediction - x.log()).square()).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite GloVe loss in epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(f"GloVe epoch {epoch}/{epochs}: loss {total_loss / len(pairs):.6f}", flush=True)
    vectors = (target.weight + context.weight).detach().cpu()
    vectors[0].zero_()
    return word_ids, vectors


def pretrained_glove(texts, path, seed):
    vocabulary = {word for tokens in texts for word in tokens}
    ids, vectors = load_embeddings(path, vocabulary, seed)
    word_ids = {word: index + 1 for word, index in ids.items()}
    word_ids[PAD] = 0
    return word_ids, torch.cat((torch.zeros(1, vectors.shape[1]), vectors))


def collate_sequences(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    return pad_sequence(list(sequences), batch_first=True, padding_value=0), lengths, torch.tensor(labels)


class SequenceDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def sequence_loader(texts, labels, indices, word_ids, batch_size, shuffle, seed):
    sequences = [torch.tensor([word_ids.get(word, word_ids[UNK]) for word in texts[i]],
                              dtype=torch.long) for i in indices]
    examples = list(zip(sequences, (int(labels[i]) for i in indices)))
    return DataLoader(SequenceDataset(examples), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_sequences, generator=torch.Generator().manual_seed(seed))


class BiLSTMAttention(nn.Module):
    def __init__(self, embeddings, num_classes, hidden=256, dropout=0.4):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embeddings.shape[1], hidden, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden * 2, hidden * 2)
        self.context = nn.Parameter(torch.empty(hidden * 2))
        nn.init.normal_(self.context, std=0.1)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden * 2, 1 if num_classes == 2 else num_classes)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        states, _ = self.lstm(packed)
        states, _ = pad_packed_sequence(states, batch_first=True, total_length=tokens.shape[1])
        scores = torch.matmul(torch.tanh(self.attention(states)), self.context)
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        scores = scores.masked_fill(positions >= lengths.to(tokens.device).unsqueeze(1), -torch.inf)
        weights = torch.softmax(scores, dim=1)
        representation = (states * weights.unsqueeze(2)).sum(dim=1)
        return self.output(self.dropout(representation))


def balanced_loss(logits, labels, alpha, counts):
    if len(counts) == 2:
        losses = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float(), reduction="none")
        return (losses * torch.where(labels == 1, alpha, 1.0 - alpha)).mean()
    inverse = 1.0 / counts
    inverse = inverse / inverse.mean()
    weights = (2.0 - 2.0 * alpha) + (2.0 * alpha - 1.0) * inverse
    return (F.cross_entropy(logits, labels, reduction="none") * weights[labels]).mean()


def predict(model, loader, device):
    model.eval()
    truth, predictions = [], []
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            logits = model(tokens.to(device), lengths)
            if logits.shape[1] == 1:
                predicted = (logits.squeeze(1) >= 0).long()
            else:
                predicted = logits.argmax(dim=1)
            truth.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
    return truth, predictions


def train_candidate(embeddings, names, alpha, counts, train_loader, val_loader,
                    seed, device, epochs, patience, hidden=256):
    seed_everything(seed)
    model = BiLSTMAttention(embeddings.clone(), len(names), hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    class_counts = torch.tensor(counts, dtype=torch.float32, device=device)
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for tokens, lengths, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(tokens.to(device), lengths)
            loss = balanced_loss(logits, labels, alpha, class_counts)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite classifier loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
        truth, predictions = predict(model, val_loader, device)
        average = "binary" if len(names) == 2 else "macro"
        score = f1_score(truth, predictions, average=average, zero_division=0)
        if score > best_f1:
            best_f1, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    return best_f1, best_state, best_epoch


def decode_gene(gene):
    return 0.5 + 0.5 * gene / GENE_MAX


def genetic_search(evaluate_gene, seed, generations, population_size=6):
    rng = random.Random(seed)
    population = [rng.randint(0, GENE_MAX) for _ in range(population_size)]
    cache, history = {}, []
    best_gene, best_score, best_state, best_epoch = None, -1.0, None, None
    for generation in range(1, generations + 1):
        for gene in population:
            if gene not in cache:
                score, state, epoch = evaluate_gene(decode_gene(gene))
                cache[gene] = (score, epoch)
                if score > best_score:
                    best_gene, best_score, best_state, best_epoch = gene, score, state, epoch
            score, epoch = cache[gene]
            history.append({"Generation": generation, "Gene": gene, "Alpha": decode_gene(gene),
                            "Validation_F1": score, "Best_Epoch": epoch})
        ranked = sorted(population, key=lambda gene: cache[gene][0], reverse=True)
        next_population = [ranked[0]]
        while len(next_population) < population_size:
            parents = rng.choices(ranked, weights=list(range(population_size, 0, -1)), k=2)
            left, right = parents
            if rng.random() < 0.6:
                split = rng.randint(1, 15)
                mask = (1 << split) - 1
                left, right = (left & ~mask) | (right & mask), (right & ~mask) | (left & mask)
            for child in (left, right):
                if rng.random() < 0.2:
                    child ^= 1 << rng.randrange(16)
                next_population.append(child)
                if len(next_population) == population_size:
                    break
        population = next_population
    return best_gene, best_score, best_state, best_epoch, history


def run_cv(texts, labels, names, output_dir, folds=10, seed=6666, epochs=30, patience=5,
           generations=5, validation_fraction=0.1, batch_size=64, device="cpu",
           glove_path=None, glove_epochs=25, hidden=256):
    counts = Counter(labels.tolist())
    if set(counts) != set(range(len(names))) or min(counts.values()) < folds:
        raise ValueError(f"Each class must have at least {folds} examples; counts: {dict(counts)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, search_rows, selections = [], [], []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (candidate, test_indices) in enumerate(splitter.split(texts, labels), 1):
        fold_seed = seed + fold
        train_indices, val_indices = train_test_split(
            candidate, test_size=validation_fraction, stratify=labels[candidate], random_state=fold_seed)
        training_texts = [texts[i] for i in train_indices]
        if glove_path is None:
            word_ids, vectors = train_glove(training_texts, fold_seed, device, epochs=glove_epochs)
        else:
            word_ids, vectors = pretrained_glove(training_texts, glove_path, fold_seed)
        val_loader = sequence_loader(texts, labels, val_indices, word_ids, batch_size, False, fold_seed)
        test_loader = sequence_loader(texts, labels, test_indices, word_ids, batch_size, False, fold_seed)
        class_counts = np.bincount(labels[train_indices], minlength=len(names))

        def evaluate_gene(alpha):
            train_loader = sequence_loader(texts, labels, train_indices, word_ids, batch_size, True, fold_seed)
            score, state, epoch = train_candidate(
                vectors, names, alpha, class_counts, train_loader, val_loader,
                fold_seed, device, epochs, patience, hidden)
            print(f"Fold {fold}/{folds}, alpha {alpha:.6f}: validation F1 {score:.6f}", flush=True)
            return score, state, epoch

        gene, score, state, epoch, history = genetic_search(evaluate_gene, fold_seed, generations)
        if gene is None or state is None:
            raise RuntimeError(f"No candidate model was selected in fold {fold}")
        for item in history:
            search_rows.append({"Fold": fold, **item})
        selections.append({"Fold": fold, "Alpha": decode_gene(gene),
                           "Validation_F1": score, "Best_Epoch": epoch})
        model = BiLSTMAttention(vectors, len(names), hidden=hidden).to(device)
        model.load_state_dict(state)
        truth, predictions = predict(model, test_loader, device)
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

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(output_dir / f"{MODEL_NAME}_fold_metrics.csv", index=False)
    aggregate = fold_df.groupby(["Model", "Class"], as_index=False).agg(
        Precision_Mean=("Precision", "mean"), Precision_Std=("Precision", "std"),
        Recall_Mean=("Recall", "mean"), Recall_Std=("Recall", "std"),
        F1_Mean=("F1", "mean"), F1_Std=("F1", "std"), Support_Mean=("Support", "mean"))
    aggregate.to_csv(output_dir / f"{MODEL_NAME}_aggregated_metrics.csv", index=False)
    pd.DataFrame(search_rows).to_csv(output_dir / f"{MODEL_NAME}_genetic_search.csv", index=False)
    pd.DataFrame(selections).to_csv(output_dir / f"{MODEL_NAME}_selected_alpha.csv", index=False)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "multi"], required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--glove", type=Path, help="Use general pre-trained 300d GloVe instead of training it")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--ga-generations", type=int, default=5)
    parser.add_argument("--glove-epochs", type=int, default=25)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-codecarbon", action="store_true")
    args = parser.parse_args()
    if args.folds < 2 or min(args.epochs, args.patience, args.ga_generations, args.glove_epochs) < 1:
        parser.error("folds must be at least 2; epoch, patience and generation counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if args.glove is not None and not args.glove.is_file():
        parser.error(f"GloVe file not found: {args.glove}")
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
        texts, labels, names = load_dataset(
            args.task, args.dataset, args.dataset_dir, tokenizer=tokenize_comment)
        run_cv(texts, labels, names, output_dir, folds=args.folds, seed=args.seed,
               epochs=args.epochs, patience=args.patience, generations=args.ga_generations,
               device=args.device, glove_path=args.glove, glove_epochs=args.glove_epochs)
    finally:
        if tracker is not None:
            tracker.stop()
    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main()
