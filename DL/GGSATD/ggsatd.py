"""Reconstruction of GGSATD for the repository's SATD experiments."""

import argparse
import copy
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "maldonado": ("maldonado.csv", "Dataset_Maldonado_All"),
    "maldonado_undersampled": ("maldonado_undersampled.csv", "Dataset_Maldonado_undersampling"),
    "yu": ("merged_Dataset_Yu.csv", "Dataset_Yu_All"),
    "yu_undersampled": ("merged_Dataset_Yu_undersampling.csv", "Dataset_Yu_undersampling"),
    "obrien": ("OBrien.csv", "Dataset_OBrien_All"),
}
MULTI_LABELS = {
    "maldonado": ["IMPLEMENTATION", "TEST", "DEFECT", "DESIGN", "DOCUMENTATION"],
    "obrien": ["CODE", "M&T", "DEFECT", "DESIGN", "DOCUMENTATION"],
}
TOKEN_RE = re.compile(r"[a-z0-9_]+")
REMOVE_RE = re.compile(r"\[\[SEP\]\]|\{@link\s+[^}]+\}|<[^>]+>|\{[^}]+\}")


def tokenize(text):
    cleaned = REMOVE_RE.sub(" ", str(text)).lower()
    return TOKEN_RE.findall(cleaned) or ["<empty>"]


def load_dataset(task, dataset, dataset_dir, tokenizer=tokenize):
    if task == "binary" and dataset not in {"maldonado", "maldonado_undersampled", "yu", "yu_undersampled"}:
        raise ValueError(f"{dataset} is not a supported binary dataset")
    if task == "multi" and dataset not in MULTI_LABELS:
        raise ValueError(f"{dataset} is not a supported multiclass dataset")

    frame = pd.read_csv(dataset_dir / DATASETS[dataset][0])
    if task == "binary":
        if dataset.startswith("yu"):
            text_col, label_col = "CommentsAssociated", "CommentsAssociatedLabel"
        else:
            text_col, label_col = "comment_text", "satd"
        labels = pd.to_numeric(frame[label_col], errors="raise")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"{label_col} must contain only 0 and 1")
        names = ["NO-SATD", "SATD"]
        y = labels.to_numpy(dtype=np.int64)
    else:
        if dataset == "maldonado":
            frame = frame.loc[frame["satd"] == 1].copy()
            text_col, label_col = "comment_text", "classification"
        else:
            frame = frame.loc[frame["Is TD?"].astype(str).str.strip().str.upper() == "YES"].copy()
            text_col, label_col = "Comment Text", "Software TD Type"
        names = MULTI_LABELS[dataset]
        frame[label_col] = frame[label_col].astype(str).str.strip().str.upper()
        frame = frame.loc[frame[label_col].isin(names)]
        y = frame[label_col].map({name: i for i, name in enumerate(names)}).to_numpy(dtype=np.int64)
    texts = [tokenizer(text) for text in frame[text_col].fillna("").astype(str)]
    return texts, y, names


def load_embeddings(glove_path, vocabulary, seed, dimension=300, require_match=True):
    """Load GloVe vectors for the requested vocabulary and initialize OOV words."""
    words = ["<unk>"] + sorted(set(vocabulary) - {"<unk>"})
    word_ids = {word: i for i, word in enumerate(words)}
    rng = np.random.default_rng(seed)
    vectors = rng.uniform(-0.01, 0.01, (len(words), dimension)).astype(np.float32)
    found = set()
    with glove_path.open(encoding="utf-8") as glove:
        for line_number, line in enumerate(glove, 1):
            parts = line.rstrip().split()
            if not parts or parts[0] not in word_ids:
                continue
            if len(parts) != dimension + 1:
                raise ValueError(f"Invalid GloVe vector on line {line_number}: expected {dimension} values")
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError as exc:
                raise ValueError(f"Invalid GloVe vector on line {line_number}") from exc
            if not np.isfinite(vector).all():
                raise ValueError(f"Non-finite GloVe vector on line {line_number}")
            vectors[word_ids[parts[0]]] = vector
            found.add(parts[0])
    if not found and require_match:
        raise ValueError("No vocabulary words found in the GloVe file; check its format and dimension")
    print(f"GloVe coverage: {len(found)}/{len(words) - 1} tokens")
    return word_ids, torch.from_numpy(vectors)


def make_graph(tokens, word_ids, window=6):
    nodes = list(dict.fromkeys(tokens))
    node_ids = {word: i for i, word in enumerate(nodes)}
    edges = set()
    for i, token in enumerate(tokens):
        source = node_ids[token]
        for other in tokens[i + 1:i + window]:
            target = node_ids[other]
            if source != target:
                edges.add((source, target))
                edges.add((target, source))
    return [word_ids.get(word, 0) for word in nodes], sorted(edges)


def collate_graphs(batch):
    nodes, sources, targets, graph_ids, labels = [], [], [], [], []
    for graph_id, (node_ids, edges, label) in enumerate(batch):
        offset = len(nodes)
        nodes.extend(node_ids)
        sources.extend(offset + source for source, _ in edges)
        targets.extend(offset + target for _, target in edges)
        graph_ids.extend([graph_id] * len(node_ids))
        labels.append(label)
    return tuple(torch.tensor(values, dtype=torch.long) for values in
                 (nodes, sources, targets, graph_ids, labels))


class GatedLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Linear(hidden, hidden, bias=False)
        self.message_gates = nn.Linear(hidden, 3 * hidden)
        self.state_gates = nn.Linear(hidden, 3 * hidden, bias=False)

    def forward(self, state, sources, targets):
        messages = torch.zeros_like(state)
        messages.index_add_(0, targets, self.message(state[sources]))
        mz, mr, mh = self.message_gates(messages).chunk(3, dim=1)
        sz, sr, _ = self.state_gates(state).chunk(3, dim=1)
        z = torch.sigmoid(mz + sz)
        r = torch.sigmoid(mr + sr)
        hidden_weight = self.state_gates.weight.chunk(3, dim=0)[2]
        candidate = torch.tanh(mh + nn.functional.linear(r * state, hidden_weight))
        return candidate * z + state * (1 - z)


class GGSATD(nn.Module):
    def __init__(self, embeddings, num_classes, hidden=96, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.input_projection = nn.Linear(embeddings.shape[1], hidden)
        self.layers = nn.ModuleList([GatedLayer(hidden), GatedLayer(hidden)])
        self.attention = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.features = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, nodes, sources, targets, graph_ids, graph_count):
        state = self.input_projection(self.embedding(nodes))
        for layer in self.layers:
            state = layer(state, sources, targets)
        scores = self.attention(state).squeeze(1)
        maximum = scores.new_full((graph_count,), -torch.inf)
        maximum.scatter_reduce_(0, graph_ids, scores, reduce="amax", include_self=True)
        weights = torch.exp(scores - maximum[graph_ids])
        denominator = weights.new_zeros(graph_count)
        denominator.index_add_(0, graph_ids, weights)
        weights = weights / denominator[graph_ids]
        enhanced = torch.sigmoid(weights.unsqueeze(1) * torch.tanh(self.features(state)))
        summed = enhanced.new_zeros((graph_count, enhanced.shape[1]))
        summed.index_add_(0, graph_ids, enhanced)
        counts = torch.bincount(graph_ids, minlength=graph_count).unsqueeze(1)
        average = summed / counts
        maximum = enhanced.new_full((graph_count, enhanced.shape[1]), -torch.inf)
        maximum.scatter_reduce_(0, graph_ids.unsqueeze(1).expand_as(enhanced),
                                enhanced, reduce="amax", include_self=True)
        return self.classifier(self.dropout(torch.cat((average, maximum), dim=1)))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(texts, labels, indices, word_ids, batch_size, shuffle, seed):
    graphs = []
    for i in indices:
        nodes, edges = make_graph(texts[i], word_ids)
        graphs.append((nodes, edges, int(labels[i])))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_graphs, generator=generator)


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, predictions, truth = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            nodes, sources, targets, graph_ids, labels = (item.to(device) for item in batch)
            logits = model(nodes, sources, targets, graph_ids, len(labels))
            loss_sum += criterion(logits, labels).item() * len(labels)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            truth.extend(labels.cpu().tolist())
    return loss_sum / len(truth), truth, predictions


def run_cv(texts, labels, names, glove_path, output_dir, folds=10, seed=6666,
           epochs=30, patience=5, validation_fraction=0.1, batch_size=32, device="cpu"):
    counts = Counter(labels.tolist())
    if set(counts) != set(range(len(names))) or min(counts.values()) < folds:
        raise ValueError(f"Each class must have at least {folds} examples; counts: {dict(counts)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (candidate, test_indices) in enumerate(splitter.split(texts, labels), 1):
        seed_everything(seed + fold)
        train_indices, val_indices = train_test_split(
            candidate, test_size=validation_fraction, stratify=labels[candidate], random_state=seed + fold)
        vocabulary = {word for i in np.concatenate((train_indices, val_indices)) for word in texts[i]}
        word_ids, embeddings = load_embeddings(glove_path, vocabulary, seed + fold)
        train_loader = make_loader(texts, labels, train_indices, word_ids, batch_size, True, seed + fold)
        val_loader = make_loader(texts, labels, val_indices, word_ids, batch_size, False, seed + fold)
        model = GGSATD(embeddings, len(names)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        best_loss, best_state, stale = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                nodes, sources, targets, graph_ids, targets_y = (item.to(device) for item in batch)
                optimizer.zero_grad()
                loss = criterion(model(nodes, sources, targets, graph_ids, len(targets_y)), targets_y)
                loss.backward()
                optimizer.step()
            val_loss, _, _ = evaluate(model, val_loader, device, criterion)
            print(f"Fold {fold}/{folds}, epoch {epoch}: validation loss {val_loss:.6f}", flush=True)
            if not np.isfinite(val_loss):
                raise RuntimeError(f"Non-finite validation loss in fold {fold}, epoch {epoch}")
            if val_loss < best_loss:
                best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        test_vocabulary = {word for i in test_indices for word in texts[i]} - set(word_ids)
        if test_vocabulary:
            extra_ids, extra_embeddings = load_embeddings(
                glove_path, test_vocabulary, seed + folds + fold, require_match=False)
            offset = len(word_ids) - 1
            word_ids.update({word: offset + index for word, index in extra_ids.items() if word != "<unk>"})
            all_embeddings = torch.cat((model.embedding.weight.detach(), extra_embeddings[1:].to(device)))
            model.embedding = nn.Embedding.from_pretrained(all_embeddings, freeze=True)
        test_loader = make_loader(texts, labels, test_indices, word_ids, batch_size, False, seed + fold)
        _, truth, predictions = evaluate(model, test_loader, device, criterion)
        report = classification_report(truth, predictions, labels=list(range(len(names))),
                                       target_names=names, output_dict=True, zero_division=0)
        for name in names + ["macro avg", "weighted avg"]:
            metrics = report[name]
            rows.append({"Fold": fold, "Model": "GGSATD", "Class": name,
                         "Precision": metrics["precision"], "Recall": metrics["recall"],
                         "F1": metrics["f1-score"], "Support": metrics["support"]})

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(output_dir / "GGSATD_fold_metrics.csv", index=False)
    aggregate = fold_df.groupby(["Model", "Class"], as_index=False).agg(
        Precision_Mean=("Precision", "mean"), Precision_Std=("Precision", "std"),
        Recall_Mean=("Recall", "mean"), Recall_Std=("Recall", "std"),
        F1_Mean=("F1", "mean"), F1_Std=("F1", "std"), Support_Mean=("Support", "mean"))
    aggregate.to_csv(output_dir / "GGSATD_aggregated_metrics.csv", index=False)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["binary", "multi"], required=True)
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--glove", type=Path, required=True, help="Local 300-dimensional GloVe text file")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "Dataset")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-codecarbon", action="store_true", help="Disable emissions tracking")
    args = parser.parse_args()
    if args.folds < 2 or args.epochs < 1 or args.patience < 1:
        parser.error("folds must be at least 2; epochs and patience must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if not args.glove.is_file():
        parser.error(f"GloVe file not found: {args.glove}")
    output_dir = args.output_dir or (ROOT / "DL" / ("Binary" if args.task == "binary" else "Multi-Class")
                                     / DATASETS[args.dataset][1])
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker = None
    if not args.no_codecarbon:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code="ITA", project_name="DL",
                                          experiment_id=f"{args.task}_GGSATD_{args.dataset}",
                                          output_dir=str(output_dir), on_csv_write="append")
        tracker.start()
    try:
        texts, labels, names = load_dataset(args.task, args.dataset, args.dataset_dir)
        run_cv(texts, labels, names, args.glove, output_dir, folds=args.folds, seed=args.seed,
               epochs=args.epochs, patience=args.patience, device=args.device)
    finally:
        if tracker is not None:
            tracker.stop()
    print(f"Results saved in {output_dir}")


if __name__ == "__main__":
    main()
