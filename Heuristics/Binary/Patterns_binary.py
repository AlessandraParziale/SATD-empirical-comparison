import re
import csv
from typing import List, Tuple, Dict, Any
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Ppreprocessing
# ----------------------------
TOKEN_RE = re.compile(r"[A-Za-z]+")
stemmer = PorterStemmer()

def preprocessing(text: str) -> List[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(text)]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def fuzzy_match(token: str, pat: str) -> bool:
    return token.startswith(pat) or token.endswith(pat)

# ----------------------------
# Load patterns
# ----------------------------
def load_composite_patterns(vocab_csv: str, column: str = "Patterns") -> List[List[str]]:
    """
    "TODO: Test" -> ["todo", "test"]
    """
    composite: List[List[str]] = []

    with open(vocab_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found. Found: {reader.fieldnames}")

        for row in reader:
            cell = (row.get(column) or "").strip()
            if not cell:
                continue

            parts = [p.strip() for p in re.split(r"[;,]", cell) if p.strip()]

            for p in parts:
                toks = preprocessing(p)
                if toks:
                    composite.append(toks)

    return composite

# ----------------------------
# Prediction
# ----------------------------
def is_satd_with_composite_patterns(comment: str, patterns: List[List[str]]) -> Tuple[int, List[str]]:
    """
    SATD if at least one pattern EXISTS
    """
    comment_tokens = preprocessing(comment)

    for pat_tokens in patterns:
        ok = True
        for t in pat_tokens:
            if not any(fuzzy_match(c, t) for c in comment_tokens):
                ok = False
                break
        if ok:
            return 1, pat_tokens  
    return 0, []

# ----------------------------
# Evaluation
# ----------------------------
def evaluate(dataset_csv: str, vocab_csv: str,
             metrics_out_csv: str = "mat_metrics.csv",
             preds_out_csv: str = "mat_predictions_per_comment.csv") -> None:

    df = pd.read_csv(dataset_csv)

    required = {"CommentsAssociated", "CommentsAssociatedLabel"}
    if not required.issubset(df.columns):
        raise ValueError(f"The dataset must contain: {required}. Found: {set(df.columns)}")

    patterns = load_composite_patterns(vocab_csv)

    # predict
    preds = []
    matched_pat = []
    for c in df["CommentsAssociated"].fillna(""):
        p, mp = is_satd_with_composite_patterns(c, patterns)
        preds.append(p)
        matched_pat.append(" ".join(mp) if mp else "")

    df["MAT_Prediction"] = preds
    df["MatchedPatternTokens"] = matched_pat

    y_true = df["CommentsAssociatedLabel"].astype(int)
    y_pred = df["MAT_Prediction"].astype(int)

    report: Dict[str, Any] = classification_report(
        y_true, y_pred,
        target_names=["Non-SATD", "SATD"],
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred) 

    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(metrics_out_csv, index=True)

    df.to_csv(preds_out_csv, index=False)

    print("=== Classification report ===")
    print(metrics_df)
    print("\n=== Confusion Matrix [[TN,FP],[FN,TP]] ===")
    print(cm)

if __name__ == "__main__":

    DATASET_CSV = "merged_Dataset_Yu_undersampling.csv"
    VOCAB_CSV = "Vocabulary_Class_v2.csv"

    METRICS_OUT = "mat_metrics.csv"
    PREDS_OUT = "mat_predictions_per_comment.csv"

    evaluate(
        dataset_csv=DATASET_CSV,
        vocab_csv=VOCAB_CSV,
        metrics_out_csv=METRICS_OUT,
        preds_out_csv=PREDS_OUT
    )


