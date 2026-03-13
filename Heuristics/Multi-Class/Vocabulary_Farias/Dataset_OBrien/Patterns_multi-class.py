import re
import csv
from typing import List, Dict, Tuple, Optional
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.metrics import classification_report, confusion_matrix
from codecarbon import OfflineEmissionsTracker
# ----------------------------
# Preprocessing 
# ----------------------------
TOKEN_RE = re.compile(r"[A-Za-z]+")
stemmer = PorterStemmer()

def preprocessing(text: str) -> List[str]:
    tokens = [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def fuzzy_match(token: str, pat: str) -> bool:
    return token.startswith(pat) or token.endswith(pat)

# ----------------------------
# Mapping Vocabulary Class
# ----------------------------
VOCAB_TO_YOURS = {
    "code debt": "Code",
    "test debt": "M&T",
    "defect debt": "Defect",
    "design debt": "Design",
    "documentation debt": "Documentation",
}
CLASSES = ["Code", "M&T", "Defect", "Design", "Documentation"]

def map_vocab_class_to_yours(vocab_class_cell: str) -> str:
    parts = [p.strip().lower() for p in str(vocab_class_cell).split(",") if p.strip()]
    for p in parts:
        if p in VOCAB_TO_YOURS:
            return VOCAB_TO_YOURS[p]
    raise ValueError(f"Unexpected unmappable class in Vocabulary: '{vocab_class_cell}'")


# ----------------------------
# Load patterns with associated class
# ----------------------------
PatternItem = Tuple[List[str], str, str]

def load_patterns_with_class(vocab_csv: str,
                             pattern_col: str = "Patterns",
                             class_col: str = "Class") -> List[PatternItem]:
    items: List[PatternItem] = []

    with open(vocab_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if pattern_col not in reader.fieldnames or class_col not in reader.fieldnames:
            raise ValueError(f"Vocabulary.csv must contain columns '{pattern_col}' and '{class_col}'. "
                             f"Found: {reader.fieldnames}")

        for row in reader:
            pat_text = (row.get(pattern_col) or "").strip()
            cls_text = (row.get(class_col) or "").strip()
            if not pat_text:
                continue

            mapped = map_vocab_class_to_yours(cls_text)
            pat_tokens = preprocessing(pat_text)
            if pat_tokens:
                items.append((pat_tokens, mapped, pat_text))

    return items

# ----------------------------
# Predict class for one comment
# ----------------------------
def predict_class_for_comment(comment: str, patterns: List[PatternItem]) -> Tuple[str, str]:
    comment_tokens = preprocessing(comment)

    best: Optional[Tuple[int, str, str]] = None  

    for pat_tokens, mapped_class, pat_text in patterns:
        ok = True
        for t in pat_tokens:
            if not any(fuzzy_match(c, t) for c in comment_tokens):
                ok = False
                break
        if ok:
            score = len(pat_tokens)
            if best is None or score > best[0]:
                best = (score, mapped_class, pat_text)

    if best is None:
        return "UNMATCHED", ""
    return best[1], best[2]

# ----------------------------
# Evaluation
# ----------------------------
def run():
    DATASET_CSV = "OBrien.csv"
    VOCAB_CSV = "Vocabulary_Farias.csv"

    METRICS_OUT = "class_metrics.csv"
    CM_OUT = "class_confusion_matrix.csv"
    PREDS_OUT = "class_predictions_per_comment.csv"

    df = pd.read_csv(DATASET_CSV)

    required = {"Comment Text", "Is TD?", "Software TD Type"}
    if not required.issubset(df.columns):
        raise ValueError(f"OBrien.csv must contain {required}. Found: {set(df.columns)}")

    df = df[df["Is TD?"].astype(str).str.strip().str.upper() == "YES"].copy()

    patterns = load_patterns_with_class(VOCAB_CSV)

    preds = []
    matched_patterns = []
    for c in df["Comment Text"].fillna(""):
        p, mp = predict_class_for_comment(c, patterns)
        preds.append(p)
        matched_patterns.append(mp)

    df["PredictedClass"] = preds
    df["MatchedPattern"] = matched_patterns


    TRUE_MAP = {
        "CODE": "Code",
        "M&T": "M&T",
        "DEFECT": "Defect",
        "DESIGN": "Design",
        "DOCUMENTATION": "Documentation",
    }

    df["classification_norm"] = (
        df["Software TD Type"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(TRUE_MAP)
    ).fillna("UNKNOWN_TRUE")

    y_true = df["classification_norm"].astype(str)
    y_pred = df["PredictedClass"].astype(str)


    report = classification_report(
        y_true, y_pred,
        labels=CLASSES,
        output_dict=True,
        zero_division=0
    )
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df.to_csv(METRICS_OUT, index=True)

    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                         columns=[f"pred_{c}" for c in CLASSES])
    cm_df.to_csv(CM_OUT, index=True)

    df.to_csv(PREDS_OUT, index=False)

    print("Saved:")
    print(f"- {METRICS_OUT} (precision/recall/f1/support)")
    print(f"- {CM_OUT} (confusion matrix)")
    print(f"- {PREDS_OUT} (predictions per comment)")
    print("\nConfusion matrix (class order):", CLASSES)
    print(cm_df)

if __name__ == "__main__":

    tracker = OfflineEmissionsTracker(
        country_iso_code="ITA",
        project_name="Heuristics",
        experiment_id="Multi_Class_Vocabulary_Farias_Dataset_Maldonado_All",
        on_csv_write="append"
    )
    tracker.start()


    run()
    
    emissions = tracker.stop()
