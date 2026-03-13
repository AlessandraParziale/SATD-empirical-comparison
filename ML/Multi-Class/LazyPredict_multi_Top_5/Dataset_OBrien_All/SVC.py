
import numpy as np
import pandas as pd
import re
import warnings
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from codecarbon import OfflineEmissionsTracker
warnings.filterwarnings("ignore")


# ----------------------------
# Cleaning
# ----------------------------
SEP_RE = re.compile(r"\[\[SEP\]\]")
JAVADOC_LINK_RE = re.compile(r"\{@link\s+[^}]+\}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BRACES_RE = re.compile(r"\{[^}]+\}")
WHITESPACE_RE = re.compile(r"\s+")

def clean_comment(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    t = SEP_RE.sub(" ", t)
    t = JAVADOC_LINK_RE.sub(" ", t)
    t = HTML_TAG_RE.sub(" ", t)
    t = BRACES_RE.sub(" ", t)
    t = t.lower()
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


# ----------------------------
# TF-IDF
# ----------------------------
def getTrainSetTFIDF(text_array, max_features=100):
    countvec = CountVectorizer(max_features=max_features)
    bow = countvec.fit_transform(text_array).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(bow).toarray()
    return pd.DataFrame(X)


# ----------------------------
# SVC
# ----------------------------
def build_models(seed=6666):
    return {
        "SVC": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        )
    }


def run_cv_and_save(dataset_path,
                    output_dir="results_satd_multiclass",
                    n_splits=10,
                    seed=6666,
                    max_features=100):

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}/")

    df = pd.read_csv(dataset_path)

    df = df[df["Is TD?"].astype(str).str.upper().str.strip() == "YES"].copy()

    allowed = ["CODE", "M&T", "DEFECT", "DESIGN", "DOCUMENTATION"]
    df["Software TD Type"] = df["Software TD Type"].astype(str).str.upper().str.strip()
    df = df[df["Software TD Type"].isin(allowed)].copy()

    X_raw = df["Comment Text"].fillna("").astype(str).values
    y = df["Software TD Type"].astype(str).values

    X_clean = np.array([clean_comment(t) for t in X_raw], dtype=object)

    X = getTrainSetTFIDF(X_clean, max_features=max_features)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    models = build_models(seed=seed)

    for model_name, model in models.items():
        print(f"\n=== Running CV for {model_name} ===")

        fold_reports = []
        fold_id = 1

        for train_idx, test_idx in skf.split(X, y):
            X_train = X.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test  = y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rep = classification_report(
                y_test, y_pred,
                labels=allowed,
                output_dict=True,
                zero_division=0
            )

            for cls in allowed:
                fold_reports.append({
                    "Fold": fold_id,
                    "Model": model_name,
                    "Class": cls,
                    "Precision": rep[cls]["precision"],
                    "Recall": rep[cls]["recall"],
                    "F1": rep[cls]["f1-score"],
                    "Support": rep[cls]["support"],
                })

            for avg_name in ["macro avg", "weighted avg"]:
                fold_reports.append({
                    "Fold": fold_id,
                    "Model": model_name,
                    "Class": avg_name,
                    "Precision": rep[avg_name]["precision"],
                    "Recall": rep[avg_name]["recall"],
                    "F1": rep[avg_name]["f1-score"],
                    "Support": rep[avg_name]["support"],
                })

            fold_id += 1

        fold_df = pd.DataFrame(fold_reports)

        agg = (fold_df
               .groupby(["Model", "Class"], as_index=False)[["Precision", "Recall", "F1", "Support"]]
               .agg({
                   "Precision": ["mean", "std"],
                   "Recall": ["mean", "std"],
                   "F1": ["mean", "std"],
                   "Support": ["mean"]
               }))

        agg.columns = [
            "Model", "Class",
            "Precision_Mean", "Precision_Std",
            "Recall_Mean", "Recall_Std",
            "F1_Mean", "F1_Std",
            "Support_Mean"
        ]

        out_path = os.path.join(output_dir, f"{model_name}_aggregated_metrics.csv")
        agg.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

    print("\nAll done.")



# -----------------------------
# Dataset OBrien
# -----------------------------

tracker = OfflineEmissionsTracker(
    country_iso_code="ITA",
    project_name="ML",
    experiment_id="Multi_Class_SVC_OBrien_All",
    on_csv_write="append"
)
tracker.start()


run_cv_and_save(
    dataset_path="OBrien.csv",
    output_dir="results_satd_multiclass",
    n_splits=10,
    seed=6666,
    max_features=100
)
emissions = tracker.stop()