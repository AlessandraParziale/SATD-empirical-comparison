import numpy as np
import pandas as pd
import re
import warnings
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

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
# Top-5 models
# ----------------------------
def build_top5_models(seed=6666):
    return {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1,
            class_weight="balanced_subsample"
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
        "LGBMClassifier": LGBMClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=500, random_state=seed, n_jobs=-1,
            eval_metric="logloss"
        ),
        "BaggingClassifier": BaggingClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
    }



def run_cv_and_save_binary(dataset_path,
                           output_dir="results_satd_binary",
                           n_splits=10,
                           seed=6666,
                           max_features=100):

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}/")

    df = pd.read_csv(dataset_path)

    X_raw = df["CommentsAssociated"].fillna("").astype(str).values
    y = df["CommentsAssociatedLabel"].astype(int).values 

    X_clean = np.array([clean_comment(t) for t in X_raw], dtype=object)
    X = getTrainSetTFIDF(X_clean, max_features=max_features)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    label_names = ["NO-SATD", "SATD"]
    label_order = [0, 1]

    models = build_top5_models(seed=seed)

    for model_name, model in models.items():
        print(f"\n=== Running CV for {model_name} ===")

        fold_rows = []
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
                labels=label_order,
                target_names=label_names,
                output_dict=True,
                zero_division=0
            )

            for cls in label_names:
                fold_rows.append({
                    "Fold": fold_id,
                    "Model": model_name,
                    "Class": cls,
                    "Precision": rep[cls]["precision"],
                    "Recall": rep[cls]["recall"],
                    "F1": rep[cls]["f1-score"],
                    "Support": rep[cls]["support"],
                })

            for avg_name in ["macro avg", "weighted avg"]:
                fold_rows.append({
                    "Fold": fold_id,
                    "Model": model_name,
                    "Class": avg_name,
                    "Precision": rep[avg_name]["precision"],
                    "Recall": rep[avg_name]["recall"],
                    "F1": rep[avg_name]["f1-score"],
                    "Support": rep[avg_name]["support"],
                })

            fold_id += 1

        fold_df = pd.DataFrame(fold_rows)

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

tracker = OfflineEmissionsTracker(
    country_iso_code="ITA", project_name="ML", 
    experiment_id="Binary_LazyPredict_Top_5", on_csv_write="append")
tracker.start()
run_cv_and_save_binary(
    dataset_path="merged_Dataset_Yu_undersampling.csv",
    output_dir="results_satd_binary",
    n_splits=10,
    seed=6666,
    max_features=100
)
emissions = tracker.stop()

