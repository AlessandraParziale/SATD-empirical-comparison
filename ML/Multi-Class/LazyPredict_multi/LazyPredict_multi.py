import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import StratifiedKFold

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
def getTrainSetTFIDF(text_array):
    countvec = CountVectorizer(max_features=100)
    bow = countvec.fit_transform(text_array).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(bow).toarray()
    training_data = pd.DataFrame(X)
    training_data.columns = training_data.columns.astype(str)
    return training_data


# ----------------------------
# Train/Test LazyPredict
# ----------------------------
def trainTestML(dataset):

    dataset = dataset[dataset["satd"] == 1].copy()
    allowed = {"IMPLEMENTATION", "TEST", "DEFECT", "DESIGN", "DOCUMENTATION"}
    dataset["classification"] = dataset["classification"].astype(str).str.upper().str.strip()
    dataset = dataset[dataset["classification"].isin(allowed)].copy()

    X_raw = dataset["comment_text"].fillna("").astype(str).values
    y = dataset["classification"].astype(str).values 

    X_clean = np.array([clean_comment(t) for t in X_raw], dtype=object)

    X = getTrainSetTFIDF(X_clean)

    fold = StratifiedKFold(n_splits=10, random_state=6666, shuffle=True)

    result_rows = []

    foldcounter = 1
    for train_index, test_index in fold.split(X, y):
        print("Processing Fold " + str(foldcounter) + " ...")

        X_train = X.iloc[train_index]
        X_test  = X.iloc[test_index]
        y_train = y[train_index]
        y_test  = y[test_index]

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        for model_name, row in models.iterrows():
            result_rows.append({
                "Fold": foldcounter,
                "Model": model_name,
                "Accuracy": round(float(row["Accuracy"]), 3),
                "F1-Score": round(float(row["F1 Score"]), 3), 
            })

        foldcounter += 1

    return pd.DataFrame(result_rows)


# -----------------------------
# Dataset Maldonado
# -----------------------------
df = pd.read_csv("maldonado.csv")  

results = trainTestML(df)
results.to_csv("lazypredict_results_multiclass.csv", index=False)
print("Saved: lazypredict_results_multiclass.csv")

print("\n=== Mean Accuracy by Model ===")
print(results.groupby('Model', as_index=False)['Accuracy'].mean().sort_values(by='Accuracy', ascending=False))

print("\n=== Mean F1-Score by Model ===")
print(results.groupby('Model', as_index=False)['F1-Score'].mean().sort_values(by='F1-Score', ascending=False))

aggreg_results = results.groupby('Model', as_index=False).agg({'Accuracy': ['mean', 'std'], 'F1-Score': ['mean', 'std']})
aggreg_results.columns = ['Model', 'Accuracy_Mean', 'Accuracy_Std', 'F1-Score_Mean', 'F1-Score_Std']
aggreg_results.to_csv("lazypredict_aggregated_results_multiclass.csv", index=False)
print("\nSaved: lazypredict_aggregated_results_multiclass.csv")
