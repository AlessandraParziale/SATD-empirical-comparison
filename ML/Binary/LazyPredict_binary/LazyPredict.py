import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import KFold


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



# -----------------------------
# Dataset_Yu
# -----------------------------
df = pd.read_csv("merged_Dataset_Yu_undersampling.csv") 

def getTrainSetTFIDF(data):
    countvec = CountVectorizer(max_features=100)
    bow = countvec.fit_transform(data).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(bow).toarray()
    training_data = pd.DataFrame(X)
    training_data.columns = training_data.columns.astype(str)
    return training_data

def trainTestML(dataset):
    result = pd.DataFrame(columns=["Fold","Model","Accuracy","F1-Score"], index=np.arange(270))
    fold = KFold(n_splits=10, random_state=6666, shuffle=True)
    X_raw = dataset["CommentsAssociated"].fillna("").astype(str).values
    y = dataset["CommentsAssociatedLabel"].astype(int).values
    X_clean = np.array([clean_comment(t) for t in X_raw], dtype=object)
    X = getTrainSetTFIDF(X_clean)
    counter = 0
    foldcounter = 1
    for train_index, test_index in fold.split(X, y):
        print("Processing Fold "+ str(foldcounter) + " ...")
        X_train, X_test, y_train, y_test = \
              X[ X.index.isin(train_index)], X[ X.index.isin(test_index)], y[train_index], y[test_index]
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models,predictions = clf.fit(X_train, X_test, y_train, y_test)
        for model in models[:].iterrows():
            result.loc[counter]["Fold"] = foldcounter
            result.loc[counter]["Model"] = model[0]
            result.loc[counter]["Accuracy"] = round(model[1][0],3)
            result.loc[counter]["F1-Score"] = round(model[1][3],3)
            counter += 1
        foldcounter += 1
    return result

results = trainTestML(df)
results.to_csv("lazypredict_results.csv", index=False)
print("Saved: lazypredict_results.csv")
print(results.groupby('Model', as_index=False)['Accuracy'].mean().sort_values(by='Accuracy', ascending=False))
print(results.groupby('Model', as_index=False)['F1-Score'].mean().sort_values(by='F1-Score', ascending=False))

aggreg_results = results.groupby('Model', as_index=False).agg({'Accuracy': ['mean', 'std'], 'F1-Score': ['mean', 'std']})
aggreg_results.columns = ['Model', 'Accuracy_Mean', 'Accuracy_Std', 'F1-Score_Mean', 'F1-Score_Std']
aggreg_results.to_csv("lazypredict_aggregated_results.csv", index=False)