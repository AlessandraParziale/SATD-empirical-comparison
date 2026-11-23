import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from ffnn import FeedForward, NLPProcessor, loss_for_logits, make_tfidf, run_cv


class FFNNTests(unittest.TestCase):
    def test_preprocessing_order_and_optional_steps(self):
        class Lemmatizer:
            def lemmatize(self, word):
                return {"mice": "mouse"}.get(word, word)

        with patch("ffnn.stopwords", SimpleNamespace(words=lambda language: ["the"])), \
                patch("ffnn.WordNetLemmatizer", Lemmatizer):
            processor = NLPProcessor()
            self.assertEqual(processor("The RUNNING mice!"), ["run", "mouse"])
            unprocessed = NLPProcessor(False, False, False, False, False)
            self.assertEqual(unprocessed("The RUNNING mice!"), ["The", "RUNNING", "mice!"])
            no_tokenization = NLPProcessor(False, False, False, False, False)
            tokenization = NLPProcessor(False, True, False, False, False)
            self.assertEqual(no_tokenization("one,two"), ["one,two"])
            self.assertEqual(tokenization("one,two"), ["one", ",", "two"])

    def test_tfidf_counts_documents_and_retains_single_character_tokens(self):
        vectorizer = make_tfidf()
        matrix = vectorizer.fit_transform([["x", "rare", "rare"], ["x", "rare"],
                                           ["x", "rare"], ["x"], ["other"]])
        self.assertEqual(vectorizer.get_feature_names_out().tolist(), ["x"])
        self.assertEqual(matrix.shape, (5, 1))

    def test_architecture_and_configurable_losses(self):
        for layers in (1, 2, 3):
            model = FeedForward(7, 5, hidden_layers=layers)
            linear = [layer for layer in model.layers if isinstance(layer, torch.nn.Linear)]
            self.assertEqual([layer.out_features for layer in linear], [10] * layers + [5])
            logits = model(torch.randn(4, 7))
            for loss, output in (("cross-entropy", "softmax"), ("mse", "sigmoid")):
                value = loss_for_logits(logits, torch.tensor([0, 1, 2, 3]), loss, output)
                self.assertTrue(torch.isfinite(value))

    def test_fit_uses_only_training_split_and_writes_multiclass_metrics(self):
        instances = []

        class RecordingExtractor:
            def __init__(self):
                self.vectorizer = make_tfidf()
                self.fit_texts = set()
                self.transform_texts = []
                instances.append(self)

            def fit_transform(self, texts):
                self.fit_texts = {tuple(text) for text in texts}
                return self.vectorizer.fit_transform(texts)

            def transform(self, texts):
                self.transform_texts.append({tuple(text) for text in texts})
                return self.vectorizer.transform(texts)

        texts = [["common", f"sample{i}"] for i in range(100)]
        labels = np.array([i % 5 for i in range(100)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_cv(texts, labels, list("ABCDE"), root, folds=2, epochs=1,
                   validation_fraction=0.2, hidden_layers=3, feature_factory=RecordingExtractor)
            for extractor in instances:
                validation, test = extractor.transform_texts
                self.assertTrue(extractor.fit_texts.isdisjoint(validation | test))
                self.assertTrue(validation.isdisjoint(test))
                self.assertEqual(len(extractor.fit_texts | validation | test), len(texts))
            folds = pd.read_csv(root / "FFNN_fold_metrics.csv")
            aggregate = pd.read_csv(root / "FFNN_aggregated_metrics.csv")
            self.assertEqual(len(folds), 14)
            self.assertEqual(set(aggregate["Class"]), set("ABCDE") | {"macro avg", "weighted avg"})

    def test_runner_executes_single_ffnn_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            with (data / "maldonado.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["comment_text", "satd"])
                writer.writerows((f"common {'debt' if i % 2 else 'clean'}", i % 2) for i in range(40))
            config = {
                "repetitions": 1, "output_dir": str(root / "outputs"),
                "experiments": [{
                    "id": "ffnn", "model": "ffnn",
                    "args": ["--task", "binary", "--dataset", "maldonado", "--dataset-dir", str(data),
                             "--folds", "2", "--epochs", "1", "--validation-fraction", "0.2",
                             "--hidden-layers", "2", "--no-stopwords", "--no-lemmatize",
                             "--no-codecarbon"],
                }],
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            runner = Path(__file__).resolve().parents[1] / "run_experiments.py"
            result = subprocess.run([sys.executable, str(runner), str(config_path)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            session = next((root / "outputs").iterdir())
            manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["runs"][0]["status"], "completed")
            output = Path(manifest["runs"][0]["output_dir"])
            self.assertEqual(len(pd.read_csv(output / "FFNN_fold_metrics.csv")), 8)


if __name__ == "__main__":
    unittest.main()
