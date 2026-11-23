import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ggsatd import (GGSATD, collate_graphs, load_dataset, load_embeddings,
                    make_graph, run_cv, tokenize)


class GGSATDTests(unittest.TestCase):
    def test_graph_uses_unique_words_and_undirected_window(self):
        nodes, edges = make_graph(["a", "b", "c", "a", "d"], {"a": 1, "b": 2, "c": 3, "d": 4}, window=2)
        self.assertEqual(nodes, [1, 2, 3, 4])
        self.assertEqual(set(edges), {(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2), (0, 3), (3, 0)})
        self.assertEqual(tokenize("[[SEP]] <b> Fix </b>"), ["fix"])
        self.assertEqual(tokenize(""), ["<empty>"])

    def test_model_handles_empty_edges_and_backpropagates(self):
        batch = collate_graphs([([1], [], 0), ([2, 3], [(0, 1), (1, 0)], 1)])
        model = GGSATD(torch.rand(4, 300), 2)
        logits = model(*batch[:4], len(batch[4]))
        self.assertEqual(tuple(logits.shape), (2, 2))
        torch.nn.functional.cross_entropy(logits, batch[4]).backward()
        self.assertIsNotNone(model.layers[0].message.weight.grad)
        self.assertTrue(torch.isfinite(logits).all())

    def test_dataset_filters_and_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"satd": [1, 1, 0], "comment_text": ["A", "B", "C"],
                          "classification": ["test", "unknown", "design"]}).to_csv(root / "maldonado.csv", index=False)
            texts, labels, names = load_dataset("multi", "maldonado", root)
            self.assertEqual(texts, [["a"]])
            self.assertEqual([names[i] for i in labels], ["TEST"])
            pd.DataFrame({"Is TD?": ["yes", "yes", "no"], "Comment Text": ["A", "B", "C"],
                          "Software TD Type": ["M&T", "requirement", "code"]}).to_csv(root / "OBrien.csv", index=False)
            texts, labels, names = load_dataset("multi", "obrien", root)
            self.assertEqual(texts, [["a"]])
            self.assertEqual([names[i] for i in labels], ["M&T"])

    def test_fold_metrics_are_written_for_multiclass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            glove = root / "glove.txt"
            glove.write_text("word " + " ".join(["0.1"] * 300) + "\n", encoding="utf-8")
            texts = [["word", f"class{i % 3}", f"unique{i}"] for i in range(36)]
            labels = np.array([i % 3 for i in range(36)])
            word_ids, embeddings = load_embeddings(glove, {"word", "missing"}, 1)
            self.assertTrue(torch.allclose(embeddings[word_ids["word"]], torch.full((300,), 0.1)))
            self.assertTrue(torch.all(embeddings[word_ids["missing"]].abs() <= 0.01))
            result = run_cv(texts, labels, ["A", "B", "C"], glove, root / "results",
                            folds=2, epochs=1, patience=1, validation_fraction=0.25, batch_size=16)
            self.assertEqual(set(result["Class"]), {"A", "B", "C", "macro avg", "weighted avg"})
            folds = pd.read_csv(root / "results" / "GGSATD_fold_metrics.csv")
            aggregate = pd.read_csv(root / "results" / "GGSATD_aggregated_metrics.csv")
            self.assertEqual(len(folds), 10)
            self.assertEqual(len(aggregate), 5)
            self.assertEqual(list(aggregate.columns), [
                "Model", "Class", "Precision_Mean", "Precision_Std", "Recall_Mean",
                "Recall_Std", "F1_Mean", "F1_Std", "Support_Mean"])


if __name__ == "__main__":
    unittest.main()
