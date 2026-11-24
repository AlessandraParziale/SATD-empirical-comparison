import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from cnn_satd import (SATDCNN, collate_comments, explain_batch,
                      run_cv, train_word2vec, weighted_cross_entropy)


class SATDCNNTests(unittest.TestCase):
    def test_word2vec_builds_training_vocabulary_and_vectors(self):
        torch.set_num_threads(1)
        texts = [["technical", "debt"], ["technical", "design"]]
        word_ids, embeddings = train_word2vec(texts, seed=3, dimension=8, epochs=1)
        self.assertEqual(set(word_ids), {"technical", "debt", "design", "<pad>", "<unk>"})
        self.assertEqual(tuple(embeddings.shape), (5, 8))
        self.assertTrue(torch.equal(embeddings[word_ids["<pad>"]], torch.zeros(8)))

    def test_cnn_pools_windows_and_returns_filter_positions(self):
        model = SATDCNN(torch.randn(6, 8), 2, filters=2, dropout=0.0,
                        filter_sizes=(1, 2, 3))
        tokens, _, _ = collate_comments([
            (torch.tensor([2]), 1, 0), (torch.tensor([3, 4, 5]), 0, 1)])
        logits, features, positions = model(tokens, return_features=True)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(features.shape), (2, 6))
        self.assertEqual([tuple(position.shape) for position in positions], [(2, 2)] * 3)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.all((positions[2][0] >= 0) & (positions[2][0] < 4)))

    def test_short_comment_uses_padded_windows_for_max_pooling(self):
        model = SATDCNN(torch.zeros(3, 1), 2, filters=1, dropout=0.0, filter_sizes=(3,))
        with torch.no_grad():
            model.embedding.weight[2, 0] = 1.0
            model.convolutions[0].weight.fill_(0.0)
            model.convolutions[0].weight[0, 0, 0] = 1.0
            model.convolutions[0].bias.zero_()
        tokens, _, _ = collate_comments(
            [(torch.tensor([2]), 1, 0)], max_length=6)

        _, features, positions = model(tokens, return_features=True)

        self.assertEqual(tokens.tolist(), [[0, 0, 2, 0, 0, 0]])
        self.assertEqual(positions[0].item(), 2)
        self.assertEqual(features.item(), 1.0)

    def test_weighted_loss_matches_article_and_multiclass_generalization(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
        labels = torch.tensor([0, 1])
        counts = torch.tensor([3.0, 1.0])
        expected = (F.cross_entropy(logits, labels, reduction="none")
                    * torch.tensor([0.25, 0.75])).mean()
        self.assertTrue(torch.allclose(weighted_cross_entropy(logits, labels, counts), expected))
        balanced = torch.tensor([2.0, 2.0])
        self.assertTrue(torch.allclose(
            weighted_cross_entropy(logits, labels, balanced),
            F.cross_entropy(logits, labels) * 0.5))

    def test_explanations_map_filter_features_to_ngrams(self):
        model = SATDCNN(torch.randn(7, 8), 2, filters=2, dropout=0.0, filter_sizes=(1, 2))
        with torch.no_grad():
            model.classifier.weight[1].fill_(2.0)
            model.classifier.weight[0].fill_(-2.0)
        tokens, labels, sample_ids = collate_comments([
            (torch.tensor([2, 3, 4]), 1, 7)])
        _, features, positions = model(tokens, return_features=True)
        rows = explain_batch(model, sample_ids, [["fix", "todo", "later"]],
                             labels, [1], features, positions, ["NO-SATD", "SATD"],
                             padding_offsets=[1])
        self.assertTrue(rows)
        self.assertEqual({row["Sample_Index"] for row in rows}, {7})
        self.assertTrue(all(row["Predicted_Label"] == "SATD" for row in rows))
        self.assertTrue(all(row["Phrase"] for row in rows))
        self.assertLessEqual(len(rows), 10)

    def test_binary_cv_writes_project_compatible_metrics(self):
        torch.set_num_threads(1)
        texts = [["debt", "todo", f"word{i % 4}"] for i in range(40)]
        labels = np.array([i % 2 for i in range(40)])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = run_cv(texts, labels, ["NO-SATD", "SATD"], output, folds=2,
                            seed=8, epochs=1, batch_size=8, embedding_epochs=1,
                            filters=2, dropout=0.0, save_explanations=True)
            self.assertEqual(set(result["Class"]), {
                "NO-SATD", "SATD", "macro avg", "weighted avg"})
            folds = pd.read_csv(output / "CNN_SATD_fold_metrics.csv")
            explanations = pd.read_csv(output / "CNN_SATD_explanations.csv")
            self.assertEqual(len(folds), 8)
            self.assertEqual(set(folds["Fold"]), {1, 2})
            self.assertTrue(set(explanations["Sample_Index"]).issubset(set(range(40))))
            self.assertEqual(list(result.columns), [
                "Model", "Class", "Precision_Mean", "Precision_Std", "Recall_Mean",
                "Recall_Std", "F1_Mean", "F1_Std", "Support_Mean"])

    def test_cv_runs_all_requested_epochs_per_fold(self):
        torch.set_num_threads(1)
        texts = [["common", f"word{i % 4}"] for i in range(40)]
        labels = np.array([i % 2 for i in range(40)])
        with tempfile.TemporaryDirectory() as directory, \
                contextlib.redirect_stdout(io.StringIO()) as log:
            run_cv(texts, labels, ["A", "B"], Path(directory), folds=2,
                   epochs=3, embedding_epochs=1, filters=2)
        self.assertIn("Fold 1/2, epoch 3/3", log.getvalue())
        self.assertIn("Fold 2/2, epoch 3/3", log.getvalue())


if __name__ == "__main__":
    unittest.main()
