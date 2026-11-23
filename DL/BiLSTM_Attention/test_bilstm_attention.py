import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from bilstm_attention import (GENE_MAX, BiLSTMAttention, balanced_loss, collate_sequences,
                              decode_gene, genetic_search, run_cv, tokenize_comment, train_glove)


class BiLSTMAttentionTests(unittest.TestCase):
    def test_preprocessing_preserves_todo_and_removes_noise(self):
        self.assertEqual(tokenize_comment("TODO: fix 12 issues at https://example.com"),
                         ["todo", "fix", "issues"])
        self.assertEqual(tokenize_comment("01-02-2020: old revision\nTODO fix"), ["todo", "fix"])
        self.assertEqual(tokenize_comment("42 !"), ["<empty>"])

    def test_padding_does_not_change_prediction(self):
        torch.manual_seed(1)
        model = BiLSTMAttention(torch.randn(5, 300), 2, hidden=4, dropout=0)
        model.eval()
        single = model(torch.tensor([[1, 2]]), torch.tensor([2]))
        padded = model(torch.tensor([[1, 2, 0, 0]]), torch.tensor([2]))
        self.assertTrue(torch.allclose(single, padded, atol=1e-6))
        batch = collate_sequences([(torch.tensor([1, 2]), 1), (torch.tensor([3]), 0)])
        self.assertEqual(batch[0].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(batch[1].tolist(), [2, 1])

    def test_binary_loss_and_multiclass_balance(self):
        logits = torch.tensor([[0.7], [-0.5]])
        labels = torch.tensor([1, 0])
        losses = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float(), reduction="none")
        self.assertTrue(torch.allclose(balanced_loss(logits, labels, 0.8, torch.tensor([3., 1.])),
                                       (losses * torch.tensor([0.8, 0.2])).mean()))
        multi_logits = torch.zeros(3, 3)
        multi_labels = torch.tensor([0, 1, 2])
        counts = torch.tensor([10., 5., 1.])
        self.assertTrue(torch.allclose(balanced_loss(multi_logits, multi_labels, 0.5, counts),
                                       F.cross_entropy(multi_logits, multi_labels)))
        rare_loss = balanced_loss(multi_logits[2:], multi_labels[2:], 1.0, counts)
        common_loss = balanced_loss(multi_logits[:1], multi_labels[:1], 1.0, counts)
        self.assertGreater(rare_loss, common_loss)

    def test_genetic_search_selects_fitness_and_caches(self):
        called = []

        def evaluate(alpha):
            called.append(alpha)
            return alpha, {"alpha": alpha}, 2

        gene, score, state, epoch, history = genetic_search(evaluate, seed=3, generations=3)
        self.assertEqual(len(history), 18)
        self.assertEqual(len(called), len(set(called)))
        self.assertEqual(score, max(row["Validation_F1"] for row in history))
        self.assertEqual(state["alpha"], decode_gene(gene))
        self.assertEqual(epoch, 2)
        self.assertEqual(decode_gene(0), 0.5)
        self.assertEqual(decode_gene(GENE_MAX), 1.0)

    def test_domain_glove_and_multiclass_pipeline(self):
        torch.set_num_threads(1)
        texts = [["todo", "fix", f"word{i % 3}"] for i in range(36)]
        labels = np.array([i % 3 for i in range(36)])
        word_ids, vectors = train_glove(texts[:12], seed=2, device="cpu", epochs=1, dimension=300)
        self.assertEqual(vectors.shape, (len(word_ids), 300))
        self.assertTrue(torch.isfinite(vectors).all())
        self.assertTrue(torch.equal(vectors[word_ids["<pad>"]], torch.zeros(300)))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            results = run_cv(texts, labels, ["A", "B", "C"], output, folds=2, epochs=1,
                             patience=1, generations=1, validation_fraction=0.25,
                             glove_epochs=1, batch_size=16, hidden=4)
            self.assertEqual(set(results["Class"]), {"A", "B", "C", "macro avg", "weighted avg"})
            folds = pd.read_csv(output / "BiLSTM_Attention_fold_metrics.csv")
            search = pd.read_csv(output / "BiLSTM_Attention_genetic_search.csv")
            selections = pd.read_csv(output / "BiLSTM_Attention_selected_alpha.csv")
            self.assertEqual(len(folds), 10)
            self.assertEqual(len(search), 12)
            self.assertEqual(len(selections), 2)
            self.assertEqual(list(results.columns), [
                "Model", "Class", "Precision_Mean", "Precision_Std", "Recall_Mean",
                "Recall_Std", "F1_Mean", "F1_Std", "Support_Mean"])


if __name__ == "__main__":
    unittest.main()
