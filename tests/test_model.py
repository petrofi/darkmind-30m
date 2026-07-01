import sys
import os
import unittest
import torch

# Proje ana dizinini sys.path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPTConfig, GPTLanguageModel

class TestGPTLanguageModel(unittest.TestCase):
    def setUp(self):
        self.config = GPTConfig(
            vocab_size=1000,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.1
        )
        self.model = GPTLanguageModel(self.config)

    def test_forward_pass_without_targets(self):
        # Batch size: 2, Sequence length: 16
        idx = torch.randint(0, self.config.vocab_size, (2, 16))
        logits, loss = self.model(idx)

        self.assertEqual(logits.shape, (2, 16, self.config.vocab_size))
        self.assertIsNone(loss)

    def test_forward_pass_with_targets(self):
        # Batch size: 2, Sequence length: 16
        idx = torch.randint(0, self.config.vocab_size, (2, 16))
        targets = torch.randint(0, self.config.vocab_size, (2, 16))
        logits, loss = self.model(idx, targets)

        self.assertEqual(logits.shape, (2, 16, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(loss.item() > 0)

    def test_sequence_length_error(self):
        # Block size'dan daha uzun bir girdi ile hata alıp almadığımızı doğrula
        idx = torch.randint(0, self.config.vocab_size, (2, self.config.block_size + 1))
        with self.assertRaises(ValueError):
            self.model(idx)

    def test_generate(self):
        idx = torch.randint(0, self.config.vocab_size, (2, 10))
        generated = self.model.generate(idx, max_new_tokens=5, temperature=0.8, top_k=10)
        self.assertEqual(generated.shape, (2, 15))

if __name__ == '__main__':
    unittest.main()
