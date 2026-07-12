"""Transformers slow-tokenizer wrapper for the frozen DarkMind v2 SentencePiece model."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import sentencepiece as spm
from transformers import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class DarkMindV2Tokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str, **kwargs: Any) -> None:
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("bos_token", "<s>")
        kwargs.setdefault("eos_token", "</s>")
        kwargs.setdefault("additional_special_tokens", ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"])
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.vocab_size())

    def get_vocab(self) -> dict[str, int]:
        return {self.sp_model.id_to_piece(index): index for index in range(self.vocab_size)}

    def _tokenize(self, text: str) -> list[str]:
        return list(self.sp_model.encode(text, out_type=str))

    def _convert_token_to_id(self, token: str) -> int:
        return int(self.sp_model.piece_to_id(token))

    def _convert_id_to_token(self, index: int) -> str:
        return str(self.sp_model.id_to_piece(index))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return str(self.sp_model.decode(tokens))

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        if token_ids_1 is not None:
            raise ValueError("DarkMind v2 is a single-sequence causal language model")
        return [self.bos_token_id, *token_ids_0, self.eos_token_id]

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{filename_prefix + '-' if filename_prefix else ''}tokenizer.model"
        destination = directory / filename
        if Path(self.vocab_file).resolve() != destination.resolve():
            shutil.copyfile(self.vocab_file, destination)
        return (str(destination),)
