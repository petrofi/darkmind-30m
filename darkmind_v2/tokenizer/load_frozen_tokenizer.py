"""Read-only adapter for the immutable DarkMind v2 BPE 24k tokenizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


TOKENIZER_NAME = "darkmind_v2_sp_bpe24k_v1"
EXPECTED_HASHES = {
    "tokenizer.model": "db116d4bcf315a6d2a7c5191cbea719d5751c9ba839778eba7e243d520253445",
    "tokenizer.vocab": "f098fecdd4f610ce5b150be09e56e7648211e1ecb076ad6f38af71cee25344ed",
    "tokenizer_freeze_manifest.json": "8e452c049f05ef1c6a94cb5fb42b6accdd1c18b76edebdb9d68bd85fbdfe538e",
}
SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<s>": 2,
    "</s>": 3,
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
    "<|end|>": 7,
}
DEFAULT_FROZEN_DIR = Path(__file__).resolve().parent / "frozen" / TOKENIZER_NAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_tokenizer(path: Path = DEFAULT_FROZEN_DIR) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"frozen tokenizer directory not found: {path}")
    for filename, expected in EXPECTED_HASHES.items():
        actual = sha256_file(path / filename)
        if actual != expected:
            raise ValueError(f"immutable tokenizer artifact changed: {filename}")

    manifest = json.loads((path / "tokenizer_freeze_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("tokenizer_name") != TOKENIZER_NAME:
        raise ValueError("unexpected frozen tokenizer name")
    if manifest.get("immutable_after_freeze") is not True:
        raise ValueError("tokenizer manifest is not immutable")
    if manifest.get("vocab_size") != 24000:
        raise ValueError("frozen tokenizer vocabulary size must be 24,000")
    if manifest.get("special_token_ids") != SPECIAL_TOKENS:
        raise ValueError("frozen tokenizer special-token IDs changed")
    if manifest.get("model_sha256") != EXPECTED_HASHES["tokenizer.model"]:
        raise ValueError("model hash reference differs from the immutable hash")
    if manifest.get("vocab_sha256") != EXPECTED_HASHES["tokenizer.vocab"]:
        raise ValueError("vocabulary hash reference differs from the immutable hash")

    hash_record = json.loads((path / "tokenizer_hashes.json").read_text(encoding="utf-8"))
    for filename, expected in hash_record.get("frozen_files", {}).items():
        if sha256_file(path / filename) != expected:
            raise ValueError(f"frozen hash record mismatch: {filename}")
    return manifest


class FrozenTokenizer:
    """SentencePiece wrapper that never trains, mutates, or extends vocabulary."""

    def __init__(self, path: Path = DEFAULT_FROZEN_DIR) -> None:
        self.path = Path(path)
        self.manifest = verify_frozen_tokenizer(self.path)
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError("sentencepiece is required to load the frozen tokenizer") from exc
        self._processor = spm.SentencePieceProcessor(model_file=str(self.path / "tokenizer.model"))
        if self._processor.vocab_size() != self.manifest["vocab_size"]:
            raise ValueError("SentencePiece model vocabulary does not match the freeze manifest")
        for piece, expected_id in SPECIAL_TOKENS.items():
            if self._processor.piece_to_id(piece) != expected_id:
                raise ValueError(f"SentencePiece special token changed: {piece}")

    @property
    def vocab_size(self) -> int:
        return int(self._processor.vocab_size())

    @property
    def pad_token_id(self) -> int:
        return SPECIAL_TOKENS["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return SPECIAL_TOKENS["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return SPECIAL_TOKENS["<s>"]

    @property
    def eos_token_id(self) -> int:
        return SPECIAL_TOKENS["</s>"]

    @property
    def end_token_id(self) -> int:
        return SPECIAL_TOKENS["<|end|>"]

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = list(self._processor.encode(text, out_type=int))
        if add_bos:
            token_ids.insert(0, self.bos_token_id)
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def encode_document(self, text: str, *, add_bos: bool = False) -> list[int]:
        return self.encode(text, add_bos=add_bos, add_eos=True)

    def decode(self, token_ids: Iterable[int]) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in ids):
            raise ValueError("token IDs are outside the frozen vocabulary")
        return self._processor.decode(ids)

    def piece_to_id(self, piece: str) -> int:
        return int(self._processor.piece_to_id(piece))

    def id_to_piece(self, token_id: int) -> str:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError("token ID is outside the frozen vocabulary")
        return str(self._processor.id_to_piece(int(token_id)))

    def is_byte_fallback_id(self, token_id: int) -> bool:
        piece = self.id_to_piece(token_id)
        return len(piece) == 6 and piece.startswith("<0x") and piece.endswith(">")

    def is_control_id(self, token_id: int) -> bool:
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError("token ID is outside the frozen vocabulary")
        return token_id in SPECIAL_TOKENS.values() or bool(self._processor.is_control(int(token_id)))

    def is_unknown_id(self, token_id: int) -> bool:
        if token_id < 0 or token_id >= self.vocab_size:
            return True
        return bool(self._processor.is_unknown(int(token_id)))

    def add_tokens(self, *_: object, **__: object) -> None:
        raise RuntimeError("the frozen tokenizer cannot add tokens")

    def train(self, *_: object, **__: object) -> None:
        raise RuntimeError("the frozen tokenizer cannot be retrained")


def load_frozen_tokenizer(path: Path = DEFAULT_FROZEN_DIR) -> FrozenTokenizer:
    return FrozenTokenizer(path)
