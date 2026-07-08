"""Build a manifest for an already-existing tokenizer directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_tokenizer_files(tokenizer_path: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(tokenizer_path.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(tokenizer_path)).replace("\\", "/")] = sha256_file(path)
    return files


def load_vocab(tokenizer_path: Path) -> dict[str, int]:
    vocab_path = tokenizer_path / "vocab.json"
    if vocab_path.exists():
        return {str(key): int(value) for key, value in json.loads(vocab_path.read_text(encoding="utf-8")).items()}

    tokenizer_json = tokenizer_path / "tokenizer.json"
    if tokenizer_json.exists():
        payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        vocab = payload.get("model", {}).get("vocab", {})
        if isinstance(vocab, dict):
            return {str(key): int(value) for key, value in vocab.items()}
    return {}


def infer_tokenizer_type(tokenizer_path: Path) -> str:
    tokenizer_json = tokenizer_path / "tokenizer.json"
    if tokenizer_json.exists():
        payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
        return str(payload.get("model", {}).get("type") or "tokenizers-json")
    if (tokenizer_path / "vocab.json").exists() and (tokenizer_path / "merges.txt").exists():
        return "ByteLevelBPE"
    if any(path.suffix == ".model" for path in tokenizer_path.iterdir() if path.is_file()):
        return "SentencePiece"
    return "unknown"


def build_manifest(
    tokenizer_path: Path,
    *,
    training_corpus_manifest_hash: str,
    tokenizer_version: str,
    normalization_rules: dict[str, Any],
    special_tokens: list[str],
    byte_fallback: bool,
    unknown_token_behavior: str,
    creation_command: str,
    immutable: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    vocab = load_vocab(tokenizer_path)
    special_token_ids = {token: vocab[token] for token in special_tokens if token in vocab}
    return {
        "tokenizer_type": infer_tokenizer_type(tokenizer_path),
        "vocabulary_size": len(vocab) if vocab else None,
        "tokenizer_file_hashes": discover_tokenizer_files(tokenizer_path),
        "training_corpus_manifest_hash": training_corpus_manifest_hash,
        "normalization_rules": normalization_rules,
        "special_tokens": special_tokens,
        "special_token_ids": special_token_ids,
        "byte_fallback": byte_fallback,
        "unknown_token_behavior": unknown_token_behavior,
        "creation_command": creation_command,
        "tokenizer_version": tokenizer_version,
        "immutable": immutable,
        "generation_timestamp": timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a DarkMind v2 tokenizer manifest without training.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--training-corpus-manifest-hash", required=True)
    parser.add_argument("--tokenizer-version", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=["<pad>", "<s>", "</s>", "<unk>"])
    parser.add_argument("--byte-fallback", action="store_true")
    parser.add_argument("--unknown-token-behavior", default="explicit <unk> token")
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()

    command = " ".join(shlex.quote(part) for part in sys.argv)
    manifest = build_manifest(
        args.tokenizer,
        training_corpus_manifest_hash=args.training_corpus_manifest_hash,
        tokenizer_version=args.tokenizer_version,
        normalization_rules={"unicode": "NFC", "mojibake_rejection": True},
        special_tokens=args.special_token,
        byte_fallback=args.byte_fallback,
        unknown_token_behavior=args.unknown_token_behavior,
        creation_command=command,
        immutable=True,
        timestamp=args.timestamp,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
