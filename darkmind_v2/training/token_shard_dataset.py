"""Memory-mapped deterministic access to validated uint16 token shards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class TokenShardDataset:
    def __init__(self, tokenized_dir: Path, split: str) -> None:
        manifest = json.loads((tokenized_dir / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("dtype") != "uint16-le" or manifest.get("vocab_size") != 24000:
            raise ValueError("incompatible tokenized corpus")
        shard_records = [item for item in manifest["shards"] if item["split"] == split]
        if not shard_records:
            raise ValueError(f"no token shards for split: {split}")
        self.split = split
        self.manifest = manifest
        self.shards: list[np.memmap] = []
        self.starts: list[int] = []
        total = 0
        for record in shard_records:
            path = tokenized_dir / record["filename"]
            values = np.memmap(path, mode="r", dtype="<u2")
            if len(values) != record["tokens"]:
                raise ValueError(f"shard token count changed: {record['filename']}")
            self.starts.append(total)
            self.shards.append(values)
            total += len(values)
        self.total_tokens = total

    def read(self, offset: int, count: int) -> np.ndarray:
        if offset < 0 or count <= 0 or offset + count > self.total_tokens:
            raise ValueError(
                f"requested token range [{offset}, {offset + count}) exceeds {self.split} total {self.total_tokens}"
            )
        remaining = count
        cursor = offset
        pieces: list[np.ndarray] = []
        for start, shard in zip(self.starts, self.shards):
            end = start + len(shard)
            if cursor >= end:
                continue
            local_start = max(0, cursor - start)
            take = min(remaining, len(shard) - local_start)
            pieces.append(np.asarray(shard[local_start : local_start + take], dtype=np.int64))
            cursor += take
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise RuntimeError("failed to assemble requested token range")
        return pieces[0].copy() if len(pieces) == 1 else np.concatenate(pieces)

    def batch(
        self,
        *,
        offset: int,
        micro_batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        count = micro_batch_size * sequence_length
        values = self.read(offset, count)
        if np.any(values < 0) or np.any(values >= 24000):
            raise ValueError("token ID outside [0, 23999]")
        return torch.from_numpy(values).to(device=device, dtype=torch.long).view(micro_batch_size, sequence_length)


def tokenized_manifest_hash(tokenized_dir: Path) -> str:
    import hashlib

    return hashlib.sha256((tokenized_dir / "tokenized_corpus_manifest.json").read_bytes()).hexdigest()


def dataset_summary(tokenized_dir: Path) -> dict[str, Any]:
    manifest = json.loads((tokenized_dir / "tokenized_corpus_manifest.json").read_text(encoding="utf-8"))
    return {
        "manifest_hash": tokenized_manifest_hash(tokenized_dir),
        "manifest_content_hash": manifest["deterministic_content_hash"],
        "split_tokens": manifest["statistics"]["split_tokens"],
        "shards": len(manifest["shards"]),
    }
