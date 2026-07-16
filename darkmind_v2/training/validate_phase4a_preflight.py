"""Fail-closed immutable preflight for the Base V1 Corpus V3 production run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from itertools import zip_longest
from pathlib import Path
from typing import Any

import numpy as np
import torch

from darkmind_v2.data_pipeline.tokenized_manifest import atomic_write_json, canonical_json_hash
from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.estimate_model_size import estimate_model_size
from darkmind_v2.modeling.model_io import model_config_hash
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM
from darkmind_v2.tokenizer.load_frozen_tokenizer import (
    EXPECTED_HASHES,
    SPECIAL_TOKENS,
    load_frozen_tokenizer,
    verify_frozen_tokenizer,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = ROOT / "darkmind_v2" / "config" / "model_base_v1.json"
MODEL_CONSTRAINTS = ROOT / "darkmind_v2" / "config" / "model_base_v1_constraints.json"
TOKENIZED_DIR = ROOT / "darkmind_v2" / "data" / "phase3c1" / "tokenized" / "tranche1_v2"
DETERMINISM_RESULT = ROOT / "darkmind_v2" / "data" / "phase3c1" / "determinism_result.json"

EXPECTED_CONFIG_SHA256 = "8e9775721b0173a92e88de15c2195428932b3aa5beec57d568674c25887c5e39"
EXPECTED_ARCHITECTURE_HASH = "3a2dda86293ceae23ca4e50ea47c840b7fc46021d293c862d330110851ac8305"
EXPECTED_CORPUS_HASH = "e75c4aa4f39cc7a3cb4fe754e2a0e85268ced300f8504a86d443540eb609e1c5"
EXPECTED_TOKENIZED_HASH = "1296caacf09d49b1c48c0fee7d5f5a523a0019e8e7e0e70132fbf68d8f023c82"
EXPECTED_BOUNDARIES_HASH = "3daae663c766575cd7526487baf713d6cfd83fdce5767cfea04722848418cdd8"
EXPECTED_SHARD_CHECKSUMS_HASH = "997ec0fe50a398e7cde90169e0c3ac94b55a8107b305444b08dbe0086148ee76"
EXPECTED_SPLIT_TOKENS = {"train": 98_082_120, "validation": 937_640, "eval": 982_329}
EXPECTED_SPLIT_DOCUMENTS = {"train": 438_255, "validation": 4_441, "eval": 4_431}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_architecture() -> dict[str, Any]:
    constraints = load_json(MODEL_CONSTRAINTS)
    config_sha256 = sha256_file(MODEL_CONFIG)
    require(config_sha256 == EXPECTED_CONFIG_SHA256, "Base V1 config hash mismatch")
    require(constraints["config_sha256"] == config_sha256, "Base V1 constraint hash mismatch")
    config = DarkMindV2Config.from_json_file(MODEL_CONFIG)
    architecture_hash = model_config_hash(config)
    require(architecture_hash == EXPECTED_ARCHITECTURE_HASH, "Base V1 architecture hash mismatch")
    require(constraints["architecture_hash"] == architecture_hash, "constraint architecture hash mismatch")
    estimate = estimate_model_size(
        vocab_size=config.vocab_size,
        n_layers=config.n_layer,
        n_heads=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        tied_embeddings=config.tie_word_embeddings,
        bias=config.bias,
        mlp_hidden_size=config.effective_mlp_hidden_size,
    )
    require(estimate.total_params == 118_056_960, "Base V1 parameter estimate mismatch")
    require(estimate.transformer_body_params == 99_624_960, "Base V1 body parameter mismatch")
    require(abs(estimate.vocab_related_percentage - 15.6128) < 0.00005, "vocabulary share mismatch")
    require(config.head_dimension == 64, "head dimension mismatch")
    require((config.n_layer, config.n_head, config.n_embd) == (14, 12, 768), "model dimensions changed")
    require(config.effective_mlp_hidden_size == 3_072, "MLP dimension changed")
    require(config.block_size == 512, "context length changed")
    require(config.attention_implementation == "sdpa", "SDPA policy changed")
    require(hasattr(torch.nn.functional, "scaled_dot_product_attention"), "PyTorch SDPA is unavailable")
    require(config.gradient_checkpointing is False, "gradient checkpointing must remain disabled")
    model = DarkMindV2ForCausalLM(config)
    require(model.parameter_count() == 118_056_960, "instantiated parameter count mismatch")
    require(model.embeddings_are_tied(), "input/output embeddings are not tied")
    require(model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr(), "tied identity mismatch")
    del model
    gc.collect()
    return {
        "result": "PASS",
        "config_sha256": config_sha256,
        "architecture_hash": architecture_hash,
        "parameters": estimate.total_params,
        "transformer_body_parameters": estimate.transformer_body_params,
        "vocabulary_share_percent": estimate.vocab_related_percentage,
        "layers": config.n_layer,
        "heads": config.n_head,
        "head_dimension": config.head_dimension,
        "embedding_dimension": config.n_embd,
        "mlp_hidden_dimension": config.effective_mlp_hidden_size,
        "context_length": config.block_size,
        "attention_implementation": config.attention_implementation,
        "gradient_checkpointing": config.gradient_checkpointing,
        "tied_embedding_identity": True,
    }


def validate_tokenizer() -> dict[str, Any]:
    manifest = verify_frozen_tokenizer()
    tokenizer = load_frozen_tokenizer()
    require(tokenizer.vocab_size == 24_000, "frozen tokenizer vocabulary changed")
    require(manifest["special_token_ids"] == SPECIAL_TOKENS, "special-token IDs changed")
    byte_ids = [tokenizer.piece_to_id(f"<0x{value:02X}>") for value in range(256)]
    require(len(set(byte_ids)) == 256, "byte-fallback pieces are incomplete")
    for value, token_id in enumerate(byte_ids):
        require(tokenizer.id_to_piece(token_id) == f"<0x{value:02X}>", "byte-fallback piece mismatch")
        require(tokenizer.is_byte_fallback_id(token_id), "byte-fallback taxonomy mismatch")
    try:
        tokenizer.add_tokens(["<dynamic-token>"])
    except RuntimeError:
        pass
    else:  # pragma: no cover - defensive fail-closed guard
        raise ValueError("frozen tokenizer accepted a dynamic token")
    require(tokenizer.vocab_size == 24_000, "tokenizer vocabulary changed after dynamic-token guard")
    return {
        "result": "PASS",
        "hashes": dict(EXPECTED_HASHES),
        "vocab_size": tokenizer.vocab_size,
        "special_token_ids": dict(SPECIAL_TOKENS),
        "byte_fallback_pieces": len(byte_ids),
        "dynamic_tokens": 0,
    }


def snapshot_compared_files(determinism: dict[str, Any]) -> dict[str, dict[str, Any]]:
    compared = determinism["comparisons"]["tokenized"]["compared"]
    snapshots: dict[str, dict[str, Any]] = {}
    for filename, expected_hash in sorted(compared.items()):
        path = TOKENIZED_DIR / filename
        require(path.is_file(), f"missing deterministic corpus file: {filename}")
        stat = path.stat()
        actual_hash = sha256_file(path)
        require(actual_hash == expected_hash, f"deterministic corpus file changed: {filename}")
        snapshots[filename] = {
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": actual_hash,
        }
    return snapshots


def validate_boundaries_and_attribution(
    manifest: dict[str, Any], shard_arrays: dict[str, np.memmap]
) -> dict[str, Any]:
    boundary_path = TOKENIZED_DIR / manifest["document_boundaries"]["filename"]
    attribution_path = TOKENIZED_DIR / "attribution_manifest.jsonl"
    split_documents = {key: 0 for key in EXPECTED_SPLIT_DOCUMENTS}
    split_tokens = {key: 0 for key in EXPECTED_SPLIT_TOKENS}
    next_offset = {name: 0 for name in shard_arrays}
    seen_ids: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    missing_license = 0
    missing_attribution = 0
    records = 0
    with boundary_path.open("r", encoding="utf-8") as boundaries, attribution_path.open(
        "r", encoding="utf-8"
    ) as attributions:
        for boundary_line, attribution_line in zip_longest(boundaries, attributions):
            require(boundary_line is not None and attribution_line is not None, "boundary/attribution count mismatch")
            boundary = json.loads(boundary_line)
            attribution = json.loads(attribution_line)
            document_id = str(boundary["id"])
            split = str(boundary["split"])
            text_hash = str(boundary["text_sha256"])
            shard_name = str(boundary["shard"])
            start = int(boundary["start_offset"])
            end = int(boundary["end_offset"])
            require(split in split_documents, f"unknown split in boundary: {split}")
            require(shard_name in shard_arrays, f"unknown shard in boundary: {shard_name}")
            require(start == next_offset[shard_name], f"non-contiguous boundary in {shard_name}")
            require(end > start and end <= len(shard_arrays[shard_name]), f"invalid boundary in {shard_name}")
            require(end - start == int(boundary["tokens"]), f"boundary token count mismatch: {document_id}")
            require(int(shard_arrays[shard_name][end - 1]) == 3, f"missing EOS boundary: {document_id}")
            prior_id_split = seen_ids.setdefault(document_id, split)
            prior_text_split = seen_text.setdefault(text_hash, split)
            require(prior_id_split == split, f"document ID crosses splits: {document_id}")
            require(prior_text_split == split, f"document content crosses splits: {document_id}")
            require(attribution.get("id") == document_id, f"attribution ID mismatch: {document_id}")
            require(attribution.get("split") == split, f"attribution split mismatch: {document_id}")
            if not attribution.get("license"):
                missing_license += 1
            if not attribution.get("attribution") or not attribution.get("source_url"):
                missing_attribution += 1
            next_offset[shard_name] = end
            split_documents[split] += 1
            split_tokens[split] += end - start
            records += 1
    for shard_name, values in shard_arrays.items():
        require(next_offset[shard_name] == len(values), f"unreferenced shard tail: {shard_name}")
    require(records == 447_127, "boundary record count mismatch")
    require(split_documents == EXPECTED_SPLIT_DOCUMENTS, "split document counts changed")
    require(split_tokens == EXPECTED_SPLIT_TOKENS, "split token counts changed")
    require(missing_license == 0, "license metadata is incomplete")
    require(missing_attribution == 0, "attribution metadata is incomplete")
    return {
        "records": records,
        "split_documents": split_documents,
        "split_tokens": split_tokens,
        "missing_license": missing_license,
        "missing_attribution": missing_attribution,
        "cross_split_document_ids": 0,
        "cross_split_text_hashes": 0,
        "missing_eos_boundaries": 0,
    }


def validate_corpus() -> dict[str, Any]:
    manifest_path = TOKENIZED_DIR / "tokenized_corpus_manifest.json"
    manifest = load_json(manifest_path)
    core = {key: value for key, value in manifest.items() if key != "deterministic_content_hash"}
    require(canonical_json_hash(core) == EXPECTED_TOKENIZED_HASH, "tokenized manifest content hash mismatch")
    require(manifest["deterministic_content_hash"] == EXPECTED_TOKENIZED_HASH, "tokenized hash changed")
    require(manifest["source"]["corpus_manifest_deterministic_hash"] == EXPECTED_CORPUS_HASH, "corpus hash changed")
    require(manifest["dtype"] == "uint16-le", "token dtype is not uint16 little-endian")
    require(int(np.frombuffer(b"\x01\x00", dtype="<u2")[0]) == 1, "little-endian decode check failed")
    require(manifest["vocab_size"] == 24_000, "corpus vocabulary changed")
    require(manifest["eos_token_id"] == 3, "corpus EOS token changed")
    require(len(manifest["shards"]) == 22, "corpus shard count changed")
    require(manifest["statistics"]["split_tokens"] == EXPECTED_SPLIT_TOKENS, "manifest split tokens changed")
    require(manifest["statistics"]["split_documents"] == EXPECTED_SPLIT_DOCUMENTS, "manifest split documents changed")
    require(manifest["statistics"]["total_tokens"] == 100_002_089, "total corpus tokens changed")
    require(manifest["statistics"]["total_bytes"] == 200_004_178, "tokenized byte count changed")
    require(manifest["statistics"]["token_range_violations"] == 0, "recorded token-range violation")
    require(manifest["statistics"]["missing_eos_boundaries"] == 0, "recorded missing EOS boundary")
    require(manifest["document_boundaries"]["sha256"] == EXPECTED_BOUNDARIES_HASH, "boundary hash changed")

    corpus_manifest = load_json(TOKENIZED_DIR / "corpus_manifest.json")
    require(corpus_manifest["deterministic_content_hash"] == EXPECTED_CORPUS_HASH, "corpus manifest hash changed")
    require(corpus_manifest["result"] == "PASS", "Corpus V3 source result is not PASS")
    require(not any(corpus_manifest["hard_gates"].values()), "Corpus V3 hard gate is non-zero")
    determinism = load_json(DETERMINISM_RESULT)
    require(determinism["result"] == "PASS", "Corpus V3 determinism result is not PASS")
    require(determinism["deterministic_mismatches"] == 0, "Corpus V3 determinism mismatch recorded")
    require(determinism["canonical_build_hash"] == EXPECTED_TOKENIZED_HASH, "canonical determinism hash changed")
    require(determinism["rebuilt_build_hash"] == EXPECTED_TOKENIZED_HASH, "rebuild determinism hash changed")
    snapshots = snapshot_compared_files(determinism)
    require(snapshots["document_boundaries.jsonl"]["sha256"] == EXPECTED_BOUNDARIES_HASH, "boundary file changed")
    require(snapshots["shard_checksums.json"]["sha256"] == EXPECTED_SHARD_CHECKSUMS_HASH, "shard checksum manifest changed")

    checksum_manifest = load_json(TOKENIZED_DIR / "shard_checksums.json")
    shard_arrays: dict[str, np.memmap] = {}
    split_tokens = {key: 0 for key in EXPECTED_SPLIT_TOKENS}
    minimum_token = 24_000
    maximum_token = -1
    for record in manifest["shards"]:
        filename = record["filename"]
        require(checksum_manifest.get(filename) == record["sha256"], f"checksum manifest mismatch: {filename}")
        require(snapshots[filename]["sha256"] == record["sha256"], f"shard hash mismatch: {filename}")
        require(snapshots[filename]["bytes"] == record["bytes"], f"shard byte count mismatch: {filename}")
        values = np.memmap(TOKENIZED_DIR / filename, mode="r", dtype="<u2")
        require(len(values) == record["tokens"], f"shard token count mismatch: {filename}")
        if len(values):
            minimum_token = min(minimum_token, int(values.min()))
            maximum_token = max(maximum_token, int(values.max()))
        shard_arrays[filename] = values
        split_tokens[record["split"]] += len(values)
    require(split_tokens == EXPECTED_SPLIT_TOKENS, "decoded split tokens changed")
    require(0 <= minimum_token <= maximum_token < 24_000, "token outside [0, 23999]")
    boundary_report = validate_boundaries_and_attribution(manifest, shard_arrays)
    return {
        "result": "PASS",
        "corpus_hash": EXPECTED_CORPUS_HASH,
        "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
        "boundary_hash": EXPECTED_BOUNDARIES_HASH,
        "shard_checksums_hash": EXPECTED_SHARD_CHECKSUMS_HASH,
        "shards": len(shard_arrays),
        "total_tokens": sum(split_tokens.values()),
        "split_tokens": split_tokens,
        "minimum_token_id": minimum_token,
        "maximum_token_id": maximum_token,
        "dtype": "uint16-le",
        "determinism_compared_files": determinism["compared_files"],
        "determinism_mismatches": determinism["deterministic_mismatches"],
        "boundary_and_attribution": boundary_report,
        "asset_snapshot": snapshots,
    }


def peak_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process(os.getpid()).memory_info().peak_wset)


def run_preflight(pass_index: int, compare_path: Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    report = {
        "schema_version": "darkmind-v2-phase4a-immutable-preflight-v1",
        "pass_index": pass_index,
        "architecture": validate_architecture(),
        "tokenizer": validate_tokenizer(),
        "corpus": validate_corpus(),
    }
    if compare_path is not None:
        previous = load_json(compare_path)
        require(previous.get("result") == "PASS", "previous preflight did not pass")
        require(
            previous["corpus"]["asset_snapshot"] == report["corpus"]["asset_snapshot"],
            "Corpus V3 runtime files changed between preflight passes",
        )
        report["cross_pass_asset_identity"] = True
    else:
        report["cross_pass_asset_identity"] = None
    report["elapsed_seconds"] = time.perf_counter() - started
    report["peak_rss_bytes"] = peak_rss_bytes()
    report["result"] = "PASS"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pass-index", type=int, choices=(1, 2), required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pass_index == 2 and args.compare is None:
        raise ValueError("pass 2 requires --compare")
    report = run_preflight(args.pass_index, args.compare)
    atomic_write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "corpus"}, indent=2, sort_keys=True))
    print(json.dumps({"corpus": {key: value for key, value in report["corpus"].items() if key != "asset_snapshot"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
