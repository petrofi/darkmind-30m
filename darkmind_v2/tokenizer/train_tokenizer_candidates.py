"""Train the configured DarkMind v2 Phase 1B SentencePiece candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .build_tokenizer_manifest import build_manifest, discover_tokenizer_files, sha256_file
    from .test_roundtrip import load_tokenizer, run_roundtrip
except ImportError:  # pragma: no cover - CLI fallback
    from build_tokenizer_manifest import build_manifest, discover_tokenizer_files, sha256_file
    from test_roundtrip import load_tokenizer, run_roundtrip


DEFAULT_CONFIG = Path("darkmind_v2/config/tokenizer_candidates.json")
DEFAULT_TRAIN = Path("darkmind_v2/data/phase1b/processed/tokenizer_train.txt")
DEFAULT_CORPUS_MANIFEST = Path("darkmind_v2/data/phase1b/processed/corpus_manifest.json")
DEFAULT_EVAL_SAMPLES = Path("darkmind_v2/tokenizer/tokenizer_eval_samples.jsonl")
DEFAULT_OUTPUT_ROOT = Path("darkmind_v2/data/phase1b/tokenizers")

CANDIDATE_DIRECTORIES = {
    "A": "candidate_a_bpe12k",
    "B": "candidate_b_bpe16k",
    "C": "candidate_c_unigram16k",
    "D": "candidate_d_bpe24k",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def count_characters(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), ""):
            total += len(chunk)
    return total


def build_sentencepiece_training_options(
    candidate: dict[str, Any],
    shared: dict[str, Any],
    train_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    special_tokens = list(shared["special_tokens"])
    special_ids = dict(shared["special_token_ids"])
    if special_tokens[:4] != ["<pad>", "<unk>", "<s>", "</s>"]:
        raise ValueError("SentencePiece core special tokens must be pad, unk, bos, eos in IDs 0..3")
    if [special_ids[token] for token in special_tokens] != list(range(len(special_tokens))):
        raise ValueError("special-token IDs must be contiguous and match tokenizer_candidates.json order")
    model_type = {
        "sentencepiece_bpe": "bpe",
        "sentencepiece_unigram": "unigram",
    }.get(candidate["algorithm"])
    if model_type is None:
        raise ValueError(f"unsupported tokenizer algorithm: {candidate['algorithm']}")
    return {
        "input": str(train_path.resolve()),
        "model_prefix": str((output_dir / "tokenizer").resolve()),
        "model_type": model_type,
        "vocab_size": int(candidate["vocab_size"]),
        "character_coverage": 1.0,
        "byte_fallback": True,
        "normalization_rule_name": "identity",
        "remove_extra_whitespaces": False,
        "add_dummy_prefix": False,
        "pad_id": special_ids["<pad>"],
        "unk_id": special_ids["<unk>"],
        "bos_id": special_ids["<s>"],
        "eos_id": special_ids["</s>"],
        "pad_piece": "<pad>",
        "unk_piece": "<unk>",
        "bos_piece": "<s>",
        "eos_piece": "</s>",
        "user_defined_symbols": special_tokens[4:],
        "num_threads": 1,
        "shuffle_input_sentence": False,
        "input_sentence_size": 0,
        "max_sentence_length": 65536,
        "hard_vocab_limit": True,
        "split_digits": True,
    }


def valid_smoke_samples(path: Path) -> list[str]:
    samples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("expected_valid"):
            samples.append(str(record["text"]))
    return samples


def capture_sentencepiece_training(log_path: Path, options: dict[str, Any]) -> None:
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - checked by CLI preflight
        raise RuntimeError("SentencePiece is required for candidate training") from exc
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write("DarkMind v2 Phase 1B SentencePiece candidate training\n")
        log.write(json.dumps(options, ensure_ascii=False, sort_keys=True) + "\n")
        log.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            spm.SentencePieceTrainer.train(**options)
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def verify_completed_candidate(output_dir: Path, expected_special_ids: dict[str, int], smoke_samples: list[str]) -> dict[str, Any]:
    manifest_path = output_dir / "tokenizer_manifest.json"
    manifest = load_json(manifest_path)
    expected_hashes = manifest.get("tokenizer_file_hashes", {})
    actual_hashes = discover_tokenizer_files(output_dir)
    hash_match = expected_hashes == actual_hashes
    special_match = manifest.get("special_token_ids") == expected_special_ids
    rebuilt_manifest = build_manifest(
        output_dir,
        training_corpus_manifest_hash=manifest["training_corpus_manifest_hash"],
        tokenizer_version=manifest["tokenizer_version"],
        normalization_rules=manifest["normalization_rules"],
        special_tokens=manifest["special_tokens"],
        byte_fallback=bool(manifest["byte_fallback"]),
        unknown_token_behavior=manifest["unknown_token_behavior"],
        creation_command=manifest["creation_command"],
        immutable=bool(manifest["immutable"]),
        timestamp=manifest["generation_timestamp"],
    )
    manifest_content_stable = manifest == rebuilt_manifest
    smoke = run_roundtrip(load_tokenizer(output_dir), smoke_samples)
    smoke_failures = sum(not item.exact_match for item in smoke)
    return {
        "result": "PASS" if hash_match and special_match and manifest_content_stable and smoke_failures == 0 else "FAIL",
        "status": "reused_completed_candidate",
        "output_dir": str(output_dir),
        "manifest_hash_match": hash_match,
        "manifest_content_stable": manifest_content_stable,
        "tokenizer_manifest_sha256": sha256_file(manifest_path),
        "special_token_match": special_match,
        "roundtrip_smoke_failures": smoke_failures,
    }


def train_candidate(
    candidate: dict[str, Any],
    shared: dict[str, Any],
    *,
    train_path: Path,
    corpus_manifest_path: Path,
    eval_samples_path: Path,
    output_root: Path,
    config_version: str,
) -> dict[str, Any]:
    candidate_id = str(candidate["id"]).upper()
    output_dir = output_root / CANDIDATE_DIRECTORIES[candidate_id]
    manifest_path = output_dir / "tokenizer_manifest.json"
    smoke_samples = valid_smoke_samples(eval_samples_path)
    expected_special_ids = {str(key): int(value) for key, value in shared["special_token_ids"].items()}
    if manifest_path.exists():
        return {"candidate_id": candidate_id, **verify_completed_candidate(output_dir, expected_special_ids, smoke_samples)}
    if output_dir.exists() and any(output_dir.iterdir()):
        return {
            "candidate_id": candidate_id,
            "result": "FAIL",
            "status": "partial_output_requires_manual_review",
            "output_dir": str(output_dir),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    options = build_sentencepiece_training_options(candidate, shared, train_path, output_dir)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    training_config = {
        "format": "darkmind-v2-phase1b-tokenizer-training-v1",
        "candidate": candidate,
        "candidate_matrix_version": config_version,
        "sentencepiece_options": options,
        "training_input": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "bytes": train_path.stat().st_size,
            "characters": count_characters(train_path),
        },
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "created_at": created_at,
        "final_tokenizer_frozen": False,
    }
    atomic_write_json(output_dir / "training_config.json", training_config)
    capture_sentencepiece_training(output_dir / "training_log.txt", options)
    if not (output_dir / "tokenizer.model").exists() or not (output_dir / "tokenizer.vocab").exists():
        raise RuntimeError(f"SentencePiece did not produce model/vocab for candidate {candidate_id}")

    creation_command = "SentencePieceTrainer.train(" + json.dumps(options, ensure_ascii=False, sort_keys=True) + ")"
    manifest_kwargs = {
        "training_corpus_manifest_hash": sha256_file(corpus_manifest_path),
        "tokenizer_version": f"phase1b-candidate-{candidate_id.lower()}-{candidate['algorithm']}-{candidate['vocab_size']}",
        "normalization_rules": {
            "unicode": shared["normalization"],
            "sentencepiece_rule": "identity",
            "remove_extra_whitespaces": False,
            "add_dummy_prefix": False,
        },
        "special_tokens": list(shared["special_tokens"]),
        "byte_fallback": True,
        "unknown_token_behavior": "explicit <unk> token; byte fallback for unseen UTF-8 bytes",
        "creation_command": creation_command,
        "immutable": True,
        "timestamp": created_at,
    }
    manifest = build_manifest(output_dir, **manifest_kwargs)
    manifest_content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(manifest_path, manifest_content)

    rebuilt_manifest = build_manifest(output_dir, **manifest_kwargs)
    rebuilt_content = json.dumps(rebuilt_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    deterministic_manifest = manifest_content == rebuilt_content
    processor = load_tokenizer(output_dir)
    smoke_results = run_roundtrip(processor, smoke_samples)
    smoke_failures = sum(not item.exact_match for item in smoke_results)
    actual_special_ids = {token: int(processor.piece_to_id(token)) for token in shared["special_tokens"]}
    pieces = {processor.id_to_piece(index) for index in range(processor.get_piece_size())}
    byte_piece_count = sum(int(f"<0x{value:02X}>" in pieces) for value in range(256))
    artifact_hashes = discover_tokenizer_files(output_dir)
    determinism = {
        "format": "darkmind-v2-phase1b-tokenizer-determinism-v1",
        "candidate_id": candidate_id,
        "result": "PASS" if deterministic_manifest else "FAIL",
        "training_determinism_note": "SentencePiece was configured with one thread, fixed input order, and no sampling; the trained candidate artifact is immutable after creation.",
        "tokenizer_model_sha256": artifact_hashes.get("tokenizer.model"),
        "tokenizer_vocab_sha256": artifact_hashes.get("tokenizer.vocab"),
        "tokenizer_manifest_sha256": sha256_file(manifest_path),
        "manifest_second_pass_sha256": hashlib.sha256(rebuilt_content.encode("utf-8")).hexdigest(),
        "manifest_content_stable": deterministic_manifest,
    }
    atomic_write_json(output_dir / "determinism_verification.json", determinism)
    special_match = actual_special_ids == expected_special_ids
    result = "PASS" if deterministic_manifest and smoke_failures == 0 and special_match and byte_piece_count == 256 else "FAIL"
    return {
        "candidate_id": candidate_id,
        "result": result,
        "status": "trained",
        "output_dir": str(output_dir),
        "vocabulary_size": int(processor.get_piece_size()),
        "special_token_ids": actual_special_ids,
        "special_token_match": special_match,
        "byte_fallback_piece_count": byte_piece_count,
        "roundtrip_smoke_samples": len(smoke_results),
        "roundtrip_smoke_failures": smoke_failures,
        "artifact_hashes": artifact_hashes,
        "manifest_sha256": sha256_file(manifest_path),
        "deterministic_manifest": deterministic_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train configured DarkMind v2 Phase 1B SentencePiece candidates.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--corpus-manifest", type=Path, default=DEFAULT_CORPUS_MANIFEST)
    parser.add_argument("--eval-samples", type=Path, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--candidate-id", action="append", default=None)
    args = parser.parse_args()
    config = load_json(args.config)
    requested = {item.upper() for item in args.candidate_id} if args.candidate_id else set(CANDIDATE_DIRECTORIES)
    unknown = requested - set(CANDIDATE_DIRECTORIES)
    if unknown:
        raise SystemExit(f"unknown candidate IDs: {', '.join(sorted(unknown))}")
    if not args.train.exists() or not args.corpus_manifest.exists():
        raise SystemExit("processed tokenizer train file or corpus manifest is missing")

    results = []
    for candidate in config["candidates"]:
        candidate_id = str(candidate["id"]).upper()
        if candidate_id not in requested:
            continue
        print(f"[tokenizer] candidate {candidate_id}: starting", file=sys.stderr, flush=True)
        try:
            result = train_candidate(
                candidate,
                config["shared_requirements"],
                train_path=args.train,
                corpus_manifest_path=args.corpus_manifest,
                eval_samples_path=args.eval_samples,
                output_root=args.output_root,
                config_version=config["candidate_matrix_version"],
            )
        except (OSError, RuntimeError, ValueError) as exc:
            result = {"candidate_id": candidate_id, "result": "FAIL", "error": str(exc)}
        results.append(result)
        print(f"[tokenizer] candidate {candidate_id}: {result['result']}", file=sys.stderr, flush=True)
    payload = {"result": "PASS" if all(item["result"] == "PASS" for item in results) else "FAIL", "candidates": results}
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "training_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
