"""Audit and compare DarkMind v2 Phase 1B tokenizer candidates."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

try:
    from .audit_tokenizer import malformed_tokens, percentile, token_script_distribution
    from .build_tokenizer_manifest import discover_tokenizer_files, load_vocab, sha256_file
    from .estimate_vocab_parameter_cost import build_cost_table
    from .test_roundtrip import encoding_ids, load_tokenizer
    from .train_tokenizer_candidates import CANDIDATE_DIRECTORIES
    from ..corpus.detect_mojibake import detect_text, looks_like_mojibake
    from ..corpus.validate_tokenizer_pilot_corpus import validate_processed_corpus
except ImportError:  # pragma: no cover - CLI fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from darkmind_v2.tokenizer.audit_tokenizer import malformed_tokens, percentile, token_script_distribution
    from darkmind_v2.tokenizer.build_tokenizer_manifest import discover_tokenizer_files, load_vocab, sha256_file
    from darkmind_v2.tokenizer.estimate_vocab_parameter_cost import build_cost_table
    from darkmind_v2.tokenizer.test_roundtrip import encoding_ids, load_tokenizer
    from darkmind_v2.tokenizer.train_tokenizer_candidates import CANDIDATE_DIRECTORIES
    from darkmind_v2.corpus.detect_mojibake import detect_text, looks_like_mojibake
    from darkmind_v2.corpus.validate_tokenizer_pilot_corpus import validate_processed_corpus


DEFAULT_CONFIG = Path("darkmind_v2/config/tokenizer_candidates.json")
DEFAULT_GATES = Path("darkmind_v2/config/tokenizer_acceptance_gates.json")
DEFAULT_PLAN = Path("darkmind_v2/config/tokenizer_pilot_corpus.json")
DEFAULT_EVAL_SAMPLES = Path("darkmind_v2/tokenizer/tokenizer_eval_samples.jsonl")
DEFAULT_PROCESSED = Path("darkmind_v2/data/phase1b/processed")
DEFAULT_TOKENIZERS = Path("darkmind_v2/data/phase1b/tokenizers")
DEFAULT_REPORTS = Path("darkmind_v2/reports")

SCORE_WEIGHTS = {
    "turkish_efficiency": 30.0,
    "english_efficiency": 20.0,
    "technical_code_efficiency": 15.0,
    "tail_sequence_lengths": 15.0,
    "vocabulary_cleanliness": 10.0,
    "parameter_cost": 10.0,
}

REQUIRED_AUDIT_KEYS = {
    "candidate_id",
    "algorithm",
    "vocabulary_size",
    "tokenizer_file_hashes",
    "special_token_ids",
    "byte_fallback",
    "unknown_token_count",
    "round_trip_failure_count",
    "mojibake_vocabulary_tokens",
    "replacement_character_tokens",
    "malformed_tokens",
    "metrics",
    "sequence_length_percentiles",
    "vocabulary_script_distribution",
    "mixed_script_risk",
    "parameter_costs",
    "hard_gate_result",
    "hard_gate_failures",
}

WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
OPERATOR_RE = re.compile(r"==|!=|<=|>=|->|=>|::|\+=|-=|\*=|/=|&&|\|\||[+\-*/%=<>]")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_fixed_samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_split_documents(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    return [document for document in content.rstrip("\n").split("\n\n") if document]


def read_split_languages(path: Path) -> dict[str, list[str]]:
    languages = {"validation": [], "eval": []}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            split = record.get("selected_split")
            if split in languages:
                languages[split].append(str(record["language"]))
    return languages


def average(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def pieces_per_word(processor: Any, texts: Iterable[str]) -> float:
    counts = []
    for text in texts:
        for word in WORD_RE.findall(text):
            counts.append(len(encoding_ids(processor.encode(word))))
    return average(counts)


def mixed_script_token_count(tokens: list[str]) -> int:
    risk = 0
    for token in tokens:
        if token.startswith("<") and token.endswith(">"):
            continue
        scripts = set()
        for char in token.replace("▁", ""):
            name = __import__("unicodedata").name(char, "")
            if "LATIN" in name:
                scripts.add("latin")
            elif "CYRILLIC" in name:
                scripts.add("cyrillic")
            elif "ARABIC" in name:
                scripts.add("arabic")
            elif "HEBREW" in name:
                scripts.add("hebrew")
            elif any(marker in name for marker in ("CJK", "HIRAGANA", "KATAKANA")):
                scripts.add("cjk_or_japanese")
        if len(scripts) > 1:
            risk += 1
    return risk


def validate_audit_schema(report: dict[str, Any]) -> list[str]:
    missing = sorted(REQUIRED_AUDIT_KEYS - set(report))
    failures = [f"missing audit field: {field}" for field in missing]
    metrics = report.get("metrics", {})
    for field in (
        "turkish_tokens_per_character",
        "english_tokens_per_character",
        "technical_code_tokens_per_character",
        "tokens_per_word",
        "turkish_suffix_fragmentation",
        "english_word_fragmentation",
        "code_operator_fragmentation",
    ):
        if field not in metrics:
            failures.append(f"missing audit metric: {field}")
    return failures


def audit_candidate(
    candidate: dict[str, Any],
    candidate_dir: Path,
    *,
    shared: dict[str, Any],
    gates: dict[str, Any],
    fixed_samples: list[dict[str, Any]],
    split_documents: dict[str, list[str]],
    split_languages: dict[str, list[str]],
    hostile_leak_count: int,
    corpus_validation_pass: bool,
) -> dict[str, Any]:
    manifest_path = candidate_dir / "tokenizer_manifest.json"
    determinism_path = candidate_dir / "determinism_verification.json"
    if not manifest_path.exists() or not determinism_path.exists():
        raise FileNotFoundError(f"candidate {candidate['id']} is missing manifest/determinism artifacts")
    processor = load_tokenizer(candidate_dir)
    vocab = load_vocab(candidate_dir)
    tokens = [token for token, _ in sorted(vocab.items(), key=lambda item: item[1])]
    manifest = load_json(manifest_path)
    determinism = load_json(determinism_path)
    expected_special_ids = {str(token): int(value) for token, value in shared["special_token_ids"].items()}
    actual_special_ids = {token: int(processor.piece_to_id(token)) for token in shared["special_tokens"]}
    actual_file_hashes = discover_tokenizer_files(candidate_dir)
    manifest_hash_match = manifest.get("tokenizer_file_hashes") == actual_file_hashes

    valid_fixed = [sample for sample in fixed_samples if sample["expected_valid"]]
    audit_items: list[dict[str, str]] = [
        {
            "id": str(sample["id"]),
            "text": str(sample["text"]),
            "language": str(sample["language"]),
            "category": str(sample["category"]),
        }
        for sample in valid_fixed
    ]
    for split in ("validation", "eval"):
        if len(split_documents[split]) != len(split_languages[split]):
            raise ValueError(f"{split} document/attribution count mismatch")
        audit_items.extend(
            {
                "id": f"{split}-{index}",
                "text": text,
                "language": language,
                "category": split,
            }
            for index, (text, language) in enumerate(zip(split_documents[split], split_languages[split]), start=1)
        )

    category_tokens: Counter[str] = Counter()
    category_characters: Counter[str] = Counter()
    language_tokens: Counter[str] = Counter()
    language_characters: Counter[str] = Counter()
    sequence_lengths: list[int] = []
    unknown_count = 0
    total_tokens = 0
    total_words = 0
    round_trip_failures: list[dict[str, Any]] = []
    roundtrip_by_language: Counter[str] = Counter()
    roundtrip_fail_by_language: Counter[str] = Counter()
    unk_id = int(processor.unk_id())
    for item in audit_items:
        ids = encoding_ids(processor.encode(item["text"]))
        decoded = processor.decode(ids)
        token_count = len(ids)
        sequence_lengths.append(token_count)
        total_tokens += token_count
        total_words += max(1, len(WORD_RE.findall(item["text"])))
        category_tokens[item["category"]] += token_count
        category_characters[item["category"]] += len(item["text"])
        language_tokens[item["language"]] += token_count
        language_characters[item["language"]] += len(item["text"])
        unknown_count += sum(int(token_id == unk_id) for token_id in ids)
        roundtrip_by_language[item["language"]] += 1
        if decoded != item["text"]:
            roundtrip_fail_by_language[item["language"]] += 1
            if len(round_trip_failures) < 100:
                round_trip_failures.append({"id": item["id"], "language": item["language"], "decoded": decoded})

    mojibake_vocab = [token for token in tokens if looks_like_mojibake(token) or detect_text(token)]
    replacement_tokens = [token for token in tokens if "\ufffd" in token]
    malformed = malformed_tokens(tokens)
    byte_pieces = {f"<0x{value:02X}>" for value in range(256)}
    byte_piece_count = len(byte_pieces.intersection(tokens))
    unknown_ratio = unknown_count / total_tokens if total_tokens else 0.0
    technical_tokens = category_tokens["technical_code_adjacent"] + category_tokens["source_code"]
    technical_chars = category_characters["technical_code_adjacent"] + category_characters["source_code"]
    suffix_texts = [
        str(sample["text"])
        for sample in valid_fixed
        if sample["language"] == "tr" and any(word in str(sample.get("notes", "")).casefold() for word in ("suffix", "agglutinative"))
    ]
    english_texts = [str(sample["text"]) for sample in valid_fixed if sample["language"] == "en"]
    code_texts = [str(sample["text"]) for sample in valid_fixed if sample["category"] == "source_code"]
    operators = [operator for text in code_texts for operator in OPERATOR_RE.findall(text)]
    mixed_script_count = mixed_script_token_count(tokens)
    metrics = {
        "turkish_tokens_per_character": language_tokens["tr"] / language_characters["tr"] if language_characters["tr"] else 0.0,
        "english_tokens_per_character": language_tokens["en"] / language_characters["en"] if language_characters["en"] else 0.0,
        "technical_code_tokens_per_character": technical_tokens / technical_chars if technical_chars else 0.0,
        "tokens_per_word": total_tokens / total_words if total_words else 0.0,
        "turkish_suffix_fragmentation": pieces_per_word(processor, suffix_texts),
        "english_word_fragmentation": pieces_per_word(processor, english_texts),
        "code_operator_fragmentation": average(len(encoding_ids(processor.encode(operator))) for operator in operators),
    }
    sequence_percentiles = {
        "p50": percentile(sequence_lengths, 50),
        "p90": percentile(sequence_lengths, 90),
        "p95": percentile(sequence_lengths, 95),
        "p99": percentile(sequence_lengths, 99),
        "maximum": max(sequence_lengths) if sequence_lengths else 0,
    }
    round_trip_failure_count = sum(roundtrip_fail_by_language.values())
    targets = gates["minimum_metric_targets"]
    hard_gate_checks = {
        "processed_corpus_gates": corpus_validation_pass,
        "special_token_mismatch": actual_special_ids == expected_special_ids,
        "roundtrip_failure": round_trip_failure_count == 0,
        "hostile_fixture_leak": hostile_leak_count == 0,
        "manifest_nondeterminism": determinism.get("result") == "PASS" and determinism.get("manifest_content_stable") is True,
        "manifest_hash_mismatch": manifest_hash_match,
        "byte_fallback": byte_piece_count == 256,
        "unknown_token_ratio": unknown_ratio <= float(targets["max_unknown_token_ratio_on_valid_samples"]),
        "turkish_efficiency": metrics["turkish_tokens_per_character"] <= float(targets["max_turkish_tokens_per_character_ratio"]),
        "english_efficiency": metrics["english_tokens_per_character"] <= float(targets["max_english_tokens_per_character_ratio"]),
        "vocabulary_cleanliness": not mojibake_vocab and not replacement_tokens and not malformed,
    }
    hard_gate_failures = [name for name, passed in hard_gate_checks.items() if not passed]
    parameter_costs = build_cost_table([len(vocab)], [384, 512])
    report = {
        "schema_version": "darkmind-v2-phase1b-tokenizer-audit-v1",
        "candidate_id": str(candidate["id"]),
        "algorithm": str(candidate["algorithm"]),
        "vocabulary_size": len(vocab),
        "tokenizer_file_hashes": actual_file_hashes,
        "tokenizer_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_manifest_content_hash": manifest.get("deterministic_content_hash"),
        "special_token_ids": actual_special_ids,
        "byte_fallback": {"enabled": manifest.get("byte_fallback") is True, "byte_piece_count": byte_piece_count},
        "unknown_token_count": unknown_count,
        "unknown_token_ratio": unknown_ratio,
        "round_trip_samples": len(audit_items),
        "round_trip_failure_count": round_trip_failure_count,
        "round_trip_failures": round_trip_failures,
        "round_trip_by_language": dict(roundtrip_by_language),
        "round_trip_failures_by_language": dict(roundtrip_fail_by_language),
        "mojibake_vocabulary_tokens": mojibake_vocab,
        "replacement_character_tokens": replacement_tokens,
        "malformed_tokens": malformed,
        "metrics": metrics,
        "sequence_length_percentiles": sequence_percentiles,
        "vocabulary_script_distribution": dict(token_script_distribution(tokens)),
        "mixed_script_risk": {
            "token_count": mixed_script_count,
            "ratio": mixed_script_count / len(tokens) if tokens else 0.0,
        },
        "parameter_costs": parameter_costs,
        "hostile_fixture_leak_count": hostile_leak_count,
        "manifest_hash_match": manifest_hash_match,
        "deterministic_manifest_verification": determinism,
        "hard_gate_checks": hard_gate_checks,
        "hard_gate_result": "PASS" if not hard_gate_failures else "FAIL",
        "hard_gate_failures": hard_gate_failures,
    }
    schema_failures = validate_audit_schema(report)
    if schema_failures:
        report["hard_gate_failures"].extend(schema_failures)
        report["hard_gate_result"] = "FAIL"
    return report


def lower_is_better_scores(values: dict[str, float]) -> dict[str, float]:
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {key: 1.0 for key in values}
    return {key: (maximum - value) / (maximum - minimum) for key, value in values.items()}


def score_candidates(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = {
        "turkish_efficiency": {report["candidate_id"]: float(report["metrics"]["turkish_tokens_per_character"]) for report in reports},
        "english_efficiency": {report["candidate_id"]: float(report["metrics"]["english_tokens_per_character"]) for report in reports},
        "technical_code_efficiency": {report["candidate_id"]: float(report["metrics"]["technical_code_tokens_per_character"]) for report in reports},
        "tail_sequence_lengths": {
            report["candidate_id"]: (float(report["sequence_length_percentiles"]["p95"]) + float(report["sequence_length_percentiles"]["p99"])) / 2
            for report in reports
        },
        "vocabulary_cleanliness": {
            report["candidate_id"]: (
                len(report["mojibake_vocabulary_tokens"])
                + len(report["replacement_character_tokens"])
                + len(report["malformed_tokens"])
                + float(report["mixed_script_risk"]["ratio"])
            )
            for report in reports
        },
        "parameter_cost": {
            report["candidate_id"]: next(
                float(row["combined_parameters"])
                for row in report["parameter_costs"]
                if row["embedding_dim"] == 384 and row["tied_output"] is True
            )
            for report in reports
        },
    }
    normalized = {metric: lower_is_better_scores(values) for metric, values in raw.items()}
    scored = []
    for report in reports:
        candidate_id = report["candidate_id"]
        components = {
            metric: round(normalized[metric][candidate_id] * weight, 6)
            for metric, weight in SCORE_WEIGHTS.items()
        }
        score = round(sum(components.values()), 6)
        eligible = report["hard_gate_result"] == "PASS"
        scored.append(
            {
                "candidate_id": candidate_id,
                "vocabulary_size": report["vocabulary_size"],
                "hard_gate_result": report["hard_gate_result"],
                "eligible": eligible,
                "raw_metrics": {metric: raw[metric][candidate_id] for metric in raw},
                "score_components": components,
                "weighted_score": score if eligible else 0.0,
            }
        )
    return sorted(scored, key=lambda item: (item["eligible"], item["weighted_score"]), reverse=True)


def recommend_candidate(scored: list[dict[str, Any]], within_points: float = 2.0) -> dict[str, Any]:
    eligible = [item for item in scored if item["eligible"]]
    if not eligible:
        return {"candidate_id": None, "strength": "none", "reason": "All candidates failed hard gates."}
    best_score = max(item["weighted_score"] for item in eligible)
    contenders = [item for item in eligible if best_score - item["weighted_score"] <= within_points]
    chosen = min(contenders, key=lambda item: (item["vocabulary_size"], -item["weighted_score"], item["candidate_id"]))
    marginal = len(contenders) > 1
    return {
        "candidate_id": chosen["candidate_id"],
        "strength": "marginal" if marginal else "strong",
        "reason": (
            f"{len(contenders)} candidates were within {within_points} points; the smaller vocabulary tie-breaker was applied."
            if marginal
            else "The leading eligible candidate was more than 2 points ahead."
        ),
        "contenders_within_2_points": [item["candidate_id"] for item in contenders],
    }


def candidate_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    sequence = report["sequence_length_percentiles"]
    lines = [
        f"# Tokenizer Candidate {report['candidate_id']}",
        "",
        f"Hard gates: **{report['hard_gate_result']}**",
        "",
        f"- Algorithm: {report['algorithm']}",
        f"- Vocabulary size: {report['vocabulary_size']}",
        f"- Byte fallback pieces: {report['byte_fallback']['byte_piece_count']} / 256",
        f"- Unknown tokens: {report['unknown_token_count']}",
        f"- Round-trip failures: {report['round_trip_failure_count']}",
        f"- Mojibake / replacement / malformed tokens: {len(report['mojibake_vocabulary_tokens'])} / {len(report['replacement_character_tokens'])} / {len(report['malformed_tokens'])}",
        "",
        "## Efficiency",
        "",
        f"- Turkish tokens/character: {metrics['turkish_tokens_per_character']:.6f}",
        f"- English tokens/character: {metrics['english_tokens_per_character']:.6f}",
        f"- Technical/code tokens/character: {metrics['technical_code_tokens_per_character']:.6f}",
        f"- Tokens/word: {metrics['tokens_per_word']:.6f}",
        f"- Turkish suffix fragmentation: {metrics['turkish_suffix_fragmentation']:.6f}",
        f"- English word fragmentation: {metrics['english_word_fragmentation']:.6f}",
        f"- Code/operator fragmentation: {metrics['code_operator_fragmentation']:.6f}",
        "",
        "## Sequence Lengths",
        "",
        f"- p50 / p90 / p95 / p99 / max: {sequence['p50']} / {sequence['p90']} / {sequence['p95']} / {sequence['p99']} / {sequence['maximum']}",
        "",
        "## Parameter Cost",
        "",
        "| Dimension | Tied | Parameters | % 45M | % 60M |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in report["parameter_costs"]:
        lines.append(
            f"| {row['embedding_dim']} | {row['tied_output']} | {row['combined_parameters']} | {row['percent_of_45m']} | {row['percent_of_60m']} |"
        )
    lines.extend(["", "## Hard Gates", ""])
    for name, passed in report["hard_gate_checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Hashes", ""])
    for name, digest in sorted(report["tokenizer_file_hashes"].items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines.append(f"- `tokenizer_manifest.json`: `{report['tokenizer_manifest_sha256']}`")
    return "\n".join(lines) + "\n"


def comparison_markdown(reports: list[dict[str, Any]], scored: list[dict[str, Any]], recommendation: dict[str, Any]) -> str:
    report_by_id = {report["candidate_id"]: report for report in reports}
    lines = [
        "# DarkMind v2 Phase 1B Tokenizer Comparison",
        "",
        "| Candidate | Algorithm | Vocab | Gates | TR t/c | EN t/c | Tech t/c | p95 | p99 | Score |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in scored:
        report = report_by_id[item["candidate_id"]]
        lines.append(
            f"| {item['candidate_id']} | {report['algorithm']} | {report['vocabulary_size']} | {report['hard_gate_result']} | "
            f"{report['metrics']['turkish_tokens_per_character']:.6f} | {report['metrics']['english_tokens_per_character']:.6f} | "
            f"{report['metrics']['technical_code_tokens_per_character']:.6f} | {report['sequence_length_percentiles']['p95']} | "
            f"{report['sequence_length_percentiles']['p99']} | {item['weighted_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"- Recommended candidate: {recommendation['candidate_id']}",
            f"- Recommendation strength: {recommendation['strength']}",
            f"- Reason: {recommendation['reason']}",
            "- This recommendation does not freeze a final tokenizer.",
            "",
            "## 24k Cost Discussion",
            "",
            "Candidate D uses 9.216M parameters at 384-dim tied (20.48% of 45M) and 18.432M untied (40.96% of 45M). At 512-dim it uses 12.288M tied (27.31% of 45M) or 24.576M untied (54.61% of 45M), so compression gains must be substantial to justify the model-capacity cost.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and compare DarkMind v2 Phase 1B tokenizer candidates.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gates", type=Path, default=DEFAULT_GATES)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--eval-samples", type=Path, default=DEFAULT_EVAL_SAMPLES)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--tokenizer-root", type=Path, default=DEFAULT_TOKENIZERS)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args()

    config = load_json(args.config)
    gates = load_json(args.gates)
    fixed_samples = read_fixed_samples(args.eval_samples)
    corpus_report, corpus_failures = validate_processed_corpus(args.processed_dir, args.plan)
    split_documents = {
        "validation": read_split_documents(args.processed_dir / "tokenizer_validation.txt"),
        "eval": read_split_documents(args.processed_dir / "tokenizer_eval.txt"),
    }
    split_languages = read_split_languages(args.processed_dir / "attribution_manifest.jsonl")
    training_text = (args.processed_dir / "tokenizer_train.txt").read_text(encoding="utf-8")
    hostile_samples = [sample for sample in fixed_samples if not sample["expected_valid"]]
    hostile_leak_count = sum(int(str(sample["text"]) in training_text) for sample in hostile_samples)

    reports = []
    for candidate in config["candidates"]:
        candidate_id = str(candidate["id"]).upper()
        candidate_dir = args.tokenizer_root / CANDIDATE_DIRECTORIES[candidate_id]
        print(f"[audit] candidate {candidate_id}", flush=True)
        report = audit_candidate(
            candidate,
            candidate_dir,
            shared=config["shared_requirements"],
            gates=gates,
            fixed_samples=fixed_samples,
            split_documents=split_documents,
            split_languages=split_languages,
            hostile_leak_count=hostile_leak_count,
            corpus_validation_pass=not corpus_failures and corpus_report.get("result") == "PASS",
        )
        reports.append(report)
        atomic_write_json(candidate_dir / "audit_report.json", report)
        atomic_write_text(args.reports_dir / f"tokenizer_candidate_{candidate_id.lower()}.md", candidate_markdown(report))

    scored = score_candidates(reports)
    recommendation = recommend_candidate(scored)
    comparison = {
        "format": "darkmind-v2-phase1b-tokenizer-comparison-v1",
        "result": "PASS" if any(item["eligible"] for item in scored) else "FAIL",
        "weights": SCORE_WEIGHTS,
        "corpus_validation": corpus_report,
        "candidates": reports,
        "scores": scored,
        "recommendation": recommendation,
        "final_tokenizer_frozen": False,
    }
    atomic_write_json(args.reports_dir / "tokenizer_comparison.json", comparison)
    atomic_write_text(args.reports_dir / "tokenizer_comparison.md", comparison_markdown(reports, scored, recommendation))
    print(json.dumps({"result": comparison["result"], "scores": scored, "recommendation": recommendation}, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if comparison["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
