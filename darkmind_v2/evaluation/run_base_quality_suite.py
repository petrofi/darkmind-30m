"""Run the deterministic Phase 5A base-continuation quality suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from darkmind_v2.data_pipeline.tokenized_manifest import canonical_json_hash, sha256_file
from darkmind_v2.evaluation.audit_public_preview import generate_record, percentile, summarize, write_manifest
from darkmind_v2.modeling.model_io import load_model_package
from darkmind_v2.tokenizer.load_frozen_tokenizer import EXPECTED_HASHES, FrozenTokenizer, verify_frozen_tokenizer
from darkmind_v2.training.phase4f_completion import (
    EXPECTED_ARCHITECTURE_HASH,
    EXPECTED_CORPUS_HASH,
    EXPECTED_TOKENIZED_HASH,
    EXPECTED_V2_FILE_HASH,
    TOKENIZED_INPUT,
    V2_CONFIG,
    _validate_shards,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "darkmind_v2" / "config" / "base_quality_suite_v1.json"
REPORT_PATH = ROOT / "darkmind_v2" / "reports" / "phase5a_base_quality_automatic.md"
RUNTIME_ROOT = Path(r"C:\DarkMindRuntime\phase5a")
FINAL_RUN = Path(r"C:\DarkMindRuntime\phase4f\runs\base_v1_first_corpus_pass_completion_v2")
FINAL_CHECKPOINT = FINAL_RUN / "checkpoints" / "step_011972_tokens_098074624"
FINAL_MODEL_HASH = "458816257836a60d804a373c17c617642c99e413c6c190d4fd1e2f73b95fd993"
FINAL_RESUME_HASH = "39e0f3e5aebb469aa9bb0ceefcbae324c0d4c0bfece71d20b012090f1380a587"
EXPORT_DIR = Path(r"C:\DarkMindRuntime\phase4f\exports\darkmind-v2-base-v1-first-pass-98m")
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|(?:system|user|assistant|end)\|>|</?s>|<pad>|<unk>")
TR_MARKERS = {"bir", "ve", "için", "ile", "bu", "olarak", "ancak", "çünkü", "sonra", "üzerinde", "göre"}
EN_MARKERS = {"the", "and", "for", "with", "this", "because", "after", "from", "that", "while", "into"}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mixed_radix_values(index: int, variables: dict[str, list[str]]) -> dict[str, str]:
    selected: dict[str, str] = {}
    cursor = index
    for key in sorted(variables):
        values = variables[key]
        if not values:
            raise ValueError(f"empty prompt variable: {key}")
        selected[key] = values[cursor % len(values)]
        cursor //= len(values)
    return selected


def expand_suite(config: dict[str, Any]) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for category in config["categories"]:
        templates = category["templates"]
        for index in range(int(category["count"])):
            template_index = index % len(templates)
            variable_index = index // len(templates)
            values = _mixed_radix_values(variable_index, category.get("variables", {}))
            language = category["languages"][index % len(category["languages"])]
            text = templates[template_index].format(**values).strip()
            prompts.append(
                {
                    "id": f"bqv1-{category['id']}-{index + 1:03d}",
                    "category": category["id"],
                    "language": language,
                    "prompt": text,
                    "source_type": "original_template",
                    "maximum_repetition_ratio": 0.35,
                    "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
    validate_prompt_suite(config, prompts)
    return prompts


def validate_prompt_suite(config: dict[str, Any], prompts: list[dict[str, Any]]) -> dict[str, Any]:
    if config.get("schema_version") != "darkmind-v2-base-quality-suite-v1":
        raise ValueError("unexpected base-quality suite schema")
    ids = [item["id"] for item in prompts]
    hashes = [item["prompt_sha256"] for item in prompts]
    texts = [item["prompt"] for item in prompts]
    if len(prompts) < 380 or len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)) or len(texts) != len(set(texts)):
        raise ValueError("prompt suite must contain at least 380 unique prompts")
    if any(item["source_type"] != "original_template" for item in prompts):
        raise ValueError("only original deterministic templates are allowed")
    maximum = int(config["prompt_policy"]["maximum_prompt_characters"])
    if any(not item["prompt"] or len(item["prompt"]) > maximum for item in prompts):
        raise ValueError("prompt length policy failed")
    counts = Counter(item["category"] for item in prompts)
    languages = Counter(item["language"] for item in prompts)
    technical_educational = sum(
        count for name, count in counts.items() if "technical" in name or "educational" in name
    )
    required = {
        "turkish": languages["tr"] >= 120,
        "english": languages["en"] >= 100,
        "technical_educational": technical_educational >= 60,
        "code_structured": counts["code_structured"] >= 40,
        "factual_context": counts["factual_context"] >= 40,
        "long_context": counts["long_context_consistency"] >= 20,
    }
    if not all(required.values()):
        raise ValueError(f"suite minimums failed: {required}")
    return {"prompt_count": len(prompts), "category_counts": dict(counts), "language_counts": dict(languages)}


def prompt_manifest_hash(prompts: list[dict[str, Any]]) -> str:
    return canonical_json_hash(prompts)


def immutable_preflight(config: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(config["checkpoint"]["path"])
    model_hash = sha256_file(checkpoint / "model" / "model.safetensors")
    resume_hash = sha256_file(checkpoint / "resume_state.pt")
    if checkpoint != FINAL_CHECKPOINT or model_hash != FINAL_MODEL_HASH or resume_hash != FINAL_RESUME_HASH:
        raise ValueError("Phase 5A checkpoint identity mismatch")
    tokenizer = verify_frozen_tokenizer()
    corpus = _validate_shards()
    authorization = load_json(ROOT / "darkmind_v2" / "config" / "train_base_v1_first_pass_completion_authorization.json")
    if corpus["manifest_content_hash"] != EXPECTED_TOKENIZED_HASH:
        raise ValueError("Corpus V3 tokenized content hash changed")
    if authorization["corpus_hashes"]["corpus"] != EXPECTED_CORPUS_HASH:
        raise ValueError("Corpus V3 source-content hash reference changed")
    if sha256_file(V2_CONFIG) != EXPECTED_V2_FILE_HASH:
        raise ValueError("V2 training policy changed")
    provenance = load_json(EXPORT_DIR / "provenance_manifest.json")
    export_failures = []
    for name, expected in provenance["files"].items():
        path = EXPORT_DIR / name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            export_failures.append(name)
    if export_failures:
        raise ValueError(f"local export hash mismatch: {export_failures}")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    hf_config = AutoConfig.from_pretrained(EXPORT_DIR, trust_remote_code=True, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(EXPORT_DIR, trust_remote_code=True, local_files_only=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(EXPORT_DIR, trust_remote_code=True, local_files_only=True)
    encoded = hf_tokenizer("Türkiye ve bilim", return_tensors="pt")
    with torch.no_grad():
        finite_forward = bool(torch.isfinite(hf_model(**encoded).logits).all())
    offline_ok = (
        hf_config.vocab_size == 24_000
        and hf_config.n_layer == 14
        and hf_model.parameter_count() == 118_056_960
        and hf_tokenizer.vocab_size == 24_000
        and finite_forward
    )
    del hf_model
    if not offline_ok:
        raise ValueError("offline Transformers validation failed")
    payload = {
        "schema_version": "darkmind-v2-phase5a-immutable-preflight-v1",
        "result": "PASS",
        "checkpoint": str(checkpoint),
        "checkpoint_model_sha256": model_hash,
        "resume_state_sha256": resume_hash,
        "optimizer_steps": int(config["checkpoint"]["optimizer_steps"]),
        "training_tokens": int(config["checkpoint"]["training_tokens"]),
        "architecture_hash": EXPECTED_ARCHITECTURE_HASH,
        "tokenizer_hashes": dict(EXPECTED_HASHES),
        "tokenizer_name": tokenizer["tokenizer_name"],
        "corpus_hash": EXPECTED_CORPUS_HASH,
        "tokenized_manifest_hash": EXPECTED_TOKENIZED_HASH,
        "v2_training_policy_sha256": EXPECTED_V2_FILE_HASH,
        "export_provenance_sha256": sha256_file(EXPORT_DIR / "provenance_manifest.json"),
        "export_file_count": len(provenance["files"]),
        "offline_transformers_load": True,
        "checkpoint_modified": False,
        "second_epoch_started": False,
    }
    atomic_write_json(RUNTIME_ROOT / "manifests" / "immutable_preflight.json", payload)
    return payload


def _prompt_loss(model: torch.nn.Module, tokenizer: FrozenTokenizer, text: str) -> tuple[float, int]:
    device = next(model.parameters()).device
    token_ids = tokenizer.encode(text, add_bos=True)[-model.config.block_size :]
    if len(token_ids) < 2:
        return 0.0, 0
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_ids).logits
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))
    return float(loss.item()), len(token_ids) - 1


def _detect_language(text: str) -> str:
    words = set(re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", text.lower()))
    tr_score = len(words & TR_MARKERS) + 2 * sum(text.count(character) for character in "çğıöşüÇĞİÖŞÜ")
    en_score = len(words & EN_MARKERS)
    if tr_score == en_score == 0:
        return "unknown"
    return "tr" if tr_score >= en_score else "en"


def _balanced_structure(text: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    for character in text:
        if character in pairs:
            stack.append(pairs[character])
        elif character in pairs.values():
            if not stack or stack.pop() != character:
                return False
    return not stack


def _longest_prompt_overlap(prompt_ids: list[int], output_ids: list[int], maximum_n: int = 12) -> int:
    for n in range(min(maximum_n, len(output_ids), len(prompt_ids)), 0, -1):
        prompt_ngrams = {tuple(prompt_ids[index : index + n]) for index in range(len(prompt_ids) - n + 1)}
        if any(tuple(output_ids[index : index + n]) in prompt_ngrams for index in range(len(output_ids) - n + 1)):
            return n
    return 0


def add_automatic_proxies(record: dict[str, Any], tokenizer: FrozenTokenizer, prompt_loss: float, loss_tokens: int) -> None:
    expected = record["language"]
    detected = _detect_language(record["output"])
    prompt_ids = tokenizer.encode(record["prompt"], add_bos=False)
    output_ids = [int(item) for item in record["token_ids"] if int(item) != tokenizer.eos_token_id]
    shared = len(set(prompt_ids) & set(output_ids)) / max(len(set(output_ids)), 1)
    code_valid = None
    if record["category"] == "code_structured":
        code_valid = _balanced_structure(record["output"]) and any(symbol in record["output"] for symbol in "=():[]{}\n")
    punctuation_complete = bool(record["output"].rstrip().endswith((".", "!", "?", ":", ";", ")", "]", "}")))
    record["automatic_proxies"] = {
        "prompt_loss": prompt_loss,
        "prompt_perplexity": math.exp(min(prompt_loss, 20.0)) if loss_tokens else None,
        "prompt_loss_tokens": loss_tokens,
        "detected_output_language": detected,
        "language_consistent": detected in {expected, "unknown"} if expected in {"tr", "en"} else True,
        "language_switch_error": detected not in {expected, "unknown"} if expected in {"tr", "en"} else False,
        "prompt_token_overlap_ratio": shared,
        "longest_prompt_overlap_tokens": _longest_prompt_overlap(prompt_ids, output_ids),
        "code_structure_valid": code_valid,
        "punctuation_completion": punctuation_complete,
        "special_token_leakage": bool(SPECIAL_TOKEN_PATTERN.search(record["output"])),
        "semantic_quality_score": None,
    }


def automatic_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    base = summarize(records)
    warnings = Counter(item for record in records for item in record["policy"]["warnings"])
    proxies = [record["automatic_proxies"] for record in records]
    lengths = [record["generated_token_count"] for record in records]
    language_eligible = [item for record, item in zip(records, proxies) if record["language"] in {"tr", "en"}]
    code = [item for item in proxies if item["code_structure_valid"] is not None]
    return {
        "generation_count": len(records),
        "repetition_rate": warnings["repetition"] / max(len(records), 1),
        "exact_loop_rate": base["exact_repeated_ngram_loop_outputs"] / max(len(records), 1),
        "eos_completion_rate": base["eos_completion_rate"],
        "empty_output_rate": base["empty_output_count"] / max(len(records), 1),
        "output_tokens": {
            "minimum": min(lengths, default=0),
            "mean": statistics.fmean(lengths) if lengths else 0.0,
            "p50": percentile(lengths, 0.5),
            "p90": percentile(lengths, 0.9),
            "maximum": max(lengths, default=0),
        },
        "mean_unique_token_ratio": base["mean_unique_token_ratio"],
        "language_id_consistency_rate": sum(item["language_consistent"] for item in language_eligible) / max(len(language_eligible), 1),
        "language_switch_error_rate": sum(item["language_switch_error"] for item in language_eligible) / max(len(language_eligible), 1),
        "unicode_health": {
            "invalid_utf8_sequences": base["invalid_utf8_sequence_count"],
            "replacement_characters": base["replacement_character_count"],
            "mojibake_outputs": base["mojibake_output_count"],
        },
        "special_token_leakage_count": sum(item["special_token_leakage"] for item in proxies),
        "mean_prompt_token_overlap_ratio": statistics.fmean(item["prompt_token_overlap_ratio"] for item in proxies),
        "maximum_prompt_overlap_tokens": max((item["longest_prompt_overlap_tokens"] for item in proxies), default=0),
        "code_structure_valid_rate": sum(bool(item["code_structure_valid"]) for item in code) / max(len(code), 1),
        "punctuation_completion_rate": sum(item["punctuation_completion"] for item in proxies) / max(len(proxies), 1),
        "human_semantic_quality_claimed": False,
    }


def _loss_groups(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[key])].append(float(record["automatic_proxies"]["prompt_loss"]))
    return {
        name: {"prompts": len(values), "mean_loss": statistics.fmean(values), "perplexity": math.exp(min(statistics.fmean(values), 20.0))}
        for name, values in sorted(grouped.items())
    }


def _fixed_probe_summary() -> dict[str, Any]:
    probes = load_json(FINAL_RUN / "evaluations.json")["11972"]["probes"]
    families: dict[str, list[float]] = defaultdict(list)
    for name, item in probes.items():
        family = "distribution" if name in {"training_distribution", "validation", "eval"} else "language" if name in {"turkish_prose", "english_prose", "turkish_technical", "english_technical"} else "source"
        families[family].append(float(item["loss"]))
    return {
        "checkpoint_step": 11972,
        "probes": {name: {"loss": float(item["loss"]), "perplexity": float(item["perplexity"])} for name, item in sorted(probes.items())},
        "source_family_mean_loss": {name: statistics.fmean(values) for name, values in sorted(families.items())},
    }


def _memorization_summary() -> dict[str, Any]:
    audit = load_json(FINAL_RUN / "memorization_audit.json")
    return {
        "result": audit["result"],
        "exact_train_continuations": audit["exact_continuation_match"]["train_count"],
        "exact_heldout_continuations": audit["exact_continuation_match"]["heldout_count"],
        "longest_true_continuation_match_tokens": audit["near_exact_similarity"]["longest_exact_span_tokens"],
        "longest_generated_ngram_in_train_tokens": audit["training_corpus_ngram"]["longest_exact_generated_ngram_in_training_tokens"],
        "material_personal_data_reproduction": audit["material_personal_data_reproduction_count"],
        "hard_release_blockers": audit["hard_release_blockers"],
        "extraction_risk_zero_claimed": False,
    }


def _render_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Phase 5A automatic base-quality review",
        "",
        "This is a deterministic base-continuation diagnostic. Automatic proxies are not human semantic-quality scores.",
        "",
        "## Immutable evaluation identity",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Model SHA-256: `{summary['checkpoint_model_sha256']}`",
        f"- Prompt manifest SHA-256: `{summary['prompt_manifest_sha256']}`",
        f"- Unique prompts: {summary['prompt_count']}",
        f"- Raw outputs: `{summary['runtime_output_dir']}` (outside Git)",
        "",
        "## Aggregate decoding health",
        "",
        "| Mode | Generations | Repetition | Exact loops | EOS | Empty | Language consistency | Switch errors | Code proxy | Punctuation proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("greedy", "seeded_sampling"):
        item = summary["modes"][mode]
        lines.append(
            f"| {mode} | {item['generation_count']} | {item['repetition_rate']:.1%} | {item['exact_loop_rate']:.1%} | "
            f"{item['eos_completion_rate']:.1%} | {item['empty_output_rate']:.1%} | {item['language_id_consistency_rate']:.1%} | "
            f"{item['language_switch_error_rate']:.1%} | {item['code_structure_valid_rate']:.1%} | {item['punctuation_completion_rate']:.1%} |"
        )
    lines.extend(["", "## Prompt perplexity by category", "", "Prompt perplexity measures model fit to the controlled prompt text; it is not a semantic score for the continuation.", "", "| Category | Prompts per mode | Greedy loss | Greedy PPL | Sampling loss | Sampling PPL |", "|---|---:|---:|---:|---:|---:|"])
    for category in sorted(summary["perplexity_by_category"]["greedy"]):
        greedy = summary["perplexity_by_category"]["greedy"][category]
        sampled = summary["perplexity_by_category"]["seeded_sampling"][category]
        lines.append(f"| {category} | {greedy['prompts']} | {greedy['mean_loss']:.4f} | {greedy['perplexity']:.1f} | {sampled['mean_loss']:.4f} | {sampled['perplexity']:.1f} |")
    lines.extend(["", "## Language fit", "", "| Mode | Language | Prompts | Mean loss | Perplexity |", "|---|---|---:|---:|---:|"])
    for mode in ("greedy", "seeded_sampling"):
        for language, item in summary["perplexity_by_language"][mode].items():
            lines.append(f"| {mode} | {language} | {item['prompts']} | {item['mean_loss']:.4f} | {item['perplexity']:.1f} |")
    mem = summary["memorization_risk_indicators"]
    lines.extend([
        "",
        "## Integrity, Unicode, and bounded memorization evidence",
        "",
        f"- Greedy/sampling special-token leakage: {summary['modes']['greedy']['special_token_leakage_count']} / {summary['modes']['seeded_sampling']['special_token_leakage_count']}.",
        f"- Greedy/sampling invalid UTF-8 sequences: {summary['modes']['greedy']['unicode_health']['invalid_utf8_sequences']} / {summary['modes']['seeded_sampling']['unicode_health']['invalid_utf8_sequences']}.",
        f"- Exact train/held-out continuations in the bounded audit: {mem['exact_train_continuations']} / {mem['exact_heldout_continuations']}.",
        f"- Longest true-continuation match: {mem['longest_true_continuation_match_tokens']} tokens; longest generated n-gram found in train shards: {mem['longest_generated_ngram_in_train_tokens']} tokens.",
        "- Extraction risk is not claimed to be zero.",
        "",
        "## Interpretation boundary",
        "",
        "The complete suite still requires blinded human review. Repetition, loop, EOS, language-ID, bracket balance, punctuation, overlap, and perplexity measurements are automatic health indicators only. They do not establish factual reliability, topical coherence, or usefulness.",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    config = load_json(config_path)
    prompts = expand_suite(config)
    output_dir = Path(config["runtime_output_dir"])
    manifest_hash = prompt_manifest_hash(prompts)
    records_by_mode: dict[str, list[dict[str, Any]]] | None = None
    if output_dir.exists() and any(output_dir.iterdir()):
        summary_path = output_dir / "automatic_summary.json"
        if summary_path.is_file():
            existing = load_json(summary_path)
            if existing.get("prompt_manifest_sha256") == manifest_hash and existing.get("checkpoint_model_sha256") == FINAL_MODEL_HASH:
                _render_report(existing)
                return existing
        prompt_path = output_dir / "prompt_manifest.json"
        mode_paths = {mode: output_dir / f"{mode}_manifest.json" for mode in ("greedy", "seeded_sampling")}
        if prompt_path.is_file() and all(path.is_file() for path in mode_paths.values()):
            prompt_payload = load_json(prompt_path)
            manifests = {mode: load_json(path) for mode, path in mode_paths.items()}
            valid = (
                prompt_payload.get("prompt_manifest_sha256") == manifest_hash
                and prompt_payload.get("prompts") == prompts
                and all(item.get("prompt_manifest_sha256") == manifest_hash for item in manifests.values())
                and all(len(item.get("results", [])) == len(prompts) for item in manifests.values())
            )
            if valid:
                records_by_mode = {mode: item["results"] for mode, item in manifests.items()}
        if records_by_mode is None:
            raise FileExistsError(f"refusing to overwrite incomplete or mismatched Phase 5A evaluation: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = RUNTIME_ROOT / "manifests" / "immutable_preflight.json"
    preflight = load_json(preflight_path) if preflight_path.is_file() else immutable_preflight(config)
    if preflight.get("result") != "PASS" or preflight.get("checkpoint_model_sha256") != FINAL_MODEL_HASH:
        raise ValueError("Phase 5A immutable preflight is not reusable")
    started = time.perf_counter()
    model = None
    if records_by_mode is None:
        atomic_write_json(output_dir / "prompt_manifest.json", {"prompts": prompts, "prompt_manifest_sha256": manifest_hash})
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model_package(FINAL_CHECKPOINT / "model", device=device)
        tokenizer = FrozenTokenizer()
        losses = {prompt["id"]: _prompt_loss(model, tokenizer, prompt["prompt"]) for prompt in prompts}
        records_by_mode = {}
        for mode in ("greedy", "seeded_sampling"):
            profile = config["decoding_profiles"][mode]
            records: list[dict[str, Any]] = []
            for index, prompt in enumerate(prompts, start=1):
                record = generate_record(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=int(profile["max_new_tokens"]),
                    do_sample=bool(profile["do_sample"]),
                    profile_name=mode,
                    seed=profile.get("seed"),
                    temperature=float(profile.get("temperature", 1.0)),
                    top_p=profile.get("top_p"),
                    top_k=profile.get("top_k"),
                    checkpoint_stage="stage1_final",
                )
                add_automatic_proxies(record, tokenizer, *losses[prompt["id"]])
                records.append(record)
                if index % 25 == 0 or index == len(prompts):
                    print(f"phase5a mode={mode} prompts={index}/{len(prompts)}", flush=True)
            records_by_mode[mode] = records
            write_manifest(
                output_dir / f"{mode}_manifest.json",
                settings={"mode": mode, **profile, "checkpoint_stage": "stage1_final", "evaluation_role": "phase5a_base_review"},
                prompt_hash=manifest_hash,
                records=records,
            )
    summary = {
        "schema_version": "darkmind-v2-phase5a-base-quality-automatic-v1",
        "result": "PASS",
        "checkpoint": str(FINAL_CHECKPOINT),
        "checkpoint_model_sha256": FINAL_MODEL_HASH,
        "checkpoint_resume_state_sha256": FINAL_RESUME_HASH,
        "prompt_manifest_sha256": manifest_hash,
        "prompt_count": len(prompts),
        "generation_count": sum(len(records) for records in records_by_mode.values()),
        "runtime_output_dir": str(output_dir),
        "decoding_profiles": config["decoding_profiles"],
        "modes": {mode: automatic_metrics(records) for mode, records in records_by_mode.items()},
        "perplexity_by_language": {mode: _loss_groups(records, "language") for mode, records in records_by_mode.items()},
        "perplexity_by_category": {mode: _loss_groups(records, "category") for mode, records in records_by_mode.items()},
        "fixed_probe_loss_by_source_family": _fixed_probe_summary(),
        "memorization_risk_indicators": _memorization_summary(),
        "immutable_preflight": preflight,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_outputs_retained_outside_git": True,
        "automatic_proxies_are_human_semantic_scores": False,
        "checkpoint_modified": False,
        "training_performed": False,
    }
    atomic_write_json(output_dir / "automatic_summary.json", summary)
    _render_report(summary)
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    print(json.dumps(run_suite(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
