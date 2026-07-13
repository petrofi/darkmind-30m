"""Build the exact Phase 3A architecture matrix from real model configs."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from darkmind_v2.modeling.configuration_darkmind_v2 import DarkMindV2Config
from darkmind_v2.modeling.estimate_model_size import estimate_model_size
from darkmind_v2.modeling.modeling_darkmind_v2 import DarkMindV2ForCausalLM


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "darkmind_v2" / "config"
REPORT_DIR = ROOT / "darkmind_v2" / "reports"
CANDIDATES = {
    "A": ("60M", CONFIG_DIR / "model_base_candidate_a_60m.json", 60_000_000),
    "B": ("80M", CONFIG_DIR / "model_base_candidate_b_80m.json", 80_000_000),
    "C": ("100M", CONFIG_DIR / "model_base_candidate_c_100m.json", 100_000_000),
    "D": ("120M", CONFIG_DIR / "model_base_candidate_d_120m.json", 120_000_000),
}


def _memory_estimates(config: DarkMindV2Config, total_parameters: int) -> dict[str, int]:
    sequence = config.block_size
    logits = sequence * config.vocab_size * 2
    hidden_workspace = sequence * config.n_embd * 2 * 8
    attention_workspace = config.n_head * sequence * sequence * 2
    inference = total_parameters * 2 + logits + hidden_workspace + attention_workspace
    retained_activations = config.n_layer * (
        sequence * config.n_embd * 2 * 16
        + config.n_head * sequence * sequence * 2 * 2
    )
    training = total_parameters * 16 + retained_activations + logits * 2
    return {
        "estimated_inference_bytes": inference,
        "estimated_training_bytes_before_measurement": training,
        "estimation_micro_batch": 1,
        "estimation_sequence_length": sequence,
    }


def build_matrix(*, verify_implementation: bool = True) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for candidate_id, (target_class, path, target_parameters) in CANDIDATES.items():
        config = DarkMindV2Config.from_json_file(path)
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
        implementation_count = None
        if verify_implementation:
            model = DarkMindV2ForCausalLM(config)
            implementation_count = model.parameter_count()
            if implementation_count != estimate.total_params:
                raise RuntimeError(
                    f"candidate {candidate_id} formula/model mismatch: "
                    f"{estimate.total_params} != {implementation_count}"
                )
            del model
            gc.collect()
        deviation = (estimate.total_params - target_parameters) / target_parameters * 100
        candidate = {
            "candidate": candidate_id,
            "target_class": target_class,
            "target_parameters": target_parameters,
            "config_path": path.relative_to(ROOT).as_posix(),
            "config": config.architecture_dict(),
            "estimate": asdict(estimate),
            "implementation_parameter_count": implementation_count,
            "implementation_count_verified": verify_implementation,
            "target_deviation_percent": round(deviation, 6),
            "within_five_percent": abs(deviation) <= 5.0,
            **_memory_estimates(config, estimate.total_params),
        }
        candidates.append(candidate)
    return {
        "schema_version": "darkmind-v2-phase3a-architecture-matrix-v1",
        "result": "PASS" if all(item["within_five_percent"] for item in candidates) else "FAIL",
        "tokenizer": "darkmind_v2_sp_bpe24k_v1",
        "vocabulary_size": 24000,
        "context_length": 512,
        "counting_policy": "exact formula verified against instantiated implementation",
        "memory_policy": "analytical pre-benchmark estimates; measured CUDA evidence supersedes these values",
        "candidates": candidates,
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Phase 3A Architecture Parameter Matrix",
        "",
        "All candidates use the frozen 24k tokenizer, tied embeddings, context 512, pre-layer normalization, and head dimension 64. Counts were verified by instantiating the real implementation one candidate at a time.",
        "",
        "| ID | Class | Layers | Width | Heads | Exact params | Transformer body | Vocabulary share | BF16 weights | AdamW state |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in matrix["candidates"]:
        config = item["config"]
        estimate = item["estimate"]
        lines.append(
            f"| {item['candidate']} | {item['target_class']} | {config['n_layer']} | "
            f"{config['n_embd']} | {config['n_head']} | {estimate['total_params']:,} | "
            f"{estimate['transformer_body_params']:,} | {estimate['vocab_related_percentage']:.4f}% | "
            f"{estimate['bf16_checkpoint_bytes'] / 2**20:.2f} MiB | "
            f"{estimate['adamw_state_bytes'] / 2**20:.2f} MiB |"
        )
    lines.extend(["", "## Exact Breakdown", ""])
    for item in matrix["candidates"]:
        estimate = item["estimate"]
        lines.extend(
            [
                f"### Candidate {item['candidate']} ({item['target_class']})",
                "",
                f"- Token embeddings: {estimate['token_embedding_params']:,}",
                f"- Positional embeddings: {estimate['position_embedding_params']:,}",
                f"- Attention (weights and biases): {estimate['attention_params']:,}",
                f"- MLP (weights and biases): {estimate['mlp_params']:,}",
                f"- Normalization parameters: {estimate['layer_norm_params']:,}",
                f"- Normalization plus all bias parameters: {estimate['normalization_and_bias_params']:,}",
                f"- LM-head-only parameters: {estimate['lm_head_params']:,} (tied)",
                f"- Total: {estimate['total_params']:,}",
                f"- Target deviation: {item['target_deviation_percent']:+.4f}%",
                f"- Estimated inference memory: {item['estimated_inference_bytes'] / 2**20:.2f} MiB",
                f"- Estimated training memory before measurement: {item['estimated_training_bytes_before_measurement'] / 2**30:.2f} GiB",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Candidate B uses 13 layers rather than the 14-layer starting shape because the 14-layer implementation exceeds the 80M class tolerance. All vocabulary shares are below the preferred 20% threshold. Memory values are conservative analytical planning estimates, not substitutes for the RTX 4060 measurements.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    matrix = build_matrix(verify_implementation=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "phase3a_architecture_parameter_matrix.json").write_text(
        json.dumps(matrix, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (REPORT_DIR / "phase3a_architecture_parameter_matrix.md").write_text(
        render_markdown(matrix), encoding="utf-8"
    )
    print(json.dumps({"result": matrix["result"], "candidates": len(matrix["candidates"])}, indent=2))


if __name__ == "__main__":
    main()
