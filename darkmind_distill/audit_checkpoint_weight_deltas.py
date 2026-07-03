from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT_DIR / "models" / "darkmind-30m-10k-step15000.pt"
DEFAULT_STUDENT = ROOT_DIR / "models" / "darkmind-30m-qwen-distill-pilot500-tr-en-v2.pt"
DEFAULT_REPORT = ROOT_DIR / "darkmind_distill" / "reports" / "checkpoint_weight_delta_audit.md"
DEFAULT_JSON = ROOT_DIR / "darkmind_distill" / "reports" / "checkpoint_weight_delta_audit.json"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path


def load_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint: {path}")
    return state


def module_group(key: str) -> str:
    if key.startswith("token_embedding."):
        return "token_embedding"
    if key.startswith("position_embedding."):
        return "position_embedding"
    if key.startswith("blocks.0."):
        return "first_transformer_block"
    if key.startswith("blocks.7."):
        return "final_transformer_block"
    if key.startswith("ln_f."):
        return "final_layer_norm"
    if key.startswith("lm_head."):
        return "lm_head"
    if key.startswith("blocks."):
        parts = key.split(".")
        return ".".join(parts[:2])
    return key.split(".")[0]


def tensor_delta(base: torch.Tensor, student: torch.Tensor) -> dict[str, float]:
    base_f = base.float()
    student_f = student.float()
    delta = student_f - base_f
    delta_norm = float(torch.linalg.vector_norm(delta).item())
    base_norm = float(torch.linalg.vector_norm(base_f).item())
    return {
        "absolute_delta_norm": delta_norm,
        "relative_delta_norm": delta_norm / max(base_norm, 1e-12),
        "max_absolute_change": float(delta.abs().max().item()),
        "base_norm": base_norm,
    }


def aggregate(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    absolute = sum(item["absolute_delta_norm"] ** 2 for item in items) ** 0.5
    base = sum(item["base_norm"] ** 2 for item in items) ** 0.5
    return {
        "absolute_delta_norm": absolute,
        "relative_delta_norm": absolute / max(base, 1e-12),
        "max_absolute_change": max(item["max_absolute_change"] for item in items),
        "base_norm": base,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base and student checkpoint weight deltas.")
    parser.add_argument("--base", default=str(DEFAULT_BASE.relative_to(ROOT_DIR)))
    parser.add_argument("--student", default=str(DEFAULT_STUDENT.relative_to(ROOT_DIR)))
    parser.add_argument("--report", default=str(DEFAULT_REPORT.relative_to(ROOT_DIR)))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON.relative_to(ROOT_DIR)))
    args = parser.parse_args()

    base_path = resolve_path(args.base)
    student_path = resolve_path(args.student)
    report_path = resolve_path(args.report)
    json_path = resolve_path(args.json_out)
    base_state = load_state(base_path)
    student_state = load_state(student_path)

    per_tensor: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    missing: list[str] = []
    shape_mismatch: list[str] = []

    for key, base_tensor in base_state.items():
        student_tensor = student_state.get(key)
        if student_tensor is None:
            missing.append(key)
            continue
        if tuple(base_tensor.shape) != tuple(student_tensor.shape):
            shape_mismatch.append(key)
            continue
        stats = tensor_delta(base_tensor, student_tensor)
        per_tensor[key] = {"shape": list(base_tensor.shape), **stats}
        groups[module_group(key)].append(stats)

    group_stats = {group: aggregate(items) for group, items in sorted(groups.items())}
    payload = {
        "base": str(base_path),
        "student": str(student_path),
        "missing_in_student": missing,
        "shape_mismatches": shape_mismatch,
        "groups": group_stats,
        "per_tensor": per_tensor,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    focus = [
        "token_embedding",
        "position_embedding",
        "first_transformer_block",
        "final_transformer_block",
        "final_layer_norm",
        "lm_head",
    ]
    lines = [
        "# Checkpoint Weight Delta Audit",
        "",
        f"Base: `{base_path}`",
        f"Student: `{student_path}`",
        f"Missing keys in student: `{missing}`",
        f"Shape mismatches: `{shape_mismatch}`",
        "",
        "## Focus Modules",
        "",
        "| Module | Abs Delta Norm | Relative Delta Norm | Max Abs Change | Base Norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in focus:
        stats = group_stats.get(group, {})
        lines.append(
            f"| {group} | {stats.get('absolute_delta_norm', 0):.6f} | "
            f"{stats.get('relative_delta_norm', 0):.6f} | "
            f"{stats.get('max_absolute_change', 0):.6f} | {stats.get('base_norm', 0):.6f} |"
        )
    lines.extend(["", "## All Module Groups", "", "| Module | Abs Delta Norm | Relative Delta Norm | Max Abs Change |", "|---|---:|---:|---:|"])
    for group, stats in group_stats.items():
        lines.append(
            f"| {group} | {stats.get('absolute_delta_norm', 0):.6f} | "
            f"{stats.get('relative_delta_norm', 0):.6f} | {stats.get('max_absolute_change', 0):.6f} |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"Report written: {report_path}")
    print(f"JSON written: {json_path}")


if __name__ == "__main__":
    main()
