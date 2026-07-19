"""Guard the Phase 4E local export; a conditional gate stop cannot export."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from darkmind_v2.training.phase4e_stage3 import FINAL_STEP, RUN_DIR, load_json


FORBIDDEN_EXPORT_PARTS = {
    "corpus_v3_tokenized",
    "document_boundaries.jsonl",
    "attribution_manifest.jsonl",
    "split_manifest.jsonl",
    "shard_checksums.json",
    "metrics.jsonl",
    "resume_state.pt",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pt",
}


def validate_export_file_list(paths: Iterable[str | Path]) -> None:
    for value in paths:
        path = Path(value)
        lowered = {part.lower() for part in path.parts}
        if lowered & FORBIDDEN_EXPORT_PARTS or path.suffix.lower() == ".bin":
            raise ValueError(f"forbidden corpus/runtime file in Phase 4E export: {path}")


def require_export_preconditions(run_dir: Path = RUN_DIR) -> dict[str, Any]:
    progress = load_json(run_dir / "progress.json")
    if int(progress["optimizer_step"]) != FINAL_STEP:
        raise PermissionError("Phase 4E local export requires the final no-wrap checkpoint")
    summary = load_json(run_dir / "final_assessment.json")
    if summary.get("classification") != "STRONG PASS":
        raise PermissionError("Phase 4E local export requires a final Strong PASS")
    memorization = load_json(run_dir / "memorization_audit.json")
    if memorization.get("result") != "PASS" or memorization.get("hard_release_blockers"):
        raise PermissionError("Phase 4E memorization hard blockers are not cleared")
    generation = load_json(run_dir / "generation_analysis.json")
    if generation.get("hard_failure_total") != 0:
        raise PermissionError("Phase 4E generation hard failures block export")
    return summary


def export_stage3() -> None:
    require_export_preconditions()
    raise NotImplementedError("Phase 4E stopped at 75M; no export implementation was activated")
