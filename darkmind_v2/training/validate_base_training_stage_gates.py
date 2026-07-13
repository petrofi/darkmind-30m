"""Validate the ordered production-base quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TOKENS = [0, 5_000_000, 25_000_000, 100_000_000, 250_000_000, 500_000_000]


def validate_stage_gates(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "darkmind-v2-base-training-stage-gates-v1":
        raise ValueError("unsupported stage-gate schema")
    stages = payload.get("stages")
    if not isinstance(stages, list) or [stage.get("tokens") for stage in stages] != EXPECTED_TOKENS:
        raise ValueError("base-training stages are incomplete or out of order")
    for stage in stages:
        if stage.get("blocking") is not True:
            raise ValueError(f"stage must be blocking: {stage.get('id')}")
        checks = stage.get("checks")
        if not isinstance(checks, list) or len(checks) < 5 or len(checks) != len(set(checks)):
            raise ValueError(f"stage checks are incomplete or duplicated: {stage.get('id')}")
    if payload.get("sft_allowed_before_required_base_stage") is not False:
        raise ValueError("SFT must remain blocked before base approval")
    release = payload.get("release_policy", {})
    if release.get("public_release_requires_stage_5") is not True:
        raise ValueError("public release must require Stage 5")
    return {"result": "PASS", "stages": len(stages), "final_tokens": EXPECTED_TOKENS[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_stage_gates(args.path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
