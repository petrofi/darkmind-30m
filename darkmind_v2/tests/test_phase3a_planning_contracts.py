import json
from pathlib import Path

import pytest

from darkmind_v2.corpus.validate_corpus_v3_targets import validate_corpus_v3_targets
from darkmind_v2.corpus.validate_source_registry_v3 import validate_source_registry_v3
from darkmind_v2.training.build_phase3a_decision_reports import WEIGHTS, score_candidates
from darkmind_v2.training.validate_base_training_stage_gates import validate_stage_gates


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "darkmind_v2" / "config"
CORPUS = ROOT / "darkmind_v2" / "corpus"
REPORTS = ROOT / "darkmind_v2" / "reports"


def test_corpus_v3_target_contract_passes() -> None:
    result = validate_corpus_v3_targets(CONFIG / "corpus_v3_targets.json")
    assert result == {"result": "PASS", "target_tokens": 500_000_000, "composition_percent": 100}


def test_corpus_v3_target_rejects_double_counted_composition(tmp_path) -> None:
    payload = json.loads((CONFIG / "corpus_v3_targets.json").read_text(encoding="utf-8"))
    payload["exclusive_composition"]["code_and_structured_text"]["target_percent"] = 5
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 100"):
        validate_corpus_v3_targets(path)


def test_source_registry_v3_counts_and_approved_budget() -> None:
    result = validate_source_registry_v3(CORPUS / "source_registry.v3.candidates.json")
    assert result["candidate_sources"] == 20
    assert result["status_counts"] == {"approved": 10, "deferred": 7, "rejected": 3}
    assert result["approved_expected_usable_tokens"] == 390_000_000
    assert result["approved_expected_download_bytes"] == 31_119_000_000


def test_source_registry_v3_rejects_approval_of_banned_web_crawl(tmp_path) -> None:
    payload = json.loads((CORPUS / "source_registry.v3.candidates.json").read_text(encoding="utf-8"))
    source = next(item for item in payload["sources"] if item["id"] == "common_crawl_bulk")
    source.update(
        {
            "approval_status": "approved",
            "rejection_reason": None,
            "expected_download_bytes": 1,
            "expected_usable_tokens": 1,
            "maximum_source_cap_tokens": 1,
            "trust_tier": "C",
        }
    )
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden source"):
        validate_source_registry_v3(path)


def test_stage_gate_contract_passes_and_blocks_sft() -> None:
    result = validate_stage_gates(CONFIG / "base_training_stage_gates.json")
    assert result == {"result": "PASS", "stages": 6, "final_tokens": 500_000_000}
    payload = json.loads((CONFIG / "base_training_stage_gates.json").read_text(encoding="utf-8"))
    assert payload["sft_allowed_before_required_base_stage"] is False
    assert payload["release_policy"]["instruct_training_requires_explicit_base_checkpoint_approval"] is True


def test_weighted_decision_uses_required_weights_and_tie_break() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    matrix = json.loads((REPORTS / "phase3a_architecture_parameter_matrix.json").read_text(encoding="utf-8"))
    benchmark = json.loads((REPORTS / "phase3a_rtx4060_benchmark.json").read_text(encoding="utf-8"))
    decision = score_candidates(matrix, benchmark)
    assert decision["rows"][0]["candidate"] == "D"
    assert decision["rows"][0]["weighted_score"] - decision["rows"][1]["weighted_score"] <= 3.0
    assert decision["selected"] == "C"
    assert decision["tie_break_applied"] is True


@pytest.mark.parametrize(
    "report_name",
    [
        "phase3a_base_candidate_decision.md",
        "phase3a_corpus_v3_design.md",
        "phase3a_corpus_source_gap.md",
        "phase3a_release_strategy.md",
        "phase3a_model_license_options.md",
        "phase3a_training_budget.md",
    ],
)
def test_phase3a_reports_preserve_planning_only_scope(report_name) -> None:
    text = (REPORTS / report_name).read_text(encoding="utf-8").lower()
    assert "hugging face upload completed" not in text
    assert "status: production-ready" not in text
    assert "license decision: finalized" not in text
    assert "no corpus v3 data downloaded" in text or "not downloaded" in text or "not training authorization" in text or "not frozen" in text or "no automatic hugging face upload" in text or "no model-weight license is finalized" in text
