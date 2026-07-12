import json
from pathlib import Path

from darkmind_v2.export.validate_public_preview_package import validate_model_card


def test_source_model_card_contains_all_public_preview_disclosures() -> None:
    text = Path("darkmind_v2/export/MODEL_CARD_STAGE1.md").read_text(encoding="utf-8")
    assert validate_model_card(text) == []


def test_corpus_attribution_summary_lists_all_sources_and_licenses() -> None:
    payload = json.loads(Path("darkmind_v2/config/corpus_attribution_summary.json").read_text(encoding="utf-8"))
    assert len(payload["sources"]) == 7
    assert all(source["license_id"] and source["license_url"] for source in payload["sources"])
    assert payload["full_attribution_manifest_sha256"] == "b820ec56a0c173604a5a97663c1ca510c7c78800d3e559722b23ffb74eb3120f"
