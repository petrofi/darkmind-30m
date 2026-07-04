from darkmind_v2.corpus.detect_mojibake import detect_text


def test_detects_common_turkish_mojibake() -> None:
    findings = detect_text("TÃ¼rkiye ve KullanÃ„Â±cÃ„Â± kaydı")
    substrings = {item.suspicious_substring for item in findings}
    assert "TÃ¼rkiye" in substrings
    assert "KullanÃ„Â±cÃ„Â±" in substrings
    assert any(item.probable_original == "Türkiye" for item in findings)


def test_detects_replacement_character() -> None:
    findings = detect_text("bozuk � karakter")
    assert findings
    assert findings[0].severity == "critical"
    assert findings[0].automatic_repair_safe is False

