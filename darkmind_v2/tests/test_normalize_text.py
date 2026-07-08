from darkmind_v2.corpus.normalize_text import normalize_text


def test_unicode_nfc_normalization() -> None:
    normalized, modifications = normalize_text("Tu\u0308rkiye")
    assert normalized == "Türkiye"
    assert any(item.reason == "applied Unicode NFC normalization" for item in modifications)


def test_turkish_characters_are_preserved() -> None:
    text = "çğıİöşü ÇĞIİÖŞÜ"
    normalized, modifications = normalize_text(text)
    assert normalized == text
    assert modifications == []


def test_control_characters_are_removed_without_transliteration() -> None:
    normalized, modifications = normalize_text("Kullanıcı\x00 adı\x01")
    assert normalized == "Kullanıcı adı"
    assert {item.reason for item in modifications} == {"removed null byte", "removed unsafe control character"}

