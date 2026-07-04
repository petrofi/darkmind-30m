"""Mojibake detection for DarkMind v2 corpus and tokenizer audits."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


EXPLICIT_REPAIRS = {
    "TÃ¼rkiye": "Türkiye",
    "TÃƒÂ¼rkiye": "Türkiye",
    "KullanÃ„Â±cÃ„Â±": "Kullanıcı",
    "KullanÄ±cÄ±": "Kullanıcı",
    "GÃ¼venlik": "Güvenlik",
    "GÃƒÂ¼venlik": "Güvenlik",
    "Ã§": "ç",
    "ÃƒÂ§": "ç",
    "Ã¶": "ö",
    "ÃƒÂ¶": "ö",
    "Ã¼": "ü",
    "ÃƒÂ¼": "ü",
    "Ã„Â±": "ı",
    "Ä±": "ı",
    "Ã…Å¸": "ş",
    "ÅŸ": "ş",
}

MOJIBAKE_CLUSTER_RE = re.compile(r"(?:Ã.|Â.|Ä.|Å.|â€[^\s]*|ï¿½){1,}")
REPEATED_BROKEN_RE = re.compile(r"((?:Ã.|Â.|Ä.|Å.){2,})")


@dataclass(frozen=True)
class MojibakeFinding:
    line_number: int
    suspicious_substring: str
    probable_original: str | None
    severity: str
    automatic_repair_safe: bool


def guess_repair(text: str) -> str | None:
    current = text
    for _ in range(3):
        try:
            repaired = current.encode("latin-1").decode("utf-8")
        except UnicodeError:
            try:
                repaired = current.encode("cp1252").decode("utf-8")
            except UnicodeError:
                return None
        if repaired == current:
            return None
        current = repaired
        if not looks_like_mojibake(current):
            return current
    return current if current != text else None


def looks_like_mojibake(text: str) -> bool:
    if "\ufffd" in text:
        return True
    if any(marker in text for marker in ("Ã", "Â", "Ä", "Å", "â€", "ï¿½")):
        return True
    return False


def detect_line(text: str, line_number: int) -> list[MojibakeFinding]:
    findings: list[MojibakeFinding] = []
    seen: set[str] = set()

    if "\ufffd" in text:
        findings.append(MojibakeFinding(line_number, "\ufffd", None, "critical", False))
        seen.add("\ufffd")

    for suspicious, original in EXPLICIT_REPAIRS.items():
        if suspicious in text and suspicious not in seen:
            findings.append(MojibakeFinding(line_number, suspicious, original, "high", True))
            seen.add(suspicious)

    for match in REPEATED_BROKEN_RE.finditer(text):
        substring = match.group(1)
        if substring not in seen:
            repaired = guess_repair(substring)
            findings.append(MojibakeFinding(line_number, substring, repaired, "high", repaired is not None))
            seen.add(substring)

    for match in MOJIBAKE_CLUSTER_RE.finditer(text):
        substring = match.group(0)
        if substring not in seen:
            repaired = EXPLICIT_REPAIRS.get(substring) or guess_repair(substring)
            severity = "medium" if repaired else "high"
            findings.append(MojibakeFinding(line_number, substring, repaired, severity, repaired is not None))
            seen.add(substring)

    return findings


def detect_text(text: str) -> list[MojibakeFinding]:
    findings: list[MojibakeFinding] = []
    for line_number, line in enumerate(text.splitlines() or [text], start=1):
        findings.extend(detect_line(line, line_number))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect mojibake in UTF-8 text.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    findings = detect_text(text)
    if args.json:
        print(json.dumps([asdict(item) for item in findings], ensure_ascii=False, indent=2, sort_keys=True))
    else:
        for item in findings:
            print(
                f"line={item.line_number} severity={item.severity} "
                f"safe_repair={item.automatic_repair_safe} "
                f"substring={item.suspicious_substring!r} probable={item.probable_original!r}"
            )
    raise SystemExit(1 if findings else 0)


if __name__ == "__main__":
    main()
