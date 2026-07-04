"""UTF-8 and Unicode normalization helpers for DarkMind v2 corpora."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SAFE_CONTROL_CHARS = {"\n", "\t"}


@dataclass(frozen=True)
class Modification:
    line_number: int
    reason: str
    before: str
    after: str


def is_unsafe_control_char(char: str) -> bool:
    return unicodedata.category(char) == "Cc" and char not in SAFE_CONTROL_CHARS


def read_utf8_text(path: Path, *, repair_invalid_utf8: bool = False) -> tuple[str, list[Modification]]:
    raw = path.read_bytes()
    if not repair_invalid_utf8:
        return raw.decode("utf-8"), []

    decoded = raw.decode("utf-8", errors="replace")
    modifications: list[Modification] = []
    if "\ufffd" in decoded:
        modifications.append(
            Modification(
                line_number=0,
                reason="invalid utf-8 bytes decoded with replacement because repair mode was enabled",
                before="<raw-bytes>",
                after="<contains U+FFFD>",
            )
        )
    return decoded, modifications


def normalize_line(line: str, line_number: int) -> tuple[str, list[Modification]]:
    modifications: list[Modification] = []
    current = line.replace("\x00", "")
    if current != line:
        modifications.append(Modification(line_number, "removed null byte", line, current))

    without_controls = "".join(char for char in current if not is_unsafe_control_char(char))
    if without_controls != current:
        modifications.append(Modification(line_number, "removed unsafe control character", current, without_controls))
    current = without_controls

    normalized = unicodedata.normalize("NFC", current)
    if normalized != current:
        modifications.append(Modification(line_number, "applied Unicode NFC normalization", current, normalized))
    return normalized, modifications


def normalize_text(text: str) -> tuple[str, list[Modification]]:
    modifications: list[Modification] = []
    if "\r" in text:
        normalized_endings = text.replace("\r\n", "\n").replace("\r", "\n")
        modifications.append(Modification(0, "normalized line endings", "<document>", "<document>"))
        text = normalized_endings

    normalized_lines: list[str] = []
    lines = text.split("\n")
    for index, line in enumerate(lines, start=1):
        normalized_line, line_modifications = normalize_line(line, index)
        normalized_lines.append(normalized_line)
        modifications.extend(line_modifications)
    return "\n".join(normalized_lines), modifications


def normalize_json_value(value: Any, line_number: int, modifications: list[Modification]) -> Any:
    if isinstance(value, str):
        normalized, value_modifications = normalize_text(value)
        modifications.extend(
            Modification(line_number, item.reason, item.before, item.after) for item in value_modifications
        )
        return normalized
    if isinstance(value, list):
        return [normalize_json_value(item, line_number, modifications) for item in value]
    if isinstance(value, dict):
        return {key: normalize_json_value(item, line_number, modifications) for key, item in value.items()}
    return value


def normalize_jsonl(text: str) -> tuple[str, list[Modification]]:
    output_lines: list[str] = []
    modifications: list[Modification] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            output_lines.append("")
            continue
        record = json.loads(line)
        normalized_record = normalize_json_value(record, line_number, modifications)
        output_lines.append(json.dumps(normalized_record, ensure_ascii=False, sort_keys=True))
    return "\n".join(output_lines) + ("\n" if text.endswith("\n") else ""), modifications


def normalize_file(
    input_path: Path,
    *,
    output_path: Path | None = None,
    report_path: Path | None = None,
    repair_invalid_utf8: bool = False,
) -> dict[str, Any]:
    text, modifications = read_utf8_text(input_path, repair_invalid_utf8=repair_invalid_utf8)
    if input_path.suffix.lower() == ".jsonl":
        normalized, json_modifications = normalize_jsonl(text)
        modifications.extend(json_modifications)
    else:
        normalized, text_modifications = normalize_text(text)
        modifications.extend(text_modifications)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(normalized, encoding="utf-8", newline="\n")

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path) if output_path else None,
        "repair_invalid_utf8": repair_invalid_utf8,
        "modification_count": len(modifications),
        "modifications": [asdict(item) for item in modifications],
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize DarkMind v2 corpus text without silent repair.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--repair-invalid-utf8", action="store_true")
    args = parser.parse_args()

    try:
        report = normalize_file(
            args.input,
            output_path=args.out,
            report_path=args.report,
            repair_invalid_utf8=args.repair_invalid_utf8,
        )
    except UnicodeDecodeError as exc:
        print(f"Invalid UTF-8 rejected: {exc}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
