from pathlib import Path
import argparse
from collections import Counter
import json


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "data" / "sources"
RAW_DIR = ROOT_DIR / "data" / "raw_collected"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "dataset_manifest.jsonl"

KNOWN_RAW_TYPES = {
    "turkish_notes": "manual",
    "coding_notes": "generated",
    "python_examples": "generated",
    "qa_pairs": "generated",
}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def relative_posix(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


def classify_source_file(path: Path) -> dict:
    return {
        "path": relative_posix(path),
        "type": "legacy_source",
        "language": "tr",
        "source": "manual",
        "license": "project-generated",
        "status": "approved",
        "notes": "Existing hand-curated project source file.",
    }


def classify_web_text(path: Path, parts: tuple[str, ...]) -> dict:
    status = "needs_review"
    license_name = "needs_review"
    source = "unknown"
    notes = "Web text requires source and license review before training use."

    if "approved" in parts:
        status = "approved"
        license_name = "reviewed-source"
        source = "approved_web"
        notes = "Approved web text. Keep source/license metadata with the file."
    elif "pending_review" in parts:
        source = "approved_web"
        notes = "Pending review web text. Do not train on this file yet."
    elif "rejected" in parts:
        status = "rejected"
        notes = "Rejected web text. Do not train on this file."

    return {
        "path": relative_posix(path),
        "type": "web_text",
        "language": "tr",
        "source": source,
        "license": license_name,
        "status": status,
        "notes": notes,
    }


def classify_raw_file(path: Path) -> dict:
    relative_raw = path.relative_to(RAW_DIR)
    parts = relative_raw.parts
    data_type = parts[0] if parts else "unknown"

    if data_type == "web_text":
        return classify_web_text(path, parts)

    if data_type in KNOWN_RAW_TYPES:
        return {
            "path": relative_posix(path),
            "type": data_type,
            "language": "tr",
            "source": KNOWN_RAW_TYPES[data_type],
            "license": "project-generated",
            "status": "approved",
            "notes": f"Known raw_collected/{data_type} project dataset file.",
        }

    return {
        "path": relative_posix(path),
        "type": data_type,
        "language": "tr",
        "source": "unknown",
        "license": "needs_review",
        "status": "needs_review",
        "notes": "Unknown raw collection folder. Review before training use.",
    }


def collect_manifest_entries() -> list[dict]:
    entries = []

    if SOURCE_DIR.exists():
        for path in sorted(SOURCE_DIR.glob("*.txt")):
            entries.append(classify_source_file(path))

    if RAW_DIR.exists():
        for path in sorted(RAW_DIR.rglob("*.txt")):
            entries.append(classify_raw_file(path))

    return entries


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a JSONL manifest for DarkMind dataset source files."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH.relative_to(ROOT_DIR)),
        help="Manifest output path.",
    )
    args = parser.parse_args()

    output_path = resolve_path(args.output_path)
    entries = collect_manifest_entries()
    write_jsonl(output_path, entries)

    status_counts = Counter(entry["status"] for entry in entries)

    print("=" * 70)
    print(f"Manifest saved: {output_path}")
    print(f"Total files: {len(entries):,}")
    print("Status counts:")

    for status, count in sorted(status_counts.items()):
        print(f"- {status}: {count:,}")

    print("=" * 70)


if __name__ == "__main__":
    main()
