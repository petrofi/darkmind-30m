"""Controlled downloader for DarkMind v2 Phase 1B approved sources."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_REGISTRY = Path(__file__).with_name("source_registry.phase1b.json")
DEFAULT_DATA_DIR = Path("darkmind_v2/data/phase1b")
CHUNK_SIZE = 1024 * 1024
TOTAL_DOWNLOAD_CAP_BYTES = 1_000_000_000


@dataclass(frozen=True)
class DownloadResult:
    source_id: str
    url: str
    final_url: str
    path: str
    compressed_bytes: int
    sha256: str
    md5: str
    sha1: str
    official_checksums_verified: bool
    resumed: bool
    elapsed_seconds: float


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def append_jsonl_atomic(path: Path, payload: dict[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    atomic_write_text(path, existing + line)


def host_matches(host: str, allowed_domains: set[str]) -> bool:
    host = host.lower()
    return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)


def require_allowed_url(url: str, allowed_domains: set[str], source_id: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"{source_id}: only HTTPS URLs are allowed: {url}")
    if not host_matches(parsed.netloc, allowed_domains):
        raise ValueError(f"{source_id}: host {parsed.netloc!r} is not allowlisted")


def head_request(url: str, allowed_domains: set[str], source_id: str) -> tuple[str, dict[str, str]]:
    require_allowed_url(url, allowed_domains, source_id)
    request = Request(url, method="HEAD", headers={"User-Agent": "DarkMind-v2-Phase1B/1.0"})
    with urlopen(request, timeout=60) as response:
        final_url = response.geturl()
        require_allowed_url(final_url, allowed_domains, source_id)
        return final_url, {key.lower(): value for key, value in response.headers.items()}


def compute_hashes(path: Path) -> dict[str, str]:
    hashers = {"sha256": hashlib.sha256(), "md5": hashlib.md5(), "sha1": hashlib.sha1()}
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            for hasher in hashers.values():
                hasher.update(chunk)
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def verify_official_checksums(source: dict[str, Any], hashes: dict[str, str]) -> bool:
    official = source.get("official_checksums") or {}
    for algorithm, expected in official.items():
        if expected and hashes.get(algorithm) != str(expected).lower():
            raise ValueError(f"{source['source_id']}: {algorithm} checksum mismatch")
    return bool(official)


def final_archive_bytes(archive_dir: Path) -> int:
    if not archive_dir.exists():
        return 0
    return sum(path.stat().st_size for path in archive_dir.iterdir() if path.is_file() and not path.name.endswith(".partial"))


def open_stream(url: str, allowed_domains: set[str], source_id: str, resume_from: int):
    headers = {"User-Agent": "DarkMind-v2-Phase1B/1.0"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    response = urlopen(Request(url, headers=headers), timeout=120)
    require_allowed_url(response.geturl(), allowed_domains, source_id)
    return response


def download_source(source: dict[str, Any], archive_dir: Path, ledger_path: Path, total_cap: int) -> DownloadResult:
    source_id = source["source_id"]
    if source.get("approved") is not True:
        raise ValueError(f"{source_id}: source is not approved")
    url = source.get("official_download_url")
    if not url:
        raise ValueError(f"{source_id}: official_download_url is required")

    allowed_domains = {domain.lower() for domain in source.get("official_domains", [])}
    final_url, headers = head_request(url, allowed_domains, source_id)
    expected = int(source.get("expected_compressed_bytes") or source.get("estimated_download_size_bytes") or 0)
    source_cap = int(source.get("max_download_bytes") or 0)
    content_length = headers.get("content-length")
    if content_length:
        size = int(content_length)
        if expected and size != expected:
            raise ValueError(f"{source_id}: Content-Length {size} does not match expected {expected}")
        if source_cap and size > source_cap:
            raise ValueError(f"{source_id}: Content-Length exceeds max_download_bytes")

    archive_dir.mkdir(parents=True, exist_ok=True)
    final_path = archive_dir / str(source.get("local_archive_name") or source["filename"])
    partial_path = final_path.with_suffix(final_path.suffix + ".partial")
    if final_path.exists():
        hashes = compute_hashes(final_path)
        verified = verify_official_checksums(source, hashes)
        return DownloadResult(source_id, url, final_url, str(final_path), final_path.stat().st_size, hashes["sha256"], hashes["md5"], hashes["sha1"], verified, False, 0.0)

    existing_bytes = final_archive_bytes(archive_dir)
    if expected and existing_bytes + expected > total_cap:
        raise ValueError(f"{source_id}: download would exceed total Phase 1B cap")

    resume_from = partial_path.stat().st_size if partial_path.exists() else 0
    resumed = False
    if resume_from and headers.get("accept-ranges", "").casefold() == "bytes":
        resumed = True
    elif resume_from:
        partial_path.unlink()
        resume_from = 0

    start = time.monotonic()
    written = resume_from
    mode = "ab" if resume_from else "wb"
    with open_stream(final_url, allowed_domains, source_id, resume_from) as response:
        if resume_from and getattr(response, "status", None) != 206:
            raise ValueError(f"{source_id}: server did not honor resume range")
        with partial_path.open(mode) as handle:
            for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                written += len(chunk)
                if source_cap and written > source_cap:
                    raise ValueError(f"{source_id}: source cap exceeded during download")
                if existing_bytes + written > total_cap:
                    raise ValueError(f"{source_id}: total download cap exceeded")
                handle.write(chunk)

    if expected and written != expected:
        raise ValueError(f"{source_id}: downloaded {written} bytes, expected {expected}")
    partial_path.replace(final_path)
    hashes = compute_hashes(final_path)
    verified = verify_official_checksums(source, hashes)
    result = DownloadResult(
        source_id,
        url,
        final_url,
        str(final_path),
        final_path.stat().st_size,
        hashes["sha256"],
        hashes["md5"],
        hashes["sha1"],
        verified,
        resumed,
        round(time.monotonic() - start, 3),
    )
    append_jsonl_atomic(ledger_path, asdict(result))
    return result


def safe_target(root: Path, member_name: str) -> Path:
    target = (root / member_name).resolve()
    root = root.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"archive path traversal blocked: {member_name}")
    return target


def safe_extract_tar(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            target = safe_target(out_dir, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def safe_extract_bz2(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = safe_target(out_dir, archive_path.name.removesuffix(".bz2"))
    with bz2.open(archive_path, "rb") as source, target.open("wb") as handle:
        shutil.copyfileobj(source, handle)


def run_downloads(registry_path: Path, data_dir: Path, source_ids: set[str] | None, extract: bool) -> dict[str, Any]:
    registry = load_registry(registry_path)
    total_cap = min(int(registry["hard_download_cap_bytes"]), TOTAL_DOWNLOAD_CAP_BYTES)
    archive_dir = data_dir / "raw" / "archives"
    ledger_path = data_dir / "download_ledger.jsonl"
    results = []
    for source in registry["sources"]:
        if source_ids and source["source_id"] not in source_ids:
            continue
        result = download_source(source, archive_dir, ledger_path, total_cap)
        results.append(result)
        if extract:
            out_dir = data_dir / "raw" / "extracted" / source["source_id"]
            if source.get("archive_type", "").startswith("tar_"):
                safe_extract_tar(Path(result.path), out_dir)
            elif source.get("archive_type") == "bz2_xml":
                safe_extract_bz2(Path(result.path), out_dir)
    total_archive_bytes = final_archive_bytes(archive_dir)
    payload = {
        "result": "PASS",
        "registry": str(registry_path),
        "downloaded_source_count": len(results),
        "selected_source_bytes": sum(item.compressed_bytes for item in results),
        "total_downloaded_bytes": total_archive_bytes,
        "hard_download_cap_bytes": total_cap,
        "remaining_download_cap_bytes": total_cap - total_archive_bytes,
        "downloads": [asdict(item) for item in results],
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(data_dir / "download_summary.json", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Download approved DarkMind v2 Phase 1B sources.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--source-id", action="append", default=None)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    try:
        payload = run_downloads(args.registry, args.data_dir, set(args.source_id) if args.source_id else None, args.extract)
    except (HTTPError, URLError, OSError, ValueError) as exc:
        print(json.dumps({"result": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2, sort_keys=True))
        sys.exit(1)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
