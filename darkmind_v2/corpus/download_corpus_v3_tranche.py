"""Fail-closed acquisition for the frozen Corpus V3 first-tranche plan."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "darkmind_v2" / "config" / "corpus_v3_tranche1.json"
REQUIRED_SOURCE_IDS = {
    "wikimedia_trwiki_20260701",
    "wikimedia_enwiki_20260701",
    "python_docs_tr_3_14_6",
    "python_docs_en_3_14_6",
}
CHUNK_BYTES = 4 * 1024 * 1024


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def file_hashes(path: Path, algorithms: tuple[str, ...] = ("sha256",)) -> dict[str, str]:
    digests = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            for digest in digests.values():
                digest.update(chunk)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def _resolve_repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _validate_https_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"download URL must use HTTPS: {url}")
    if parsed.hostname not in allowed_hosts:
        raise ValueError(f"download URL host is not approved: {url}")
    if not parsed.path or parsed.query or parsed.fragment:
        raise ValueError(f"download URL must be an exact path without query or fragment: {url}")


def validate_config(config: dict[str, Any], *, registry_path: Path | None = None) -> dict[str, Any]:
    if config.get("schema_version") != "darkmind-v2-corpus-v3-tranche-v1":
        raise ValueError("unexpected tranche config schema")
    allowed_hosts = set(config.get("allowed_hosts", []))
    if allowed_hosts != {"docs.python.org", "dumps.wikimedia.org"}:
        raise ValueError("the first tranche must use only the two frozen official hosts")

    sources = list(config.get("sources", []))
    source_ids = [source.get("source_id") for source in sources]
    if set(source_ids) != REQUIRED_SOURCE_IDS or len(source_ids) != len(REQUIRED_SOURCE_IDS):
        raise ValueError("the first tranche must contain each of the four approved sources exactly once")

    registry_path = registry_path or _resolve_repository_path(str(config["approval_registry"]))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    approved = {
        source["id"]: source
        for source in registry.get("sources", [])
        if source.get("phase3a_approval_status", source.get("approval_status")) == "approved"
    }

    expected_bytes = 0
    files: list[dict[str, Any]] = []
    source_targets: dict[str, int] = {}
    language_targets: dict[str, int] = {"tr": 0, "en": 0}
    category_targets: dict[str, int] = {}
    filenames: set[str] = set()

    for source in sources:
        source_id = str(source["source_id"])
        registry_source = approved.get(source_id)
        if registry_source is None or source.get("approval_status") != "approved":
            raise ValueError(f"source is not approved by the frozen registry: {source_id}")
        snapshot = str(source.get("snapshot", ""))
        if not snapshot or "latest" in snapshot.casefold():
            raise ValueError(f"source does not use an exact snapshot: {source_id}")
        if source_id.startswith("wikimedia_") and not re.fullmatch(r"2026-07-01", snapshot):
            raise ValueError(f"Wikimedia source snapshot changed: {source_id}")
        if source_id.startswith("python_docs_") and not snapshot.startswith("3.14.6-"):
            raise ValueError(f"Python documentation release changed: {source_id}")
        registry_cap = int(registry_source["maximum_source_cap_tokens"])
        if int(source["approval_registry_cap_tokens"]) != registry_cap:
            raise ValueError(f"approval registry cap mismatch: {source_id}")
        if int(source["tranche_source_cap_tokens"]) > registry_cap:
            raise ValueError(f"tranche source cap exceeds approval: {source_id}")
        if int(source["target_tokens"]) > int(source["tranche_source_cap_tokens"]):
            raise ValueError(f"source target exceeds tranche source cap: {source_id}")

        source_targets[source_id] = int(source["target_tokens"])
        language_targets[source["language"]] += int(source["target_tokens"])
        category = str(source["category"])
        category_targets[category] = category_targets.get(category, 0) + int(source["target_tokens"])

        checksum_required = source.get("official_checksum_required", True)
        if not checksum_required and not source.get("official_checksum_unavailable_reason"):
            raise ValueError(f"official checksum bypass lacks a recorded reason: {source_id}")
        for item in source.get("files", []):
            _validate_https_url(str(item["url"]), allowed_hosts)
            filename = str(item["filename"])
            if Path(filename).name != filename or filename in filenames:
                raise ValueError(f"unsafe or duplicate archive filename: {filename}")
            filenames.add(filename)
            if checksum_required and not (item.get("md5") or item.get("sha1")):
                raise ValueError(f"official checksum is required for {filename}")
            if not checksum_required and not (
                item.get("expected_last_modified") and item.get("expected_etag")
            ):
                raise ValueError(f"checksum-less source must lock HTTP metadata: {filename}")
            expected_bytes += int(item["expected_bytes"])
            files.append({**item, "source_id": source_id, "kind": "archive"})

    checksum_manifests = list(config.get("checksum_manifests", []))
    for item in checksum_manifests:
        _validate_https_url(str(item["url"]), allowed_hosts)
        filename = str(item["filename"])
        if Path(filename).name != filename or filename in filenames:
            raise ValueError(f"unsafe or duplicate checksum-manifest filename: {filename}")
        filenames.add(filename)
        expected_bytes += int(item["expected_bytes"])
        files.append({**item, "source_id": "official-checksum-manifests", "kind": "checksum_manifest"})

    cap = int(config["maximum_raw_download_bytes"])
    if expected_bytes != int(config["expected_raw_download_bytes"]):
        raise ValueError("frozen expected raw byte total does not match file entries")
    if expected_bytes > cap:
        raise ValueError("planned raw downloads exceed the hard cap")
    if language_targets != config["targets"]["languages"]:
        raise ValueError("source targets do not add up to the frozen language targets")
    if category_targets != config["targets"]["categories"]:
        raise ValueError("source targets do not add up to the frozen category targets")
    if sum(source_targets.values()) != int(config["targets"]["total_tokens"]):
        raise ValueError("source targets do not add up to the tranche total")

    return {
        "result": "PASS",
        "source_ids": source_ids,
        "source_targets": source_targets,
        "language_targets": language_targets,
        "category_targets": category_targets,
        "expected_raw_download_bytes": expected_bytes,
        "maximum_raw_download_bytes": cap,
        "files": files,
    }


def preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = validate_config(config)
    runtime_root = _resolve_repository_path(str(config["runtime_root"]))
    runtime_root.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(runtime_root.parent)
    existing_bytes = 0
    raw_root = runtime_root / "raw"
    for item in plan["files"]:
        if item["kind"] == "checksum_manifest":
            path = raw_root / "checksum_manifests" / item["filename"]
        else:
            path = raw_root / item["source_id"] / item["filename"]
        if path.exists() and path.stat().st_size == int(item["expected_bytes"]):
            existing_bytes += path.stat().st_size
    required_bytes = plan["expected_raw_download_bytes"] - existing_bytes
    if disk.free < required_bytes + 1024 * 1024 * 1024:
        raise OSError("insufficient disk space for the frozen plan plus a 1 GiB safety margin")
    return {
        **plan,
        "config": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
        "available_disk_bytes": disk.free,
        "already_verified_size_bytes": existing_bytes,
        "remaining_download_bytes": required_bytes,
        "raw_root": str(raw_root.relative_to(ROOT)),
    }


def _quarantine(path: Path) -> Path:
    index = 1
    while True:
        target = path.with_name(f"{path.name}.quarantine.{index}")
        if not target.exists():
            os.replace(path, target)
            return target
        index += 1


def _verify_file(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    actual_bytes = path.stat().st_size
    if actual_bytes != int(item["expected_bytes"]):
        raise ValueError(f"byte-count mismatch for {item['filename']}: {actual_bytes}")
    algorithms = ["sha256"]
    algorithms.extend(name for name in ("md5", "sha1") if item.get(name))
    hashes = file_hashes(path, tuple(algorithms))
    for algorithm in ("md5", "sha1"):
        expected = item.get(algorithm)
        if expected and hashes[algorithm].casefold() != str(expected).casefold():
            raise ValueError(f"{algorithm} mismatch for {item['filename']}")
    return {"bytes": actual_bytes, **hashes}


def _download_once(url: str, partial: Path, item: dict[str, Any], *, timeout: int) -> dict[str, str]:
    offset = partial.stat().st_size if partial.exists() else 0
    expected = int(item["expected_bytes"])
    if offset > expected:
        raise ValueError(f"partial file exceeds expected size: {partial}")
    request = urllib.request.Request(url, headers={"User-Agent": "DarkMind-v2-CorpusV3/1.0"})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(getattr(response, "status", response.getcode()))
        mode = "ab" if offset and status == 206 else "wb"
        if offset and status != 206:
            offset = 0
        headers = {key.casefold(): value for key, value in response.headers.items()}
        expected_last_modified = item.get("expected_last_modified")
        expected_etag = item.get("expected_etag")
        if expected_last_modified and headers.get("last-modified") != expected_last_modified:
            raise ValueError(f"Last-Modified changed for {item['filename']}")
        if expected_etag and headers.get("etag") != expected_etag:
            raise ValueError(f"ETag changed for {item['filename']}")
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                offset += len(chunk)
                if offset > expected:
                    raise ValueError(f"server exceeded frozen byte count for {item['filename']}")
                if offset and offset % (128 * 1024 * 1024) < CHUNK_BYTES:
                    print(f"download progress file={item['filename']} bytes={offset}/{expected}", flush=True)
        return headers


def _parse_curl_headers(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="iso-8859-1")
    blocks = [block for block in re.split(r"\r?\n\r?\n", text) if block.startswith("HTTP/")]
    if not blocks:
        raise ValueError("curl did not record an HTTP response header")
    headers: dict[str, str] = {}
    for line in blocks[-1].splitlines()[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().casefold()] = value.strip()
    return headers


def _download_once_curl(url: str, partial: Path, item: dict[str, Any], *, timeout: int) -> dict[str, str]:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("curl is not available for the Windows resumable-download backend")
    header_path = partial.with_name(partial.name + ".headers")
    command = [
        curl,
        "--fail",
        "--location",
        "--show-error",
        "--retry",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        str(timeout),
        "--speed-limit",
        "1024",
        "--speed-time",
        str(timeout),
        "--continue-at",
        "-",
        "--output",
        str(partial),
        "--dump-header",
        str(header_path),
        "--user-agent",
        "DarkMind-v2-CorpusV3/1.0",
        url,
    ]
    subprocess.run(command, check=True)
    headers = _parse_curl_headers(header_path)
    expected_last_modified = item.get("expected_last_modified")
    expected_etag = item.get("expected_etag")
    if expected_last_modified and headers.get("last-modified") != expected_last_modified:
        raise ValueError(f"Last-Modified changed for {item['filename']}")
    if expected_etag and headers.get("etag") != expected_etag:
        raise ValueError(f"ETag changed for {item['filename']}")
    return headers


def _download_segment(
    curl: str,
    url: str,
    segment_root: Path,
    index: int,
    start: int,
    end: int,
    *,
    timeout: int,
    attempts: int,
) -> tuple[int, dict[str, str]]:
    final = segment_root / f"part-{index:05d}"
    partial = segment_root / f"part-{index:05d}.partial"
    header_path = segment_root / f"part-{index:05d}.headers"
    expected = end - start + 1
    if final.exists():
        if final.stat().st_size != expected:
            raise ValueError(f"completed segment has unexpected size: {final}")
        return index, _parse_curl_headers(header_path)
    if partial.exists() and partial.stat().st_size > expected:
        raise ValueError(f"partial segment exceeds its frozen range: {partial}")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        existing = partial.stat().st_size if partial.exists() else 0
        if existing == expected:
            os.replace(partial, final)
            return index, _parse_curl_headers(header_path)
        range_start = start + existing
        command = [
            curl,
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--connect-timeout",
            str(timeout),
            "--speed-limit",
            "1024",
            "--speed-time",
            str(timeout),
            "--range",
            f"{range_start}-{end}",
            "--output",
            "-",
            "--dump-header",
            str(header_path),
            "--user-agent",
            "DarkMind-v2-CorpusV3/1.0",
            url,
        ]
        try:
            with partial.open("ab") as output:
                process = subprocess.Popen(command, stdout=subprocess.PIPE)
                assert process.stdout is not None
                for chunk in iter(lambda: process.stdout.read(CHUNK_BYTES), b""):
                    output.write(chunk)
                    if output.tell() > expected:
                        process.kill()
                        raise ValueError(f"server exceeded frozen segment range: {index}")
                return_code = process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            if partial.stat().st_size != expected:
                raise IOError(
                    f"segment response ended early: index={index} "
                    f"bytes={partial.stat().st_size}/{expected}"
                )
            os.replace(partial, final)
            return index, _parse_curl_headers(header_path)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(min(8, 2 ** (attempt - 1)))
    raise RuntimeError(f"segment {index} failed after {attempts} attempts: {last_error}")


def _download_once_segmented(
    url: str,
    partial: Path,
    item: dict[str, Any],
    *,
    timeout: int,
    attempts: int = 4,
) -> dict[str, str]:
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("curl is not available for segmented downloads")
    expected = int(item["expected_bytes"])
    if partial.exists() and partial.stat().st_size == expected:
        return {}
    segment_root = partial.with_name(partial.name + ".parts")
    if partial.exists() and not segment_root.exists():
        legacy = partial.with_name(partial.name + ".single-connection")
        suffix = 1
        while legacy.exists():
            legacy = partial.with_name(partial.name + f".single-connection.{suffix}")
            suffix += 1
        os.replace(partial, legacy)
    segment_root.mkdir(parents=True, exist_ok=True)
    segment_count = min(32, max(4, math.ceil(expected / (8 * 1024 * 1024))))
    segment_size = math.ceil(expected / segment_count)
    ranges = [
        (index, index * segment_size, min(expected - 1, (index + 1) * segment_size - 1))
        for index in range(segment_count)
        if index * segment_size < expected
    ]
    print(
        f"segmented download file={item['filename']} segments={len(ranges)} bytes={expected}",
        flush=True,
    )
    headers_by_index: dict[int, dict[str, str]] = {}
    host = urllib.parse.urlparse(url).hostname
    worker_limit = 2 if host == "dumps.wikimedia.org" else 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(worker_limit, len(ranges))) as executor:
        futures = {
            executor.submit(
                _download_segment,
                curl,
                url,
                segment_root,
                index,
                start,
                end,
                timeout=timeout,
                attempts=attempts,
            ): index
            for index, start, end in ranges
        }
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index, headers = future.result()
            headers_by_index[index] = headers
            completed += 1
            print(
                f"segment progress file={item['filename']} completed={completed}/{len(ranges)}",
                flush=True,
            )

    assembling = partial.with_name(partial.name + ".assembling")
    with assembling.open("wb") as output:
        for index, start, end in ranges:
            part = segment_root / f"part-{index:05d}"
            if part.stat().st_size != end - start + 1:
                raise ValueError(f"segment size changed before assembly: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=CHUNK_BYTES)
    if assembling.stat().st_size != expected:
        raise ValueError(f"segmented assembly byte mismatch for {item['filename']}")
    os.replace(assembling, partial)
    headers = headers_by_index[min(headers_by_index)]
    expected_last_modified = item.get("expected_last_modified")
    expected_etag = item.get("expected_etag")
    if expected_last_modified and headers.get("last-modified") != expected_last_modified:
        raise ValueError(f"Last-Modified changed for {item['filename']}")
    if expected_etag and headers.get("etag") != expected_etag:
        raise ValueError(f"ETag changed for {item['filename']}")
    return headers


def download_file(
    item: dict[str, Any],
    destination: Path,
    *,
    attempts: int = 4,
    timeout: int = 60,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            verification = _verify_file(destination, item)
        except ValueError:
            quarantined = _quarantine(destination)
            raise ValueError(f"existing file failed verification and was quarantined at {quarantined}")
        return {"status": "reused_verified", "headers": {}, **verification}

    partial = destination.with_name(destination.name + ".partial")
    headers: dict[str, str] = {}
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if os.name == "nt" and (shutil.which("curl.exe") or shutil.which("curl")):
                headers = _download_once_segmented(
                    str(item["url"]), partial, item, timeout=timeout, attempts=attempts
                )
            else:
                headers = _download_once(str(item["url"]), partial, item, timeout=timeout)
            if partial.stat().st_size != int(item["expected_bytes"]):
                raise IOError(
                    f"incomplete response for {item['filename']}: "
                    f"{partial.stat().st_size}/{item['expected_bytes']}"
                )
            verification = _verify_file(partial, item)
            os.replace(partial, destination)
            return {"status": "downloaded_verified", "headers": headers, **verification}
        except (OSError, ValueError, urllib.error.URLError, subprocess.CalledProcessError) as exc:
            last_error = exc
            if isinstance(exc, ValueError) and "mismatch" in str(exc).casefold() and partial.exists():
                quarantined = _quarantine(partial)
                raise ValueError(f"verification failed; partial quarantined at {quarantined}: {exc}") from exc
            if attempt < attempts:
                delay = 2 ** (attempt - 1)
                print(f"download retry file={item['filename']} attempt={attempt} error={exc}", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"download failed after {attempts} attempts: {item['filename']}: {last_error}")


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]+)\s+\*?(.+)", line.strip())
        if match:
            checksums[match.group(2)] = match.group(1).casefold()
    if not checksums:
        raise ValueError(f"official checksum manifest is empty or malformed: {path}")
    return checksums


def verify_official_manifests(config: dict[str, Any], raw_root: Path) -> dict[str, Any]:
    parsed: dict[tuple[str, str], dict[str, str]] = {}
    for item in config["checksum_manifests"]:
        project = item["id"].split("-")[0]
        parsed[(project, item["algorithm"])] = parse_checksum_manifest(
            raw_root / "checksum_manifests" / item["filename"]
        )
    verified: list[dict[str, str]] = []
    for source in config["sources"]:
        if not source["source_id"].startswith("wikimedia_"):
            continue
        project = "trwiki" if source["language"] == "tr" else "enwiki"
        for item in source["files"]:
            for algorithm in ("md5", "sha1"):
                official = parsed[(project, algorithm)].get(item["filename"])
                if official != item[algorithm].casefold():
                    raise ValueError(
                        f"frozen {algorithm} does not match official manifest for {item['filename']}"
                    )
                verified.append(
                    {"filename": item["filename"], "algorithm": algorithm, "checksum": official}
                )
    return {"result": "PASS", "verified_entries": verified}


def render_inventory_report(config: dict[str, Any], result: dict[str, Any]) -> str:
    source_lookup = {source["source_id"]: source for source in config["sources"]}
    lines = [
        "# Phase 3C Download Inventory",
        "",
        f"Status: **{result['result']}**",
        "",
        f"Hard raw-download cap: {config['maximum_raw_download_bytes']:,} bytes",
        f"Frozen planned bytes: {config['expected_raw_download_bytes']:,} bytes",
        f"Verified bytes in this acquisition run: {result.get('actual_raw_download_bytes', 0):,} bytes",
        f"Full plan complete: {result.get('full_plan_complete', False)}",
        f"Available disk at preflight: {result['available_disk_bytes']:,} bytes",
        "",
        "| Source | Snapshot | File | Expected bytes | Checksum policy | Target tokens | Status |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    records = {record["filename"]: record for record in result.get("downloads", [])}
    for source in config["sources"]:
        checksum_policy = (
            "official MD5 + SHA-1"
            if source.get("official_checksum_required", True)
            else "official checksum unavailable; byte + Last-Modified + ETag lock, local SHA-256"
        )
        for item in source["files"]:
            status = records.get(item["filename"], {}).get("status")
            if status is None:
                status = "not_downloaded_source_quota_gate" if result.get("full_plan_complete") is False else "planned"
            lines.append(
                f"| {source['source_id']} | {source['snapshot']} | {item['filename']} | "
                f"{item['expected_bytes']:,} | {checksum_policy} | {source['target_tokens']:,} | {status} |"
            )
    lines.extend(
        [
            "",
            "The four official checksum manifests contribute 98,064 bytes to the verified acquisition total. Raw archives are preserved unchanged under the ignored Phase 3C runtime directory.",
            "",
            "No unapproved source, undated `latest` endpoint, or silent checksum substitution is permitted.",
            "",
        ]
    )
    return "\n".join(lines)


def download_tranche(
    config_path: Path = DEFAULT_CONFIG,
    *,
    only_source_ids: set[str] | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    flight = preflight(config_path)
    selected_ids = only_source_ids or REQUIRED_SOURCE_IDS
    if not selected_ids or not selected_ids <= REQUIRED_SOURCE_IDS:
        raise ValueError(f"source selection is outside the frozen first-tranche list: {sorted(selected_ids)}")
    full_plan = selected_ids == REQUIRED_SOURCE_IDS
    runtime_root = _resolve_repository_path(str(config["runtime_root"]))
    raw_root = runtime_root / "raw"
    records: list[dict[str, Any]] = []

    manifest_items = [
        {**item, "source_id": "official-checksum-manifests", "kind": "checksum_manifest"}
        for item in config["checksum_manifests"]
    ]
    for item in manifest_items:
        destination = raw_root / "checksum_manifests" / item["filename"]
        outcome = download_file(item, destination)
        records.append({**item, **outcome, "path": str(destination.relative_to(ROOT))})
        atomic_write_json(runtime_root / "download_progress.json", {"downloads": records})

    official_verification = verify_official_manifests(config, raw_root)
    selected_sources = [source for source in config["sources"] if source["source_id"] in selected_ids]
    for source in sorted(selected_sources, key=lambda value: int(value["source_priority"])):
        for file_item in source["files"]:
            item = {**file_item, "source_id": source["source_id"], "kind": "archive"}
            destination = raw_root / source["source_id"] / item["filename"]
            outcome = download_file(item, destination)
            records.append({**item, **outcome, "path": str(destination.relative_to(ROOT))})
            atomic_write_json(runtime_root / "download_progress.json", {"downloads": records})

    expected_selected_bytes = sum(int(item["expected_bytes"]) for item in config["checksum_manifests"])
    expected_selected_bytes += sum(
        int(item["expected_bytes"])
        for source in selected_sources
        for item in source["files"]
    )
    result = {
        "schema_version": "darkmind-v2-corpus-v3-download-manifest-v1",
        "result": "PASS" if full_plan else "PASS_SELECTED_SOURCES",
        "full_plan_complete": full_plan,
        "selected_source_ids": sorted(selected_ids),
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "config_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        "available_disk_bytes": flight["available_disk_bytes"],
        "expected_raw_download_bytes": expected_selected_bytes,
        "full_plan_expected_raw_download_bytes": flight["expected_raw_download_bytes"],
        "actual_raw_download_bytes": sum(int(record["bytes"]) for record in records),
        "official_manifest_verification": official_verification,
        "downloads": records,
    }
    if result["actual_raw_download_bytes"] != result["expected_raw_download_bytes"]:
        raise ValueError("completed download byte total differs from frozen plan")
    manifest_name = "download_manifest.json" if full_plan else "download_manifest.selected.json"
    atomic_write_json(runtime_root / manifest_name, result)
    provenance_lines = [
        json.dumps(
            {
                "source_id": record["source_id"],
                "url": record["url"],
                "filename": record["filename"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "status": record["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        for record in records
    ]
    provenance_name = "provenance.jsonl" if full_plan else "provenance.selected.jsonl"
    atomic_write_text(runtime_root / provenance_name, "\n".join(provenance_lines) + "\n")
    report_path = _resolve_repository_path(str(config["reports_root"])) / "phase3c_download_inventory.md"
    atomic_write_text(report_path, render_inventory_report(config, result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "download"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-id", action="append", default=None)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.config)
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report_path = _resolve_repository_path(str(config["reports_root"])) / "phase3c_download_inventory.md"
        atomic_write_text(report_path, render_inventory_report(config, result))
    else:
        result = download_tranche(
            args.config,
            only_source_ids=set(args.source_id) if args.source_id else None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
