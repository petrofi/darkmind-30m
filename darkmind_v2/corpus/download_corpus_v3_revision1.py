"""Acquire frozen Corpus V3 revision-1 Wikimedia inputs with fail-closed retries."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import email.utils
import hashlib
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "darkmind_v2" / "config" / "corpus_v3_tranche1_revision1.json"
SUPPLEMENTAL_SOURCE_IDS = {
    "wikimedia_trwikibooks_20260701",
    "wikimedia_enwikiversity_20260201",
    "wikimedia_enwikibooks_20260701",
}
WIKIPEDIA_SOURCE_IDS = {
    "wikimedia_trwiki_20260701",
    "wikimedia_enwiki_20260701",
}
REMOTE_SOURCE_IDS = SUPPLEMENTAL_SOURCE_IDS | WIKIPEDIA_SOURCE_IDS
CHUNK_BYTES = 4 * 1024 * 1024


class RetryableDownloadError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


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


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def file_hashes(path: Path, algorithms: tuple[str, ...] = ("sha256",)) -> dict[str, str]:
    digests = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            for digest in digests.values():
                digest.update(chunk)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def repository_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_transfer_concurrency(value: int) -> int:
    if value < 1 or value > 2:
        raise ValueError("Wikimedia transfer concurrency must be between 1 and 2")
    return value


def retry_after_seconds(value: str | None, *, now: dt.datetime | None = None) -> float | None:
    if not value:
        return None
    stripped = value.strip()
    if stripped.isdigit():
        return max(0.0, float(stripped))
    try:
        target = email.utils.parsedate_to_datetime(stripped)
    except (TypeError, ValueError):
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=dt.timezone.utc)
    current = now or dt.datetime.now(dt.timezone.utc)
    return max(0.0, (target - current).total_seconds())


def retry_delay(
    attempt: int,
    policy: dict[str, Any],
    *,
    retry_after: float | None = None,
    random_value: float | None = None,
) -> float:
    if attempt < 1:
        raise ValueError("attempt must be positive")
    base = min(
        float(policy["maximum_backoff_seconds"]),
        float(policy["initial_backoff_seconds"]) * (2 ** (attempt - 1)),
    )
    if retry_after is not None:
        base = max(base, retry_after)
    jitter_fraction = float(policy["jitter_fraction"])
    sample = random.random() if random_value is None else random_value
    jitter = base * jitter_fraction * sample
    return min(float(policy["maximum_backoff_seconds"]), base + jitter)


def _validate_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in allowed_hosts:
        raise ValueError(f"unapproved download URL: {url}")
    if not parsed.path or parsed.query or parsed.fragment or "latest" in parsed.path.casefold():
        raise ValueError(f"download URL must be an exact dated path: {url}")


def _metadata_url(item: dict[str, Any]) -> str:
    return f"https://dumps.wikimedia.org/{item['project']}/{item['snapshot']}/{item['filename']}"


def _source_project(source: dict[str, Any]) -> str:
    return urllib.parse.urlparse(source["files"][0]["url"]).path.split("/")[1]


def _metadata_for_sources(config: dict[str, Any], source_ids: set[str]) -> list[dict[str, Any]]:
    selected = [source for source in config["sources"] if source["source_id"] in source_ids]
    projects = {(_source_project(source), source["snapshot"].replace("-", "")) for source in selected}
    result = []
    for item in config["metadata_files"]:
        if (item["project"], item["snapshot"]) in projects:
            result.append(
                {
                    **item,
                    "url": _metadata_url(item),
                    "storage_filename": item.get("local_filename", item["filename"]),
                }
            )
    return result


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "darkmind-v2-corpus-v3-tranche-revision1-v1":
        raise ValueError("unexpected revision-1 schema")
    if config["allocation_policy"].get("source_minimums_enabled") is not False:
        raise ValueError("revision 1 must use aggregate quotas")
    targets = config["targets"]
    if sum(targets["languages"].values()) != targets["total_tokens"]:
        raise ValueError("language targets do not add up to total")
    if sum(targets["categories"].values()) != targets["total_tokens"]:
        raise ValueError("category targets do not add up to total")
    policy = config["rate_limit_policy"]
    if int(policy["metadata_concurrency"]) != 1:
        raise ValueError("metadata concurrency must remain one")
    validate_transfer_concurrency(int(policy["maximum_wikimedia_transfers"]))
    validate_transfer_concurrency(int(policy["default_wikimedia_transfers"]))
    if int(policy["default_wikimedia_transfers"]) > int(policy["maximum_wikimedia_transfers"]):
        raise ValueError("default transfer count exceeds maximum")
    if not policy.get("honor_retry_after"):
        raise ValueError("Retry-After must be honored")

    registry = json.loads(repository_path(config["approval_registry"]).read_text(encoding="utf-8"))
    approved = {
        item["id"]: item
        for item in registry["sources"]
        if item.get("phase3a_approval_status", item.get("approval_status")) == "approved"
    }
    allowed_hosts = set(config["allowed_hosts"])
    seen: set[str] = set()
    for source in config["sources"]:
        source_id = source["source_id"]
        if source_id in seen:
            raise ValueError(f"duplicate source id: {source_id}")
        seen.add(source_id)
        if source.get("approval_status") != "approved":
            raise ValueError(f"source is not approved: {source_id}")
        if source_id not in REMOTE_SOURCE_IDS:
            continue
        registry_id = source["approval_registry_id"]
        registry_source = approved.get(registry_id)
        if registry_source is None:
            raise ValueError(f"source lacks frozen registry approval: {source_id}")
        if int(source["maximum_source_cap_tokens"]) > int(registry_source["maximum_source_cap_tokens"]):
            raise ValueError(f"source cap exceeds registry: {source_id}")
        snapshot = source.get("snapshot", "")
        if not re.fullmatch(r"20\d{2}-\d{2}-\d{2}", snapshot):
            raise ValueError(f"source lacks exact dated snapshot: {source_id}")
        if not source.get("files"):
            raise ValueError(f"source has no frozen article file: {source_id}")
        for item in source["files"]:
            _validate_url(item["url"], allowed_hosts)
            if Path(item["filename"]).name != item["filename"]:
                raise ValueError(f"unsafe filename: {item['filename']}")
            if not item.get("md5") or not item.get("sha1") or int(item["expected_bytes"]) <= 0:
                raise ValueError(f"archive lacks official byte/checksum lock: {item['filename']}")
    if not REMOTE_SOURCE_IDS <= seen:
        raise ValueError("revision-1 config lacks an authorized remote source")

    for item in config["metadata_files"]:
        _validate_url(_metadata_url(item), allowed_hosts)
        if int(item["expected_bytes"]) <= 0 or item["kind"] not in {"md5", "sha1", "dumpstatus"}:
            raise ValueError("invalid metadata file declaration")
    return {"result": "PASS", "source_ids": sorted(seen)}


def stage_source_ids(stage: str) -> set[str]:
    if stage == "supplemental":
        return set(SUPPLEMENTAL_SOURCE_IDS)
    if stage == "wikipedia":
        return set(WIKIPEDIA_SOURCE_IDS)
    raise ValueError(f"unknown acquisition stage: {stage}")


def build_plan(config: dict[str, Any], stage: str, transfers: int) -> dict[str, Any]:
    validate_config(config)
    transfers = validate_transfer_concurrency(transfers)
    if transfers > int(config["rate_limit_policy"]["maximum_wikimedia_transfers"]):
        raise ValueError("requested transfer count exceeds frozen policy")
    source_ids = stage_source_ids(stage)
    sources = [source for source in config["sources"] if source["source_id"] in source_ids]
    metadata = _metadata_for_sources(config, source_ids)
    archives = [
        {**item, "source_id": source["source_id"], "project": _source_project(source)}
        for source in sources
        for item in source["files"]
    ]
    planned = sum(int(item["expected_bytes"]) for item in metadata + archives)
    prior_verified = int(config["already_verified_phase3c_bytes"])
    if stage == "wikipedia":
        supplemental_ids = stage_source_ids("supplemental")
        supplemental_sources = [
            source for source in config["sources"] if source["source_id"] in supplemental_ids
        ]
        supplemental_metadata = _metadata_for_sources(config, supplemental_ids)
        prior_verified += sum(int(item["expected_bytes"]) for item in supplemental_metadata)
        prior_verified += sum(
            int(item["expected_bytes"])
            for source in supplemental_sources
            for item in source["files"]
        )
    cumulative = prior_verified + planned
    if cumulative > int(config["maximum_raw_download_bytes"]):
        raise ValueError("planned cumulative raw bytes exceed the 8 GB hard cap")
    return {
        "result": "PASS",
        "stage": stage,
        "source_ids": sorted(source_ids),
        "sources": sources,
        "metadata": metadata,
        "archives": archives,
        "planned_download_bytes": planned,
        "prior_verified_raw_bytes": prior_verified,
        "cumulative_raw_bytes": cumulative,
        "transfers": transfers,
    }


def _verify_file(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != int(item["expected_bytes"]):
        raise ValueError(f"byte-count mismatch for {path.name}: {actual_bytes}")
    algorithms = ["sha256"] + [name for name in ("md5", "sha1") if item.get(name)]
    hashes = file_hashes(path, tuple(algorithms))
    for algorithm in ("md5", "sha1"):
        if item.get(algorithm) and hashes[algorithm].casefold() != item[algorithm].casefold():
            raise ValueError(f"{algorithm} mismatch for {path.name}")
    return {"bytes": actual_bytes, **hashes}


def _download_once(
    item: dict[str, Any],
    partial: Path,
    *,
    user_agent: str,
    timeout: int,
) -> dict[str, str]:
    offset = partial.stat().st_size if partial.exists() else 0
    expected = int(item["expected_bytes"])
    if offset > expected:
        raise ValueError(f"partial exceeds frozen bytes: {partial}")
    request = urllib.request.Request(item["url"], headers={"User-Agent": user_agent})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        retry_after = retry_after_seconds(exc.headers.get("Retry-After")) if exc.code == 429 else None
        if exc.code == 429 or 500 <= exc.code <= 599:
            raise RetryableDownloadError(f"HTTP {exc.code} for {item['filename']}", retry_after=retry_after) from exc
        raise
    with response:
        status = int(getattr(response, "status", response.getcode()))
        if offset and status != 206:
            raise RetryableDownloadError(
                f"server did not honor Range for preserved partial {item['filename']}"
            )
        if not offset and status != 200:
            raise RetryableDownloadError(f"unexpected HTTP {status} for {item['filename']}")
        headers = {key.casefold(): value for key, value in response.headers.items()}
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                offset += len(chunk)
                if offset > expected:
                    raise ValueError(f"response exceeded frozen bytes: {item['filename']}")
                if offset and offset % (128 * 1024 * 1024) < CHUNK_BYTES:
                    print(f"download progress file={item['filename']} bytes={offset}/{expected}", flush=True)
        return headers


def download_file(
    item: dict[str, Any],
    destination: Path,
    policy: dict[str, Any],
    *,
    timeout: int = 120,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verification = _verify_file(destination, item)
        return {"status": "reused_verified", "headers": {}, **verification}
    partial = destination.with_name(destination.name + ".partial")
    attempts = int(policy["maximum_attempts"])
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            headers = _download_once(
                item,
                partial,
                user_agent=policy["user_agent"],
                timeout=timeout,
            )
            if partial.stat().st_size != int(item["expected_bytes"]):
                raise RetryableDownloadError(
                    f"incomplete response {partial.stat().st_size}/{item['expected_bytes']} for {item['filename']}"
                )
            verification = _verify_file(partial, item)
            os.replace(partial, destination)
            return {"status": "downloaded_verified", "headers": headers, **verification}
        except (OSError, ValueError, urllib.error.URLError, RetryableDownloadError) as exc:
            last_error = exc
            if isinstance(exc, ValueError) and "mismatch" in str(exc).casefold():
                raise ValueError(f"checksum verification failed; preserved partial: {partial}") from exc
            if attempt == attempts:
                break
            retry_after = exc.retry_after if isinstance(exc, RetryableDownloadError) else None
            delay = retry_delay(attempt, policy, retry_after=retry_after)
            print(
                f"download retry file={item['filename']} attempt={attempt} delay={delay:.2f}s error={exc}",
                flush=True,
            )
            sleep(delay)
    raise RuntimeError(f"download failed after {attempts} attempts: {item['filename']}: {last_error}")


def parse_checksum_manifest(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]+)\s+\*?(.+)", line.strip())
        if match:
            result[match.group(2)] = match.group(1).casefold()
    if not result:
        raise ValueError(f"empty or malformed checksum manifest: {path}")
    return result


def verify_official_metadata(plan: dict[str, Any], metadata_root: Path) -> dict[str, Any]:
    parsed: dict[tuple[str, str, str], Any] = {}
    for item in plan["metadata"]:
        path = metadata_root / item["storage_filename"]
        key = (item["project"], item["snapshot"], item["kind"])
        if item["kind"] in {"md5", "sha1"}:
            parsed[key] = parse_checksum_manifest(path)
        else:
            parsed[key] = json.loads(path.read_text(encoding="utf-8"))
    verified: list[dict[str, Any]] = []
    for source in plan["sources"]:
        project = _source_project(source)
        snapshot = source["snapshot"].replace("-", "")
        status = parsed[(project, snapshot, "dumpstatus")]
        jobs = status.get("jobs", {})
        for item in source["files"]:
            for algorithm in ("md5", "sha1"):
                official = parsed[(project, snapshot, algorithm)].get(item["filename"])
                if official != item[algorithm].casefold():
                    raise ValueError(f"official {algorithm} mismatch for {item['filename']}")
            article_files: dict[str, Any] = {}
            for job_name in ("articlesdump", "articlesmultistreamdump"):
                job = jobs.get(job_name, {})
                if job.get("status") == "done":
                    article_files.update(job.get("files", {}))
            status_item = article_files.get(item["filename"])
            if not status_item or int(status_item["size"]) != int(item["expected_bytes"]):
                raise ValueError(f"completed article job metadata mismatch for {item['filename']}")
            if status_item["md5"] != item["md5"] or status_item["sha1"] != item["sha1"]:
                raise ValueError(f"dumpstatus checksum mismatch for {item['filename']}")
            verified.append({"filename": item["filename"], "project": project, "snapshot": snapshot})
    return {"result": "PASS", "verified_archives": verified}


def render_inventory(config: dict[str, Any], plan: dict[str, Any], result: dict[str, Any] | None = None) -> str:
    available = shutil.disk_usage(repository_path(config["runtime_root"]).parent).free
    records = {item["filename"]: item for item in (result or {}).get("downloads", [])}
    title = "Supplemental" if plan["stage"] == "supplemental" else "Wikipedia"
    lines = [
        f"# Phase 3C.1 {title} Download Inventory",
        "",
        f"Status: **{(result or {}).get('result', 'PREFLIGHT_PASS')}**",
        "",
        f"Wikimedia transfer concurrency: {plan['transfers']}",
        "Metadata concurrency: 1",
        f"Planned download bytes: {plan['planned_download_bytes']:,}",
        f"Cumulative raw bytes including verified Phase 3C inputs: {plan['cumulative_raw_bytes']:,}",
        f"Hard raw cap: {config['maximum_raw_download_bytes']:,}",
        f"Available disk bytes: {available:,}",
        "",
        "| Project | Snapshot | File | Compressed bytes | Max extracted bytes | Estimated usable tokens | Status |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    source_by_id = {source["source_id"]: source for source in plan["sources"]}
    for item in plan["archives"]:
        source = source_by_id[item["source_id"]]
        status = records.get(item["filename"], {}).get("status", "planned")
        lines.append(
            f"| {item['project']} | {source['snapshot']} | {item['filename']} | "
            f"{item['expected_bytes']:,} | {item.get('maximum_expected_extracted_bytes', 0):,} | "
            f"{item.get('estimated_usable_tokens', 0):,} | {status} |"
        )
    lines.extend(
        [
            "",
            "Only exact dated article-content archives are included. Histories, media, logs, users, and SQL tables are excluded.",
            "",
            "MD5, SHA-1, and dumpstatus metadata are fetched serially and preserved before archive extraction is allowed.",
            "",
        ]
    )
    return "\n".join(lines)


def acquire(config_path: Path, stage: str, transfers: int) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    plan = build_plan(config, stage, transfers)
    runtime_root = repository_path(config["runtime_root"])
    raw_root = runtime_root / "raw"
    metadata_root = raw_root / "official_metadata"
    policy = config["rate_limit_policy"]
    downloads: list[dict[str, Any]] = []

    for item in plan["metadata"]:
        destination = metadata_root / item["storage_filename"]
        outcome = download_file(item, destination, policy)
        downloads.append({**item, **outcome, "path": str(destination.relative_to(ROOT))})
        atomic_write_json(runtime_root / f"{stage}_download_progress.json", {"downloads": downloads})
    official = verify_official_metadata(plan, metadata_root)

    def acquire_archive(item: dict[str, Any]) -> dict[str, Any]:
        destination = raw_root / item["source_id"] / item["filename"]
        outcome = download_file(item, destination, policy)
        return {**item, **outcome, "path": str(destination.relative_to(ROOT))}

    with concurrent.futures.ThreadPoolExecutor(max_workers=plan["transfers"]) as executor:
        futures = [executor.submit(acquire_archive, item) for item in plan["archives"]]
        for future in concurrent.futures.as_completed(futures):
            downloads.append(future.result())
            atomic_write_json(runtime_root / f"{stage}_download_progress.json", {"downloads": downloads})

    actual = sum(int(item["bytes"]) for item in downloads)
    if actual != plan["planned_download_bytes"]:
        raise ValueError("verified download bytes differ from the frozen plan")
    result = {
        "schema_version": "darkmind-v2-corpus-v3-revision1-download-v1",
        "result": "PASS",
        "stage": stage,
        "source_ids": plan["source_ids"],
        "config_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        "planned_download_bytes": plan["planned_download_bytes"],
        "verified_download_bytes": actual,
        "cumulative_raw_bytes": plan["cumulative_raw_bytes"],
        "official_metadata_verification": official,
        "downloads": sorted(downloads, key=lambda item: (item.get("source_id", "metadata"), item["filename"])),
    }
    atomic_write_json(runtime_root / f"{stage}_download_manifest.json", result)
    report_name = (
        "phase3c1_supplemental_download_inventory.md"
        if stage == "supplemental"
        else "phase3c1_wikipedia_download_inventory.md"
    )
    atomic_write_text(repository_path(config["reports_root"]) / report_name, render_inventory(config, plan, result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "download"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("supplemental", "wikipedia"), required=True)
    parser.add_argument("--transfers", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    transfers = args.transfers or int(config["rate_limit_policy"]["default_wikimedia_transfers"])
    if args.command == "preflight":
        plan = build_plan(config, args.stage, transfers)
        report_name = (
            "phase3c1_supplemental_download_inventory.md"
            if args.stage == "supplemental"
            else "phase3c1_wikipedia_download_inventory.md"
        )
        atomic_write_text(repository_path(config["reports_root"]) / report_name, render_inventory(config, plan))
        result: dict[str, Any] = plan
    else:
        result = acquire(args.config, args.stage, transfers)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
