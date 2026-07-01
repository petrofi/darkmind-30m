from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib import robotparser
import argparse
import json
import re
import time
import urllib.request

from check_web_source_policy import (
    DEFAULT_CONFIG_PATH,
    ROOT_DIR,
    load_config,
    resolve_path,
    validate_url_against_config,
)


DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "raw_collected" / "web_text" / "pending_review"
DEFAULT_METADATA_DIR = ROOT_DIR / "data" / "raw_collected" / "web_text" / "metadata"
SKIP_TAGS = {"script", "style", "nav", "header", "footer", "form", "noscript", "svg"}
BLOCK_TAGS = {"p", "div", "br", "li", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6"}


class ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()

        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return

        if self.skip_depth:
            return

        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
            return

        if self.skip_depth:
            return

        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return

        text = data.strip()

        if text:
            self.parts.append(text)

    def text(self) -> str:
        raw_text = " ".join(self.parts)
        raw_text = re.sub(r"[ \t]+", " ", raw_text)
        raw_text = re.sub(r" *\n+ *", "\n", raw_text)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)

        lines = [
            line.strip()
            for line in raw_text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines).strip()


def safe_name(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "web_source"


def fetch_html(url: str, user_agent: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent},
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        content_type = response.headers.get("Content-Type", "")
        charset = response.headers.get_content_charset() or "utf-8"
        raw_bytes = response.read()

    if "html" not in content_type.lower() and "text" not in content_type.lower():
        raise ValueError(f"Unsupported content type: {content_type}")

    return raw_bytes.decode(charset, errors="replace")


def extract_readable_text(html: str, max_chars: int) -> str:
    parser = ReadableHTMLParser()
    parser.feed(html)
    text = parser.text()

    if max_chars > 0:
        text = text[:max_chars].strip()

    return text


def robots_allowed(url: str, source: dict, user_agent: str) -> tuple[bool, str]:
    robots_url = urljoin(source["base_url"].rstrip("/") + "/", "robots.txt")
    parser = robotparser.RobotFileParser()
    parser.set_url(robots_url)

    try:
        parser.read()
    except Exception as exc:
        return True, f"Could not read robots.txt ({exc}); continuing with allowlist policy."

    if parser.can_fetch(user_agent, url):
        return True, "robots.txt allows this URL."

    return False, f"robots.txt blocks this URL for user agent '{user_agent}'."


def save_fetch_result(
    url: str,
    source: dict,
    text: str,
    output_dir: Path,
    metadata_dir: Path,
    max_chars: int,
) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{safe_name(source['name'])}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    text_path = output_dir / f"{base_name}.txt"
    metadata_path = metadata_dir / f"{base_name}.json"

    text_path.write_text(text, encoding="utf-8")

    metadata = {
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source["name"],
        "license": source.get("license", "unknown"),
        "usage_status": source.get("usage_status", "unknown"),
        "max_chars": max_chars,
        "character_count": len(text),
        "note": "Pending human review before training use.",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return text_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch text from an allowlisted URL into pending review."
    )
    parser.add_argument("--url", required=True, help="Allowlisted URL to fetch.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.relative_to(ROOT_DIR)),
        help="Allowlist config path.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT_DIR)),
        help="Pending review output directory.",
    )
    parser.add_argument(
        "--metadata_dir",
        default=str(DEFAULT_METADATA_DIR.relative_to(ROOT_DIR)),
        help="Metadata output directory.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=20000,
        help="Maximum extracted characters to save.",
    )
    args = parser.parse_args()

    config = load_config(resolve_path(args.config))
    allowed, reason, source = validate_url_against_config(args.url, config)

    if not allowed or source is None:
        print(f"BLOCKED: {reason}")
        raise SystemExit(1)

    settings = config.get("global_settings", {})
    user_agent = settings.get(
        "user_agent",
        "DarkMindResearchBot/0.1 educational dataset collection",
    )
    rate_limit_seconds = float(settings.get("rate_limit_seconds", 2))

    robots_ok, robots_reason = robots_allowed(args.url, source, user_agent)
    print(robots_reason)

    if not robots_ok:
        raise SystemExit(1)

    time.sleep(max(rate_limit_seconds, 0.0))

    html = fetch_html(args.url, user_agent)
    text = extract_readable_text(html, args.max_chars)

    if not text:
        raise ValueError("No readable text was extracted from the page.")

    text_path, metadata_path = save_fetch_result(
        url=args.url,
        source=source,
        text=text,
        output_dir=resolve_path(args.output_dir),
        metadata_dir=resolve_path(args.metadata_dir),
        max_chars=args.max_chars,
    )

    print("=" * 70)
    print("Fetched web text into pending review.")
    print("=" * 70)
    print(f"URL: {args.url}")
    print(f"Source: {source['name']}")
    print(f"License: {source.get('license', 'unknown')}")
    print(f"Usage status: {source.get('usage_status', 'unknown')}")
    print(f"Text file: {text_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Characters: {len(text):,}")
    print("Pending human review before training use.")
    print("=" * 70)


if __name__ == "__main__":
    main()
