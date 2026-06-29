from pathlib import Path
from urllib.parse import urlparse
import argparse
import json


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "web_sources_allowlist.json"


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalized_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid base_url in allowlist: {base_url}")

    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def path_is_allowed(request_path: str, allowed_path: str) -> bool:
    if not allowed_path.startswith("/"):
        allowed_path = f"/{allowed_path}"

    if allowed_path == "/":
        return True

    normalized_allowed = allowed_path.rstrip("/")

    return (
        request_path == normalized_allowed
        or request_path.startswith(f"{normalized_allowed}/")
    )


def validate_url_against_config(url: str, config: dict) -> tuple[bool, str, dict | None]:
    parsed_url = urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        return False, "Only http and https URLs are supported.", None

    if not parsed_url.netloc:
        return False, "URL has no domain.", None

    request_base = f"{parsed_url.scheme.lower()}://{parsed_url.netloc.lower()}"
    request_path = parsed_url.path or "/"

    for source in config.get("sources", []):
        source_base = normalized_base_url(source["base_url"])

        if request_base != source_base:
            continue

        allowed_paths = source.get("allowed_paths", [])

        if not allowed_paths:
            return (
                False,
                f"Source '{source['name']}' has no allowed_paths entries.",
                source,
            )

        if any(path_is_allowed(request_path, allowed_path) for allowed_path in allowed_paths):
            return True, f"Allowed by source '{source['name']}'.", source

        return (
            False,
            f"URL path '{request_path}' is outside allowed_paths for source '{source['name']}'.",
            source,
        )

    return False, "URL domain is not in the allowlist.", None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether a URL is allowed by the DarkMind web allowlist."
    )
    parser.add_argument(
        "--url",
        required=True,
        help="URL to check. No network request is made.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH.relative_to(ROOT_DIR)),
        help="Allowlist config path.",
    )
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    config = load_config(config_path)
    allowed, reason, source = validate_url_against_config(args.url, config)

    print("=" * 70)
    print("DarkMind Web Source Policy Check")
    print("=" * 70)
    print(f"URL: {args.url}")
    print(f"Config: {config_path}")
    print(f"Status: {'ALLOWED' if allowed else 'BLOCKED'}")
    print(f"Reason: {reason}")

    if source:
        print(f"Source: {source.get('name')}")
        print(f"License: {source.get('license', 'unknown')}")
        print(f"Usage status: {source.get('usage_status', 'unknown')}")

    print("=" * 70)

    if not allowed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
