#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def parse_pairs(raw: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"invalid config item {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"invalid config item {item!r}; key is empty")
        config[key] = value
    return config


def coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def normalize(config: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(config):
        value = config[key]
        if isinstance(value, str):
            normalized[key] = coerce(value)
        elif isinstance(value, list):
            normalized[key] = [coerce(item) if isinstance(item, str) else item for item in value]
        else:
            normalized[key] = value
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add a config field. Repeat for every field that should contribute to the fingerprint.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = normalize(parse_pairs(args.set))
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
