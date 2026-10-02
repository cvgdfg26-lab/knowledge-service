"""Run with the backend interpreter. Copies credentials without echoing their values."""

import argparse
import json

from knowledge_service.core.env_import import prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="prompt/.env")
    parser.add_argument("--output", default=".env.existing")
    args = parser.parse_args()
    try:
        report = prepare(args.source, args.output)
    except (ValueError, OSError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
