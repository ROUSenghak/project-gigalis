#!/usr/bin/env python3
"""Check the seal without breaking it, and report how often it has been opened.

Run this before quoting any sealed-test number. It confirms the file still
matches the checksum it was sealed with, and prints the opening history so the
write-up can state how many times the seal was broken rather than implying it
never was.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boamp_pipeline.sealed_split import verify  # noqa: E402

DEFAULT_SEALED = Path("data/processed/boamp_v2/benchmark_v3/sealed/benchmark_v3_test.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sealed", type=Path, default=DEFAULT_SEALED)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    sealed = args.sealed if args.sealed.is_absolute() else project_root / args.sealed

    result = verify(sealed, project_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
