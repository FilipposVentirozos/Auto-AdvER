# -*- coding: utf-8 -*-
"""Strip gold spans from a test fold, so an LLM cannot see them.

Port of the original span-stripping script, which hard-coded the three folds and the
input and output directories.  Everything is a flag now, and one file is handled
per call.

The original wrote the stripped fold into a separate ``intermediate/``
directory; ``predict.py`` overwrites ``spans`` regardless, so this is belt and
braces -- useful when you want to eyeball the file that goes to the model, or
when you pass a fold to some other tool.

Example::

    python src/llm/clear_test.py --in data/folds/f1/test.jsonl \\
        --out work/f1_test_blank.jsonl

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List


def load_records(path: str) -> List[dict]:
    """Read JSON Lines, or a single top-level JSON array."""
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def clear_spans(records: List[dict], field: str = "spans") -> List[dict]:
    """Blank *field* on every record, in place, and return them."""
    for record in records:
        record[field] = []
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clear_test.py",
        description="Copy a fold file, blanking its gold spans.")
    parser.add_argument("--in", dest="input", required=True,
                        help="input fold (JSON Lines or a JSON array)")
    parser.add_argument("--out", required=True, help="output JSON Lines path")
    parser.add_argument("--field", default="spans",
                        help="record field to blank (default: %(default)s)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    records = clear_spans(load_records(args.input), field=args.field)
    directory = os.path.dirname(os.path.abspath(args.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} records with an empty '{args.field}' field "
          f"to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
