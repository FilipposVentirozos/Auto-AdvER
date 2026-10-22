#!/usr/bin/env python3
"""Convert Auto-AdvER documents to a Hugging Face token-classification file.

Output is a JSON list of ``{"ner_tags": [...], "tokens": [...]}`` objects, the
shape ``run_ner.py`` reads.  This is the port of ``hugging_face_ner_convert.py``.

Two tag schemes:

``--iob2`` (default)
    ``B-LABEL`` on the first token of an entity, ``I-LABEL`` after, ``O``
    elsewhere.  Because a flat tag sequence cannot express nested or
    overlapping entities, overlapping spans are resolved first: spans are
    considered longest-first from the left, and a span overlapping one already
    placed is dropped.  The number dropped is reported, never silenced.

``--plain-labels``
    The original scheme, which wrote the bare label on every token of an
    entity and let a later span overwrite an earlier one wherever they
    overlapped.  It is kept because it is what the encoder models in the paper
    were trained on, and it reproduces the shipped ``*_hf.json`` files exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_tools.records import Document, load_document_list  # noqa: E402


def resolve_overlaps(
    spans: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop spans that overlap one already accepted; return them and the count.

    Spans are ordered by start token and then by decreasing length, so the
    outermost of a nested pair wins.
    """
    ordered = sorted(
        spans, key=lambda s: (s["token_start"], -(s["token_end"] - s["token_start"]))
    )
    kept: List[Dict[str, Any]] = []
    taken: set = set()
    dropped = 0
    for span in ordered:
        window = set(range(span["token_start"], span["token_end"] + 1))
        if window & taken:
            dropped += 1
            continue
        taken |= window
        kept.append(span)
    return kept, dropped


def tags_for_document(
    doc: Document, iob2: bool = True
) -> Tuple[List[str], int]:
    """Return the tag sequence for one document and the number of spans dropped."""
    tokens = doc.tokens
    if tokens is None:
        raise SystemExit("document %s carries no tokens" % doc.doc_id)
    tags = ["O"] * len(tokens)
    spans = doc.token_spans()
    if not iob2:
        # The original scheme: file order, later spans overwrite earlier ones.
        for span in spans:
            for index in range(span["token_start"], span["token_end"] + 1):
                tags[index] = span["label"]
        return tags, 0
    kept, dropped = resolve_overlaps(spans)
    for span in kept:
        for offset, index in enumerate(range(span["token_start"], span["token_end"] + 1)):
            tags[index] = ("B-" if offset == 0 else "I-") + span["label"]
    return tags, dropped


def convert(path: str, iob2: bool = True) -> Tuple[List[Dict[str, Any]], int]:
    """Convert a JSONL file; return the HF records and the spans dropped."""
    records: List[Dict[str, Any]] = []
    dropped_total = 0
    for doc in load_document_list(path):
        tags, dropped = tags_for_document(doc, iob2=iob2)
        dropped_total += dropped
        records.append({"ner_tags": tags, "tokens": list(doc.tokens or [])})
    return records, dropped_total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="to_hf.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="JSONL file (Prodigy or public)")
    parser.add_argument("--out", required=True, help="output JSON path")
    scheme = parser.add_mutually_exclusive_group()
    scheme.add_argument(
        "--iob2",
        dest="iob2",
        action="store_true",
        default=True,
        help="write IOB2 tags (the default)",
    )
    scheme.add_argument(
        "--plain-labels",
        dest="iob2",
        action="store_false",
        help="write the original bare-label scheme, which reproduces the shipped "
        "*_hf.json files exactly",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="JSON indentation (4 matches the shipped files)",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print a summary")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records, dropped = convert(args.input, iob2=args.iob2)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=args.indent)
    if not args.quiet:
        print(
            "wrote %d documents to %s (%s tags)"
            % (len(records), args.out, "IOB2" if args.iob2 else "plain")
        )
        if dropped:
            print(
                "%d overlapping span(s) dropped: a flat tag sequence cannot "
                "represent them" % dropped,
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
