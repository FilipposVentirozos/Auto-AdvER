#!/usr/bin/env python3
"""Corpus statistics for Auto-AdvER, in markdown.

Reports, for one or more JSONL files in either format: the number of adverts,
the number of tokens, how many adverts carry no entity, the number of entities
per label, and the token and character coverage per label, plus the advertiser
type and region distributions.

**On the paper's Table 1.** Its "# Tokens" column is in fact a character count:
the numbers published there (13,107 / 21,227 / 38,358) are the sums of
``end - start + 1`` over each label's spans, i.e. inclusive character lengths.
The true token counts are 2,395 / 3,887 / 6,058.  Both are reported below, each
under its own heading, so the two can never be confused again.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_tools.records import LABELS_DEFAULT, iter_documents  # noqa: E402


def analyse(path: str) -> Dict[str, Any]:
    """Collect the statistics for one file."""
    n_docs = 0
    n_tokens = 0
    n_empty = 0
    spans_per_label: Counter = Counter()
    tokens_per_label: Counter = Counter()
    chars_per_label: Counter = Counter()
    advertiser_type: Counter = Counter()
    region: Counter = Counter()

    for doc in iter_documents(path):
        n_docs += 1
        if doc.tokens is not None:
            n_tokens += len(doc.tokens)
        token_spans = doc.token_spans()
        char_spans = doc.char_spans()
        if not token_spans:
            n_empty += 1
        for span in token_spans:
            spans_per_label[span["label"]] += 1
            tokens_per_label[span["label"]] += span["token_end"] - span["token_start"] + 1
        for span in char_spans:
            # Inclusive character length, which is what the paper's Table 1 sums.
            chars_per_label[span["label"]] += span["end"] - span["start"] + 1
        advertiser_type[doc.meta_value("advertiser_type")] += 1
        region[doc.meta_value("region")] += 1

    return {
        "path": path,
        "docs": n_docs,
        "tokens": n_tokens,
        "empty_docs": n_empty,
        "spans_per_label": spans_per_label,
        "tokens_per_label": tokens_per_label,
        "chars_per_label": chars_per_label,
        "advertiser_type": advertiser_type,
        "region": region,
    }


def _labels_in(results: Sequence[Dict[str, Any]], labels: Sequence[str]) -> List[str]:
    seen = [label for label in labels]
    for result in results:
        for label in result["spans_per_label"]:
            if label not in seen:
                seen.append(label)
    return seen


def column_names(results: Sequence[Dict[str, Any]]) -> List[str]:
    """Shortest distinct labels for the columns: file names, else parent/file."""
    names = [os.path.basename(result["path"]) for result in results]
    if len(set(names)) == len(names):
        return names
    return [
        os.path.join(os.path.basename(os.path.dirname(result["path"])), name)
        for result, name in zip(results, names)
    ]


def render_markdown(results: Sequence[Dict[str, Any]], labels: Sequence[str]) -> str:
    names = column_names(results)
    labels = _labels_in(results, labels)
    lines: List[str] = ["# Auto-AdvER corpus statistics", ""]
    lines.append("| Metric | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")

    def row(title: str, values: Sequence[Any]) -> None:
        lines.append("| %s | %s |" % (title, " | ".join(str(value) for value in values)))

    row("Adverts", [result["docs"] for result in results])
    row("Tokens", ["{:,}".format(result["tokens"]) for result in results])
    row(
        "Mean tokens per advert",
        [
            "%.1f" % (result["tokens"] / result["docs"] if result["docs"] else 0)
            for result in results
        ],
    )
    row(
        "Adverts without entities",
        [
            "%d (%.1f%%)"
            % (
                result["empty_docs"],
                result["empty_docs"] / result["docs"] * 100 if result["docs"] else 0,
            )
            for result in results
        ],
    )
    row("Entities", [sum(result["spans_per_label"].values()) for result in results])

    lines.append("")
    lines.append("## Entities per label")
    lines.append("")
    lines.append("| Label | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for label in labels:
        row(label, [result["spans_per_label"][label] for result in results])
    row("Total", [sum(result["spans_per_label"].values()) for result in results])

    lines.append("")
    lines.append("## Tokens covered per label (token_end − token_start + 1)")
    lines.append("")
    lines.append("| Label | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for label in labels:
        row(label, ["{:,}".format(result["tokens_per_label"][label]) for result in results])
    row(
        "Total",
        ["{:,}".format(sum(result["tokens_per_label"].values())) for result in results],
    )

    lines.append("")
    lines.append(
        "## Characters covered per label (end − start + 1)"
    )
    lines.append("")
    lines.append("| Label | " + " | ".join(names) + " |")
    lines.append("|---" * (len(names) + 1) + "|")
    for label in labels:
        row(label, ["{:,}".format(result["chars_per_label"][label]) for result in results])
    row(
        "Total",
        ["{:,}".format(sum(result["chars_per_label"].values())) for result in results],
    )

    for field in ("advertiser_type", "region"):
        keys: List[Any] = []
        for result in results:
            for key, _ in result[field].most_common():
                if key not in keys:
                    keys.append(key)
        if keys == [None]:
            continue
        lines.append("")
        lines.append("## %s" % field.replace("_", " ").capitalize())
        lines.append("")
        lines.append("| Value | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 1) + "|")
        for key in keys:
            row(
                "—" if key is None else str(key),
                [
                    "%d (%.1f%%)"
                    % (
                        result[field][key],
                        result[field][key] / result["docs"] * 100 if result["docs"] else 0,
                    )
                    for result in results
                ],
            )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stats.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="one or more JSONL files to describe")
    parser.add_argument(
        "--labels",
        default=",".join(LABELS_DEFAULT),
        help="comma-separated label order for the tables; any further label found "
        "in the data is appended",
    )
    parser.add_argument("--out", help="write markdown here instead of stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    labels = [label for label in args.labels.split(",") if label]
    results = [analyse(path) for path in args.inputs]
    text = render_markdown(results, labels)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
