#!/usr/bin/env python3
"""Turn per-token label sequences into span-level predictions.

The CRF baselines and the fine-tuned encoders both produce one label per
token; the scorer works on spans.  This is the conversion that builds the
shipped prediction files of the eight token-label configurations from the raw
label sequences under ``predictions/*/raw_token_labels/``.

The two kinds of label list differ in length.  A CRF list holds one label per
token, whitespace-only tokens (``"\n"``, ``" "``, ``"\n\n"``, ``"\t"``,
``"\r\n"``) included, because its features were built over every token.
Hugging Face's ``run_ner.py`` writes one label per *word that produced at
least one sub-word*, and a whitespace-only word produces none, so an encoder
list is shorter than its token list by the number of whitespace-only tokens
the document contains.  ``--alignment`` decides how a label list is laid over
a token list:

``auto`` (the default)
    ``positional`` when ``len(labels) == len(tokens)``; ``skip-whitespace``
    when ``len(labels)`` equals the number of non-whitespace tokens, and also
    when it is smaller than that (a document truncated by the model's maximum
    sequence length: the labels are laid over the leading non-whitespace
    tokens, the tail is left unlabelled, and a warning names the document and
    the shortfall).  Any other length is refused, because there is no defensible
    way to line the two up.
``positional``
    zip labels onto tokens by index; the two lists must have the same length.
``skip-whitespace``
    lay the labels over the non-whitespace tokens only; whitespace-only tokens
    stay ``O`` and therefore break a span in two if one runs across them.

A whitespace-only token is one whose text is empty after ``str.strip()``.

Input labels may be plain (``SALES_OPTIONS``) or IOB2 (``B-SALES_OPTIONS``);
the scheme is detected automatically.  Under the plain scheme a span ends only
when the label changes or becomes ``O``, so two adjacent entities of the same
type merge into one span.  A span still open at the end of a document closes
on the document's last token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_tools.records import (  # noqa: E402
    Document,
    FormatError,
    load_document_list,
    write_jsonl,
)


def is_iob2(label_lists: Sequence[Sequence[str]]) -> bool:
    """True if any label carries a ``B-``/``I-`` prefix."""
    return any(
        label[:2] in ("B-", "I-") for labels in label_lists for label in labels
    )


def spans_from_labels(
    labels: Sequence[str], iob2: bool = False
) -> List[Dict[str, Any]]:
    """Convert one document's token labels into inclusive token spans.

    A span still open at the end of the sequence closes on its last label.
    """
    spans: List[Dict[str, Any]] = []
    token_start: Optional[int] = None
    current: Optional[str] = None

    for index, raw in enumerate(labels):
        if raw == "O":
            if current is not None:
                spans.append(
                    {"token_start": token_start, "token_end": index - 1, "label": current}
                )
                current = None
                token_start = None
            continue
        prefix, _, stripped = raw.partition("-")
        label = stripped if iob2 and prefix in ("B", "I") else raw
        begins = iob2 and prefix == "B"
        if current is None:
            token_start = index
            current = label
        elif label != current or begins:
            spans.append(
                {"token_start": token_start, "token_end": index - 1, "label": current}
            )
            token_start = index
            current = label

    if current is not None:
        spans.append(
            {"token_start": token_start, "token_end": len(labels) - 1, "label": current}
        )
    return spans


ALIGNMENTS = ("auto", "positional", "skip-whitespace")


def is_whitespace_token(token: str) -> bool:
    """True for a token that is empty once stripped (``"\n"``, ``" "``, ...)."""
    return token.strip() == ""


def count_content_tokens(tokens: Sequence[str]) -> int:
    """How many of ``tokens`` are not whitespace-only."""
    return sum(1 for token in tokens if not is_whitespace_token(token))


def align_labels(
    labels: Sequence[str],
    tokens: Sequence[str],
    alignment: str = "auto",
    doc_id: str = "?",
) -> Tuple[List[str], Optional[str]]:
    """Lay ``labels`` over ``tokens`` and return ``(per-token labels, warning)``.

    Under ``skip-whitespace`` the result always has one entry per token, with
    ``O`` on every whitespace-only token and on any tail the label list does not
    reach.  Under ``positional`` the label list is returned unchanged and must
    hold one label per token.

    Raises :class:`FormatError` when the two cannot be lined up.
    """
    if alignment not in ALIGNMENTS:
        raise FormatError(
            "unknown alignment %r; expected one of %s"
            % (alignment, ", ".join(ALIGNMENTS))
        )
    n_labels = len(labels)
    n_tokens = len(tokens)
    n_content = count_content_tokens(tokens)
    warning: Optional[str] = None

    if alignment == "auto":
        if n_labels == n_tokens:
            alignment = "positional"
        elif n_labels <= n_content:
            alignment = "skip-whitespace"
        else:
            raise FormatError(
                "document %s: %d labels for %d tokens (%d of them not "
                "whitespace-only); the label list matches neither the tokens "
                "nor the non-whitespace tokens, so it cannot be aligned."
                % (doc_id, n_labels, n_tokens, n_content)
            )

    if alignment == "positional":
        if n_labels != n_tokens:
            raise FormatError(
                "document %s: %d tokens but %d labels; --alignment positional "
                "needs one label per token" % (doc_id, n_tokens, n_labels)
            )
        return list(labels), warning

    # skip-whitespace
    if n_labels > n_content:
        raise FormatError(
            "document %s: %d labels but only %d non-whitespace tokens; "
            "--alignment skip-whitespace needs at most one label per "
            "non-whitespace token" % (doc_id, n_labels, n_content)
        )
    if n_labels < n_content:
        warning = (
            "document %s: %d labels for %d non-whitespace tokens (%d short, a "
            "truncated document); the labels are laid over the leading tokens "
            "and the tail is left unlabelled"
            % (doc_id, n_labels, n_content, n_content - n_labels)
        )
    aligned = ["O"] * n_tokens
    position = 0
    for index, token in enumerate(tokens):
        if is_whitespace_token(token):
            continue
        if position < n_labels:
            aligned[index] = labels[position]
        position += 1
    return aligned, warning


def _public_entity(
    span: Dict[str, Any], doc: Document
) -> Dict[str, Any]:
    entity: Dict[str, Any] = {
        "label": span["label"],
        "token_start": span["token_start"],
        "token_end": span["token_end"],
    }
    offsets = doc.offsets
    start, end = span["token_start"], span["token_end"]
    if offsets is not None and 0 <= start <= end < len(offsets):
        entity = {
            "start": offsets[start][0],
            "end": offsets[end][1],
            "label": span["label"],
            "token_start": start,
            "token_end": end,
        }
    return entity


def convert(
    gold_path: str,
    labels_path: str,
    ids_path: Optional[str] = None,
    alignment: str = "auto",
) -> Tuple[List[Document], List[List[Dict[str, Any]]], List[str]]:
    """Read a gold fold and a token-label file and build the predicted spans.

    Returns the gold documents in the order they were paired, the span list per
    document, and a list of human-readable warnings (length mismatches and
    truncated documents).  ``alignment`` is one of :data:`ALIGNMENTS`.
    """
    docs = load_document_list(gold_path)
    with open(labels_path, "r", encoding="utf-8") as handle:
        label_lists = json.load(handle)
    if not isinstance(label_lists, list) or not all(
        isinstance(item, list) for item in label_lists
    ):
        raise FormatError("%s must hold a JSON list of per-document label lists" % labels_path)

    if ids_path:
        with open(ids_path, "r", encoding="utf-8") as handle:
            ids = [str(doc_id) for doc_id in json.load(handle)]
        if len(ids) != len(label_lists):
            raise FormatError(
                "%s lists %d ids but %s holds %d label lists"
                % (ids_path, len(ids), labels_path, len(label_lists))
            )
        by_id = {doc.doc_id: doc for doc in docs}
        missing = [doc_id for doc_id in ids if doc_id not in by_id]
        if missing:
            raise FormatError(
                "%d document ids from %s are not in %s (e.g. %s)"
                % (len(missing), ids_path, gold_path, missing[:5])
            )
        paired = [by_id[doc_id] for doc_id in ids]
    else:
        if len(docs) != len(label_lists):
            raise FormatError(
                "%s holds %d documents but %s holds %d label lists; pass --ids to "
                "align by document id instead of file order"
                % (gold_path, len(docs), labels_path, len(label_lists))
            )
        paired = docs

    iob2 = is_iob2(label_lists)

    warnings: List[str] = []
    all_spans: List[List[Dict[str, Any]]] = []
    for doc, labels in zip(paired, label_lists):
        if doc.tokens is None:
            if alignment == "skip-whitespace":
                raise FormatError(
                    "document %s carries no tokens, so whitespace-only tokens "
                    "cannot be skipped; --alignment skip-whitespace needs a "
                    "gold file with tokens" % doc.doc_id
                )
            aligned, warning = list(labels), None
        else:
            aligned, warning = align_labels(
                labels, doc.tokens, alignment=alignment, doc_id=doc.doc_id
            )
        if warning:
            warnings.append(warning)
        all_spans.append(spans_from_labels(aligned, iob2=iob2))
    return paired, all_spans, warnings


def to_records(
    docs: Sequence[Document],
    spans: Sequence[List[Dict[str, Any]]],
    out_format: str,
) -> List[Dict[str, Any]]:
    """Render the predictions as Prodigy-style or public-style records."""
    records: List[Dict[str, Any]] = []
    for doc, doc_spans in zip(docs, spans):
        if out_format == "prodigy":
            record = dict(doc.raw)
            record["spans"] = doc_spans
            records.append(record)
        else:
            records.append(
                {
                    "id": doc.doc_id,
                    "entities": [_public_entity(span, doc) for span in doc_spans],
                }
            )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tags_to_spans.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gold", required=True, help="gold fold file (Prodigy or public)")
    parser.add_argument(
        "--labels",
        required=True,
        help="JSON file holding a list of per-document token-label lists",
    )
    parser.add_argument(
        "--ids",
        help="optional JSON list of document ids matching --labels, used to pair "
        "them with the gold documents by id instead of by file order",
    )
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--out-format",
        choices=("prodigy", "public"),
        default="public",
        help="record layout to write. 'prodigy' copies the gold record and "
        "replaces its spans, which reproduces the shipped prediction files; "
        "given a public gold file it therefore writes public fields with a "
        "'spans' key added, so pair it with a Prodigy --gold",
    )
    parser.add_argument(
        "--alignment",
        choices=ALIGNMENTS,
        default="auto",
        help="how to lay the label list over the token list. 'auto' picks "
        "'positional' when the two are the same length and 'skip-whitespace' "
        "when the labels match the non-whitespace tokens (or fall short of "
        "them, as in a truncated document), and refuses any other length. "
        "'positional' needs one label per token, as in the CRF files; "
        "'skip-whitespace' leaves whitespace-only tokens unlabelled, as the "
        "encoder files need",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="do not print the summary line"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    docs, spans, warnings = convert(
        args.gold,
        args.labels,
        ids_path=args.ids,
        alignment=args.alignment,
    )
    written = write_jsonl(args.out, to_records(docs, spans, args.out_format))
    if not args.quiet:
        total = sum(len(doc_spans) for doc_spans in spans)
        print(
            "wrote %d documents and %d spans to %s (%s alignment)"
            % (written, total, args.out, args.alignment)
        )
    for warning in warnings:
        print("warning: %s" % warning, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
