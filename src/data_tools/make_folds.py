#!/usr/bin/env python3
"""Re-cut the Auto-AdvER cross-validation folds.

Port of ``prepping_for_ner.py``, the script that produced the published
partition.  The recipe is:

1. read the corpus, keep the accepted annotations, and de-duplicate by document
   id, keeping the first occurrence's position;
2. ``KFold(n_splits=7, shuffle=True, random_state=42)`` over that list, whose
   held-out part becomes the fold's test set;
3. ``train_test_split(test_size=0.1765, random_state=42)`` over the remainder,
   giving roughly 70 % train and 15 % validation.

Only folds 1–3 were used in the paper and only those are written by default;
``--all-folds`` writes all seven.

The partition depends on the input order, because ``KFold`` shuffles indices
rather than documents.  It reproduces the published partition exactly from the
corpus **in annotation order**; the released ``data/auto_adver.jsonl`` is sorted
by advert id instead, so re-cutting from it yields a different, equally valid
partition.  Pass ``--order`` with a JSON list of document ids to impose the
original order, or simply use the fold files as released.  ``--verify-against``
compares the computed id sets with an existing set of fold files and reports the
outcome per fold and split.

This is the one tool in the package with a third-party dependency: the exact
partition is a property of scikit-learn's shuffling, so it has to be
scikit-learn that does it.
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
    iter_documents,
    write_jsonl,
)

SPLITS = ("train", "val", "test")


def load_corpus(path: str, accepted_only: bool = True) -> Tuple[List[Document], Dict[str, int]]:
    """Load the corpus in file order, de-duplicated by id, with a small report.

    Prodigy records carry an ``answer`` field; only ``accept`` counts as an
    annotated advert, as in the original script.  A repeated id replaces the
    earlier record's *content* but keeps its *position*, which is what a Python
    dict does and therefore what the original did.
    """
    by_id: "Dict[str, Document]" = {}
    stats = {"read": 0, "rejected": 0, "duplicates": 0}
    for doc in iter_documents(path):
        stats["read"] += 1
        if accepted_only and doc.fmt == "prodigy":
            if doc.raw.get("answer") not in (None, "accept"):
                stats["rejected"] += 1
                continue
        if doc.doc_id in by_id:
            stats["duplicates"] += 1
        by_id[doc.doc_id] = doc
    return list(by_id.values()), stats


def reorder(documents: Sequence[Document], order_path: str) -> List[Document]:
    """Reorder the corpus to the document ids listed in a JSON file."""
    with open(order_path, "r", encoding="utf-8") as handle:
        order = [str(doc_id) for doc_id in json.load(handle)]
    by_id = {doc.doc_id: doc for doc in documents}
    missing = [doc_id for doc_id in order if doc_id not in by_id]
    if missing:
        raise SystemExit(
            "%d ids in %s are not in the corpus (e.g. %s)"
            % (len(missing), order_path, missing[:5])
        )
    if len(order) != len(documents):
        raise SystemExit(
            "%s lists %d ids but the corpus holds %d adverts"
            % (order_path, len(order), len(documents))
        )
    return [by_id[doc_id] for doc_id in order]


def make_folds(
    documents: Sequence[Document], n_splits: int = 7
) -> List[Dict[str, List[int]]]:
    """Return the index lists per fold, exactly as the original script did."""
    from sklearn.model_selection import KFold, train_test_split  # noqa: PLC0415

    samples = list(documents)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds: List[Dict[str, List[int]]] = []
    for train_val_indices, test_indices in kfold.split(samples):
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=0.1765, random_state=42
        )
        folds.append(
            {
                "train": list(train_indices),
                "val": list(val_indices),
                "test": list(test_indices),
            }
        )
    return folds


def fold_path(out_dir: str, fold: int, split: str, layout: str) -> str:
    """Where a fold file goes: ``f1_train.jsonl`` or ``f1/train.jsonl``."""
    if layout == "nested":
        return os.path.join(out_dir, "f%d" % fold, "%s.jsonl" % split)
    return os.path.join(out_dir, "f%d_%s.jsonl" % (fold, split))


def write_folds(
    documents: Sequence[Document],
    folds: Sequence[Dict[str, List[int]]],
    out_dir: str,
    layout: str,
) -> List[str]:
    """Write the fold files and return the paths written."""
    written: List[str] = []
    for number, fold in enumerate(folds, start=1):
        for split in SPLITS:
            path = fold_path(out_dir, number, split, layout)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            write_jsonl(path, [documents[index].raw for index in fold[split]])
            written.append(path)
    return written


def verify(
    documents: Sequence[Document],
    folds: Sequence[Dict[str, List[int]]],
    reference_dir: str,
    layout: str,
) -> List[Dict[str, Any]]:
    """Compare the computed id sets with an existing set of fold files."""
    rows: List[Dict[str, Any]] = []
    for number, fold in enumerate(folds, start=1):
        for split in SPLITS:
            path = fold_path(reference_dir, number, split, layout)
            if not os.path.exists(path):
                rows.append(
                    {"fold": number, "split": split, "status": "reference missing", "path": path}
                )
                continue
            expected = {doc.doc_id for doc in iter_documents(path)}
            produced = {documents[index].doc_id for index in fold[split]}
            rows.append(
                {
                    "fold": number,
                    "split": split,
                    "status": "identical" if produced == expected else "DIFFERENT",
                    "n_produced": len(produced),
                    "n_expected": len(expected),
                    "only_produced": len(produced - expected),
                    "only_expected": len(expected - produced),
                    "path": path,
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_folds.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus", required=True, help="corpus JSONL file (Prodigy or public format)"
    )
    parser.add_argument("--out-dir", help="directory to write the fold files into")
    parser.add_argument(
        "--order",
        metavar="JSON",
        help="JSON list of document ids giving the input order. The published "
        "partition is a property of that order, so this is needed whenever the "
        "corpus file is not in its original annotation order",
    )
    parser.add_argument(
        "--layout",
        choices=("flat", "nested"),
        default="nested",
        help="flat writes f1_train.jsonl, nested writes f1/train.jsonl",
    )
    parser.add_argument(
        "--all-folds",
        action="store_true",
        help="write all seven folds instead of only the three the paper used",
    )
    parser.add_argument(
        "--n-splits", type=int, default=7, help="number of cross-validation folds"
    )
    parser.add_argument(
        "--verify-against",
        metavar="DIR",
        help="compare the computed document id sets against the fold files in DIR "
        "and report per fold and split, instead of (or as well as) writing",
    )
    parser.add_argument(
        "--verify-layout",
        choices=("flat", "nested"),
        help="layout of --verify-against (defaults to --layout)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the verification report as JSON"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.out_dir and not args.verify_against:
        raise SystemExit("nothing to do: pass --out-dir, --verify-against, or both")

    documents, stats = load_corpus(args.corpus)
    print(
        "read %d records from %s: %d rejected, %d duplicate ids, %d adverts kept"
        % (
            stats["read"],
            args.corpus,
            stats["rejected"],
            stats["duplicates"],
            len(documents),
        )
    )
    if args.order:
        documents = reorder(documents, args.order)
        print("reordered the corpus to the %d ids in %s" % (len(documents), args.order))
    folds = make_folds(documents, n_splits=args.n_splits)
    keep = folds if args.all_folds else folds[:3]

    if args.out_dir:
        written = write_folds(documents, keep, args.out_dir, args.layout)
        sizes = ", ".join(
            "f%d %d/%d/%d"
            % (number, len(fold["train"]), len(fold["val"]), len(fold["test"]))
            for number, fold in enumerate(keep, start=1)
        )
        print("wrote %d files to %s (%s)" % (len(written), args.out_dir, sizes))

    if args.verify_against:
        rows = verify(
            documents, keep, args.verify_against, args.verify_layout or args.layout
        )
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print()
            print("| Fold | Split | Produced | Expected | Status |")
            print("|---|---|---|---|---|")
            for row in rows:
                print(
                    "| f%s | %s | %s | %s | %s |"
                    % (
                        row["fold"],
                        row["split"],
                        row.get("n_produced", "—"),
                        row.get("n_expected", "—"),
                        row["status"],
                    )
                )
        if any(row["status"] != "identical" for row in rows):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
