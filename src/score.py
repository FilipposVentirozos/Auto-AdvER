#!/usr/bin/env python3
"""Auto-AdvER scorer — the paper's partial-match (MUC-style) evaluation.

One tool replaces the seven scoring scripts used for the paper.  It implements
the paper's matching rules and nothing else, so that every published number
reproduces from the released files.

Four sub-commands:

``score.py models``
    Score one or more model prediction files against the gold test folds.  Each
    fold's COR/INC/PAR/MIS/SPU counts are pooled over its documents, exact and
    partial precision/recall/F1 are computed from the pooled counts, and the
    per-fold percentages are then averaged over the folds with a population
    standard deviation — the shape of the paper's results table.

``score.py boundaries``
    Where a partial match's boundaries go wrong: each PAR of the ``models``
    matching is classified as too wide, too narrow or staggered, and the three
    shares are reported per model, pooled over the folds.  This is the table
    behind the paper's error analysis.

``score.py iaa``
    Inter-annotator agreement: pairwise partial-match F1 for each pair of
    annotators (both directions pooled), and the per-label table over all six
    ordered pairs that the paper's appendix reports.

``score.py families``
    The same mention under all three labels: how the gold spans of a handful of
    recurring mention families (MOT, service, tyre, owner, mileage) divide over
    the three labels, how the models fare on each family, where the two expert
    annotators disagree about the label of an identically bounded span, and
    short candidate snippets for the paper's example table.

All of them accept the released public format and the internal Prodigy format; see
``data_tools.records``.  Matching is always on inclusive token indices, so
character offsets in public records are converted first (with the gold
document's tokens when a prediction file carries no tokens of its own).

Definitions, as in the paper (§ evaluation):

    possible = COR + INC + PAR + MIS
    actual   = COR + INC + PAR + SPU
    exact    P = COR / actual                R = COR / possible
    partial  P = (COR + 0.5·PAR) / actual    R = (COR + 0.5·PAR) / possible
    F1       = 2PR / (P + R)

British spelling is used throughout the documentation; JSON field names follow
the data files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_tools.records import (  # noqa: E402  (path set up above)
    LABELS_DEFAULT,
    Document,
    FormatError,
    load_documents,
)

COUNT_KEYS = ("cor", "inc", "par", "mis", "spu")

_DOC_CACHE: "Dict[str, Dict[str, Document]]" = {}


def documents(path: str) -> Dict[str, Document]:
    """Load a JSONL file once per process (the per-label passes re-read files)."""
    cached = _DOC_CACHE.get(path)
    if cached is None:
        cached = load_documents(path)
        _DOC_CACHE[path] = cached
    return cached


class Span:
    """A span as the paper's scorer sees it: inclusive token indices + label.

    ``get_score`` is a byte-for-byte port of ``Span.get_score`` as it appears,
    duplicated verbatim, in all seven of the project's original scoring scripts
    (``iaa_scoring.py``, ``iaa_scoring_per_label*.py``, ``f1_scoring_*.py``).
    It is kept exactly as it was, together with the counting loops in
    :func:`score_pair`, because the published numbers depend on its behaviour.

    Two properties of this scheme are worth stating plainly, since they differ
    from a one-to-one matcher such as SemEval-2013's:

    1. The overlap test fires only when one of the compared span's endpoints
       falls inside the reference span.  If the compared span *strictly*
       contains the reference span on both sides, the test returns ``None``.
       Because the counting loop calls the test from the gold side for MIS and
       from the predicted side for SPU, a prediction that swallows a gold span
       whole is counted as a miss but not as a spurious prediction: it drops out
       of ``actual``, so recall is depressed and precision is not.  In the
       mirror case, where a gold span strictly contains a prediction, the gold
       span is matched as PAR while the prediction is still counted spurious.
    2. Matching is greedy first-match and not one-to-one: the loop breaks on the
       first span that matches, and the same span may be matched by several
       spans on the other side.

    Both are inherent to the published evaluation and are preserved here
    deliberately; the release notes quantify their effect.  The only additions
    to the original loops are the optional ``partial_pairs`` and ``outcomes``
    collectors of :func:`score_spans`, which record what the loop already
    counts and cannot change a count.
    """

    __slots__ = ("start", "end", "label")

    def get_score(self, cls: "Span") -> Optional[str]:
        if self.start == cls.start and self.end == cls.end:
            if self.label != cls.label:
                return "inc"
            return "cor"
        if self.end >= cls.start >= self.start or self.end >= cls.end >= self.start:
            if self.label != cls.label:
                return "inc"
            return "par"
        return None

    def __init__(self, span: Dict[str, Any]) -> None:
        self.start = span["token_start"]
        self.end = span["token_end"]
        self.label = span["label"]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Span(%d, %d, %r)" % (self.start, self.end, self.label)


def zero_counts() -> Dict[str, int]:
    """A fresh COR/INC/PAR/MIS/SPU accumulator."""
    return {key: 0 for key in COUNT_KEYS}


def add_counts(into: Dict[str, int], other: Dict[str, int]) -> Dict[str, int]:
    """Add ``other`` into ``into`` in place and return it."""
    for key in COUNT_KEYS:
        into[key] += other[key]
    return into


def score_spans(
    reference: Sequence[Dict[str, Any]],
    compared: Sequence[Dict[str, Any]],
    label: Optional[str] = None,
    partial_pairs: Optional[List[Tuple["Span", "Span"]]] = None,
    outcomes: Optional[List[Tuple["Span", str, Optional["Span"]]]] = None,
) -> Dict[str, int]:
    """Count COR/INC/PAR/MIS/SPU for one document, as the original scripts did.

    ``reference`` and ``compared`` are lists of dicts with ``token_start``,
    ``token_end`` (inclusive) and ``label``.  Both sides are sorted by
    ``token_end``, exactly as the original loops did, before matching.

    ``label`` restricts the count to one entity type using the paper's own
    definition: the filter is applied to the side being iterated only.  A gold
    CONDITION span may therefore still be matched against a predicted HISTORIC
    span and counted as INC in the CONDITION pass.  Filtering both sides
    instead would move per-label F1 by up to 1.7 points, so the paper's
    definition is the one implemented here.

    ``partial_pairs``, when given, is extended with one ``(reference_span,
    compared_span)`` pair for every PAR the loop counts, in the order it counts
    them; :func:`boundary_direction` classifies those pairs.  It is a pure
    observer of the loop below and cannot change a count.

    ``outcomes``, when given, is extended with one ``(reference_span, outcome,
    compared_span)`` triple for every reference span the loop *iterates* —
    every span that passes the ``label`` filter, misses included, where a miss
    has ``None`` for its compared span.  It is the same information the four
    reference-side counters hold, itemised, and is likewise a pure observer.
    The triples arrive in the loop's own order, which is ``reference`` sorted by
    ``token_end``; :func:`family_outcomes` relies on that being a stable sort of
    the argument.
    """
    spans = [Span(s) for s in sorted(reference, key=lambda x: x["token_end"])]
    spans_ = [Span(s) for s in sorted(compared, key=lambda x: x["token_end"])]

    cor = inc = par = mis = spu = 0
    for span in spans:
        if label is not None and span.label != label:
            continue
        matched = False
        for span_ in spans_:
            result = span.get_score(span_)
            if result == "cor":
                cor += 1
                if outcomes is not None:
                    outcomes.append((span, "cor", span_))
                matched = True
                break
            if result == "inc":
                inc += 1
                if outcomes is not None:
                    outcomes.append((span, "inc", span_))
                matched = True
                break
            if result == "par":
                par += 1
                if partial_pairs is not None:
                    partial_pairs.append((span, span_))
                if outcomes is not None:
                    outcomes.append((span, "par", span_))
                matched = True
                break
        if not matched:
            mis += 1
            if outcomes is not None:
                outcomes.append((span, "mis", None))

    for span_ in spans_:
        if label is not None and span_.label != label:
            continue
        found = False
        for span in spans:
            if span_.get_score(span):
                found = True
                break
        if not found:
            spu += 1

    return dict(cor=cor, inc=inc, par=par, mis=mis, spu=spu)


def score_pair(
    gold: Document,
    pred: Document,
    label: Optional[str] = None,
) -> Dict[str, int]:
    """Score one gold document against one prediction document."""
    return score_spans(gold.token_spans(), pred.token_spans(reference=gold), label=label)


def prf(counts: Dict[str, int], partial: bool = True) -> Tuple[float, float, float]:
    """Precision, recall and F1 from pooled counts (fractions, not percent)."""
    possible = counts["cor"] + counts["inc"] + counts["par"] + counts["mis"]
    actual = counts["cor"] + counts["inc"] + counts["par"] + counts["spu"]
    numerator = counts["cor"] + (0.5 * counts["par"] if partial else 0.0)
    precision = numerator / actual if actual else 0.0
    recall = numerator / possible if possible else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and population standard deviation (the paper's convention)."""
    if not values:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# document alignment
# ---------------------------------------------------------------------------


def align_exact(
    gold: Dict[str, Document], pred: Dict[str, Document], gold_name: str, pred_name: str
) -> List[str]:
    """Return the shared document ids, insisting that the two sets are equal.

    Silent misalignment is the failure mode that would quietly change every
    number, so a mismatch is an error and never a warning.
    """
    missing = sorted(set(gold) - set(pred))
    extra = sorted(set(pred) - set(gold))
    if missing or extra:
        raise SystemExit(
            "document ids do not match between %s (%d docs) and %s (%d docs):\n"
            "  missing from predictions: %d %s\n"
            "  not present in gold:      %d %s"
            % (
                gold_name,
                len(gold),
                pred_name,
                len(pred),
                len(missing),
                missing[:5],
                len(extra),
                extra[:5],
            )
        )
    return [doc_id for doc_id in gold]


def align_subset(
    left: Dict[str, Document], right: Dict[str, Document], left_name: str, right_name: str
) -> List[str]:
    """Return the shared ids of two annotator files, insisting on containment.

    The annotator files are not all the same size — the full-corpus pass covers
    every advert while the calibration passes cover a subset — so equality is
    the wrong test here.  One id set must contain the other; anything else is a
    misalignment and raises.
    """
    left_ids, right_ids = set(left), set(right)
    if not (left_ids <= right_ids or right_ids <= left_ids):
        raise SystemExit(
            "document ids of %s (%d) and %s (%d) are not nested: %d only in the "
            "first, %d only in the second"
            % (
                left_name,
                len(left_ids),
                right_name,
                len(right_ids),
                len(left_ids - right_ids),
                len(right_ids - left_ids),
            )
        )
    common = left_ids & right_ids
    return [doc_id for doc_id in left if doc_id in common]


# ---------------------------------------------------------------------------
# models sub-command
# ---------------------------------------------------------------------------


def gold_documents(gold_path: str) -> Dict[str, Document]:
    """Read a gold fold file: the released corpus annotations on its adverts."""
    return documents(gold_path)


def score_fold(
    gold_path: str,
    pred_path: str,
    label: Optional[str] = None,
) -> Tuple[Dict[str, int], int]:
    """Pooled counts for one fold, plus the number of documents scored."""
    gold = gold_documents(gold_path)
    pred = documents(pred_path)
    doc_ids = align_exact(gold, pred, gold_path, pred_path)
    totals = zero_counts()
    for doc_id in doc_ids:
        add_counts(totals, score_pair(gold[doc_id], pred[doc_id], label=label))
    return totals, len(doc_ids)


def fill_template(template: str, fold: str) -> str:
    """Substitute ``{fold}`` in a path template."""
    return template.format(fold=fold)


def score_model(
    gold_template: str,
    pred_template: str,
    folds: Sequence[str],
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Score one model over the folds and summarise as the paper's table does."""
    per_fold: List[Dict[str, Any]] = []
    for fold in folds:
        counts, support = score_fold(
            fill_template(gold_template, fold),
            fill_template(pred_template, fold),
            label,
        )
        exact = prf(counts, partial=False)
        partial = prf(counts, partial=True)
        per_fold.append(
            {
                "fold": fold,
                "support": support,
                "counts": counts,
                "exact": {
                    "precision": exact[0] * 100,
                    "recall": exact[1] * 100,
                    "f1": exact[2] * 100,
                },
                "partial": {
                    "precision": partial[0] * 100,
                    "recall": partial[1] * 100,
                    "f1": partial[2] * 100,
                },
            }
        )
    summary: Dict[str, Any] = {}
    for scheme in ("exact", "partial"):
        summary[scheme] = {}
        for metric in ("precision", "recall", "f1"):
            mean, std = mean_std([f[scheme][metric] for f in per_fold])
            summary[scheme][metric] = {"mean": mean, "std": std}
    return {"folds": per_fold, "summary": summary}


# ---------------------------------------------------------------------------
# boundaries sub-command
# ---------------------------------------------------------------------------

BOUNDARY_KEYS = ("wide", "narrow", "staggered")

def boundary_direction(reference: Span, compared: Span) -> str:
    """Classify one partial match by which way its boundaries are wrong.

    ``reference`` is the gold span and ``compared`` the prediction that the
    greedy walk of :func:`score_spans` matched it to as PAR.  Three outcomes:

    ``"wide"``
        the prediction reaches past the gold span on at least one side and
        stops short on neither (``compared.start <= reference.start`` and
        ``compared.end >= reference.end``);
    ``"narrow"``
        the prediction lies inside the gold span (``compared.start >=
        reference.start`` and ``compared.end <= reference.end``);
    ``"staggered"``
        everything else: the prediction begins before the gold span and ends
        before it, or begins after it and ends after it.

    A PAR never has both boundaries equal — that pair would have been counted
    COR or INC — so the two containment tests cannot both fire, and "wide" is
    tested first only for definiteness.  Note also that a "wide" PAR always
    shares one boundary with the gold span: the overlap test of
    :meth:`Span.get_score` is one-sided, so a prediction that strictly contains
    the gold span on *both* sides never matches it from the gold side at all
    (it is counted MIS, not PAR).  That is why the definition reads "on at
    least one side".
    """
    if compared.start <= reference.start and compared.end >= reference.end:
        return "wide"
    if compared.start >= reference.start and compared.end <= reference.end:
        return "narrow"
    return "staggered"


def zero_boundaries() -> Dict[str, int]:
    """A fresh PAR/too-wide/too-narrow/staggered accumulator."""
    counts = {"par": 0}
    counts.update({key: 0 for key in BOUNDARY_KEYS})
    return counts


def boundary_pair(
    gold: Document, pred: Document, label: Optional[str] = None
) -> Dict[str, int]:
    """Classify every partial match between one gold and one prediction document."""
    pairs: List[Tuple[Span, Span]] = []
    score_spans(
        gold.token_spans(),
        pred.token_spans(reference=gold),
        label=label,
        partial_pairs=pairs,
    )
    counts = zero_boundaries()
    counts["par"] = len(pairs)
    for reference, compared in pairs:
        counts[boundary_direction(reference, compared)] += 1
    return counts


def boundary_fold(
    gold_path: str,
    pred_path: str,
    label: Optional[str] = None,
) -> Tuple[Dict[str, int], int]:
    """Pooled boundary counts for one fold, plus the number of documents scored."""
    gold = gold_documents(gold_path)
    pred = documents(pred_path)
    doc_ids = align_exact(gold, pred, gold_path, pred_path)
    totals = zero_boundaries()
    for doc_id in doc_ids:
        counts = boundary_pair(gold[doc_id], pred[doc_id], label=label)
        for key in totals:
            totals[key] += counts[key]
    return totals, len(doc_ids)


def boundary_model(
    gold_template: str,
    pred_template: str,
    folds: Sequence[str],
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Boundary directions for one model, per fold and pooled over the folds.

    The shares are percentages of the model's own PAR count, which is the
    pooled ``par`` of ``score.py models`` for the same files; the paper's error
    analysis quotes the pooled row.
    """
    per_fold: List[Dict[str, Any]] = []
    pooled = zero_boundaries()
    for fold in folds:
        counts, support = boundary_fold(
            fill_template(gold_template, fold),
            fill_template(pred_template, fold),
            label,
        )
        for key in pooled:
            pooled[key] += counts[key]
        per_fold.append({"fold": fold, "support": support, "counts": counts})
    par = pooled["par"]
    shares = {
        key: (100.0 * pooled[key] / par if par else 0.0) for key in BOUNDARY_KEYS
    }
    return {"folds": per_fold, "pooled": pooled, "shares": shares}


# ---------------------------------------------------------------------------
# iaa sub-command
# ---------------------------------------------------------------------------


def iaa_pairwise(
    annotators: Dict[str, str], label: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Partial-match agreement per unordered pair, both directions pooled.

    Pooling the two directions of a pair makes ``possible`` equal ``actual``
    (each miss in one direction is a spurious span in the other), so precision,
    recall and F1 coincide — the single agreement figure the paper quotes.
    """
    docs = {name: documents(path) for name, path in annotators.items()}
    names = list(annotators)
    results = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            common = align_subset(
                docs[left], docs[right], annotators[left], annotators[right]
            )
            totals = zero_counts()
            for reference, compared in ((left, right), (right, left)):
                for doc_id in common:
                    add_counts(
                        totals,
                        score_pair(docs[reference][doc_id], docs[compared][doc_id], label),
                    )
            exact = prf(totals, partial=False)
            partial = prf(totals, partial=True)
            results.append(
                {
                    "pair": [left, right],
                    "support": len(common),
                    "counts": totals,
                    "exact": {
                        "precision": exact[0] * 100,
                        "recall": exact[1] * 100,
                        "f1": exact[2] * 100,
                    },
                    "partial": {
                        "precision": partial[0] * 100,
                        "recall": partial[1] * 100,
                        "f1": partial[2] * 100,
                    },
                }
            )
    return results


def iaa_per_label(
    annotators: Dict[str, str], labels: Sequence[str]
) -> List[Dict[str, Any]]:
    """Per-label agreement pooled over all six ordered annotator pairs.

    This is the aggregation the paper's appendix table uses: raw counts are
    summed across every ordered (reference, compared) pair before precision,
    recall and F1 are computed.
    """
    docs = {name: documents(path) for name, path in annotators.items()}
    names = list(annotators)
    ordered = [(a, b) for a in names for b in names if a != b]
    rows = []
    for label in labels:
        totals = zero_counts()
        for reference, compared in ordered:
            common = align_subset(
                docs[reference], docs[compared], annotators[reference], annotators[compared]
            )
            for doc_id in common:
                add_counts(
                    totals,
                    score_pair(docs[reference][doc_id], docs[compared][doc_id], label),
                )
        exact = prf(totals, partial=False)
        partial = prf(totals, partial=True)
        rows.append(
            {
                "label": label,
                "counts": totals,
                "exact": {
                    "precision": exact[0] * 100,
                    "recall": exact[1] * 100,
                    "f1": exact[2] * 100,
                },
                "partial": {
                    "precision": partial[0] * 100,
                    "recall": partial[1] * 100,
                    "f1": partial[2] * 100,
                },
                "share": {
                    key: (totals[key] / sum(totals.values()) * 100)
                    if sum(totals.values())
                    else 0.0
                    for key in COUNT_KEYS
                },
            }
        )
    return rows


# ---------------------------------------------------------------------------
# families sub-command
# ---------------------------------------------------------------------------

# The mention families of the paper's contextual-labelling table.  Each is a
# case-insensitive regular expression over the *text* of a span, deliberately
# blunt: the point is to collect every mention of a recurring topic without
# presupposing which label it carries.  ``\bservic`` catches "service",
# "serviced" and "servicing"; ``\bmile`` catches "miles", "mileage" and
# "mile"; the two plural forms are spelled out.  Being blunt, they also catch a
# handful of unrelated uses — "fire service", "owners manual", a "100,000 mile
# battery warranty" — which is harmless in the counts and is why the snippets
# below are candidates for a human to choose from rather than a final table.
FAMILIES: Tuple[Tuple[str, str], ...] = (
    ("MOT", r"\bmot\b"),
    ("service", r"\bservic"),
    ("tyre", r"\btyres?\b"),
    ("owner", r"\bowners?\b"),
    ("mileage", r"\bmile"),
)

FAMILY_KEYS: Tuple[str, ...] = tuple(name for name, _ in FAMILIES)

# The row that holds every gold span matching none of the families, and the row
# that holds every gold span whatever it matches.
OTHER_KEY = "all other spans"
EVERY_KEY = "every gold span"

# The reference-side counts; a family row has no SPU, because "spurious" is a
# property of a prediction and a prediction has no gold family.
FAMILY_COUNT_KEYS = ("cor", "inc", "par", "mis")

# Patterns a candidate snippet must not contain.  The corpus is public and
# already scrubbed, so this is a belt-and-braces filter over the span texts the
# snippet table would print: a personal name cannot be matched by regular
# expression and is excluded by eye instead.
PII_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("email", r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    ("url", r"(?:https?://|www\.)\S+|\b[\w-]+\.(?:co|org|ac)?\.?(?:uk|com|net|org)\b"),
    ("phone", r"(?:\+44|\b0)(?:[\s-]?\d){9,}"),
    ("postcode", r"\b[a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2}\b"),
    ("plate", r"\b[a-z]{2}\d{2}\s?[a-z]{3}\b"),
)

_FAMILY_RE = tuple((name, re.compile(pattern, re.I)) for name, pattern in FAMILIES)
_PII_RE = tuple((name, re.compile(pattern, re.I)) for name, pattern in PII_PATTERNS)


def span_families(text: str) -> Tuple[str, ...]:
    """The families whose regex matches ``text``, in the order of :data:`FAMILIES`.

    A span may belong to several families at once ("last serviced at 8,291
    miles" is both *service* and *mileage*), so the family rows of every table
    below overlap; the count of such spans is reported beside them.
    """
    return tuple(name for name, regex in _FAMILY_RE if regex.search(text))


def family_keys(text: str) -> Tuple[str, ...]:
    """The rows one gold span contributes to: its families, or :data:`OTHER_KEY`."""
    return span_families(text) or (OTHER_KEY,)


def pii_pattern(text: str) -> Optional[str]:
    """The name of the first :data:`PII_PATTERNS` entry ``text`` matches, if any."""
    for name, regex in _PII_RE:
        if regex.search(text):
            return name
    return None


def flatten(text: str) -> str:
    """One line: every newline and tab becomes a space, and the ends are trimmed."""
    return text.replace("\r", " ").replace("\n", " ").replace("\t", " ").strip()


def gold_span_texts(doc: Document) -> List[str]:
    """The text each of ``doc``'s spans covers, in file order.

    Character offsets are used, not the token span, so the text is exactly what
    the annotator selected.
    """
    if doc.text is None:
        raise FormatError("document %r carries no text" % doc.doc_id)
    return [doc.text[span["start"] : span["end"]] for span in doc.char_spans()]


def zero_family_counts() -> Dict[str, int]:
    """A fresh COR/INC/PAR/MIS accumulator for one family row."""
    return {key: 0 for key in FAMILY_COUNT_KEYS}


def family_corpus_counts(
    gold_path: str,
    labels: Sequence[str] = LABELS_DEFAULT,
) -> Dict[str, Any]:
    """How the gold spans of each family divide over the labels, over one file."""
    docs = gold_documents(gold_path)
    rows = {key: {label: 0 for label in labels} for key in FAMILY_KEYS}
    rows[OTHER_KEY] = {label: 0 for label in labels}
    rows[EVERY_KEY] = {label: 0 for label in labels}
    multi = 0
    for doc in docs.values():
        for text, span in zip(gold_span_texts(doc), doc.spans):
            label = span["label"]
            if label not in rows[EVERY_KEY]:
                continue
            rows[EVERY_KEY][label] += 1
            keys = family_keys(text)
            if len(keys) > 1:
                multi += 1
            for key in keys:
                rows[key][label] += 1
    return {
        "documents": len(docs),
        "labels": list(labels),
        "rows": rows,
        "multi_family": multi,
    }


def family_outcomes(
    gold: Document, pred: Document
) -> List[Tuple[Tuple[str, ...], str]]:
    """Per gold span: the rows it belongs to and its COR/INC/PAR/MIS outcome.

    The outcome is the one the paper's own matching gives it — the greedy walk of
    :func:`score_spans`, observed through its ``outcomes`` collector, with no
    label filter so that every gold span is iterated exactly once.  The walk
    sees ``reference`` sorted by ``token_end``; the same stable sort is applied
    to the indices here to recover which span each outcome belongs to, and the
    correspondence is checked rather than assumed.
    """
    token_spans = gold.token_spans()
    texts = gold_span_texts(gold)
    order = sorted(
        range(len(token_spans)), key=lambda index: token_spans[index]["token_end"]
    )
    observed: List[Tuple[Span, str, Optional[Span]]] = []
    score_spans(token_spans, pred.token_spans(reference=gold), outcomes=observed)
    if len(observed) != len(order):
        raise FormatError(
            "document %r: %d gold spans but %d outcomes observed"
            % (gold.doc_id, len(order), len(observed))
        )
    out: List[Tuple[Tuple[str, ...], str]] = []
    for position, (span, outcome, _compared) in zip(order, observed):
        reference = token_spans[position]
        if (span.start, span.end, span.label) != (
            reference["token_start"],
            reference["token_end"],
            reference["label"],
        ):
            raise FormatError(
                "document %r: the observed gold order does not match the "
                "reference order" % gold.doc_id
            )
        out.append((family_keys(texts[position]), outcome))
    return out


def family_fold(
    gold_path: str, pred_path: str
) -> Tuple[Dict[str, Dict[str, int]], int]:
    """Per-family COR/INC/PAR/MIS for one fold, plus the documents scored."""
    gold = gold_documents(gold_path)
    pred = documents(pred_path)
    doc_ids = align_exact(gold, pred, gold_path, pred_path)
    totals = {key: zero_family_counts() for key in FAMILY_KEYS}
    totals[OTHER_KEY] = zero_family_counts()
    totals[EVERY_KEY] = zero_family_counts()
    for doc_id in doc_ids:
        for keys, outcome in family_outcomes(gold[doc_id], pred[doc_id]):
            totals[EVERY_KEY][outcome] += 1
            for key in keys:
                totals[key][outcome] += 1
    return totals, len(doc_ids)


def family_rates(counts: Dict[str, int]) -> Dict[str, float]:
    """The two rates the paper quotes, as percentages.

    ``confusion`` is INC / (COR + INC + PAR): of the gold spans the model did
    find, the share it put under the wrong label.  ``miss`` is MIS / (COR + INC
    + PAR + MIS): the share it did not find at all.
    """
    found = counts["cor"] + counts["inc"] + counts["par"]
    possible = found + counts["mis"]
    return {
        "confusion": 100.0 * counts["inc"] / found if found else 0.0,
        "miss": 100.0 * counts["mis"] / possible if possible else 0.0,
    }


def family_model(
    gold_template: str,
    pred_template: str,
    folds: Sequence[str],
) -> Dict[str, Any]:
    """One model's per-family counts, pooled over the folds."""
    pooled = {key: zero_family_counts() for key in FAMILY_KEYS}
    pooled[OTHER_KEY] = zero_family_counts()
    pooled[EVERY_KEY] = zero_family_counts()
    support = 0
    for fold in folds:
        counts, documents_scored = family_fold(
            fill_template(gold_template, fold),
            fill_template(pred_template, fold),
        )
        support += documents_scored
        for key, row in counts.items():
            for count_key in FAMILY_COUNT_KEYS:
                pooled[key][count_key] += row[count_key]
    return {
        "support": support,
        "rows": pooled,
        "rates": {key: family_rates(row) for key, row in pooled.items()},
    }


def family_snippets(
    gold_path: str,
    min_tokens: int = 2,
    max_tokens: int = 8,
) -> Dict[str, Any]:
    """Candidate example snippets per family and label, ordered deterministically.

    A candidate is the text of a gold span that is ``min_tokens``–``max_tokens``
    tokens long, matches the family, and matches none of
    :data:`PII_PATTERNS`.  Newlines and tabs become spaces.  Identical texts are
    pooled and the order is by descending frequency, then by length and
    case-insensitively by text — short examples first, and stable across runs.
    Returned as
    ``{"excluded": n, "candidates": {family: {label: [(text, count), ...]}}}``,
    where ``excluded`` counts the spans the PII patterns dropped.
    """
    docs = gold_documents(gold_path)
    tally: Dict[Tuple[str, str], Dict[str, int]] = {}
    excluded = 0
    for doc in docs.values():
        token_spans = doc.token_spans()
        for text, span, token_span in zip(
            gold_span_texts(doc), doc.char_spans(), token_spans
        ):
            length = token_span["token_end"] - token_span["token_start"] + 1
            if not min_tokens <= length <= max_tokens:
                continue
            families = span_families(text)
            if not families:
                continue
            snippet = flatten(text)
            if pii_pattern(snippet):
                excluded += 1
                continue
            for family in families:
                counts = tally.setdefault((family, span["label"]), {})
                counts[snippet] = counts.get(snippet, 0) + 1
    candidates: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}
    for family in FAMILY_KEYS:
        candidates[family] = {}
        for label in LABELS_DEFAULT:
            counts = tally.get((family, label), {})
            candidates[family][label] = sorted(
                counts.items(),
                key=lambda item: (-item[1], len(item[0]), item[0].casefold(), item[0]),
            )
    return {"excluded": excluded, "candidates": candidates}


def identical_span_disagreements(
    left: Document, right: Document
) -> Tuple[int, List[Dict[str, Any]]]:
    """Spans two annotators bound identically: how many, and which differ in label.

    Boundaries are compared on the inclusive token indices the paper's matcher
    uses, so this is exactly the INC that :meth:`Span.get_score` reports for a
    pair whose ``start`` and ``end`` both agree.  Every pairing of an identically
    bounded span on the left with one on the right is counted, which matters only
    if an annotator repeated a span.
    """
    left_spans = list(zip(left.token_spans(), gold_span_texts(left)))
    right_spans = right.token_spans()
    index: Dict[Tuple[int, int], List[str]] = {}
    for span in right_spans:
        key = (span["token_start"], span["token_end"])
        index.setdefault(key, []).append(span["label"])
    identical = 0
    rows: List[Dict[str, Any]] = []
    for span, text in left_spans:
        key = (span["token_start"], span["token_end"])
        for label in index.get(key, []):
            identical += 1
            if label == span["label"]:
                continue
            rows.append(
                {
                    "doc_id": left.doc_id,
                    "token_start": key[0],
                    "token_end": key[1],
                    "text": flatten(text),
                    "families": list(span_families(text)),
                    "left": span["label"],
                    "right": label,
                }
            )
    return identical, rows


def disagreement_pairs(annotators: Dict[str, str]) -> List[Dict[str, Any]]:
    """:func:`identical_span_disagreements` over every unordered annotator pair."""
    docs = {name: documents(path) for name, path in annotators.items()}
    names = list(annotators)
    results = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            common = align_subset(
                docs[left], docs[right], annotators[left], annotators[right]
            )
            identical = 0
            rows: List[Dict[str, Any]] = []
            for doc_id in common:
                count, found = identical_span_disagreements(
                    docs[left][doc_id], docs[right][doc_id]
                )
                identical += count
                rows.extend(found)
            results.append(
                {
                    "pair": [left, right],
                    "support": len(common),
                    "identical": identical,
                    "different": len(rows),
                    "share": 100.0 * len(rows) / identical if identical else 0.0,
                    "rows": rows,
                }
            )
    return results


def disagreement_predictions(
    rows: Sequence[Dict[str, Any]],
    gold_template: str,
    models: Dict[str, str],
    folds: Sequence[str],
) -> List[Dict[str, Any]]:
    """What each model predicts for the disagreed spans that fall in a test fold.

    The gold label reported beside them is the one the fold file carries, so the
    reader can see which of the two experts the adjudicated gold sided with.  The
    prediction is the one the paper's own greedy walk matches to the span, and
    is described by its label plus how its boundaries sit against the span:
    ``exact``, or the direction :func:`boundary_direction` gives it.
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        for fold in folds:
            gold_path = fill_template(gold_template, fold)
            gold = documents(gold_path)
            document = gold.get(row["doc_id"])
            if document is None:
                continue
            reference = {
                "token_start": row["token_start"],
                "token_end": row["token_end"],
                "label": row["left"],
            }
            gold_label = None
            for span in document.token_spans():
                if (span["token_start"], span["token_end"]) == (
                    row["token_start"],
                    row["token_end"],
                ):
                    gold_label = span["label"]
                    break
            predicted: Dict[str, Optional[str]] = {}
            for name, template in models.items():
                pred = documents(fill_template(template, fold))[row["doc_id"]]
                observed: List[Tuple[Span, str, Optional[Span]]] = []
                score_spans(
                    [reference], pred.token_spans(reference=document), outcomes=observed
                )
                _span, _outcome, compared = observed[0]
                if compared is None:
                    predicted[name] = None
                    continue
                where = (
                    "exact"
                    if (compared.start, compared.end)
                    == (reference["token_start"], reference["token_end"])
                    else boundary_direction(Span(reference), compared)
                )
                predicted[name] = "%s (%s)" % (compared.label, where)
            out.append(
                dict(row, fold=fold, gold_label=gold_label, predicted=predicted)
            )
            break
    return out

# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _counts_cell(counts: Dict[str, int]) -> str:
    return "/".join(str(counts[key]) for key in COUNT_KEYS)


def render_models_markdown(
    results: Dict[str, Dict[str, Any]],
    folds: Sequence[str],
    per_label: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    labels: Sequence[str] = LABELS_DEFAULT,
) -> str:
    lines: List[str] = []
    lines.append("### Per-fold pooled counts and scores")
    lines.append("")
    lines.append(
        "| Model | Fold | Docs | COR/INC/PAR/MIS/SPU | Exact P | Exact R | Exact F1 "
        "| Partial P | Partial R | Partial F1 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for model, result in results.items():
        for fold in result["folds"]:
            lines.append(
                "| %s | %s | %d | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |"
                % (
                    model,
                    fold["fold"],
                    fold["support"],
                    _counts_cell(fold["counts"]),
                    fold["exact"]["precision"],
                    fold["exact"]["recall"],
                    fold["exact"]["f1"],
                    fold["partial"]["precision"],
                    fold["partial"]["recall"],
                    fold["partial"]["f1"],
                )
            )
    lines.append("")
    lines.append(
        "### Partial match, mean ± population standard deviation over folds %s"
        % ", ".join(folds)
    )
    lines.append("")
    lines.append("| Model | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    for model, result in results.items():
        summary = result["summary"]["partial"]
        lines.append(
            "| %s | %.1f ± %.1f | %.1f ± %.1f | %.1f ± %.1f |"
            % (
                model,
                summary["precision"]["mean"],
                summary["precision"]["std"],
                summary["recall"]["mean"],
                summary["recall"]["std"],
                summary["f1"]["mean"],
                summary["f1"]["std"],
            )
        )
    lines.append("")
    lines.append("### Exact match, mean ± population standard deviation")
    lines.append("")
    lines.append("| Model | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    for model, result in results.items():
        summary = result["summary"]["exact"]
        lines.append(
            "| %s | %.1f ± %.1f | %.1f ± %.1f | %.1f ± %.1f |"
            % (
                model,
                summary["precision"]["mean"],
                summary["precision"]["std"],
                summary["recall"]["mean"],
                summary["recall"]["std"],
                summary["f1"]["mean"],
                summary["f1"]["std"],
            )
        )
    if per_label:
        lines.append("")
        lines.append(
            "### Per-label partial F1, mean ± population standard deviation "
            "(label filter on the reference side, as in the paper)"
        )
        lines.append("")
        lines.append("| Model | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        for model in results:
            cells = []
            for label in labels:
                summary = per_label[model][label]["summary"]["partial"]["f1"]
                cells.append("%.1f ± %.1f" % (summary["mean"], summary["std"]))
            lines.append("| %s | %s |" % (model, " | ".join(cells)))
    return "\n".join(lines) + "\n"


def render_boundaries_markdown(
    results: Dict[str, Dict[str, Any]], folds: Sequence[str]
) -> str:
    lines: List[str] = []
    lines.append(
        "### Boundary direction of partial matches, pooled over folds %s"
        % ", ".join(folds)
    )
    lines.append("")
    lines.append(
        "| Model | PAR | Too wide | Too narrow | Staggered |"
    )
    lines.append("|---|---|---|---|---|")
    for model, result in results.items():
        pooled = result["pooled"]
        shares = result["shares"]
        lines.append(
            "| %s | %d | %s |"
            % (
                model,
                pooled["par"],
                " | ".join(
                    "%d (%.1f%%)" % (pooled[key], shares[key])
                    for key in BOUNDARY_KEYS
                ),
            )
        )
    lines.append("")
    lines.append("### Per fold")
    lines.append("")
    lines.append("| Model | Fold | Docs | PAR | Too wide | Too narrow | Staggered |")
    lines.append("|---|---|---|---|---|---|---|")
    for model, result in results.items():
        for fold in result["folds"]:
            counts = fold["counts"]
            lines.append(
                "| %s | %s | %d | %d | %s |"
                % (
                    model,
                    fold["fold"],
                    fold["support"],
                    counts["par"],
                    " | ".join(str(counts[key]) for key in BOUNDARY_KEYS),
                )
            )
    return "\n".join(lines) + "\n"


def render_iaa_markdown(
    pairs: List[Dict[str, Any]], per_label: List[Dict[str, Any]]
) -> str:
    lines: List[str] = []
    lines.append("### Pairwise agreement (both directions pooled, so P = R = F1)")
    lines.append("")
    lines.append(
        "| Pair | Docs | COR/INC/PAR/MIS/SPU | Exact F1 | Partial F1 |"
    )
    lines.append("|---|---|---|---|---|")
    for row in pairs:
        lines.append(
            "| %s–%s | %d | %s | %.2f | %.2f |"
            % (
                row["pair"][0],
                row["pair"][1],
                row["support"],
                _counts_cell(row["counts"]),
                row["exact"]["f1"],
                row["partial"]["f1"],
            )
        )
    mean, std = mean_std([row["partial"]["f1"] for row in pairs])
    lines.append("")
    lines.append("Mean partial F1 over the pairs: %.2f (population sd %.2f)." % (mean, std))
    lines.append("")
    lines.append("### Per label, pooled over all six ordered annotator pairs")
    lines.append("")
    lines.append(
        "| Label | COR/INC/PAR/MIS/SPU | Exact P | Exact R | Exact F1 | "
        "Partial P | Partial R | Partial F1 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in per_label:
        lines.append(
            "| %s | %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |"
            % (
                row["label"],
                _counts_cell(row["counts"]),
                row["exact"]["precision"],
                row["exact"]["recall"],
                row["exact"]["f1"],
                row["partial"]["precision"],
                row["partial"]["recall"],
                row["partial"]["f1"],
            )
        )
    lines.append("")
    lines.append("### Per label, share of each count type (%)")
    lines.append("")
    lines.append("| Label | COR | INC | PAR | MIS | SPU |")
    lines.append("|---|---|---|---|---|---|")
    for row in per_label:
        lines.append(
            "| %s | %s |"
            % (
                row["label"],
                " | ".join("%.1f" % row["share"][key] for key in COUNT_KEYS),
            )
        )
    return "\n".join(lines) + "\n"


def _family_regex_cell(key: str) -> str:
    for name, pattern in FAMILIES:
        if name == key:
            return "`%s`" % pattern
    return "—"


def render_families_markdown(payload: Dict[str, Any]) -> str:
    """The four tables of the ``families`` sub-command."""
    lines: List[str] = []
    corpus = payload.get("corpus")
    if corpus:
        labels = corpus["labels"]
        lines.append(
            "### Gold spans of each mention family, by label (%d adverts)"
            % corpus["documents"]
        )
        lines.append("")
        lines.append("| Family | Regex | " + " | ".join(labels) + " | Total |")
        lines.append("|---" * (len(labels) + 3) + "|")
        for key in FAMILY_KEYS + (OTHER_KEY, EVERY_KEY):
            row = corpus["rows"][key]
            lines.append(
                "| %s | %s | %s | %d |"
                % (
                    key,
                    _family_regex_cell(key),
                    " | ".join(str(row[label]) for label in labels),
                    sum(row.values()),
                )
            )
        lines.append("")
        lines.append(
            "%d of the %d gold spans match more than one family, so the family "
            "rows overlap; `%s` holds the spans that match none."
            % (
                corpus["multi_family"],
                sum(corpus["rows"][EVERY_KEY].values()),
                OTHER_KEY,
            )
        )
    models = payload.get("models")
    if models:
        support = next(iter(models.values()))["support"]
        lines.append("")
        lines.append(
            "### Model behaviour on the families, folds %s pooled (%d adverts)"
            % (", ".join(payload["folds"]), support)
        )
        lines.append("")
        lines.append(
            "| Model | Family | Gold spans | COR | INC | PAR | MIS | "
            "INC/(COR+INC+PAR) | MIS/(COR+INC+PAR+MIS) |"
        )
        lines.append("|---" * 9 + "|")
        for model, result in models.items():
            for key in FAMILY_KEYS + (OTHER_KEY, EVERY_KEY):
                row = result["rows"][key]
                rates = result["rates"][key]
                lines.append(
                    "| %s | %s | %d | %s | %.1f%% | %.1f%% |"
                    % (
                        model,
                        key,
                        sum(row.values()),
                        " | ".join(str(row[count]) for count in FAMILY_COUNT_KEYS),
                        rates["confusion"],
                        rates["miss"],
                    )
                )
        lines.append("")
        lines.append(
            "`Gold spans` is COR + INC + PAR + MIS, the gold spans of that "
            "family the model was scored against; the `%s` row is the model's "
            "pooled count over every gold span and so agrees with the "
            "`score.py models` counts, SPU aside." % EVERY_KEY
        )
    pairs = payload.get("disagreements")
    if pairs:
        lines.append("")
        lines.append(
            "### Identically bounded spans two annotators labelled differently"
        )
        lines.append("")
        lines.append(
            "| Pair | Adverts | Identically bounded spans | Same label | "
            "Different label | Share different |"
        )
        lines.append("|---" * 6 + "|")
        for row in pairs:
            lines.append(
                "| %s–%s | %d | %d | %d | %d | %.1f%% |"
                % (
                    row["pair"][0],
                    row["pair"][1],
                    row["support"],
                    row["identical"],
                    row["identical"] - row["different"],
                    row["different"],
                    row["share"],
                )
            )
        expert = pairs[0]
        lines.append("")
        lines.append(
            "The first pair is the two experts. Their %d disagreements, in "
            "corpus order:" % expert["different"]
        )
        lines.append("")
        lines.append(
            "| Span | Family | %s | %s |" % (expert["pair"][0], expert["pair"][1])
        )
        lines.append("|---" * 4 + "|")
        for row in expert["rows"]:
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    row["text"],
                    ", ".join(row["families"]) or "—",
                    row["left"],
                    row["right"],
                )
            )
    predicted = payload.get("disagreement_predictions")
    if predicted:
        names = list(payload["models"])
        lines.append("")
        lines.append(
            "### What the models predict for the expert-disagreed spans in a test fold"
        )
        lines.append("")
        lines.append(
            "| Span | Fold | Adjudicated gold | " + " | ".join(names) + " |"
        )
        lines.append("|---" * (len(names) + 3) + "|")
        for row in predicted:
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    row["text"],
                    row["fold"],
                    row["gold_label"] or "—",
                    " | ".join(row["predicted"].get(name) or "—" for name in names),
                )
            )
        lines.append("")
        lines.append(
            "The adjudicated gold is the label the fold file carries, so it "
            "shows which expert the gold sided with. A cell gives "
            "the label of the prediction the paper's greedy walk matches to the "
            "span, and whether its boundaries are `exact` or too wide, too "
            "narrow or staggered; `—` means no prediction overlapped the span."
        )
    snippets = payload.get("snippets")
    if snippets:
        lines.append("")
        lines.append(
            "### Candidate example snippets, %d–%d tokens, with their corpus count"
            % (payload["snippet_tokens"][0], payload["snippet_tokens"][1])
        )
        lines.append("")
        lines.append("| Family | Label | " + " | ".join(
            "Candidate %d" % (index + 1) for index in range(payload["snippet_max"])
        ) + " |")
        lines.append("|---" * (payload["snippet_max"] + 2) + "|")
        for family in FAMILY_KEYS:
            for label in LABELS_DEFAULT:
                cells = [
                    "%s (%d)" % (text, count)
                    for text, count in snippets["candidates"][family][label][
                        : payload["snippet_max"]
                    ]
                ]
                cells += ["—"] * (payload["snippet_max"] - len(cells))
                lines.append("| %s | %s | %s |" % (family, label, " | ".join(cells)))
        lines.append("")
        lines.append(
            "Candidates are gold spans of that family and label, ordered by "
            "descending corpus count and then by length and text, so the list "
            "is stable across runs; spans matching an email, URL, telephone, "
            "postcode or "
            "registration-plate pattern are dropped (%d in all). The paper's "
            "example table picks one snippet per cell from these."
            % snippets["excluded"]
        )
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------


PER_LABEL_HELP = (
    "also report F1 per entity type using the paper's own per-label definition: "
    "the label filter is applied to the side being iterated only, so a gold "
    "CONDITION span may still match a predicted HISTORIC span and be counted as "
    "INC in the CONDITION pass. Filtering both sides instead moves per-label F1 "
    "by up to 1.7 points; the paper's definition is what is implemented."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="score.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser(
        "models",
        help="score model predictions against the gold test folds",
        description=(
            "Score one or more prediction files against the gold test folds. "
            "Counts are pooled within a fold, then the per-fold percentages are "
            "averaged over folds with a population standard deviation. Document "
            "ids must match exactly between a gold fold and its prediction file; "
            "a mismatch is an error, never a warning."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    models.add_argument(
        "--gold",
        required=True,
        help="gold fold path template, with {fold} standing for the fold name, "
        "e.g. data/folds/{fold}/test.jsonl",
    )
    models.add_argument(
        "--pred",
        required=True,
        action="append",
        metavar="[NAME=]TEMPLATE",
        help="prediction path template, with {fold}; may be repeated. Prefix "
        "with NAME= to label the row, e.g. CRF=predictions/crf/{fold}_test.jsonl",
    )
    models.add_argument(
        "--folds",
        default="f1,f2,f3",
        help="comma-separated fold names substituted into the templates",
    )
    models.add_argument(
        "--labels",
        default=",".join(LABELS_DEFAULT),
        help="comma-separated entity types for --per-label",
    )
    models.add_argument("--per-label", action="store_true", help=PER_LABEL_HELP)
    models.add_argument(
        "--format", choices=("json", "markdown"), default="markdown", help="output format"
    )
    models.add_argument("--out", help="write the output here instead of stdout")

    boundaries = sub.add_parser(
        "boundaries",
        help="classify each partial match as too wide, too narrow or staggered",
        description=(
            "Take the partial matches of the same greedy walk that "
            "``score.py models`` counts, and classify each by which way its "
            "boundaries are wrong: too wide (the prediction reaches past the "
            "gold span on at least one side and stops short on neither), too "
            "narrow (the prediction lies inside the gold span) or staggered "
            "(it begins and ends on the same side of the gold span). The "
            "shares are percentages of the model's own PAR count, pooled over "
            "the folds; this is the table behind the paper's error analysis."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    boundaries.add_argument(
        "--gold",
        required=True,
        help="gold fold path template, with {fold} standing for the fold name, "
        "e.g. data/folds/{fold}/test.jsonl",
    )
    boundaries.add_argument(
        "--pred",
        required=True,
        action="append",
        metavar="[NAME=]TEMPLATE",
        help="prediction path template, with {fold}; may be repeated. Prefix "
        "with NAME= to label the row, e.g. CRF=predictions/crf/{fold}_test.jsonl",
    )
    boundaries.add_argument(
        "--folds",
        default="f1,f2,f3",
        help="comma-separated fold names substituted into the templates",
    )
    boundaries.add_argument(
        "--label",
        help="restrict the count to one entity type, with the paper's "
        "per-label definition (the filter applies to the gold side)",
    )
    boundaries.add_argument(
        "--format", choices=("json", "markdown"), default="markdown", help="output format"
    )
    boundaries.add_argument("--out", help="write the output here instead of stdout")

    iaa = sub.add_parser(
        "iaa",
        help="inter-annotator agreement",
        description=(
            "Pairwise partial-match agreement (both directions of each pair "
            "pooled, which makes precision, recall and F1 coincide) and the "
            "per-label table pooled over all six ordered pairs, as in the "
            "paper's appendix. The annotator files need not be the same size: "
            "one id set must contain the other, and scoring runs on the "
            "intersection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    iaa.add_argument(
        "--annotators",
        required=True,
        nargs="+",
        metavar="NAME=FILE",
        help="two or more annotator files, e.g. a1=data/iaa/annotator_1.jsonl",
    )
    iaa.add_argument(
        "--labels",
        default=",".join(LABELS_DEFAULT),
        help="comma-separated entity types for the per-label table",
    )
    iaa.add_argument(
        "--format", choices=("json", "markdown"), default="markdown", help="output format"
    )
    iaa.add_argument("--out", help="write the output here instead of stdout")

    families = sub.add_parser(
        "families",
        help="how a mention family divides over the three labels, and who "
        "gets it right",
        description=(
            "Mention families under the three labels. A family is a blunt "
            "case-insensitive regex over a span's text (MOT, service, tyre, "
            "owner, mileage), and a span belongs to every family it matches. "
            "Up to four tables, each switched on by its own inputs: the gold "
            "spans of each family per label over --corpus; per-family "
            "COR/INC/PAR/MIS with the label-confusion and miss rates for each "
            "--pred over the --gold folds; the identically bounded spans two "
            "--annotators labelled differently, with what the models predict "
            "for the ones that fall in a test fold; and candidate example "
            "snippets per family and label from --corpus."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    families.add_argument(
        "--corpus",
        help="a single gold file for the corpus-level counts and the snippet "
        "candidates, e.g. data/auto_adver.jsonl",
    )
    families.add_argument(
        "--gold",
        help="gold fold path template, with {fold}, for the per-model table",
    )
    families.add_argument(
        "--pred",
        action="append",
        metavar="[NAME=]TEMPLATE",
        help="prediction path template, with {fold}; may be repeated. Prefix "
        "with NAME= to label the row",
    )
    families.add_argument(
        "--folds",
        default="f1,f2,f3",
        help="comma-separated fold names substituted into the templates",
    )
    families.add_argument(
        "--annotators",
        nargs="+",
        metavar="NAME=FILE",
        help="two or more annotator files for the disagreement tables; the "
        "first pair listed is taken to be the expert pair",
    )
    families.add_argument(
        "--labels",
        default=",".join(LABELS_DEFAULT),
        help="comma-separated entity types for the corpus table's columns",
    )
    families.add_argument(
        "--snippet-tokens",
        default="2,8",
        help="MIN,MAX token length a candidate snippet must have",
    )
    families.add_argument(
        "--snippet-max",
        type=int,
        default=3,
        help="how many candidate snippets to show per family and label",
    )
    families.add_argument(
        "--format", choices=("json", "markdown"), default="markdown", help="output format"
    )
    families.add_argument("--out", help="write the output here instead of stdout")
    return parser


def _split_named(items: Iterable[str], what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" in item:
            name, _, path = item.partition("=")
        else:
            name, path = os.path.basename(os.path.dirname(item)) or item, item
        if name in out:
            raise SystemExit("duplicate %s name %r" % (what, name))
        out[name] = path
    return out


def _emit(text: str, out_path: Optional[str]) -> None:
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # ``boundaries`` takes a single --label rather than a --labels list.
    labels = [label for label in getattr(args, "labels", "").split(",") if label]

    if args.command == "models":
        folds = [fold for fold in args.folds.split(",") if fold]
        models = _split_named(args.pred, "model")
        results = {
            name: score_model(args.gold, template, folds)
            for name, template in models.items()
        }
        per_label = None
        if args.per_label:
            per_label = {
                name: {
                    label: score_model(args.gold, template, folds, label=label)
                    for label in labels
                }
                for name, template in models.items()
            }
        if args.format == "json":
            payload = {
                "folds": folds,
                "matching": "paper (partial match)",
                "models": results,
            }
            if per_label:
                payload["per_label"] = per_label
            _emit(json.dumps(payload, indent=2) + "\n", args.out)
        else:
            _emit(
                render_models_markdown(results, folds, per_label, labels),
                args.out,
            )
        return 0

    if args.command == "boundaries":
        folds = [fold for fold in args.folds.split(",") if fold]
        models = _split_named(args.pred, "model")
        results = {
            name: boundary_model(args.gold, template, folds, label=args.label)
            for name, template in models.items()
        }
        if args.format == "json":
            _emit(
                json.dumps(
                    {
                        "folds": folds,
                        "matching": "paper (partial match)",
                        "label": args.label,
                        "models": results,
                    },
                    indent=2,
                )
                + "\n",
                args.out,
            )
        else:
            _emit(render_boundaries_markdown(results, folds), args.out)
        return 0

    if args.command == "families":
        folds = [fold for fold in args.folds.split(",") if fold]
        try:
            min_tokens, max_tokens = (
                int(part) for part in args.snippet_tokens.split(",")
            )
        except ValueError:
            raise SystemExit("--snippet-tokens wants MIN,MAX, e.g. 2,8")
        payload: Dict[str, Any] = {
            "matching": "paper (partial match)",
            "folds": folds,
            "snippet_tokens": [min_tokens, max_tokens],
            "snippet_max": args.snippet_max,
        }
        if args.corpus:
            payload["corpus"] = family_corpus_counts(args.corpus, labels)
            payload["snippets"] = family_snippets(args.corpus, min_tokens, max_tokens)
        models = _split_named(args.pred or [], "model")
        if models:
            if not args.gold:
                raise SystemExit("--pred needs --gold")
            payload["models"] = {
                name: family_model(args.gold, template, folds)
                for name, template in models.items()
            }
        if args.annotators:
            annotators = _split_named(args.annotators, "annotator")
            if len(annotators) < 2:
                raise SystemExit("--annotators needs at least two NAME=FILE arguments")
            payload["disagreements"] = disagreement_pairs(annotators)
            if models and args.gold:
                payload["disagreement_predictions"] = disagreement_predictions(
                    payload["disagreements"][0]["rows"], args.gold, models, folds
                )
        if args.format == "json":
            _emit(json.dumps(payload, indent=2) + "\n", args.out)
        else:
            _emit(render_families_markdown(payload), args.out)
        return 0

    annotators = _split_named(args.annotators, "annotator")
    if len(annotators) < 2:
        raise SystemExit("--annotators needs at least two NAME=FILE arguments")
    pairs = iaa_pairwise(annotators)
    per_label_rows = iaa_per_label(annotators, labels)
    if args.format == "json":
        _emit(
            json.dumps(
                {
                    "matching": "paper (partial match)",
                    "pairs": pairs,
                    "per_label": per_label_rows,
                },
                indent=2,
            )
            + "\n",
            args.out,
        )
    else:
        _emit(render_iaa_markdown(pairs, per_label_rows), args.out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FormatError as error:
        raise SystemExit("format error: %s" % error)
