#!/usr/bin/env python3
"""Part-of-speech profile of the annotated entities.

Port of ``Analytics/pos_analysis.py``: the surface form of every entity is
re-tokenised and tagged with spaCy, and the coarse (``pos_``) and fine
(``tag_``) tag counts are reported per label.  The original analysed CONDITION
only; this covers every label, and the markdown output is what the appendix
table is built from.

spaCy is an optional dependency — nothing else in the release needs it — so it
is imported only when this tool runs, and a missing model or missing package is
reported plainly rather than as a traceback::

    pip install spacy && python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_tools.records import LABELS_DEFAULT, iter_documents  # noqa: E402


def load_spacy(model: str) -> Any:
    """Load a spaCy pipeline, or exit with an instruction on how to get one."""
    try:
        import spacy  # noqa: PLC0415
    except ImportError:
        raise SystemExit(
            "pos_stats.py needs spaCy, which is not installed. Install it with\n"
            "    pip install spacy\n"
            "    python -m spacy download %s" % model
        )
    try:
        return spacy.load(model)
    except OSError:
        raise SystemExit(
            "the spaCy model %r is not installed. Install it with\n"
            "    python -m spacy download %s" % (model, model)
        )


def entity_texts(paths: Sequence[str]) -> Dict[str, List[str]]:
    """Collect the surface form of every entity, grouped by label."""
    texts: Dict[str, List[str]] = {}
    for path in paths:
        for doc in iter_documents(path):
            if doc.text is None:
                raise SystemExit("%s holds records without text" % path)
            for span in doc.char_spans():
                texts.setdefault(span["label"], []).append(
                    doc.text[span["start"] : span["end"]]
                )
    return texts


def tag_counts(nlp: Any, texts: Sequence[str]) -> Dict[str, Counter]:
    """Coarse and fine part-of-speech counts over a list of entity strings."""
    pos: Counter = Counter()
    tag: Counter = Counter()
    for parsed in nlp.pipe(texts):
        for token in parsed:
            pos[token.pos_] += 1
            tag[token.tag_] += 1
    return {"pos": pos, "tag": tag}


def render_markdown(
    counts: Dict[str, Dict[str, Counter]], labels: Sequence[str], top: int
) -> str:
    lines: List[str] = ["# Part-of-speech profile of Auto-AdvER entities", ""]
    for label in labels:
        if label not in counts:
            continue
        lines.append("## %s" % label)
        lines.append("")
        for scheme, title in (("pos", "Coarse tag (pos_)"), ("tag", "Fine tag (tag_)")):
            table = counts[label][scheme]
            total = sum(table.values()) or 1
            lines.append("| %s | Tokens | Share |" % title)
            lines.append("|---|---|---|")
            for name, count in table.most_common(top):
                lines.append("| %s | %d | %.1f%% |" % (name, count, count / total * 100))
            lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pos_stats.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="one or more JSONL files (either format)")
    parser.add_argument(
        "--model", default="en_core_web_sm", help="spaCy pipeline to tag the entities with"
    )
    parser.add_argument(
        "--labels",
        default=",".join(LABELS_DEFAULT),
        help="comma-separated labels to report, in order",
    )
    parser.add_argument("--top", type=int, default=15, help="tags to list per table")
    parser.add_argument("--out", help="write markdown here instead of stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    labels = [label for label in args.labels.split(",") if label]
    texts = entity_texts(args.inputs)
    nlp = load_spacy(args.model)
    counts = {label: tag_counts(nlp, texts[label]) for label in labels if label in texts}
    text = render_markdown(counts, labels, args.top)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
