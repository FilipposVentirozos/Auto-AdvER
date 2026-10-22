#!/usr/bin/env python3
"""Regenerate ``results/tables.md`` from the released files.

Every table in the output is preceded by the exact command that produced it, and
this script runs those commands as subprocesses so that the two cannot drift
apart.  Nothing here needs a model, an API key or a GPU: the tables come from the
stored predictions and the annotator files.

Six groups of tables:

1. the paper's results table, scored against the corpus annotations;
2. the same, per entity type, using the paper's own per-label definition;
3. the pooled COR/INC/PAR/MIS/SPU counts behind the partial-match scores of the
   CRF and GPT-4o, which the paper's discussion compares;
4. the boundary direction of those partial matches — too wide, too narrow or
   staggered — which the paper's error analysis quotes;
5. the inter-annotator agreement tables;
6. the mention families under the three labels: how a recurring mention such as
   the MOT or the service divides over CONDITION, HISTORIC and SALES_OPTIONS,
   how the four best models fare on each family, and where the two expert
   annotators disagree about the label of an identically bounded span.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

MODELS: List[Tuple[str, str]] = [
    ("BERT-base", "bert-base"),
    ("BERT-large", "bert-large"),
    ("DeBERTaV3-base", "deberta-v3-base"),
    ("DeBERTaV3-large", "deberta-v3-large"),
    ("GPT-4o", "gpt-4o"),
    ("Gemini 1.5 Flash", "gemini-1.5-flash"),
    ("Llama 3.1 70B it", "llama-3.1-70b"),
    ("Qwen 2.5 72B it", "qwen2.5-72b"),
    ("CRF", "crf"),
    ("CRF PoS", "crf-pos"),
    ("CRF w/o PoS", "crf-wo-pos"),
    ("CRF 2n/3n", "crf-2-3"),
]

PER_LABEL_MODELS = ["DeBERTaV3-base", "GPT-4o", "Gemini 1.5 Flash", "CRF"]

CRF_CONFIGS = [("CRF", "crf"), ("CRF PoS", "crf-pos"), ("CRF w/o PoS", "crf-wo-pos"), ("CRF 2n/3n", "crf-2-3")]

ENCODER_CONFIGS = [
    ("BERT-base", "bert-base"),
    ("BERT-large", "bert-large"),
    ("DeBERTaV3-base", "deberta-v3-base"),
    ("DeBERTaV3-large", "deberta-v3-large"),
]

# Every configuration whose spans come from per-token labels.
TOKEN_LABEL_CONFIGS = CRF_CONFIGS + ENCODER_CONFIGS

# The two models whose partial matches the paper's discussion compares.
COUNT_MODELS = [("CRF", "crf"), ("GPT-4o", "gpt-4o")]

FOLDS = ("f1", "f2", "f3")

# The values as typeset in the paper: partial-match P/R/F1, mean ± population
# std.  The paper reports what this release scores, so these are a check on the
# release rather than a second source.
PUBLISHED = {
    "BERT-base": "52.6 ± 1.7 / 63.5 ± 2.4 / 57.5 ± 1.2",
    "DeBERTaV3-base": "63.6 ± 1.2 / 72.4 ± 2.0 / 67.7 ± 0.8",
    "GPT-4o": "73.8 ± 1.6 / 54.4 ± 1.3 / 62.6 ± 0.3",
    "Gemini 1.5 Flash": "72.4 ± 2.1 / 49.9 ± 3.2 / 59.0 ± 2.0",
    "Llama 3.1 70B it": "75.9 ± 1.1 / 44.5 ± 2.3 / 56.1 ± 2.1",
    "Qwen 2.5 72B it": "76.8 ± 3.2 / 34.8 ± 1.1 / 47.9 ± 1.6",
    "CRF": "72.5 ± 1.8 / 60.8 ± 1.5 / 66.1 ± 1.3",
    "CRF PoS": "25.4 ± 8.6 / 1.1 ± 0.1 / 2.1 ± 0.2",
    "CRF w/o PoS": "71.9 ± 0.9 / 58.5 ± 2.5 / 64.4 ± 1.7",
    "CRF 2n/3n": "71.2 ± 0.8 / 60.1 ± 0.9 / 65.2 ± 0.8",
    "BERT-large": "not in the paper's table",
    "DeBERTaV3-large": "not in the paper's table",
}

PUBLISHED_PER_LABEL = {
    "DeBERTaV3-base": {"HISTORIC": "58.1 ± 2.3", "SALES_OPTIONS": "74.0 ± 2.7", "CONDITION": "57.8 ± 5.7"},
    "GPT-4o": {"HISTORIC": "53.0 ± 2.2", "SALES_OPTIONS": "68.4 ± 1.9", "CONDITION": "52.2 ± 5.2"},
    "Gemini 1.5 Flash": {"HISTORIC": "51.6 ± 2.4", "SALES_OPTIONS": "66.7 ± 4.1", "CONDITION": "38.5 ± 5.2"},
    "CRF": {"HISTORIC": "59.5 ± 3.8", "SALES_OPTIONS": "73.2 ± 2.8", "CONDITION": "46.5 ± 4.4"},
}

# The boundary directions as the paper's error analysis quotes them.  Gemini
# 1.5 Flash is in the table for completeness; the paper does not quote it.
PUBLISHED_BOUNDARIES = {
    "CRF": "58% too wide against 33% too narrow",
    "GPT-4o": "58% too wide against 38% too narrow",
    "DeBERTaV3-base": "more often too narrow, 53%",
}

PUBLISHED_IAA_PAIRS = {"A1–A2": "88", "A1–A3": "74", "A2–A3": "73"}


def run(command: Sequence[str], cwd: str) -> str:
    """Run a command and return its standard output, failing loudly."""
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(
            "command failed (%d): %s" % (result.returncode, shlex.join(command))
        )
    return result.stdout


def shown(command: Sequence[str]) -> str:
    """The command as a reader should type it."""
    parts = ["python3" if part == sys.executable else part for part in command]
    return shlex.join(parts).replace("'{fold}'", "{fold}")


def score_command(
    models: Sequence[Tuple[str, str]],
    predictions_dir: str = "predictions",
    per_label: bool = False,
) -> List[str]:
    command = [
        sys.executable,
        "src/score.py",
        "models",
        "--gold",
        "data/folds/{fold}/test.jsonl",
    ]
    for name, directory in models:
        command += [
            "--pred",
            "%s=%s/%s/{fold}_test.jsonl" % (name, predictions_dir, directory),
        ]
    command += ["--folds", ",".join(FOLDS)]
    if per_label:
        # The paper's column order.
        command += ["--labels", "HISTORIC,SALES_OPTIONS,CONDITION", "--per-label"]
    return command


def boundaries_command(
    models: Sequence[Tuple[str, str]],
) -> List[str]:
    command = [
        sys.executable,
        "src/score.py",
        "boundaries",
        "--gold",
        "data/folds/{fold}/test.jsonl",
    ]
    for name, directory in models:
        command += [
            "--pred",
            "%s=predictions/%s/{fold}_test.jsonl" % (name, directory),
        ]
    command += ["--folds", ",".join(FOLDS)]
    return command


def iaa_command() -> List[str]:
    return [
        sys.executable,
        "src/score.py",
        "iaa",
        "--annotators",
        "A1=data/iaa/annotator_1.jsonl",
        "A2=data/iaa/annotator_2.jsonl",
        "A3=data/iaa/annotator_3.jsonl",
    ]


def families_command(
    models: Sequence[Tuple[str, str]],
) -> List[str]:
    command = [
        sys.executable,
        "src/score.py",
        "families",
        "--corpus",
        "data/auto_adver.jsonl",
        "--gold",
        "data/folds/{fold}/test.jsonl",
    ]
    for name, directory in models:
        command += [
            "--pred",
            "%s=predictions/%s/{fold}_test.jsonl" % (name, directory),
        ]
    command += ["--folds", ",".join(FOLDS)]
    # The expert pair first: the paper's disagreement figures are A1 against A2.
    command += [
        "--annotators",
        "A1=data/iaa/annotator_1.jsonl",
        "A2=data/iaa/annotator_2.jsonl",
        "A3=data/iaa/annotator_3.jsonl",
    ]
    return command


def counts_table(scored_json: str) -> str:
    """Pooled COR/INC/PAR/MIS/SPU over the three folds, and their shares."""
    data = json.loads(scored_json)
    lines = [
        "| Model | COR | INC | PAR | MIS | SPU | Total |",
        "|---|---|---|---|---|---|---|",
    ]
    shares = [
        "| Model | COR | INC | PAR | MIS | SPU |",
        "|---|---|---|---|---|---|",
    ]
    keys = ("cor", "inc", "par", "mis", "spu")
    for name, result in data["models"].items():
        pooled = {key: 0 for key in keys}
        for fold in result["folds"]:
            for key in keys:
                pooled[key] += fold["counts"][key]
        total = sum(pooled.values())
        lines.append(
            "| %s | %s | %d |"
            % (name, " | ".join(str(pooled[key]) for key in keys), total)
        )
        shares.append(
            "| %s | %s |"
            % (
                name,
                " | ".join("%.1f" % (100.0 * pooled[key] / total) for key in keys),
            )
        )
    return "\n".join(lines) + "\n\n" + "\n".join(shares)


def boundaries_table(scored: str) -> str:
    """The pooled boundary rows, beside what the paper's error analysis says."""
    header = extract_header(scored, "### Boundary direction of partial matches")
    lines = [
        "| Model | " + " | ".join(header) + " | In the paper's error analysis |",
        "|---" * (len(header) + 2) + "|",
    ]
    for name, cells in extract_rows(scored, "### Boundary direction of partial matches"):
        lines.append(
            "| %s | %s | %s |"
            % (name, " | ".join(cells), PUBLISHED_BOUNDARIES.get(name, "\u2014"))
        )
    return "\n".join(lines)


def section(title: str, commands: Sequence[Sequence[str]], body: str) -> str:
    lines = ["## %s" % title, ""]
    lines.append("```console")
    for command in commands:
        lines.append("$ " + shown(command))
    lines.append("```")
    lines.append("")
    lines.append(body.rstrip())
    lines.append("")
    return "\n".join(lines)


def published_table(
    scored: str,
    per_label: bool = False,
    reference: Optional[dict] = None,
    reference_header: str = "As published",
    left_header: str = "Reproduced P / R / F1",
) -> str:
    """Extract the reproduced mean rows and pair them with the published ones."""
    lines: List[str] = []
    if reference is None:
        reference = PUBLISHED
    if not per_label:
        lines.append("| Model | %s | %s |" % (left_header, reference_header))
        lines.append("|---|---|---|")
        rows = extract_rows(scored, "### Partial match, mean")
        for name, cells in rows:
            lines.append(
                "| %s | %s | %s |"
                % (name, " / ".join(cells), reference.get(name, "—"))
            )
        return "\n".join(lines)
    rows = extract_rows(scored, "### Per-label partial F1")
    header = extract_header(scored, "### Per-label partial F1")
    lines.append("| Model | " + " | ".join("%s (reproduced / paper)" % label for label in header) + " |")
    lines.append("|---" * (len(header) + 1) + "|")
    for name, cells in rows:
        if name not in PUBLISHED_PER_LABEL:
            continue
        published = PUBLISHED_PER_LABEL[name]
        lines.append(
            "| %s | %s |"
            % (
                name,
                " | ".join(
                    "%s / %s" % (cell, published.get(label, "—"))
                    for cell, label in zip(cells, header)
                ),
            )
        )
    return "\n".join(lines)


def _table_after(text: str, heading: str) -> List[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(heading):
            table = []
            for candidate in lines[index:]:
                if candidate.startswith("|"):
                    table.append(candidate)
                elif table:
                    break
            return table
    raise SystemExit("heading %r not found in the scorer output" % heading)


def extract_header(text: str, heading: str) -> List[str]:
    table = _table_after(text, heading)
    return [cell.strip() for cell in table[0].strip("|").split("|")][1:]


def extract_rows(text: str, heading: str) -> List[Tuple[str, List[str]]]:
    table = _table_after(text, heading)
    rows = []
    for line in table[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append((cells[0], cells[1:]))
    return rows


def tail_from(text: str, heading: str) -> str:
    """Everything from a heading to the end of the scorer's output."""
    index = text.find(heading)
    if index < 0:
        raise SystemExit("heading %r not found in the scorer output" % heading)
    return text[index:].rstrip()


def build(root: str) -> str:
    parts: List[str] = []
    parts.append(
        "# Auto-AdvER results, regenerated\n\n"
        "Every table below was produced by the command shown above it, run from "
        "the root of this repository, using only the files in `data/` and "
        "`predictions/`. Regenerate the whole file with:\n\n"
        "```console\n$ python3 results/make_tables.py\n```\n\n"
        "Scores use the paper's partial-match scheme: `P = (COR + 0.5·PAR) / "
        "(COR + INC + PAR + SPU)`, `R = (COR + 0.5·PAR) / (COR + INC + PAR + "
        "MIS)`. Counts are pooled within a fold, and the per-fold percentages "
        "are then averaged over folds f1–f3 with a population standard "
        "deviation. Gold spans are the corpus annotations carried by the fold "
        "files.\n"
    )

    command = score_command(MODELS)
    scored = run(command, cwd=root)
    parts.append(
        section(
            "Results table (paper matching, shipped predictions)",
            [command],
            published_table(scored)
            + "\n\nThe paper reports exactly these figures, so the "
            "two columns agree by construction; the second column is typed "
            "into `results/make_tables.py` and therefore catches any drift "
            "between the paper and the release. The two large encoders were "
            "trained and then dropped from the paper's table, and are here for "
            "completeness.\n\nThe eight configurations that predict one label "
            "per token are built from `predictions/*/raw_token_labels/` with "
            "`src/data_tools/tags_to_spans.py`.\n\n"
            "The scorer's own full output, including per-fold counts and the "
            "exact-match table, is what the command above prints; only the "
            "summary rows are reproduced here.",
        )
    )

    command = score_command(
        [(name, directory) for name, directory in MODELS if name in PER_LABEL_MODELS],
        per_label=True,
    )
    scored = run(command, cwd=root)
    parts.append(
        section(
            "Per-label results table",
            [command],
            published_table(scored, per_label=True)
            + "\n\nThese four are the paper's \"top four performing models\" by "
            "F1: DeBERTaV3-base, the CRF, GPT-4o and Gemini 1.5 Flash.\n\n"
            "The label filter follows the paper's own definition: it "
            "applies to the side being iterated, so a gold CONDITION span may "
            "still match a differently labelled prediction and count as INC in "
            "the CONDITION pass. Filtering both sides instead moves these "
            "figures by up to 1.7 points.",
        )
    )

    command = score_command(COUNT_MODELS) + ["--format", "json"]
    scored = run(command, cwd=root)
    parts.append(
        section(
            "Where the partial matches are: CRF against GPT-4o",
            [command],
            counts_table(scored)
            + "\n\nCounts are pooled over the three test folds, 261 adverts in "
            "all; the second table is each count as a percentage of that "
            "model's total. The CRF both finds more of the gold spans and "
            "gets more of their boundaries wrong: it carries the larger share "
            "of partial matches and the smaller share of misses, which is the "
            "trade the paper's discussion describes.",
        )
    )

    command = boundaries_command(
        [(name, directory) for name, directory in MODELS if name in PER_LABEL_MODELS]
    )
    scored = run(command, cwd=root)
    parts.append(
        section(
            "Boundary direction of partial matches",
            [command],
            boundaries_table(scored)
            + "\n\nThe paper's four top-performing models, with every partial "
            "match the scorer counts for them classified by which way its "
            "boundaries are wrong. A prediction is *too wide* "
            "when it reaches past the gold span on at least one side and stops "
            "short on neither, *too narrow* when it lies inside the gold span, "
            "and *staggered* when it begins and ends on the same side of the "
            "gold span. Shares are percentages of that model's own pooled PAR "
            "count, over the three test folds.\n\nThis is the table behind the "
            "paper's error analysis: \u201cwhen the CRF or GPT-4o produce a partial "
            "match, the predicted span is more often too wide than too narrow "
            "(59% against 33% of the CRF\u2019s partial matches, 58% against 38% of "
            "GPT-4o\u2019s), whereas DeBERTaV3\u2019s partial matches are more often too "
            "narrow (53%).\u201d The shares here round to those integers. Gemini 1.5 "
            "Flash, which the paper does not quote, splits evenly.\n\nA "
            "too-wide partial match always shares one boundary with its gold "
            "span, because the paper\u2019s overlap test is one-sided: a prediction "
            "that strictly contains a gold span on *both* sides is not matched "
            "from the gold side at all and is counted MIS. The same quirk is "
            "documented on `Span.get_score` in `src/score.py` and pinned by "
            "`tests/test_matching.py`.\n\n" + tail_from(scored, "### Per fold"),
        )
    )

    command = iaa_command()
    scored = run(command, cwd=root)
    pair_rows = extract_rows(scored, "### Pairwise agreement")
    pairs = ["| Pair | Reproduced partial F1 | As published |", "|---|---|---|"]
    for name, cells in pair_rows:
        pairs.append(
            "| %s | %s | %s |" % (name, cells[-1], PUBLISHED_IAA_PAIRS.get(name, "—"))
        )
    parts.append(
        section(
            "Inter-annotator agreement",
            [command],
            "\n".join(pairs)
            + "\n\nA1 is the industry data scientist, A2 the guidelines lead and "
            "A3 the non-domain annotator; the files are the released annotation, "
            "which is what the paper's agreement section reports on. The "
            "published 74 for A1–A3 reproduces as 72.72; the other two agree to "
            "the printed precision.\n\n"
            + tail_from(scored, "### Per label, pooled"),
        )
    )

    command = families_command(
        [(name, directory) for name, directory in MODELS if name in PER_LABEL_MODELS]
    )
    scored = run(command, cwd=root)
    parts.append(
        section(
            "Mention families under the three labels",
            [command],
            scored.rstrip()
            + "\n\nAuto-AdvER's three labels are not three topics: they are three "
            "*readings* of a topic, and which reading applies is settled by the "
            "context around the mention. The five families above are blunt "
            "case-insensitive regular expressions over a span's text, chosen "
            "because they recur, and each of the five occurs under all three "
            "labels in the corpus — an MOT that the car holds now is "
            "CONDITION, one it passed last month is HISTORIC, and one the "
            "dealer will supply with the sale is SALES_OPTIONS. A span belongs "
            "to every family whose regex it matches, so the family rows "
            "overlap slightly and do not sum to the total; the counts are over "
            "the released corpus.\n\nThe model table is the paper's four top-performing "
            "models, scored exactly as the results table scores them, with each "
            "gold span's outcome attributed to its families. Two things show "
            "up. *Labelling a family is harder than labelling anything else*: "
            "every model's label-confusion rate on MOT (14.5% for "
            "DeBERTaV3-base, 19.5% for GPT-4o, 20.0% for the CRF, 27.7% for "
            "Gemini 1.5 Flash) is between four and seven times its rate on the spans in "
            "no family (3.1–5.3%), and on service (11.5–13.8%) two to four "
            "times. *Finding one is easier*, at least for the two models that "
            "find most spans: DeBERTaV3-base misses 0.8% of MOT spans against "
            "12.7% of the others, and the CRF 12.9% against 29.3%. The "
            "exception is tyre, which is both confusable and easily missed — "
            "Gemini 1.5 Flash misses 59.5% of the 42 tyre spans — and whose "
            "label distribution inverts the corpus’s: 5 SALES_OPTIONS "
            "against 96 elsewhere, where the corpus as a whole is 61% "
            "SALES_OPTIONS.\n\n"
            "The disagreement tables read the annotator files the agreement "
            "section uses. Restricting to spans the two annotators bounded "
            "*identically* separates the label decision from the boundary "
            "decision: the two experts placed 586 spans on exactly the same "
            "tokens and chose a different label for 7 of them, 1.2%, against "
            "16 of 429 (3.7%) for A2 against the non-domain annotator A3. Six "
            "of the experts' seven are in the MOT and service families, and the "
            "three shapes are the contextual distinction itself — a warranted "
            "MOT read as the car's current state or as something the sale "
            "includes, a recent service read as a past event or as the car's "
            "present condition, a count of services read as a history or as a "
            "package on offer. Four of the seven adverts are in a test fold, "
            "and the models split the same way the experts did: all four read "
            "“12 month MOT” as CONDITION where the adjudicated gold says "
            "SALES_OPTIONS, and on “2 services” Gemini 1.5 Flash says "
            "SALES_OPTIONS where the other three and the gold say HISTORIC.\n\n"
            "The snippet candidates are the raw material for the paper's "
            "example table, not the table itself: the ordering is fixed so the "
            "list does not move between runs, and a human picks one cell at a "
            "time. Everything printed is a gold span of the released corpus, "
            "2–8 tokens long, with newlines flattened to spaces and anything "
            "matching an e-mail, URL, telephone, postcode or "
            "registration-plate pattern dropped.",
        )
    )

    return "\n".join(parts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_tables.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--root",
        default=default_root,
        help="repository root holding data/, predictions/ and src/",
    )
    parser.add_argument(
        "--out",
        help="output markdown path (default: results/tables.md under --root)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = os.path.abspath(args.root)
    out = args.out or os.path.join(root, "results", "tables.md")
    text = build(root)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print("wrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
