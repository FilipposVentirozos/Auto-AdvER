# -*- coding: utf-8 -*-
"""Prompt construction for the Auto-AdvER LLM baselines.

This module is a faithful port of the original prompt-construction module.  It is the single most
important reproducibility artefact of the LLM experiments, so the prompt strings
are reproduced **verbatim**, including the two typographical errors that the
models actually saw:

* the HISTORIC definition reads "affected its condition mechanically condition
  or bodywork";
* the SALES_OPTIONS definition reads "The SALES OPTIONS label refer to" (and
  spells the label without the underscore).

Do not tidy these up: the stored predictions in ``predictions/`` were generated
with exactly these strings.

Structure of a prompt (one prompt per label, per document)
----------------------------------------------------------
1. a ``system`` turn with the system instruction;
2. a ``user`` turn stating the task and quoting the label definition, and the
   fixed ``assistant`` acknowledgement (the "framing" pair);
3. ``2 * n`` alternating in-context turns: the advert text as ``user``, the same
   text with ``@@``/``##`` inserted around the gold spans of that label as
   ``assistant``.  Documents with no span of the label become negative examples
   (the assistant turn repeats the text unchanged);
4. finally the advert under test as a ``user`` turn (added in ``generate.py``).

Known discrepancy with the paper
--------------------------------
The paper states "random 100 samples from the training data".  The original code
never sampled or shuffled: examples were appended in **file order**.  Moreover
only the Hugging Face path was capped, at ``chat_history[:202]`` -- two framing
turns plus 200 messages, i.e. **100 examples**.  The GPT-4o path was **uncapped
in every commit**, so with ``f{n}_train.jsonl`` it received all **426** training
adverts.  The Gemini runs are lost and their example count is unrecorded.

For the public release, :func:`select_icl_examples` adds an explicit
``--icl-mode`` with two modes:

``first``
    file order, i.e. the historical behaviour.  This is the default, and with
    ``--n-icl 100`` it reproduces the Hugging Face (Llama/Qwen) prompts exactly.
``random``
    a seeded :func:`random.Random.sample`, i.e. what the paper describes.  This
    was never used for the published numbers.

``--n-icl`` applies to **all** backends in this release, including ``openai``.
That is a deliberate divergence from the original GPT-4o path: re-running the
OpenAI backend with ``--n-icl 100`` will *not* reproduce the paper's GPT-4o
predictions, which used 426 examples.  Pass ``--n-icl 0`` (meaning "no cap") to
restore the original, uncapped GPT-4o behaviour.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "LABELS",
    "LABEL_DEFINITIONS",
    "SYSTEM_INSTRUCTION",
    "N_FRAMING_TURNS",
    "load_records",
    "normalise_record",
    "select_icl_examples",
    "fill_in_icl_examples",
    "icl_prompt_template",
    "task_turns",
]

#: Label construction order, as in the original ``icl.py``.  Note that the
#: prediction driver *iterates* labels in a different order; see
#: ``predict.py --labels``.
LABELS: List[str] = ["HISTORIC", "CONDITION", "SALES_OPTIONS"]

# VERBATIM from the original prompt module.  Typos intentional; the
# CONDITION definition uses a curly apostrophe in "car's" and has no full stop.
LABEL_DEFINITIONS: Dict[str, str] = {
    "HISTORIC": """The HISTORIC label identifies any word or span of words that presents events that have happened to the car in the past and affected its condition mechanically condition or bodywork.""",

    "CONDITION": """The CONDITION label identifies any word or span of words in car advertisements that indicate the present material or documented condition of the vehicle, both positive and negative. This includes descriptions that reflect the car’s current operational performance, mechanical state, aesthetic appearance, maintenance status, and legal classifications affecting its value or usability""",

    "SALES_OPTIONS": """The SALES OPTIONS label refer to any word or span of words that indicate something tangible or a service offered during or after the sale apart from the car itself, and enhances the sale proposition.""",
}

# VERBATIM from the original prompt module.
SYSTEM_INSTRUCTION = ("""You are a Natural Language Processing assistant specializing in car-related named entity recognition. In the following conversation, perform the tasks exactly as requested, without providing explanations or reasoning.""")

#: The two framing turns that precede the in-context examples.  There are two,
#: so the original ``chat_history[:202]`` cap equals ``2 + 2 * n_icl``.
N_FRAMING_TURNS = 2


def _nested_defaultdict():
    return defaultdict(list)


def _read_json_any(path: str) -> List[dict]:
    """Read JSON Lines, or a single top-level JSON array."""
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        while first and first.isspace():
            first = handle.read(1)
        handle.seek(0)
        if first == "[":
            return json.load(handle)
        return [json.loads(line) for line in handle if line.strip()]


def _is_prodigy(record: dict) -> bool:
    tokens = record.get("tokens")
    if "spans" in record:
        return True
    return isinstance(tokens, list) and bool(tokens) and isinstance(tokens[0], dict)


def normalise_record(record: dict) -> dict:
    """Return ``{'id', 'text', 'tokens', 'spans'}`` for either on-disk format.

    Two record layouts occur in this project.  The **public** release format has
    string ``tokens`` and ``entities`` with character offsets; the internal
    **Prodigy** format has ``tokens`` as dicts carrying their own offsets and
    ``spans``.  Both are accepted here, and the normalised shape is the Prodigy
    one, because that is what the original prompt and span-conversion code
    consumed.

    For public records the per-token character offsets have to be reconstructed
    from the text.  That is delegated to ``src/data_tools/records.py``, the
    repository's single implementation of that conversion, so this code and the
    scorer cannot disagree about it.
    """
    if _is_prodigy(record):
        return {
            "id": record.get("id") or (record.get("meta") or {}).get("metadata_search_id"),
            "text": record["text"],
            "tokens": record.get("tokens") or [],
            "spans": [{"start": span["start"], "end": span["end"],
                       "label": span["label"]}
                      for span in record.get("spans", [])
                      if span.get("start") is not None],
            "raw": record,
        }

    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    try:
        from data_tools import records as records_mod
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "this looks like a public-format record (string tokens, 'entities'), "
            "whose per-token character offsets must be reconstructed from the "
            "text. That conversion lives in src/data_tools/records.py, which "
            "could not be imported."
        ) from exc

    document = records_mod.document_from_record(record)
    if document.text is None or document.offsets is None:
        raise ValueError(
            f"record {document.doc_id!r} carries no text or no tokens, so it "
            "cannot be used for prompting"
        )
    tokens = [{"text": surface, "start": start, "end": end, "id": index}
              for index, (surface, (start, end))
              in enumerate(zip(document.tokens, document.offsets))]
    return {"id": document.doc_id, "text": document.text, "tokens": tokens,
            "spans": document.char_spans(), "raw": record}


def load_records(path: str) -> List[dict]:
    """Load a gold or fold file and normalise every record.

    See :func:`normalise_record`: both the public release format and the
    internal Prodigy format are accepted, and the result always has Prodigy-shaped
    ``tokens`` and ``spans``.  The untouched input record is kept under ``raw``,
    so predictions can be written back in the format they came from.
    """
    return [normalise_record(record) for record in _read_json_any(path)]


def select_icl_examples(
    records: Sequence[dict],
    n: int = 100,
    seed: int = 0,
    mode: str = "first",
) -> List[dict]:
    """Choose the in-context documents from *records*.

    Parameters
    ----------
    records:
        Gold records, in file order.
    n:
        How many documents to keep.  ``n <= 0`` or ``n >= len(records)`` keeps
        all of them (the original, uncapped GPT-4o behaviour).
    seed:
        Seed for ``mode="random"``.  Ignored by ``mode="first"``.
    mode:
        ``"first"`` takes the first *n* records in file order -- the historical
        behaviour, and the default.  ``"random"`` takes a seeded sample,
        preserving file order among the chosen records so the prompt is
        deterministic for a given seed.

    Returns
    -------
    A new list; *records* is not mutated.
    """
    if mode not in ("first", "random"):
        raise ValueError(f"unknown ICL mode {mode!r}; expected 'first' or 'random'")
    records = list(records)
    if n is None or n <= 0 or n >= len(records):
        return records
    if mode == "first":
        return records[:n]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(records)), n))
    return [records[i] for i in chosen]


def _mark_spans(record: dict, label: str) -> str:
    """Insert ``@@``/``##`` around every *label* span of *record*.

    Reproduces the original algorithm exactly, including its treatment of nested
    or overlapping same-label spans: starts and ends are each sorted descending
    and then zipped *independently*, so for nested spans the pairing is not the
    original one.  This is what the models were shown, so it is kept.
    """
    idx = defaultdict(_nested_defaultdict)
    for span in record["spans"]:
        if span["label"] in LABELS:
            idx[span["label"]]["start"].append(span["start"])
            idx[span["label"]]["end"].append(span["end"])
    starts = idx[label]["start"]
    starts.sort(reverse=True)
    ends = idx[label]["end"]
    ends.sort(reverse=True)
    output = copy.copy(record["text"])
    for s, e in zip(starts, ends):
        output = output[:e] + "##" + output[e:]
        output = output[:s] + "@@" + output[s:]
    return output


def fill_in_icl_examples(records: Iterable[dict]) -> Dict[str, List[Tuple[str, str]]]:
    """Build ``{label: [(text, marked_text), ...]}`` from gold *records*.

    Every record yields one example for **every** label, so records without a
    span of a given label become negative examples for it.  Order follows
    *records*.
    """
    examples: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for record in records:
        for label in LABELS:
            examples[label].append((record["text"], _mark_spans(record, label)))
    return examples


def task_turns(label: str) -> List[dict]:
    """Return the two framing turns for *label*, verbatim from the original."""
    return [
        {"role": "user", "content":
            f"""The task is to label {label} entities in the given text. {label} is characterised as '{LABEL_DEFINITIONS[label]}'. Your task is to detect {label} span words by inserting '@@' at the beginning of the span and '##' at the end, and generating the remaining text verbatim, whitespaces and newlines included for parsing purposes."""},
        {"role": "assistant", "content":
            f"Sure, I can help with that. Please provide the text you want me to label for {label} entities."},
    ]


def icl_prompt_template(
    gold_path: Optional[str] = None,
    n_icl: int = 0,
    seed: int = 0,
    mode: str = "first",
    records: Optional[Sequence[dict]] = None,
) -> Tuple[str, Dict[str, List[dict]]]:
    """Build the system instruction and the per-label chat history.

    Either *gold_path* (a fold file) or *records* (already loaded) must be
    given.  Example selection happens once, at record level, so all three labels
    see the same documents -- as in the original.

    Returns ``(system_instruction, {label: [turn, ...]})``.  The returned chat
    histories do **not** include the system turn or the advert under test; the
    backends in ``generate.py`` add both.
    """
    if records is None:
        if gold_path is None:
            raise ValueError("pass either gold_path or records")
        records = load_records(gold_path)
    records = select_icl_examples(records, n=n_icl, seed=seed, mode=mode)
    examples = fill_in_icl_examples(records)

    template: Dict[str, List[dict]] = {}
    for label in LABELS:
        turns = task_turns(label)
        for text, marked in examples[label]:
            turns.append({"role": "user", "content": text})
            turns.append({"role": "assistant", "content": marked})
        template[label] = turns
    return SYSTEM_INSTRUCTION, template


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Print the Auto-AdvER ICL prompt for inspection.")
    parser.add_argument("--icl-from", required=True,
                        help="gold fold file, e.g. data/folds/f1/train.jsonl")
    parser.add_argument("--label", default="HISTORIC", choices=LABELS)
    parser.add_argument("--n-icl", type=int, default=100,
                        help="number of in-context documents; 0 means no cap")
    parser.add_argument("--icl-mode", default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=6,
                        help="how many chat turns to print (0 = all)")
    args = parser.parse_args(argv)

    system_instruction, template = icl_prompt_template(
        args.icl_from, n_icl=args.n_icl, seed=args.seed, mode=args.icl_mode)
    turns = template[args.label]
    print("=== system ===")
    print(system_instruction)
    shown = turns if args.max_turns <= 0 else turns[: args.max_turns]
    for turn in shown:
        print(f"=== {turn['role']} ===")
        print(turn["content"])
    if len(shown) < len(turns):
        print(f"... {len(turns) - len(shown)} further turns "
              f"({(len(turns) - N_FRAMING_TURNS) // 2} examples in total)")


if __name__ == "__main__":
    _main()
