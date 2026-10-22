# -*- coding: utf-8 -*-
"""Run an LLM over a test fold and write NER predictions.

This replaces the original labelling script, whose fold, model and output
suffix were chosen by editing ``__main__``.  Everything is now a flag, nothing
heavy is imported at module load, and no credentials, cache paths or CUDA
assignments are baked in.

Pipeline, per test document, per label
-------------------------------------
1. build the in-context prompt for the label (``icl.py``);
2. generate (``generate.py``) and parse the ``@@``/``##`` markers into character
   spans over the advert text;
3. convert character offsets to Prodigy token ids by **strict containment**
   (``token.start >= span.start and token.end <= span.end``), as the original;
4. resolve cross-label overlaps that share a start offset: keep the span with the
   highest ``@@`` log probability, or -- when the backend gives no log
   probabilities, i.e. every local model -- pick one arbitrarily.

Two behavioural notes, both deliberate:

* The original tie-break used an **unseeded** ``random.choice``, so the
  Llama/Qwen/Gemini predictions in ``predictions/`` are not reproducible.  Here
  the tie-break is seeded from ``--seed`` when that is given, and is otherwise
  the first span of the group in document order (a deterministic default).
* The original deleted losing spans group by group while re-using indices
  computed before the deletions, so a document with two or more start-collisions
  could delete the wrong span or raise ``IndexError``, in which case *all* spans
  of the document were dropped ("generation error").  This port collects the
  losers across all groups and deletes them once, descending.  The stored
  predictions came from the original code, so on such documents a re-run will not
  match byte-for-byte.

Environment
-----------
No cache directory or GPU list is set here.  Export ``HF_HOME`` and
``CUDA_VISIBLE_DEVICES`` yourself; see ``src/llm/README.md``.  Credentials come
from a gitignored ``.env`` at the repository root (``env.py``).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import icl  # noqa: E402  (stdlib-only)
from env import load_env  # noqa: E402

#: Label iteration order.  This is ``ner_labelling.py``'s order, which differs
#: from ``icl.py``'s construction order and matters because it decides the order
#: spans are appended and therefore which span wins an unscored overlap.
DEFAULT_LABELS = ["SALES_OPTIONS", "HISTORIC", "CONDITION"]

OUT_FORMATS = ("prodigy", "public")


# --------------------------------------------------------------------------- #
# Span post-processing
# --------------------------------------------------------------------------- #

def get_token_ids_from_offsets(tokens: Sequence[dict],
                               spans: Optional[Sequence[dict]]) -> Optional[List[dict]]:
    """Attach Prodigy ``token_start``/``token_end`` to character spans.

    A token belongs to a span when it is **fully contained** in it, exactly as in
    the original.  A span containing no fully-contained token (for instance one
    that starts mid-token) is dropped with a warning -- the original dropped it
    silently.  ``spans=None`` returns ``None``, again as before.
    """
    if spans is None:
        return None
    result: List[dict] = []
    for span in spans:
        start, end = span["start"], span["end"]
        token_ids = [token["id"] for token in tokens
                     if token["start"] >= start and token["end"] <= end]
        if not token_ids:
            warnings.warn(
                f"dropping span [{start}:{end}] {span.get('span', '')!r}: "
                "no token is fully contained in it"
            )
            continue
        converted = {"start": start, "end": end,
                     "token_start": token_ids[0], "token_end": token_ids[-1]}
        if "logprob" in span:
            converted["logprob"] = span["logprob"]
        result.append(converted)
    return result


def resolve_overlaps(spans: List[dict], rng: Optional[random.Random] = None) -> List[dict]:
    """Keep one span per start offset, across labels.

    Spans from different labels that share a ``start`` are competing readings of
    the same text, so only one survives.  The winner is the span with the highest
    ``logprob`` if every candidate has one (the OpenAI path); otherwise the first
    candidate in document order, or a pick from *rng* when one is supplied.

    Returns a new list in the original order, minus the losers.
    """
    groups: Dict[int, List[int]] = defaultdict(list)
    for index, span in enumerate(spans):
        groups[span["start"]].append(index)

    losers = set()
    for _, candidates in groups.items():
        if len(candidates) < 2:
            continue
        if all("logprob" in spans[i] for i in candidates):
            winner = max(candidates, key=lambda i: spans[i]["logprob"])
        elif rng is not None:
            winner = rng.choice(candidates)
        else:
            winner = candidates[0]
        losers.update(i for i in candidates if i != winner)

    return [span for index, span in enumerate(spans) if index not in losers]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def _public_record(record: dict, index: int, warned: List[bool]) -> dict:
    """Reduce a record to the release schema, ``{id, entities}``."""
    doc_id = record.get("id")
    if doc_id is None:
        if not warned[0]:
            warnings.warn(
                "records carry no 'id' field; falling back to the positional "
                "index. Join by position against the gold fold, or add ids first."
            )
            warned[0] = True
        doc_id = index
    entities = [{"start": span["start"], "end": span["end"], "label": span["label"]}
                for span in record.get("spans", [])]
    return {"id": doc_id, "entities": entities}


def write_predictions(records: Sequence[dict], out_path: str,
                      out_format: str = "prodigy") -> None:
    """Write JSON Lines predictions.

    ``prodigy`` writes the input record back out with ``spans`` replaced by the
    predictions -- the historical format, and what the stored LLM predictions
    look like.  When the input was in the public format, the record keeps its own
    fields and gains a ``spans`` list.  ``public`` emits the release schema,
    ``{id, entities}`` with character offsets, ``end`` exclusive.
    """
    if out_format not in OUT_FORMATS:
        raise ValueError(f"unknown --out-format {out_format!r}; expected {OUT_FORMATS}")
    directory = os.path.dirname(os.path.abspath(out_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    warned = [False]
    with open(out_path, "w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            if out_format == "public":
                payload = _public_record(record, index, warned)
            else:
                payload = dict(record.get("raw") or record)
                payload.pop("raw", None)
                spans = record.get("spans", [])
                payload["spans"] = spans
                if "entities" in payload:
                    # A public input record still carries its GOLD entities and
                    # IOB2 labels. Replace and drop them, so the file cannot be
                    # read as a mixture of gold and prediction.
                    payload["entities"] = [
                        {"start": span["start"], "end": span["end"],
                         "label": span["label"]} for span in spans]
                    payload.pop("labels", None)
            handle.write(json.dumps(payload) + "\n")


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def predict_fold(gold_fold: str, icl_from: str, backend_name: str,
                 model: Optional[str] = None, n_icl: int = 100,
                 icl_mode: str = "first", seed: Optional[int] = None,
                 labels: Sequence[str] = tuple(DEFAULT_LABELS),
                 progress: bool = True) -> List[dict]:
    """Predict spans for every document of *gold_fold*.

    *gold_fold* may still carry gold spans; they are ignored and overwritten (use
    ``clear_test.py`` if you would rather be certain the model cannot see them).
    Returns the records with ``spans`` replaced by the predictions.
    """
    import generate  # local import keeps `--help` free of heavy dependencies

    load_env()
    # Build the backend before reading anything, so an unusable backend (vertex)
    # fails immediately rather than after loading the folds.
    backend = generate.get_backend(backend_name, model=model, seed=seed)

    records = icl.load_records(gold_fold)
    icl_records = icl.load_records(icl_from)
    system_instruction, template = icl.icl_prompt_template(
        records=icl_records, n_icl=n_icl, seed=seed or 0, mode=icl_mode)

    rng = random.Random(seed) if seed is not None else None

    iterator = enumerate(records)
    if progress:
        try:
            import tqdm  # optional
            iterator = tqdm.tqdm(iterator, total=len(records))
        except ImportError:
            print(f"predicting {len(records)} documents "
                  f"({len(labels)} prompts each)", file=sys.stderr)

    for index, record in iterator:
        spans: List[dict] = []
        for label in labels:
            try:
                _, parsed = backend.generate(
                    chat_history=template[label],
                    system_instruction=system_instruction,
                    prompt=record["text"],
                )
            except AssertionError as exc:  # kept from the original
                print(exc)
                continue
            if not parsed:
                continue
            converted = get_token_ids_from_offsets(record["tokens"], parsed)
            if not converted:
                continue
            for span in converted:
                span["label"] = label
            spans.extend(converted)
        record["spans"] = resolve_overlaps(spans, rng=rng)
        if progress and not spans:
            print(f"document {index}: no spans predicted", file=sys.stderr)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description="Run an LLM over an Auto-AdvER test fold and write NER "
                    "predictions. Re-scoring the shipped predictions with "
                    "src/score.py needs no model, key or GPU; this script is the "
                    "optional generation path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples\n"
            "  # Qwen on fold 1, the paper's Hugging Face setting (needs GPUs):\n"
            "  python src/llm/predict.py --gold-fold data/folds/f1/test.jsonl \\\n"
            "      --icl-from data/folds/f1/train.jsonl --backend hf \\\n"
            "      --model Qwen/Qwen2.5-72B-Instruct --n-icl 100 --seed 42 \\\n"
            "      --out predictions_rerun/qwen2.5-72b/f1_test.jsonl\n"
            "\n"
            "  # GPT-4o as originally run: uncapped in-context examples. Costs money.\n"
            "  python src/llm/predict.py --gold-fold data/folds/f1/test.jsonl \\\n"
            "      --icl-from data/folds/f1/train.jsonl --backend openai --n-icl 0 \\\n"
            "      --out predictions_rerun/gpt-4o/f1_test.jsonl\n"
        ),
    )
    parser.add_argument("--gold-fold", required=True,
                        help="test fold to predict on (JSON Lines or a JSON array); "
                             "any gold spans it carries are overwritten")
    parser.add_argument("--icl-from", required=True,
                        help="gold file the in-context examples are drawn from, "
                             "normally the matching train fold")
    parser.add_argument("--backend", required=True, choices=("openai", "hf", "vertex"),
                        help="openai = gpt-4o-2024-11-20; hf = local transformers "
                             "pipeline (Llama/Qwen); vertex = Gemini, NOT implemented")
    parser.add_argument("--model", default=None,
                        help="model id; defaults to gpt-4o-2024-11-20 for openai and "
                             "Qwen/Qwen2.5-72B-Instruct for hf")
    parser.add_argument("--n-icl", type=int, default=100,
                        help="number of in-context documents, applied to ALL backends. "
                             "100 (default) reproduces the Llama/Qwen prompts; 0 means "
                             "no cap, which is what the original GPT-4o path did (426 "
                             "documents). See src/llm/icl.py for the discrepancy with "
                             "the paper's 'random 100'.")
    parser.add_argument("--icl-mode", default="first", choices=("first", "random"),
                        help="first = file order (the historical behaviour, default); "
                             "random = seeded sample (what the paper describes, never "
                             "actually used)")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for the random ICL sample, transformers.set_seed, "
                             "and the overlap tie-break. Without it the tie-break is "
                             "deterministic (first span in document order) and "
                             "--icl-mode random falls back to seed 0.")
    parser.add_argument("--out", required=True, help="output JSON Lines path")
    parser.add_argument("--out-format", default="prodigy", choices=OUT_FORMATS,
                        help="prodigy = full records with predicted spans (default); "
                             "public = {id, entities} with character offsets")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS),
                        help="comma-separated labels, in the order they are queried "
                             "(default: %(default)s)")
    parser.add_argument("--no-progress", action="store_true",
                        help="suppress the progress bar")
    parser.add_argument("--dry-run", action="store_true",
                        help="load the data, build the prompts and report their size, "
                             "then exit without contacting any model")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    unknown = [label for label in labels if label not in icl.LABEL_DEFINITIONS]
    if unknown:
        raise SystemExit(f"unknown label(s) {unknown}; expected any of "
                         f"{sorted(icl.LABEL_DEFINITIONS)}")

    if args.dry_run:
        records = icl.load_records(args.gold_fold)
        icl_records = icl.load_records(args.icl_from)
        _, template = icl.icl_prompt_template(
            records=icl_records, n_icl=args.n_icl, seed=args.seed or 0,
            mode=args.icl_mode)
        n_examples = (len(template[labels[0]]) - icl.N_FRAMING_TURNS) // 2
        chars = sum(len(turn["content"]) for turn in template[labels[0]])
        print(f"backend           : {args.backend} "
              f"(model {args.model or 'default'}) -- NOT contacted (--dry-run)")
        print(f"documents to label: {len(records)} from {args.gold_fold}")
        print(f"ICL pool          : {len(icl_records)} from {args.icl_from}")
        print(f"ICL examples used : {n_examples} (mode {args.icl_mode}, "
              f"n-icl {args.n_icl}, seed {args.seed})")
        print(f"labels            : {', '.join(labels)}")
        print(f"prompts to send   : {len(records) * len(labels)}")
        print(f"prompt size       : ~{chars:,} characters of chat history per prompt")
        print(f"would write       : {args.out} (format {args.out_format})")
        return 0

    records = predict_fold(
        gold_fold=args.gold_fold,
        icl_from=args.icl_from,
        backend_name=args.backend,
        model=args.model,
        n_icl=args.n_icl,
        icl_mode=args.icl_mode,
        seed=args.seed,
        labels=labels,
        progress=not args.no_progress,
    )
    write_predictions(records, args.out, out_format=args.out_format)
    print(f"wrote {len(records)} records to {args.out} ({args.out_format} format)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
