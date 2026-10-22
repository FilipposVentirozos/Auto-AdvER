# -*- coding: utf-8 -*-
"""List the highest-weighted CRF state features per label.

This regenerates the paper's "CRF top features" appendix.  It is a port of the
tail of the original top-features script, including its monkey-patch of
``pycrfsuite._dumpparser`` (the CRFsuite text dump contains lines the shipped
regular expression cannot parse; without the guard, reading
``crf.state_features_`` raises).

Which model produced the appendix
---------------------------------
The published appendix came from that script, **not** from the searched model
behind the results table.  It differed in three ways: ``en_core_web_sm`` instead
of ``en_core_web_trf``; fixed ``c1 = c2 = 0.1`` instead of the searched values;
fold 1 only, trained on the train split with no validation set.  Its features
were also extracted from ``" ".join(tokens)`` rather than the raw advert text.

So that the README can state exactly which model produced the table, this script
makes the choice explicit:

* ``--model model.pkl`` lists the features of a model pickled by
  ``train_predict.py --save-model`` -- the searched, reported model;
* otherwise it trains a fresh CRF with the given ``--c1 --c2 --spacy-model
  --config --text-from``.  The defaults reproduce the published appendix exactly:
  ``--c1 0.1 --c2 0.1 --spacy-model en_core_web_sm --config all --text-from
  tokens``.

Heavy dependencies are imported lazily, so ``--help`` works without them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import features as feats  # noqa: E402
from train_predict import _texts_for, load_hf, prepare_data  # noqa: E402

APPENDIX_DEFAULTS = {
    "c1": 0.1, "c2": 0.1, "spacy_model": "en_core_web_sm",
    "config": "all", "text_from": "tokens",
}


def patch_dump_parser() -> None:
    """Guard ``pycrfsuite``'s dump parser against lines it cannot parse.

    Verbatim in effect from ``process.py``; applied lazily, immediately before
    ``state_features_`` is touched.
    """
    import re

    import pycrfsuite._dumpparser as dumpparser

    original = dumpparser.CRFsuiteDumpParser.parse_STATE_FEATURES

    def safe_parse_state(self, line):
        if re.match(r"\(\d+\) (.+) --> (.+): ([+-]?\d+\.\d+)", line) is None:
            return
        original(self, line)

    dumpparser.CRFsuiteDumpParser.parse_STATE_FEATURES = safe_parse_state


def train(args):
    """Train a CRF with fixed ``c1``/``c2`` and return it."""
    import sklearn_crfsuite
    import spacy

    nlp = spacy.load(args.spacy_model)
    train_data = load_hf(args.train)
    texts = _texts_for(args.train, args.train_text, args.text_from, train_data)
    X_train, y_train, _ = prepare_data(nlp, train_data, texts, args.config,
                                       drop_mismatched=True, name="train",
                                       scheme=args.label_scheme)
    print(f"training on {len(X_train)} documents "
          f"({feats.describe_config(args.config)}; spaCy {args.spacy_model}; "
          f"c1={args.c1} c2={args.c2})")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=args.c1,
        c2=args.c2,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    return crf


def top_features(crf, top_n: int = 10) -> dict:
    """Return ``{label: [(feature, weight), ...]}``, highest weight first."""
    from collections import Counter

    patch_dump_parser()
    state_features = Counter(crf.state_features_)
    ranked = {}
    for label in sorted(crf.classes_):
        if label == "O":
            continue
        ranked[label] = [(attr, weight)
                         for (attr, lbl), weight in state_features.most_common()
                         if lbl == label][:top_n]
    return ranked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="top_features.py",
        description="List the top-weighted CRF state features per label "
                    "(the paper's CRF top-features appendix).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The published appendix was produced by a DIFFERENT model from the\n"
            "results table: en_core_web_sm, fixed c1 = c2 = 0.1, fold 1, features\n"
            "from ' '.join(tokens). Those are this script's defaults, so\n"
            "\n"
            "  python src/crf/top_features.py --train data/folds/f1/train_hf.json\n"
            "\n"
            "reproduces the appendix, whereas\n"
            "\n"
            "  python src/crf/top_features.py --model work/crf/f1_all_model.pkl\n"
            "\n"
            "lists the features of the searched model that produced the reported\n"
            "CRF numbers (pickle it with train_predict.py --save-model).\n"
        ),
    )
    source = parser.add_argument_group("model source")
    source.add_argument("--model", default=None,
                        help="pickled estimator from train_predict.py --save-model; "
                             "when given, no training happens and --c1/--c2/"
                             "--spacy-model/--config are ignored")
    source.add_argument("--train", default=None,
                        help="train fold *_hf.json, to retrain a model instead")
    source.add_argument("--train-text", default=None,
                        help="raw-text JSON Lines for --train (default: --train with "
                             "_hf.json -> .jsonl); only used with --text-from text")
    train_group = parser.add_argument_group("retraining settings (appendix defaults)")
    train_group.add_argument("--c1", type=float, default=APPENDIX_DEFAULTS["c1"],
                             help="L1 coefficient (default: %(default)s)")
    train_group.add_argument("--c2", type=float, default=APPENDIX_DEFAULTS["c2"],
                             help="L2 coefficient (default: %(default)s)")
    train_group.add_argument("--spacy-model", default=APPENDIX_DEFAULTS["spacy_model"],
                             help="spaCy pipeline (default: %(default)s, as in the "
                                  "appendix; the results table used en_core_web_trf)")
    train_group.add_argument("--config", default=APPENDIX_DEFAULTS["config"],
                             choices=feats.CONFIGS,
                             help="feature configuration (default: %(default)s)")
    train_group.add_argument("--label-scheme", default="plain",
                             choices=("plain", "iob2"),
                             help="default: %(default)s, as in the appendix")
    train_group.add_argument("--text-from", default=APPENDIX_DEFAULTS["text_from"],
                             choices=("text", "tokens"),
                             help="default: %(default)s, as in the appendix")
    parser.add_argument("--top-n", type=int, default=10,
                        help="features per label (default: %(default)s)")
    parser.add_argument("--out", default=None,
                        help="also write the table as JSON to this path")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model and not args.train:
        raise SystemExit("pass either --model (a pickled estimator) or --train "
                         "(a fold to retrain on)")

    if args.model:
        import pickle
        with open(args.model, "rb") as handle:
            crf = pickle.load(handle)
        provenance = f"pickled model {args.model}"
    else:
        crf = train(args)
        provenance = (f"retrained: config={args.config}, c1={args.c1}, c2={args.c2}, "
                      f"spacy={args.spacy_model}, text_from={args.text_from}, "
                      f"train={args.train}")

    ranked = top_features(crf, top_n=args.top_n)
    print(f"\nsource: {provenance}")
    for label, rows in ranked.items():
        print(f"\n=== Top {args.top_n} features for {label} ===")
        for attr, weight in rows:
            print(f"  {weight:+.6f}  {attr}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"provenance": provenance,
                       "top_features": {k: [[a, w] for a, w in v]
                                        for k, v in ranked.items()}},
                      handle, indent=4)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
