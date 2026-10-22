# -*- coding: utf-8 -*-
"""Train and predict with the Auto-AdvER CRF baselines.

Port of the original cross-validation script that produced all four CRF
rows of the paper.  Changes, all of them deliberate:

* the four feature configurations are ``--config {all,pos,wo_pos,2_3}`` instead
  of a commented-out block that had to be edited between runs (see
  ``features.py``);
* absolute paths are replaced by ``--train/--val/--test`` and ``--out``;
* ``--seed`` sets ``RandomizedSearchCV(random_state=...)``, which the original
  lacked, so the sampled ``c1``/``c2`` are now reproducible;
* ``--save-params`` writes the winning ``c1``/``c2`` and the library versions to
  JSON.  The original only printed them, which is why the exact regularisation
  behind the published CRF numbers is unknown;
* ``--save-model`` pickles the refitted estimator so ``top_features.py`` can list
  the features of *this* model rather than retraining a different one;
* a feature/label length mismatch drops the document with a warning.  The
  original printed a message but had the ``continue`` commented out, so the
  features of a misaligned document were zipped against the wrong labels.

The search itself is unchanged: ``sklearn_crfsuite.CRF(algorithm='lbfgs',
max_iterations=100, all_possible_transitions=True)``, ``c1 ~ expon(scale=0.5)``,
``c2 ~ expon(scale=0.05)``, ``n_iter=100``, ``n_jobs=1``, scored by weighted flat
F1 over the non-``O`` labels, on a ``PredefinedSplit`` that uses the validation
fold only.  ``refit=True`` means the final model is retrained on train+val --
that was the original behaviour, flagged by the author's own ``# todo``.

Input format
------------
``--train/--val/--test`` are the Hugging Face-style JSON files
(``[{"tokens": [...], "ner_tags": [...]}, ...]``).  Their labels may be flat
(``CONDITION``) or IOB2 (``B-CONDITION``); ``--label-scheme`` decides which the
model is trained on, and converts if needed.  The published runs used flat
labels, which is the default.  The
features, however, are extracted by running spaCy over the **raw advert text**,
not over the tokens: that is what the original did, and it is why the token count
of a document can differ from its label count.  The raw text comes from the
matching Prodigy JSON Lines file, which by default is the input path with
``_hf.json`` replaced by ``.jsonl``; override with ``--train-text`` etc.

``--text-from tokens`` instead reproduces ``process.py``'s behaviour
(``nlp(" ".join(tokens))``).  That is *not* the paper's CRF setting -- it is the
setting behind the top-features appendix -- so it is not the default.

Output
------
``--out`` is a JSON array of per-document label lists, the same shape the
original wrote and what ``src/data_tools/tags_to_spans.py`` consumes.  Test
documents are never dropped, so the output stays positionally aligned with the
gold test fold even when a document's features and labels disagree in length.

``spacy`` and ``sklearn_crfsuite`` are imported lazily, so ``--help`` works
without them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from typing import List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import features as feats  # noqa: E402  (stdlib-only)

DEFAULT_SPACY_MODEL = "en_core_web_trf"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_hf(path: str) -> List[dict]:
    """Load a Hugging Face-format fold: a JSON array of ``{tokens, ner_tags}``."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str) -> List[dict]:
    """Load a Prodigy-style JSON Lines fold (needed only for the raw ``text``)."""
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def default_text_path(hf_path: str) -> str:
    """``.../f1_train_hf.json`` -> ``.../f1_train.jsonl``; ``.../train_hf.json`` -> ``.../train.jsonl``."""
    return re.sub(r"_hf\.json$", ".jsonl", hf_path)


def _texts_for(hf_path: str, text_path: Optional[str], text_from: str,
               data: Sequence[dict]) -> List[str]:
    """Return one raw text per document, per the ``--text-from`` policy."""
    if text_from == "tokens":
        return [" ".join(item["tokens"]) for item in data]
    path = text_path or default_text_path(hf_path)
    if not os.path.exists(path):
        raise SystemExit(
            f"cannot find the raw-text file {path!r} for {hf_path!r}. The paper's "
            f"CRF features are extracted from the raw advert text, so this file is "
            f"required. Pass it explicitly, or use --text-from tokens to featurise "
            f"' '.join(tokens) instead (the top-features setting, not the paper's)."
        )
    supp = load_jsonl(path)
    if len(supp) != len(data):
        raise SystemExit(
            f"{path} has {len(supp)} records but {hf_path} has {len(data)}; "
            f"they must be the same fold in the same order."
        )
    return [item["text"] for item in supp]


# --------------------------------------------------------------------------- #
# Featurisation
# --------------------------------------------------------------------------- #

def prepare_data(nlp, data: Sequence[dict], texts: Sequence[str], config: str,
                 drop_mismatched: bool, name: str,
                 scheme: str = "plain") -> Tuple[List, List, List[int]]:
    """Featurise *data*, returning ``(X, y, dropped_indices)``.

    When the spaCy tokenisation of a document yields a different number of tokens
    from its label list, the document is reported.  With *drop_mismatched* it is
    dropped (train and validation); without, it is kept so that positional
    alignment with the gold file survives (test).
    """
    X, y, dropped = [], [], []
    for index, (item, text) in enumerate(zip(data, texts)):
        doc = nlp(text)
        features = feats.sent2features(doc, config=config)
        labels = feats.convert_scheme([feats.sent2labels(item["ner_tags"])],
                                      scheme)[0]
        if len(features) != len(labels):
            warnings.warn(
                f"{name} document {index}: {len(features)} spaCy tokens but "
                f"{len(labels)} labels"
                + (" -- dropped" if drop_mismatched else " -- kept (test alignment)")
            )
            if drop_mismatched:
                dropped.append(index)
                continue
        X.append(features)
        y.append(labels)
    return X, y, dropped


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def run(args) -> int:
    import itertools

    import numpy as np
    import scipy.stats
    import sklearn_crfsuite
    import spacy
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
    from sklearn_crfsuite import metrics

    print(f"config: {feats.describe_config(args.config)}")
    print(f"spaCy model: {args.spacy_model}")
    nlp = spacy.load(args.spacy_model)

    train_data, val_data, test_data = (load_hf(p) for p in (args.train, args.val, args.test))
    train_texts = _texts_for(args.train, args.train_text, args.text_from, train_data)
    val_texts = _texts_for(args.val, args.val_text, args.text_from, val_data)
    test_texts = _texts_for(args.test, args.test_text, args.text_from, test_data)

    found = feats.detect_scheme(item["ner_tags"] for item in train_data)
    print(f"label scheme: file is {found}, training on {args.label_scheme}"
          + ("" if found == args.label_scheme else " (converted)"))

    X_train, y_train, _ = prepare_data(nlp, train_data, train_texts, args.config,
                                       drop_mismatched=True, name="train",
                                       scheme=args.label_scheme)
    X_val, y_val, _ = prepare_data(nlp, val_data, val_texts, args.config,
                                   drop_mismatched=True, name="val",
                                   scheme=args.label_scheme)
    # Never drop test documents: --out must stay aligned with the gold fold.
    X_test, y_test, _ = prepare_data(nlp, test_data, test_texts, args.config,
                                     drop_mismatched=False, name="test",
                                     scheme=args.label_scheme)
    print(f"documents: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        max_iterations=100,
        all_possible_transitions=True,
    )
    params_space = {
        "c1": scipy.stats.expon(scale=0.5),
        "c2": scipy.stats.expon(scale=0.05),
    }

    # Train and validation are concatenated and split back by PredefinedSplit:
    # -1 = train, 0 = the single validation fold.
    X_train_val = X_train + X_val
    y_train_val = y_train + y_val
    ps = PredefinedSplit([-1] * len(X_train) + [0] * len(X_val))

    labels = np.unique(list(itertools.chain(*y_train)))
    labels = labels[labels != "O"]
    f1_scorer = make_scorer(metrics.flat_f1_score, average="weighted", labels=labels)

    search = RandomizedSearchCV(
        crf, params_space,
        cv=ps,
        verbose=1,
        n_jobs=1,
        n_iter=args.n_iter,
        scoring=f1_scorer,
        random_state=args.seed,
    )
    search.fit(X_train_val, y_train_val)
    print("Best hyperparameters:", search.best_params_)
    print("Best CV score:", search.best_score_)
    # refit=True (the default, and the original behaviour): best_estimator_ is
    # retrained on train + val.

    y_pred = search.best_estimator_.predict(X_test)

    eval_labels = list(search.best_estimator_.classes_)
    if "O" in eval_labels:
        eval_labels.remove("O")  # the O class is not evaluated
    print(metrics.flat_classification_report(y_test, y_pred, labels=eval_labels,
                                             digits=3))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(list(y_pred), handle, indent=4)
    print(f"wrote {len(y_pred)} token-label lists to {args.out}")

    if args.save_params:
        record = {
            "config": args.config,
            "paper_row": feats.CONFIG_LABELS[args.config],
            "c1": search.best_params_["c1"],
            "c2": search.best_params_["c2"],
            "best_val_weighted_f1": search.best_score_,
            "n_iter": args.n_iter,
            "seed": args.seed,
            "spacy_model": args.spacy_model,
            "spacy_model_version": nlp.meta.get("version"),
            "spacy_version": spacy.__version__,
            "sklearn_crfsuite_version": getattr(sklearn_crfsuite, "__version__", "unknown"),
            "text_from": args.text_from,
            "label_scheme": args.label_scheme,
            "train": args.train, "val": args.val, "test": args.test,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_params)) or ".",
                    exist_ok=True)
        with open(args.save_params, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=4)
        print(f"wrote best hyper-parameters to {args.save_params}")

    if args.save_model:
        import pickle
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)) or ".",
                    exist_ok=True)
        with open(args.save_model, "wb") as handle:
            pickle.dump(search.best_estimator_, handle)
        print(f"pickled the refitted estimator to {args.save_model}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_predict.py",
        description="Train the Auto-AdvER CRF on one fold and write test "
                    "predictions as token-label lists.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Feature configurations (the paper's four CRF rows)\n"
            + "".join(f"  {name:7s} {feats.describe_config(name)}\n"
                      for name in feats.CONFIGS)
            + "\nExample\n"
            "  python src/crf/train_predict.py --config all --seed 42 \\\n"
            "      --train data/folds/f1/train_hf.json \\\n"
            "      --val   data/folds/f1/val_hf.json \\\n"
            "      --test  data/folds/f1/test_hf.json \\\n"
            "      --out   work/crf/f1_pred.json \\\n"
            "      --save-params work/crf/f1_params.json\n"
        ),
    )
    parser.add_argument("--config", default="all", choices=feats.CONFIGS,
                        help="feature configuration (default: %(default)s = the "
                             "paper's 'CRF' row)")
    parser.add_argument("--train", required=True, help="train fold, *_hf.json")
    parser.add_argument("--val", required=True, help="validation fold, *_hf.json")
    parser.add_argument("--test", required=True, help="test fold, *_hf.json")
    parser.add_argument("--train-text", default=None,
                        help="Prodigy JSON Lines file with the raw advert text for "
                             "--train (default: --train with _hf.json -> .jsonl)")
    parser.add_argument("--val-text", default=None, help="likewise for --val")
    parser.add_argument("--test-text", default=None, help="likewise for --test")
    parser.add_argument("--text-from", default="text", choices=("text", "tokens"),
                        help="'text' featurises the raw advert text, as the paper's "
                             "CRF did (default); 'tokens' featurises "
                             "' '.join(tokens), as the top-features script did")
    parser.add_argument("--label-scheme", default="plain", choices=("plain", "iob2"),
                        help="'plain' trains on flat labels (CONDITION), which is "
                             "what the published CRF runs used and so the default; "
                             "'iob2' trains on B-/I- prefixed labels, which the "
                             "released folds carry. Whichever you choose, the labels "
                             "in --out use it, and tags_to_spans.py detects it.")
    parser.add_argument("--spacy-model", default=DEFAULT_SPACY_MODEL,
                        help="spaCy pipeline for tokenisation and PoS tags "
                             "(default: %(default)s, as in process_cv.py). The "
                             "top-features appendix used en_core_web_sm instead.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random_state for RandomizedSearchCV, so the sampled "
                             "c1/c2 are reproducible. The original had none, which "
                             "is why the published c1/c2 are unknown. (default: "
                             "%(default)s)")
    parser.add_argument("--n-iter", type=int, default=100,
                        help="RandomizedSearchCV iterations (default: %(default)s, "
                             "as in the paper)")
    parser.add_argument("--out", required=True,
                        help="output JSON: a list of per-document label lists, for "
                             "src/data_tools/tags_to_spans.py")
    parser.add_argument("--save-params", default=None,
                        help="write the winning c1/c2 and library versions to this "
                             "JSON file")
    parser.add_argument("--save-model", default=None,
                        help="pickle the refitted estimator here, for "
                             "src/crf/top_features.py --model")
    return parser


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
