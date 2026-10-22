# -*- coding: utf-8 -*-
"""Feature extraction for the Auto-AdvER CRF baselines.

The paper reports four CRF configurations.  In the original ``process_cv.py``
they were produced by hand-editing one commented-out feature block and re-running
the script, so no single file could regenerate all four.  Here they are four
named configurations:

============  =========================================================
``all``       "CRF" in the paper: every token feature, neighbours at +/-1.
``pos``       "CRF PoS": ``bias`` + ``postag`` + ``postag_detail``, and the
              neighbours' ``postag`` only.
``wo_pos``    "CRF w/o PoS": ``all`` minus ``postag``/``postag_detail`` and
              minus the neighbours' ``postag``.
``2_3``       "CRF 2n/3n": ``all`` plus the same neighbour block at +/-2 and
              +/-3.
============  =========================================================

Feature **key strings** are byte-identical to the originals (``'token.lower()'``,
``'-1:postag'``, ``'BOS'``, ...), because the CRF top-features appendix prints
them.

This module imports nothing outside the standard library.  Tokens are duck-typed
on the spaCy ``Token`` API -- ``lower_``, ``text``, ``is_upper``, ``is_title``,
``is_digit``, ``pos_``, ``tag_``, ``shape_``, ``is_alpha``, ``is_stop`` -- so the
configurations can be unit-tested with a stub token and no spaCy install.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

__all__ = ["CONFIGS", "CONFIG_LABELS", "token2features", "sent2features",
           "sent2labels", "describe_config", "detect_scheme", "to_plain",
           "to_iob2", "convert_scheme"]

#: ``--config`` value -> the paper's name for that row of the results table.
CONFIG_LABELS = {
    "all": "CRF",
    "pos": "CRF PoS",
    "wo_pos": "CRF w/o PoS",
    "2_3": "CRF 2n/3n",
}

CONFIGS = tuple(CONFIG_LABELS)

# Which token-level features each configuration keeps.
_CORE = (
    "token.lower()", "token[-3:]", "token[-2:]", "token.is_upper",
    "token.is_title", "token.is_digit", "postag", "postag_detail", "shape",
    "is_alpha", "is_stop",
)
_POS_ONLY = ("postag", "postag_detail")
_WO_POS = tuple(f for f in _CORE if f not in _POS_ONLY)

# Which neighbour features each configuration keeps, and at which offsets.
_NEIGHBOUR_ALL = ("token.lower()", "postag", "is_digit")

_SPEC = {
    "all":    {"token": _CORE,    "neighbour": _NEIGHBOUR_ALL,           "offsets": (1,)},
    "pos":    {"token": _POS_ONLY, "neighbour": ("postag",),             "offsets": (1,)},
    "wo_pos": {"token": _WO_POS,  "neighbour": ("token.lower()", "is_digit"), "offsets": (1,)},
    "2_3":    {"token": _CORE,    "neighbour": _NEIGHBOUR_ALL,           "offsets": (1, 2, 3)},
}


def _token_feature(token, name: str):
    if name == "token.lower()":
        return token.lower_
    if name == "token[-3:]":
        return token.text[-3:]
    if name == "token[-2:]":
        return token.text[-2:]
    if name == "token.is_upper":
        return token.is_upper
    if name == "token.is_title":
        return token.is_title
    if name == "token.is_digit":
        return token.is_digit
    if name == "postag":
        return token.pos_
    if name == "postag_detail":
        return token.tag_
    if name == "shape":
        return token.shape_
    if name == "is_alpha":
        return token.is_alpha
    if name == "is_stop":
        return token.is_stop
    raise KeyError(name)


def _neighbour_feature(token, name: str):
    if name == "token.lower()":
        return token.lower_
    if name == "postag":
        return token.pos_
    if name == "is_digit":
        return token.is_digit
    raise KeyError(name)


def token2features(sent: Sequence, i: int, config: str = "all") -> Dict[str, object]:
    """Features for token *i* of *sent* under *config*.

    ``bias`` is always present, as in every original variant.  ``BOS``/``EOS``
    are emitted when the +/-1 neighbour is missing -- only for the +/-1 offset,
    matching the original (the +/-2 and +/-3 blocks had no such marker and simply
    contributed nothing at the edges).
    """
    if config not in _SPEC:
        raise ValueError(f"unknown CRF config {config!r}; expected one of {CONFIGS}")
    spec = _SPEC[config]
    token = sent[i]

    features: Dict[str, object] = {"bias": 1.0}
    for name in spec["token"]:
        features[name] = _token_feature(token, name)

    for offset in spec["offsets"]:
        if i - offset >= 0:
            previous = sent[i - offset]
            for name in spec["neighbour"]:
                features[f"-{offset}:{name}"] = _neighbour_feature(previous, name)
        elif offset == 1:
            features["BOS"] = True  # Beginning of sentence

        if i + offset < len(sent):
            following = sent[i + offset]
            for name in spec["neighbour"]:
                features[f"+{offset}:{name}"] = _neighbour_feature(following, name)
        elif offset == 1:
            features["EOS"] = True  # End of sentence

    return features


def sent2features(sent: Sequence, config: str = "all") -> List[Dict[str, object]]:
    """Features for every token of *sent*."""
    return [token2features(sent, i, config=config) for i in range(len(sent))]


def sent2labels(labels):
    """Identity, kept for symmetry with the original scripts."""
    return labels


# --------------------------------------------------------------------------- #
# Label schemes
# --------------------------------------------------------------------------- #
# The published CRF runs trained on **flat** labels ("CONDITION"), because that
# is what the Hugging Face-format folds carried at the time.  The released folds
# carry IOB2 ("B-CONDITION", "I-CONDITION"), which is the better convention and
# the one the span converter prefers.  Both are supported, and the scheme is an
# explicit choice rather than a property of whichever file you happen to pass.

def detect_scheme(sequences) -> str:
    """Return ``"iob2"`` if any label carries a ``B-``/``I-`` prefix, else ``"plain"``."""
    for sequence in sequences:
        for label in sequence:
            if label != "O" and label[:2] in ("B-", "I-"):
                return "iob2"
    return "plain"


def to_plain(sequence):
    """Strip ``B-``/``I-`` prefixes: ``B-CONDITION`` -> ``CONDITION``."""
    return [label if label == "O" or label[:2] not in ("B-", "I-") else label[2:]
            for label in sequence]


def to_iob2(sequence):
    """Add ``B-``/``I-`` prefixes, opening a new entity whenever the type changes.

    Two adjacent entities of the same type cannot be told apart in the plain
    scheme, so this conversion merges them -- the same information loss the
    original flat folds already had.
    """
    out, previous = [], "O"
    for label in sequence:
        if label == "O":
            out.append("O")
        elif label[:2] in ("B-", "I-"):
            out.append(label)
        else:
            out.append(("I-" if previous == label else "B-") + label)
        previous = label
    return out


def convert_scheme(sequences, scheme: str):
    """Convert every sequence to *scheme* (``"plain"`` or ``"iob2"``)."""
    if scheme == "plain":
        return [to_plain(sequence) for sequence in sequences]
    if scheme == "iob2":
        return [to_iob2(sequence) for sequence in sequences]
    raise ValueError(f"unknown label scheme {scheme!r}; expected 'plain' or 'iob2'")


def describe_config(config: str) -> str:
    """One-line human-readable summary, for ``--help`` and log headers."""
    spec = _SPEC[config]
    offsets = ", ".join(f"+/-{o}" for o in spec["offsets"])
    return (f"{CONFIG_LABELS[config]}: token features "
            f"[bias, {', '.join(spec['token'])}]; neighbour features "
            f"[{', '.join(spec['neighbour'])}] at {offsets}")


if __name__ == "__main__":
    for name in CONFIGS:
        print(f"--config {name:7s} {describe_config(name)}")
