"""Where the tests look for data, and how they skip when it is not there.

The public release is self-contained: ``data/`` and ``predictions/`` sit beside
``src/`` and every test uses them.  A test that needs a file which is missing
skips with a message naming it; no test hard-codes a path into a tool.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Optional

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SRC = os.path.join(REPO_ROOT, "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

PUBLIC_DATA = os.path.join(REPO_ROOT, "data")
PUBLIC_PREDICTIONS = os.path.join(REPO_ROOT, "predictions")

def public_fold(fold: str, split: str) -> str:
    return os.path.join(PUBLIC_DATA, "folds", fold, "%s.jsonl" % split)


def public_fold_template(split: str = "test") -> str:
    return os.path.join(PUBLIC_DATA, "folds", "{fold}", "%s.jsonl" % split)


def public_prediction_template(model: str) -> str:
    return os.path.join(PUBLIC_PREDICTIONS, model, "{fold}_test.jsonl")


def public_iaa(number: int) -> str:
    return os.path.join(PUBLIC_DATA, "iaa", "annotator_%d.jsonl" % number)


def skip_unless_exists(*paths: str) -> Optional[str]:
    """Return a skip reason if any path is missing, else ``None``."""
    for path in paths:
        if not path or not os.path.exists(path):
            return "data not present: %s" % (path or "<unset>")
    return None


LFS_POINTER = b"version https://git-lfs"


def is_lfs_pointer(path: str) -> bool:
    """True if the file is a Git LFS stub rather than the file itself."""
    try:
        with open(path, "rb") as handle:
            return handle.read(len(LFS_POINTER)) == LFS_POINTER
    except OSError:
        return False


def require(*paths: str) -> None:
    """Skip the calling test unless every path exists.

    A Git LFS pointer is not a missing file and must not be skipped over: it
    would turn into an unreadable JSON error deep inside a tool.  It stops the
    test with the command that fixes it instead.
    """
    reason = skip_unless_exists(*paths)
    if reason:
        raise unittest.SkipTest(reason)
    stubs = [path for path in paths if is_lfs_pointer(path)]
    if stubs:
        raise RuntimeError(
            "Git LFS pointer, not the file: %s. "
            "Run `git lfs install` and then `git lfs pull` in this clone."
            % ", ".join(os.path.relpath(path, REPO_ROOT) for path in stubs)
        )
