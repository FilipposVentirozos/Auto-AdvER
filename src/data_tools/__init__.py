"""Auto-AdvER data tools: fold creation, format conversion and statistics.

Every module in this package is a command-line tool; run any of them with
``--help``.  Only :mod:`data_tools.make_folds` needs a third-party dependency
(scikit-learn), because the published folds were cut with scikit-learn's
``KFold`` and ``train_test_split`` and the exact partition has to be
reproducible.  Everything else is Python standard library only.
"""

__all__ = ["records"]
