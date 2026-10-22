"""Regression test: the inter-annotator agreement tables.

The appendix table is asserted to two decimal places, which is how exactly it
reproduced from the released annotator files, and the pairwise figures likewise.
"""

from __future__ import annotations

import unittest

import _paths

import score

# annotator pair -> partial-match F1, both directions pooled
EXPECTED_PAIRS = {
    ("A1", "A2"): 88.27,
    ("A1", "A3"): 72.72,
    ("A2", "A3"): 72.63,
}

# label -> (exact P/R/F1, partial P/R/F1), pooled over all six ordered pairs
EXPECTED_PER_LABEL = {
    "CONDITION": (75.60, 81.64),
    "SALES_OPTIONS": (64.88, 76.83),
    "HISTORIC": (67.28, 77.83),
}

EXPECTED_COUNTS = {
    "CONDITION": dict(cor=564, inc=53, par=90, mis=39, spu=39),
    "SALES_OPTIONS": dict(cor=1578, inc=57, par=581, mis=216, spu=216),
    "HISTORIC": dict(cor=654, inc=47, par=205, mis=66, spu=66),
}


class AgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.annotators = {
            "A1": _paths.public_iaa(1),
            "A2": _paths.public_iaa(2),
            "A3": _paths.public_iaa(3),
        }
        _paths.require(*cls.annotators.values())

    def test_pairwise_agreement(self) -> None:
        rows = score.iaa_pairwise(self.annotators)
        found = {tuple(row["pair"]): row for row in rows}
        self.assertEqual(set(found), set(EXPECTED_PAIRS))
        for pair, f1 in EXPECTED_PAIRS.items():
            with self.subTest(pair=pair):
                row = found[pair]
                self.assertEqual(row["support"], 113)
                self.assertAlmostEqual(row["partial"]["f1"], f1, places=2)
                # Pooling both directions makes precision, recall and F1 equal.
                self.assertAlmostEqual(
                    row["partial"]["precision"], row["partial"]["recall"], places=6
                )

    def test_mean_pairwise_agreement(self) -> None:
        rows = score.iaa_pairwise(self.annotators)
        mean, _ = score.mean_std([row["partial"]["f1"] for row in rows])
        self.assertAlmostEqual(mean, 77.87, places=2)

    def test_per_label_appendix_table(self) -> None:
        rows = score.iaa_per_label(self.annotators, list(EXPECTED_PER_LABEL))
        found = {row["label"]: row for row in rows}
        for label, (exact_f1, partial_f1) in EXPECTED_PER_LABEL.items():
            with self.subTest(label=label):
                row = found[label]
                self.assertEqual(row["counts"], EXPECTED_COUNTS[label])
                for scheme, expected in (("exact", exact_f1), ("partial", partial_f1)):
                    for metric in ("precision", "recall", "f1"):
                        self.assertAlmostEqual(
                            row[scheme][metric], expected, places=2
                        )

    def test_unrelated_files_are_an_error(self) -> None:
        """Annotator id sets must be nested; unrelated files must not be scored."""
        _paths.require(_paths.public_fold("f1", "test"), _paths.public_fold("f2", "test"))
        with self.assertRaises(SystemExit):
            score.iaa_pairwise(
                {
                    "left": _paths.public_fold("f1", "test"),
                    "right": _paths.public_fold("f2", "test"),
                }
            )


if __name__ == "__main__":
    unittest.main()
