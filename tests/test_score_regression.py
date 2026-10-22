"""Regression test: the paper's results tables, from the released files.

Every figure asserted here was reproduced independently from the stored
predictions before the release was assembled, and each is checked to within
0.15 percentage points.  Where the paper's typeset value differs slightly from
the reproduction (rounding), the reproduced value is the one asserted and the
published one is noted beside it; ``results/tables.md`` sets the two side by
side.

``EXPECTED`` is what the shipped prediction files score, and what the paper
reports.  The eight token-label configurations are built from their raw
per-token labels with ``src/data_tools/tags_to_spans.py``.  The gold spans are
the corpus annotations carried by the fold files.
"""

from __future__ import annotations

import unittest

import _paths

import score

TOLERANCE = 0.15

# model directory -> (precision, recall, F1), partial match, mean over f1-f3,
# from the files this release ships
EXPECTED = {
    "bert-base": (52.6, 63.5, 57.5),
    "bert-large": (53.6, 65.0, 58.7),
    "deberta-v3-base": (63.6, 72.4, 67.7),
    "deberta-v3-large": (63.9, 74.0, 68.5),
    "gpt-4o": (73.8, 54.4, 62.6),
    "gemini-1.5-flash": (72.4, 49.9, 59.0),
    "llama-3.1-70b": (75.9, 44.5, 56.1),
    "qwen2.5-72b": (76.8, 34.8, 47.9),
    "crf": (72.5, 60.8, 66.1),
    "crf-pos": (25.4, 1.1, 2.1),
    "crf-wo-pos": (71.9, 58.5, 64.4),
    "crf-2-3": (71.2, 60.1, 65.2),
}

# model directory -> {label: partial F1 mean}, the paper's per-label definition
EXPECTED_PER_LABEL = {
    "crf": {"CONDITION": 46.5, "HISTORIC": 59.5, "SALES_OPTIONS": 73.2},
    "gpt-4o": {"CONDITION": 52.2, "HISTORIC": 53.0, "SALES_OPTIONS": 68.4},
    "gemini-1.5-flash": {"CONDITION": 38.5, "HISTORIC": 51.6, "SALES_OPTIONS": 66.7},
    "deberta-v3-base": {"CONDITION": 57.8, "HISTORIC": 58.1, "SALES_OPTIONS": 74.0},
    "qwen2.5-72b": {"CONDITION": 35.8, "HISTORIC": 42.6, "SALES_OPTIONS": 52.7},
}

TOKEN_LABEL_CONFIGS = (
    "crf",
    "crf-pos",
    "crf-wo-pos",
    "crf-2-3",
    "bert-base",
    "bert-large",
    "deberta-v3-base",
    "deberta-v3-large",
)

FOLDS = ("f1", "f2", "f3")


class PaperTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gold = _paths.public_fold_template("test")
        _paths.require(*(cls.gold.format(fold=fold) for fold in FOLDS))

    def test_overall_table(self) -> None:
        for model, (precision, recall, f1) in EXPECTED.items():
            template = _paths.public_prediction_template(model)
            _paths.require(*(template.format(fold=fold) for fold in FOLDS))
            with self.subTest(model=model):
                result = score.score_model(self.gold, template, FOLDS)
                summary = result["summary"]["partial"]
                self.assertEqual(
                    [fold["support"] for fold in result["folds"]], [87, 87, 87]
                )
                self.assertAlmostEqual(
                    summary["precision"]["mean"], precision, delta=TOLERANCE
                )
                self.assertAlmostEqual(summary["recall"]["mean"], recall, delta=TOLERANCE)
                self.assertAlmostEqual(summary["f1"]["mean"], f1, delta=TOLERANCE)

    def test_per_label_table(self) -> None:
        for model, labels in EXPECTED_PER_LABEL.items():
            template = _paths.public_prediction_template(model)
            _paths.require(*(template.format(fold=fold) for fold in FOLDS))
            for label, f1 in labels.items():
                with self.subTest(model=model, label=label):
                    result = score.score_model(self.gold, template, FOLDS, label=label)
                    self.assertAlmostEqual(
                        result["summary"]["partial"]["f1"]["mean"], f1, delta=TOLERANCE
                    )

    def test_mismatched_document_ids_are_an_error(self) -> None:
        """Alignment is on document id and a gap must stop the run."""
        gold = _paths.public_fold("f1", "test")
        other = _paths.public_fold("f2", "test")
        _paths.require(gold, other)
        with self.assertRaises(SystemExit):
            score.score_fold(gold, other)


# model directory -> (PAR, too wide %, too narrow %), pooled over f1-f3, as the
# paper's error analysis quotes them: "when the CRF or GPT-4o produce a partial
# match, the predicted span is more often too wide than too narrow (58% against
# 33% of the CRF's partial matches, 58% against 38% of GPT-4o's), whereas
# DeBERTaV3's partial matches are more often too narrow (53%)."
EXPECTED_BOUNDARIES = {
    "crf": (261, 58, 33),
    "gpt-4o": (218, 58, 38),
    "deberta-v3-base": (340, 42, 53),
}


class BoundaryDirectionTests(unittest.TestCase):
    """The error analysis' too-wide / too-narrow shares, from the same files.

    ``results/tables.md`` prints this under "Boundary direction of partial
    matches"; the shares here are rounded to the integers the paper typesets.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.gold = _paths.public_fold_template("test")
        _paths.require(*(cls.gold.format(fold=fold) for fold in FOLDS))

    def test_boundary_shares(self) -> None:
        for model, (par, wide, narrow) in EXPECTED_BOUNDARIES.items():
            template = _paths.public_prediction_template(model)
            _paths.require(*(template.format(fold=fold) for fold in FOLDS))
            with self.subTest(model=model):
                result = score.boundary_model(self.gold, template, FOLDS)
                self.assertEqual(result["pooled"]["par"], par)
                self.assertEqual(round(result["shares"]["wide"]), wide)
                self.assertEqual(round(result["shares"]["narrow"]), narrow)

    def test_par_agrees_with_the_scorer(self) -> None:
        """The boundary walk and the scored table count the same partials."""
        for model in EXPECTED_BOUNDARIES:
            template = _paths.public_prediction_template(model)
            _paths.require(*(template.format(fold=fold) for fold in FOLDS))
            with self.subTest(model=model):
                scored = score.score_model(self.gold, template, FOLDS)
                self.assertEqual(
                    score.boundary_model(self.gold, template, FOLDS)["pooled"]["par"],
                    sum(fold["counts"]["par"] for fold in scored["folds"]),
                )


if __name__ == "__main__":
    unittest.main()
