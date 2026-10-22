"""The mention families: the regexes, the PII filter and the disagreement test.

The regexes are deliberately blunt, so what matters is exactly where they stop:
``\\bmot\\b`` must not fire on "motorway" and ``\\bservic`` must fire on every
inflection.  The disagreement test is checked on a hand-built pair of records
first and then on the released annotator files, where the expert pair's figure
is the one the paper quotes.
"""

from __future__ import annotations

import unittest

import _paths

import score
from data_tools.records import document_from_record


def record(doc_id: str, text: str, tokens: list, spans: list) -> dict:
    """A public-format record: spans are ``(start, end, label)`` triples."""
    return {
        "id": doc_id,
        "text": text,
        "tokens": tokens,
        "entities": [
            {"start": start, "end": end, "label": label} for start, end, label in spans
        ],
    }


class FamilyRegexTests(unittest.TestCase):
    def test_families_match_their_inflections_and_stop_at_word_boundaries(self) -> None:
        cases = {
            "MOT": ("MOT", "mot", "MoT", "12 months MOT", "new MOT."),
            "service": ("service", "Serviced", "SERVICING", "full service history"),
            "tyre": ("tyre", "Tyres", "4 new tyres"),
            "owner": ("owner", "OWNERS", "2 previous owners"),
            "mileage": ("mile", "miles", "MILEAGE", "100,000 mile warranty"),
        }
        for family, texts in cases.items():
            for text in texts:
                with self.subTest(family=family, text=text):
                    self.assertIn(family, score.span_families(text))
        # "motorway" is not an MOT, and a span in no family goes to its own row.
        self.assertEqual(score.span_families("motorway miles"), ("mileage",))
        self.assertEqual(score.span_families("smote the kerb"), ())
        self.assertEqual(score.span_families("2 owner"), ("owner",))
        self.assertEqual(score.span_families("ownership history"), ())
        self.assertEqual(score.family_keys("air conditioning"), (score.OTHER_KEY,))
        # A span may belong to several families at once, in FAMILIES order.
        self.assertEqual(
            score.span_families("last serviced at 8,291 miles"),
            ("service", "mileage"),
        )

    def test_pii_filter_catches_contact_details_and_not_dates(self) -> None:
        for text, pattern in (
            ("call 07123 456789", "phone"),
            ("ring 0161 247 1234", "phone"),
            ("tel +44 7911 123456", "phone"),
            ("sales@example.co.uk", "email"),
            ("see www.example.co.uk", "url"),
            ("collect from M15 6BH", "postcode"),
            ("reg AB12 CDE", "plate"),
        ):
            with self.subTest(text=text):
                self.assertEqual(score.pii_pattern(text), pattern)
        for text in (
            "latest service on 09-03-2024",
            "MOT until 09/2025",
            "12 months MOT",
            "81446 miles",
            "Recent Full Service @ 99,508 miles",
        ):
            with self.subTest(text=text):
                self.assertIsNone(score.pii_pattern(text))


class DisagreementTests(unittest.TestCase):
    def documents(self, left: list, right: list) -> tuple:
        text = "12 month MOT and a full service history"
        tokens = text.split(" ")
        return (
            document_from_record(record("0001", text, tokens, left)),
            document_from_record(record("0001", text, tokens, right)),
        )

    def test_identical_boundaries_and_different_labels_are_reported(self) -> None:
        left, right = self.documents(
            [(0, 12, "CONDITION")], [(0, 12, "SALES_OPTIONS")]
        )
        identical, rows = score.identical_span_disagreements(left, right)
        self.assertEqual(identical, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["text"], "12 month MOT")
        self.assertEqual(rows[0]["families"], ["MOT"])
        self.assertEqual(
            (rows[0]["left"], rows[0]["right"]), ("CONDITION", "SALES_OPTIONS")
        )
        # The same label on the same tokens is agreement, and overlapping but
        # differing boundaries are a boundary disagreement, not a label one.
        left, right = self.documents(
            [(0, 12, "CONDITION")], [(0, 12, "CONDITION")]
        )
        identical, rows = score.identical_span_disagreements(left, right)
        self.assertEqual((identical, rows), (1, []))
        left, right = self.documents(
            [(0, 12, "CONDITION")], [(3, 12, "SALES_OPTIONS")]
        )
        identical, rows = score.identical_span_disagreements(left, right)
        self.assertEqual((identical, rows), (0, []))

    def test_the_expert_pair_disagrees_on_seven_of_their_shared_spans(self) -> None:
        annotators = {"A1": _paths.public_iaa(1), "A2": _paths.public_iaa(2)}
        _paths.require(*annotators.values())
        (pair,) = score.disagreement_pairs(annotators)
        self.assertEqual(pair["support"], 113)
        self.assertEqual(pair["identical"], 586)
        self.assertEqual(pair["different"], 7)
        families = [family for row in pair["rows"] for family in row["families"]]
        self.assertEqual(families.count("MOT"), 3)
        self.assertEqual(families.count("service"), 3)


class OutcomeTests(unittest.TestCase):
    def test_per_span_outcomes_reproduce_the_counts(self) -> None:
        """The observer itemises the four reference-side counts and nothing else."""
        text = "12 month MOT and a full service history"
        tokens = text.split(" ")
        gold = document_from_record(
            record(
                "0001",
                text,
                tokens,
                [(0, 12, "CONDITION"), (21, 38, "SALES_OPTIONS")],
            )
        )
        pred = document_from_record(
            record(
                "0001",
                text,
                tokens,
                [(0, 12, "SALES_OPTIONS"), (26, 38, "SALES_OPTIONS")],
            )
        )
        counts = score.score_spans(gold.token_spans(), pred.token_spans())
        outcomes = score.family_outcomes(gold, pred)
        self.assertEqual(len(outcomes), len(gold.spans))
        tally = {key: 0 for key in score.FAMILY_COUNT_KEYS}
        for _keys, outcome in outcomes:
            tally[outcome] += 1
        self.assertEqual(tally, {key: counts[key] for key in score.FAMILY_COUNT_KEYS})
        self.assertEqual(
            outcomes, [(("MOT",), "inc"), (("service",), "par")]
        )


if __name__ == "__main__":
    unittest.main()
