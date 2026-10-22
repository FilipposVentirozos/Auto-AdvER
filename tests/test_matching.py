"""Fixtures pinning the paper's matching rules, quirks included.

These assertions are deliberately written against the behaviour of the original
scorer rather than against what an ideal one-to-one matcher would do.  If a
future change "fixes" the matching, these tests fail, which is the point: the
published numbers depend on exactly this behaviour.
"""

from __future__ import annotations

import unittest

import _paths  # noqa: F401  (puts src/ on the path)

from score import Span, boundary_direction, score_spans


def span(start: int, end: int, label: str = "CONDITION") -> dict:
    return {"token_start": start, "token_end": end, "label": label}


class GetScoreTests(unittest.TestCase):
    def score(self, reference: tuple, compared: tuple) -> str:
        return Span(span(*reference)).get_score(Span(span(*compared)))

    def test_identical_boundaries_and_label_are_correct(self) -> None:
        self.assertEqual(self.score((5, 6, "HISTORIC"), (5, 6, "HISTORIC")), "cor")

    def test_identical_boundaries_different_label_is_incorrect(self) -> None:
        self.assertEqual(self.score((5, 6, "HISTORIC"), (5, 6, "CONDITION")), "inc")

    def test_overlap_same_label_is_partial(self) -> None:
        self.assertEqual(self.score((5, 8), (7, 12)), "par")

    def test_overlap_different_label_is_incorrect(self) -> None:
        self.assertEqual(self.score((5, 8, "HISTORIC"), (7, 12, "CONDITION")), "inc")

    def test_no_overlap_is_no_match(self) -> None:
        self.assertIsNone(self.score((5, 6), (9, 10)))

    def test_shared_boundary_containment_is_partial(self) -> None:
        self.assertEqual(self.score((5, 6), (5, 10)), "par")
        self.assertEqual(self.score((5, 10), (5, 6)), "par")

    def test_strict_containment_is_not_detected(self) -> None:
        """The documented quirk: neither endpoint of the wider span is inside."""
        self.assertIsNone(self.score((5, 6), (3, 10)))
        # The reverse direction does see it, which is what makes it asymmetric.
        self.assertEqual(self.score((3, 10), (5, 6)), "par")


class ScoreSpansTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        counts = score_spans([span(5, 6)], [span(5, 6)])
        self.assertEqual(counts, dict(cor=1, inc=0, par=0, mis=0, spu=0))

    def test_overlap_counts_as_partial_on_both_sides(self) -> None:
        counts = score_spans([span(5, 8)], [span(7, 12)])
        self.assertEqual(counts, dict(cor=0, inc=0, par=1, mis=0, spu=0))

    def test_disjoint_spans_are_a_miss_and_a_spurious(self) -> None:
        counts = score_spans([span(5, 6)], [span(20, 21)])
        self.assertEqual(counts, dict(cor=0, inc=0, par=0, mis=1, spu=1))

    def test_prediction_swallowing_gold_is_a_miss_but_not_spurious(self) -> None:
        """The containment asymmetry, as it reaches the counts."""
        counts = score_spans([span(5, 6)], [span(3, 10)])
        self.assertEqual(counts, dict(cor=0, inc=0, par=0, mis=1, spu=0))

    def test_gold_swallowing_prediction_is_partial_and_also_spurious(self) -> None:
        """The mirror of the case above: matched from the gold side, spurious
        from the predicted side, because the one-sided overlap test fails when
        the gold span strictly contains the prediction."""
        counts = score_spans([span(3, 10)], [span(5, 6)])
        self.assertEqual(counts, dict(cor=0, inc=0, par=1, mis=0, spu=1))

    def test_one_prediction_can_satisfy_two_gold_spans(self) -> None:
        """Greedy first match, not one-to-one: PAR is counted twice."""
        counts = score_spans([span(0, 1), span(5, 6)], [span(0, 6)])
        self.assertEqual(counts, dict(cor=0, inc=0, par=2, mis=0, spu=0))

    def test_label_filter_applies_to_the_iterated_side_only(self) -> None:
        """The paper's per-label definition: cross-label pairs still count."""
        gold = [span(5, 6, "CONDITION")]
        pred = [span(5, 6, "HISTORIC")]
        # In the CONDITION pass the gold span is counted, and matches the
        # differently labelled prediction as INC; the prediction itself is
        # skipped by the filter on the spurious loop.
        condition = score_spans(gold, pred, label="CONDITION")
        self.assertEqual(condition, dict(cor=0, inc=1, par=0, mis=0, spu=0))
        # In the HISTORIC pass the gold span is skipped, and the prediction is
        # not spurious because it still finds the CONDITION span.
        historic = score_spans(gold, pred, label="HISTORIC")
        self.assertEqual(historic, dict(cor=0, inc=0, par=0, mis=0, spu=0))

    def test_first_match_wins_after_sorting_by_end(self) -> None:
        """Both sides are sorted by token_end before the greedy walk, so the
        gold span is matched against (3, 4) and never against (8, 20)."""
        gold = [span(4, 9)]
        pred = [span(8, 20), span(3, 4)]
        counts = score_spans(gold, pred)
        self.assertEqual(counts, dict(cor=0, inc=0, par=1, mis=0, spu=0))
        # An exact match later in the sorted order is still reached, because
        # only a match ends the walk.
        counts = score_spans([span(4, 9)], [span(20, 30), span(4, 9)])
        self.assertEqual(counts, dict(cor=1, inc=0, par=0, mis=0, spu=1))


class BoundaryDirectionTests(unittest.TestCase):
    """The three directions a partial match's boundaries can be wrong in."""

    def direction(self, gold: tuple, pred: tuple) -> str:
        return boundary_direction(Span(span(*gold)), Span(span(*pred)))

    def test_prediction_reaching_past_the_gold_span_is_too_wide(self) -> None:
        # Shares the start, runs on past the end, and the mirror case.
        self.assertEqual(self.direction((5, 8), (5, 10)), "wide")
        self.assertEqual(self.direction((5, 8), (3, 8)), "wide")

    def test_prediction_inside_the_gold_span_is_too_narrow(self) -> None:
        self.assertEqual(self.direction((5, 8), (5, 6)), "narrow")
        self.assertEqual(self.direction((5, 8), (7, 8)), "narrow")

    def test_prediction_sliding_off_one_end_is_staggered(self) -> None:
        self.assertEqual(self.direction((5, 8), (7, 12)), "staggered")
        self.assertEqual(self.direction((5, 8), (3, 6)), "staggered")

    def test_every_direction_is_reachable_through_the_matching_loop(self) -> None:
        """The collector sees exactly the pairs the loop counts as PAR."""
        gold = [span(5, 8)]
        for pred, expected in (
            ((5, 10), "wide"),
            ((5, 6), "narrow"),
            ((7, 12), "staggered"),
        ):
            with self.subTest(pred=pred):
                pairs = []
                counts = score_spans(gold, [span(*pred)], partial_pairs=pairs)
                self.assertEqual(counts["par"], 1)
                self.assertEqual(len(pairs), 1)
                self.assertEqual(boundary_direction(*pairs[0]), expected)

    def test_collector_matches_the_partial_count_and_changes_nothing(self) -> None:
        """PAR and the collector cannot drift: the loop fills both."""
        gold = [span(0, 1), span(5, 8), span(20, 21), span(30, 31, "HISTORIC")]
        pred = [span(0, 6), span(5, 6), span(20, 21), span(30, 33, "CONDITION")]
        without = score_spans(gold, pred)
        pairs = []
        with_collector = score_spans(gold, pred, partial_pairs=pairs)
        self.assertEqual(without, with_collector)
        self.assertEqual(len(pairs), with_collector["par"])
        # Every collected pair is a genuine partial match of the same walk.
        for reference, compared in pairs:
            self.assertEqual(reference.get_score(compared), "par")


if __name__ == "__main__":
    unittest.main()
