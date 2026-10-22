"""The token-label to span conversion."""

from __future__ import annotations

import json
import os
import unittest

import _paths

from data_tools import tags_to_spans
from data_tools.records import FormatError, load_document_list, load_documents

CRF_CONFIGS = ("crf", "crf-pos", "crf-wo-pos", "crf-2-3")
ENCODER_CONFIGS = (
    "bert-base",
    "bert-large",
    "deberta-v3-base",
    "deberta-v3-large",
)
TOKEN_LABEL_CONFIGS = CRF_CONFIGS + ENCODER_CONFIGS


class SpanBuildingTests(unittest.TestCase):
    def test_plain_labels_merge_adjacent_entities_of_one_type(self) -> None:
        """Only a change of label closes a span."""
        labels = ["O", "HISTORIC", "HISTORIC", "O"]
        self.assertEqual(
            tags_to_spans.spans_from_labels(labels),
            [{"token_start": 1, "token_end": 2, "label": "HISTORIC"}],
        )

    def test_label_change_starts_a_new_span(self) -> None:
        labels = ["HISTORIC", "CONDITION", "O"]
        self.assertEqual(
            tags_to_spans.spans_from_labels(labels),
            [
                {"token_start": 0, "token_end": 0, "label": "HISTORIC"},
                {"token_start": 1, "token_end": 1, "label": "CONDITION"},
            ],
        )

    def test_iob2_b_prefix_splits_neighbouring_entities(self) -> None:
        labels = ["B-HISTORIC", "I-HISTORIC", "B-HISTORIC"]
        self.assertEqual(
            tags_to_spans.spans_from_labels(labels, iob2=True),
            [
                {"token_start": 0, "token_end": 1, "label": "HISTORIC"},
                {"token_start": 2, "token_end": 2, "label": "HISTORIC"},
            ],
        )

    def test_span_open_at_the_end_closes_on_the_last_token(self) -> None:
        labels = ["O", "O", "CONDITION", "CONDITION"]
        self.assertEqual(
            tags_to_spans.spans_from_labels(labels),
            [{"token_start": 2, "token_end": 3, "label": "CONDITION"}],
        )

    def test_iob2_detection(self) -> None:
        self.assertTrue(tags_to_spans.is_iob2([["O", "B-HISTORIC"]]))
        self.assertFalse(tags_to_spans.is_iob2([["O", "HISTORIC"]]))


class ShippedPredictionTests(unittest.TestCase):
    """The shipped span files of the eight token-label configurations come from
    the shipped raw labels."""

    def raw_labels(self, config: str, fold: str) -> tuple:
        directory = os.path.join(
            _paths.PUBLIC_PREDICTIONS, config, "raw_token_labels"
        )
        return (
            os.path.join(directory, "%s_test.json" % fold),
            os.path.join(directory, "%s_test_ids.json" % fold),
        )

    def test_default_mode_reproduces_the_shipped_spans(self) -> None:
        """Every shipped token-label prediction file comes from these labels.

        The default ``auto`` alignment is what the release pipeline runs: for
        the four CRF configurations (one label per token,
        so ``auto`` resolves to a positional zip) and for the four encoders
        (one label per non-whitespace token, so it resolves to whitespace
        skipping).
        """
        for fold in ("f1", "f2", "f3"):
            for config in TOKEN_LABEL_CONFIGS:
                labels, ids = self.raw_labels(config, fold)
                shipped = os.path.join(
                    _paths.PUBLIC_PREDICTIONS, config, "%s_test.jsonl" % fold
                )
                gold = _paths.public_fold(fold, "test")
                _paths.require(labels, ids, shipped, gold)
                with self.subTest(config=config, fold=fold):
                    docs, spans, _ = tags_to_spans.convert(gold, labels, ids_path=ids)
                    expected = load_documents(shipped)
                    self.assertEqual(len(docs), 87)
                    for doc, doc_spans in zip(docs, spans):
                        produced = [
                            (span["token_start"], span["token_end"], span["label"])
                            for span in doc_spans
                        ]
                        reference = [
                            (span["token_start"], span["token_end"], span["label"])
                            for span in expected[doc.doc_id].spans
                        ]
                        self.assertEqual(produced, reference, doc.doc_id)

    def test_crf_labels_align_positionally_with_no_warning(self) -> None:
        """The CRF has one label per token, whitespace tokens included."""
        fold = "f1"
        for config in CRF_CONFIGS:
            labels, ids = self.raw_labels(config, fold)
            gold = _paths.public_fold(fold, "test")
            _paths.require(labels, ids, gold)
            with self.subTest(config=config):
                docs, _, warnings = tags_to_spans.convert(gold, labels, ids_path=ids)
                self.assertEqual(warnings, [])
                with open(labels, encoding="utf-8") as handle:
                    label_lists = json.load(handle)
                for doc, doc_labels in zip(docs, label_lists):
                    self.assertEqual(doc.n_tokens, len(doc_labels), doc.doc_id)

    def test_encoder_labels_are_short_by_the_whitespace_tokens(self) -> None:
        """One label per non-whitespace token — except two truncated documents."""
        expected_short = {("bert-base", "f3"): 2, ("bert-large", "f3"): 2}
        for fold in ("f1", "f2", "f3"):
            for config in ENCODER_CONFIGS:
                labels, ids = self.raw_labels(config, fold)
                gold = _paths.public_fold(fold, "test")
                _paths.require(labels, ids, gold)
                with self.subTest(config=config, fold=fold):
                    docs = {doc.doc_id: doc for doc in load_document_list(gold)}
                    with open(labels, encoding="utf-8") as handle:
                        label_lists = json.load(handle)
                    with open(ids, encoding="utf-8") as handle:
                        doc_ids = json.load(handle)
                    self.assertEqual(len(label_lists), 87)
                    short = 0
                    for doc_id, doc_labels in zip(doc_ids, label_lists):
                        content = tags_to_spans.count_content_tokens(
                            docs[doc_id].tokens
                        )
                        self.assertLessEqual(len(doc_labels), content, doc_id)
                        if len(doc_labels) < content:
                            short += 1
                    self.assertEqual(short, expected_short.get((config, fold), 0))

    def test_whitespace_tokens_are_present_in_the_folds(self) -> None:
        """The alignment only matters because the folds hold such tokens."""
        for fold, expected in (("f1", 302), ("f2", 297), ("f3", 315)):
            gold = _paths.public_fold(fold, "test")
            _paths.require(gold)
            with self.subTest(fold=fold):
                docs = load_document_list(gold)
                blank = sum(
                    1
                    for doc in docs
                    for token in doc.tokens
                    if tags_to_spans.is_whitespace_token(token)
                )
                self.assertEqual(blank, expected)

    def test_public_output_carries_character_offsets(self) -> None:
        fold = "f1"
        labels, ids = self.raw_labels("crf", fold)
        gold = _paths.public_fold(fold, "test")
        _paths.require(labels, ids, gold)
        docs, spans, _ = tags_to_spans.convert(gold, labels, ids_path=ids)
        records = tags_to_spans.to_records(docs, spans, "public")
        self.assertEqual(len(records), 87)
        for record, doc in zip(records, docs):
            self.assertEqual(record["id"], doc.doc_id)
            for entity in record["entities"]:
                self.assertIn("start", entity)
                # The character offsets must name the same tokens as the indices.
                self.assertEqual(
                    doc.offsets[entity["token_start"]][0], entity["start"]
                )
                self.assertEqual(doc.offsets[entity["token_end"]][1], entity["end"])


class AlignmentTests(unittest.TestCase):
    """``align_labels``: how a label list is laid over a token list."""

    TOKENS = ["Full", "service", "\n", "history", "\n\n", "MOT"]

    def test_whitespace_token_is_recognised_by_stripping(self) -> None:
        for blank in ("\n", " ", "\n\n", "\t", "\r\n", "", "  \n "):
            self.assertTrue(tags_to_spans.is_whitespace_token(blank), repr(blank))
        for filled in ("MOT", " MOT", "-"):
            self.assertFalse(tags_to_spans.is_whitespace_token(filled), repr(filled))
        self.assertEqual(tags_to_spans.count_content_tokens(self.TOKENS), 4)

    def test_skip_whitespace_leaves_blank_tokens_unlabelled(self) -> None:
        labels = ["HISTORIC", "HISTORIC", "HISTORIC", "O"]
        aligned, warning = tags_to_spans.align_labels(labels, self.TOKENS)
        self.assertIsNone(warning)
        self.assertEqual(
            aligned, ["HISTORIC", "HISTORIC", "O", "HISTORIC", "O", "O"]
        )

    def test_skip_whitespace_splits_a_span_across_a_newline(self) -> None:
        """A whitespace token inside an entity breaks it in two; deliberate."""
        labels = ["HISTORIC", "HISTORIC", "HISTORIC", "O"]
        aligned, _ = tags_to_spans.align_labels(labels, self.TOKENS)
        self.assertEqual(
            tags_to_spans.spans_from_labels(aligned),
            [
                {"token_start": 0, "token_end": 1, "label": "HISTORIC"},
                {"token_start": 3, "token_end": 3, "label": "HISTORIC"},
            ],
        )

    def test_positional_requires_one_label_per_token(self) -> None:
        labels = ["HISTORIC", "HISTORIC", "HISTORIC", "O"]
        with self.assertRaises(FormatError) as raised:
            tags_to_spans.align_labels(
                labels, self.TOKENS, alignment="positional", doc_id="0002"
            )
        self.assertIn("0002", str(raised.exception))
        self.assertIn("one label per token", str(raised.exception))

    def test_equal_lengths_are_zipped_positionally_with_no_warning(self) -> None:
        labels = ["O", "O", "O", "CONDITION", "O", "O"]
        aligned, warning = tags_to_spans.align_labels(labels, self.TOKENS)
        self.assertIsNone(warning)
        self.assertEqual(aligned, labels)

    def test_a_truncated_label_list_leaves_the_tail_unlabelled(self) -> None:
        labels = ["HISTORIC", "HISTORIC"]
        aligned, warning = tags_to_spans.align_labels(
            labels, self.TOKENS, doc_id="0326"
        )
        self.assertEqual(aligned, ["HISTORIC", "HISTORIC", "O", "O", "O", "O"])
        self.assertIn("0326", warning)
        self.assertIn("2 short", warning)
        self.assertIn("truncated", warning)

    def test_a_length_between_the_two_is_refused(self) -> None:
        """Five labels for six tokens, four of them content: unalignable."""
        labels = ["O"] * 5
        with self.assertRaises(FormatError) as raised:
            tags_to_spans.align_labels(labels, self.TOKENS, doc_id="0001")
        self.assertIn("0001", str(raised.exception))
        self.assertIn("cannot be aligned", str(raised.exception))

    def test_more_labels_than_tokens_is_refused(self) -> None:
        with self.assertRaises(FormatError):
            tags_to_spans.align_labels(["O"] * 7, self.TOKENS)

    def test_skip_whitespace_refuses_more_labels_than_content_tokens(self) -> None:
        with self.assertRaises(FormatError) as raised:
            tags_to_spans.align_labels(
                ["O"] * 5, self.TOKENS, alignment="skip-whitespace"
            )
        self.assertIn("non-whitespace", str(raised.exception))

    def test_unknown_alignment_is_refused(self) -> None:
        with self.assertRaises(FormatError):
            tags_to_spans.align_labels(["O"] * 4, self.TOKENS, alignment="nearest")

    def test_tokens_without_whitespace_behave_the_same_either_way(self) -> None:
        tokens = ["Full", "service", "history"]
        labels = ["HISTORIC", "HISTORIC", "O"]
        auto, _ = tags_to_spans.align_labels(labels, tokens)
        skipped, _ = tags_to_spans.align_labels(
            labels, tokens, alignment="skip-whitespace"
        )
        self.assertEqual(auto, skipped)


if __name__ == "__main__":
    unittest.main()
