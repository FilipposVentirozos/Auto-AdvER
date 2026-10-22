"""The shared record reader: format detection and the offset conversions."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import _paths

from data_tools.records import (
    FormatError,
    char_span_to_token_span,
    detect_format,
    document_from_record,
    load_documents,
    token_offsets_from_text,
)

PUBLIC_RECORD = {
    "id": "0001",
    "text": "Full service history , 2 owners .",
    "tokens": ["Full", "service", "history", ",", "2", "owners", "."],
    "labels": ["B-HISTORIC", "I-HISTORIC", "I-HISTORIC", "O", "B-HISTORIC", "I-HISTORIC", "O"],
    "entities": [
        {"start": 0, "end": 20, "label": "HISTORIC"},
        {"start": 23, "end": 31, "label": "HISTORIC"},
    ],
    "advertiser_type": "PRIVATE",
    "region": "NORTH WEST",
}

PRODIGY_RECORD = {
    "text": "Full service history",
    "tokens": [
        {"text": "Full", "start": 0, "end": 4, "id": 0},
        {"text": "service", "start": 5, "end": 12, "id": 1},
        {"text": "history", "start": 13, "end": 20, "id": 2},
    ],
    "spans": [{"start": 0, "end": 20, "token_start": 0, "token_end": 2, "label": "HISTORIC"}],
    "meta": {"metadata_search_id": "202401010000001", "advertiser_type": "TRADE"},
}


class FormatDetectionTests(unittest.TestCase):
    def test_public_record(self) -> None:
        self.assertEqual(detect_format(PUBLIC_RECORD), "public")

    def test_prodigy_record(self) -> None:
        self.assertEqual(detect_format(PRODIGY_RECORD), "prodigy")

    def test_prediction_only_record_is_public(self) -> None:
        self.assertEqual(detect_format({"id": "0001", "entities": []}), "public")

    def test_unknown_record_raises(self) -> None:
        with self.assertRaises(FormatError):
            detect_format({"foo": "bar"})


class TokenOffsetTests(unittest.TestCase):
    def test_single_spaces(self) -> None:
        text = "Full service history"
        offsets = token_offsets_from_text(text, ["Full", "service", "history"])
        self.assertEqual(offsets, [(0, 4), (5, 12), (13, 20)])
        for (start, end), token in zip(offsets, ["Full", "service", "history"]):
            self.assertEqual(text[start:end], token)

    def test_double_space_becomes_a_whitespace_token_after_the_separator(self) -> None:
        text = "miles,  Manual"
        tokens = ["miles", ",", " ", "Manual"]
        self.assertEqual(
            token_offsets_from_text(text, tokens), [(0, 5), (5, 6), (7, 8), (8, 14)]
        )

    def test_newline_token_takes_no_separator(self) -> None:
        text = "CLUTCH\nMOT"
        self.assertEqual(
            token_offsets_from_text(text, ["CLUTCH", "\n", "MOT"]),
            [(0, 6), (6, 7), (7, 10)],
        )

    def test_absent_token_raises(self) -> None:
        with self.assertRaises(FormatError):
            token_offsets_from_text("one two", ["one", "three"])

    def test_agrees_with_prodigy_offsets_on_a_real_fold(self) -> None:
        path = _paths.public_fold("f1", "test")
        _paths.require(path)
        checked = 0
        for doc in load_documents(path).values():
            self.assertEqual(
                doc.offsets, token_offsets_from_text(doc.text, doc.tokens)
            )
            checked += 1
        self.assertGreater(checked, 0)


class CharToTokenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.offsets = token_offsets_from_text(
            PUBLIC_RECORD["text"], PUBLIC_RECORD["tokens"]
        )

    def test_token_aligned_span(self) -> None:
        self.assertEqual(char_span_to_token_span(self.offsets, 0, 20), (0, 2))

    def test_single_token_span(self) -> None:
        self.assertEqual(char_span_to_token_span(self.offsets, 23, 24), (4, 4))

    def test_span_starting_inside_a_token_prefers_the_contained_tokens(self) -> None:
        # "ull service history" -> the tokens wholly inside are 1 and 2.
        self.assertEqual(char_span_to_token_span(self.offsets, 1, 20), (1, 2))

    def test_span_narrower_than_a_token_falls_back_to_overlap(self) -> None:
        self.assertEqual(char_span_to_token_span(self.offsets, 1, 3), (0, 0))

    def test_span_outside_the_text_raises(self) -> None:
        with self.assertRaises(FormatError):
            char_span_to_token_span(self.offsets, 900, 950)

    def test_public_entities_convert_to_the_expected_token_spans(self) -> None:
        doc = document_from_record(PUBLIC_RECORD)
        self.assertEqual(
            doc.token_spans(),
            [
                {"token_start": 0, "token_end": 2, "label": "HISTORIC"},
                {"token_start": 4, "token_end": 5, "label": "HISTORIC"},
            ],
        )

    def test_prediction_without_tokens_uses_the_gold_document(self) -> None:
        gold = document_from_record(PUBLIC_RECORD)
        pred = document_from_record(
            {"id": "0001", "entities": [{"start": 0, "end": 20, "label": "HISTORIC"}]}
        )
        self.assertEqual(
            pred.token_spans(reference=gold),
            [{"token_start": 0, "token_end": 2, "label": "HISTORIC"}],
        )
        with self.assertRaises(FormatError):
            pred.token_spans()

    def test_prodigy_token_spans_are_used_as_they_are(self) -> None:
        doc = document_from_record(PRODIGY_RECORD)
        self.assertEqual(doc.doc_id, "202401010000001")
        self.assertEqual(
            doc.token_spans(), [{"token_start": 0, "token_end": 2, "label": "HISTORIC"}]
        )
        self.assertEqual(doc.meta_value("advertiser_type"), "TRADE")


class LoadTests(unittest.TestCase):
    def test_duplicate_ids_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dup.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(PUBLIC_RECORD) + "\n")
                handle.write(json.dumps(PUBLIC_RECORD) + "\n")
            with self.assertRaises(FormatError):
                load_documents(path)

    def test_empty_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.jsonl")
            open(path, "w", encoding="utf-8").close()
            with self.assertRaises(FormatError):
                load_documents(path)


if __name__ == "__main__":
    unittest.main()
