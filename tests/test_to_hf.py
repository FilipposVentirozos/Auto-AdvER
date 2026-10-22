"""The Hugging Face conversion: both tag schemes, and a round trip."""

from __future__ import annotations

import json
import os
import unittest

import _paths

from data_tools import to_hf
from data_tools.records import document_from_record, load_document_list
from data_tools.tags_to_spans import spans_from_labels

RECORD = {
    "id": "0001",
    "text": "Full service history , new clutch and MOT",
    "tokens": ["Full", "service", "history", ",", "new", "clutch", "and", "MOT"],
    "entities": [
        {"start": 0, "end": 20, "label": "HISTORIC"},
        {"start": 23, "end": 33, "label": "CONDITION"},
        {"start": 38, "end": 41, "label": "HISTORIC"},
    ],
}

NESTED = {
    "id": "0002",
    "text": "full service history from new",
    "tokens": ["full", "service", "history", "from", "new"],
    "entities": [
        {"start": 0, "end": 29, "label": "HISTORIC"},
        {"start": 5, "end": 12, "label": "SALES_OPTIONS"},
    ],
}


class TagSchemeTests(unittest.TestCase):
    def test_iob2_tags(self) -> None:
        tags, dropped = to_hf.tags_for_document(document_from_record(RECORD), iob2=True)
        self.assertEqual(dropped, 0)
        self.assertEqual(
            tags,
            [
                "B-HISTORIC",
                "I-HISTORIC",
                "I-HISTORIC",
                "O",
                "B-CONDITION",
                "I-CONDITION",
                "O",
                "B-HISTORIC",
            ],
        )

    def test_plain_tags(self) -> None:
        tags, _ = to_hf.tags_for_document(document_from_record(RECORD), iob2=False)
        self.assertEqual(
            tags,
            [
                "HISTORIC",
                "HISTORIC",
                "HISTORIC",
                "O",
                "CONDITION",
                "CONDITION",
                "O",
                "HISTORIC",
            ],
        )

    def test_nested_span_is_dropped_under_iob2_and_reported(self) -> None:
        tags, dropped = to_hf.tags_for_document(document_from_record(NESTED), iob2=True)
        self.assertEqual(dropped, 1)
        self.assertEqual(tags, ["B-HISTORIC"] + ["I-HISTORIC"] * 4)

    def test_nested_span_overwrites_under_the_plain_scheme(self) -> None:
        tags, dropped = to_hf.tags_for_document(document_from_record(NESTED), iob2=False)
        self.assertEqual(dropped, 0)
        self.assertEqual(
            tags,
            ["HISTORIC", "SALES_OPTIONS", "HISTORIC", "HISTORIC", "HISTORIC"],
        )


class RoundTripTests(unittest.TestCase):
    def test_iob2_tags_convert_back_to_the_same_spans(self) -> None:
        doc = document_from_record(RECORD)
        tags, _ = to_hf.tags_for_document(doc, iob2=True)
        recovered = spans_from_labels(tags, iob2=True)
        self.assertEqual(recovered, doc.token_spans())

    def test_round_trip_over_a_released_fold(self) -> None:
        path = _paths.public_fold("f1", "test")
        _paths.require(path)
        checked = 0
        for doc in load_document_list(path):
            tags, dropped = to_hf.tags_for_document(doc, iob2=True)
            expected = [
                span
                for span in to_hf.resolve_overlaps(doc.token_spans())[0]
            ]
            expected.sort(key=lambda span: span["token_start"])
            recovered = spans_from_labels(tags, iob2=True)
            self.assertEqual(recovered, expected, doc.doc_id)
            self.assertEqual(dropped, len(doc.token_spans()) - len(expected))
            checked += 1
        self.assertEqual(checked, 87)


class ShippedFileTests(unittest.TestCase):
    def test_released_hf_files_match_the_iob2_conversion(self) -> None:
        for fold in ("f1", "f2", "f3"):
            for split in ("train", "val", "test"):
                source = _paths.public_fold(fold, split)
                target = os.path.join(
                    _paths.PUBLIC_DATA, "folds", fold, "%s_hf.json" % split
                )
                _paths.require(source, target)
                with self.subTest(fold=fold, split=split):
                    records, _ = to_hf.convert(source, iob2=True)
                    with open(target, "r", encoding="utf-8") as handle:
                        self.assertEqual(records, json.load(handle))


if __name__ == "__main__":
    unittest.main()
