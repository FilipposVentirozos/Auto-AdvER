"""Reading Auto-AdvER documents in either of the two on-disk formats.

Two record layouts occur in this project and every tool here accepts both:

**Public format** (the released ``data/*.jsonl`` and ``predictions/*.jsonl``)::

    {"id": "0412", "text": "...", "tokens": ["Very", "good", ...],
     "labels": ["O", "O", ...], "entities": [{"start": 36, "end": 53, "label": "HISTORIC"}],
     "advertiser_type": "PRIVATE", "region": "YORKSHIRE AND THE HUMBER"}

``entities`` carry **character** offsets with an **exclusive** end.  Prediction
records may carry only ``id`` and ``entities``, and may additionally carry
``token_start``/``token_end`` per entity.

**Prodigy format** (the internal annotation and prediction files)::

    {"text": "...", "tokens": [{"text": "2020", "start": 0, "end": 4, "id": 0, ...}],
     "spans": [{"start": 43, "end": 55, "token_start": 10, "token_end": 11, "label": "CONDITION"}],
     "meta": {"metadata_search_id": "202401...", ...}}

``spans`` carry **inclusive token indices** plus character offsets with an
exclusive end.

The paper's evaluation matches spans on inclusive token indices, so public
records have to be converted.  That conversion is done here, once, by
:func:`token_offsets_from_text` and :func:`char_span_to_token_span`, so that
the scorer and the data tools cannot disagree about it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

LABELS_DEFAULT: Tuple[str, ...] = ("CONDITION", "HISTORIC", "SALES_OPTIONS")

PRODIGY = "prodigy"
PUBLIC = "public"


class FormatError(ValueError):
    """Raised when a record cannot be interpreted as either supported format."""


def detect_format(record: Dict[str, Any]) -> str:
    """Return ``"prodigy"`` or ``"public"`` for a single decoded JSON record.

    Prodigy records are recognised by ``meta.metadata_search_id`` or by
    ``spans``/dict-shaped ``tokens``; public records by a top-level ``id`` with
    ``entities`` and/or string-shaped ``tokens``.
    """
    meta = record.get("meta")
    if isinstance(meta, dict) and "metadata_search_id" in meta:
        return PRODIGY
    if "spans" in record:
        return PRODIGY
    tokens = record.get("tokens")
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], dict):
        return PRODIGY
    if "entities" in record or "id" in record:
        return PUBLIC
    raise FormatError(
        "record is neither Prodigy (spans / meta.metadata_search_id) nor "
        "public (id / entities); keys: %s" % sorted(record)
    )


def token_offsets_from_text(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Locate each token in ``text`` in order and return ``(start, end)`` pairs.

    ``end`` is exclusive.  The tokenisation used throughout Auto-AdvER is
    spaCy's, which preserves every character of the text: consecutive tokens
    are separated by either nothing at all or by exactly one space (any longer
    run of whitespace, and every newline, becomes a whitespace token in its own
    right).  The reconstruction therefore only has to decide, for each token,
    between a gap of zero and a gap of one space.  The single ambiguous case —
    a whitespace token that could equally begin at the separator or just after
    it — is resolved in favour of consuming the separator, which is what spaCy
    does.

    A token that matches at neither position is looked up with ``str.find``
    from the cursor as a last resort, and :class:`FormatError` is raised if it
    is not present at all, rather than guessing.

    Verified against the stored per-token offsets of all 8,546 Prodigy records
    in the project (gold, folds, IAA and predictions): zero mismatches.
    """
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        length = len(token)
        at_cursor = text[cursor : cursor + length] == token
        after_space = (
            text[cursor : cursor + 1] == " "
            and text[cursor + 1 : cursor + 1 + length] == token
        )
        if at_cursor and not (token[:1] == " " and after_space):
            start = cursor
        elif after_space:
            start = cursor + 1
        else:
            start = text.find(token, cursor)
            if start < 0:
                raise FormatError(
                    "token %d (%r) not found in text at or after character %d"
                    % (index, token, cursor)
                )
        offsets.append((start, start + length))
        cursor = start + length
    return offsets


def char_span_to_token_span(
    offsets: Sequence[Tuple[int, int]], start: int, end: int
) -> Tuple[int, int]:
    """Convert a character span (``end`` exclusive) to inclusive token indices.

    The tokens wholly contained in ``[start, end)`` are preferred, which is the
    convention the annotation tool used: verified against the stored
    ``token_start``/``token_end`` of all 5,863 character-carrying spans in the
    project (gold, IAA and predictions), zero mismatches.  Ten of those spans
    (GPT-4o predictions) begin or end inside a token, and only the containment
    rule places them where the stored indices do.

    If no token is wholly contained — a span narrower than a single token — the
    rule falls back to overlap: the first token reaching past ``start`` and the
    last token beginning before ``end``.
    """
    contained = [
        index
        for index, (tok_start, tok_end) in enumerate(offsets)
        if tok_start >= start and tok_end <= end
    ]
    if contained:
        return contained[0], contained[-1]
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    for index, (tok_start, tok_end) in enumerate(offsets):
        if token_start is None and tok_end > start:
            token_start = index
        if tok_start < end:
            token_end = index
    if token_start is None or token_end is None or token_end < token_start:
        raise FormatError(
            "character span (%d, %d) does not cover any token" % (start, end)
        )
    return token_start, token_end


class Document:
    """One advert, normalised across the two formats.

    Attributes:
        doc_id: ``meta.metadata_search_id`` (Prodigy) or ``id`` (public).
        text: the advert text, or ``None`` for prediction-only records.
        tokens: token surface forms, or ``None`` when the record carries none.
        offsets: per-token ``(start, end)`` character offsets, ``end``
            exclusive; ``None`` when the record carries no tokens.
        spans: the record's spans in **file order** (order matters for the
            greedy matcher's tie-breaks and for the plain-label HF conversion).
            Each is a dict with ``label`` plus whichever of ``start``/``end``
            (characters) and ``token_start``/``token_end`` (inclusive tokens)
            the file supplied.
        fmt: ``"prodigy"`` or ``"public"``.
        raw: the decoded record, untouched.
    """

    __slots__ = ("doc_id", "text", "tokens", "offsets", "spans", "fmt", "raw")

    def __init__(
        self,
        doc_id: str,
        text: Optional[str],
        tokens: Optional[List[str]],
        offsets: Optional[List[Tuple[int, int]]],
        spans: List[Dict[str, Any]],
        fmt: str,
        raw: Dict[str, Any],
    ) -> None:
        self.doc_id = doc_id
        self.text = text
        self.tokens = tokens
        self.offsets = offsets
        self.spans = spans
        self.fmt = fmt
        self.raw = raw

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Document(id=%r, fmt=%r, tokens=%s, spans=%d)" % (
            self.doc_id,
            self.fmt,
            len(self.tokens) if self.tokens is not None else None,
            len(self.spans),
        )

    @property
    def n_tokens(self) -> int:
        if self.tokens is None:
            raise FormatError("document %r carries no tokens" % self.doc_id)
        return len(self.tokens)

    def token_spans(
        self, reference: Optional["Document"] = None
    ) -> List[Dict[str, Any]]:
        """Return the spans with inclusive ``token_start``/``token_end`` filled in.

        Spans that already carry token indices are used as they are.  Spans
        that carry only character offsets are converted with this document's
        token offsets, or with ``reference``'s if this document has none (the
        case for public prediction records, which hold only ``id`` and
        ``entities``).
        """
        offsets = self.offsets
        if offsets is None and reference is not None:
            offsets = reference.offsets
        out: List[Dict[str, Any]] = []
        for span in self.spans:
            if span.get("token_start") is not None and span.get("token_end") is not None:
                out.append(
                    {
                        "token_start": int(span["token_start"]),
                        "token_end": int(span["token_end"]),
                        "label": span["label"],
                    }
                )
                continue
            if offsets is None:
                raise FormatError(
                    "document %r has character-offset spans but no tokens to "
                    "convert them with (pass the matching gold document)"
                    % self.doc_id
                )
            token_start, token_end = char_span_to_token_span(
                offsets, int(span["start"]), int(span["end"])
            )
            out.append(
                {"token_start": token_start, "token_end": token_end, "label": span["label"]}
            )
        return out

    def char_spans(
        self, reference: Optional["Document"] = None
    ) -> List[Dict[str, Any]]:
        """Return the spans with character ``start``/``end`` (end exclusive) filled in."""
        offsets = self.offsets
        if offsets is None and reference is not None:
            offsets = reference.offsets
        out: List[Dict[str, Any]] = []
        for span in self.spans:
            if span.get("start") is not None and span.get("end") is not None:
                out.append(
                    {
                        "start": int(span["start"]),
                        "end": int(span["end"]),
                        "label": span["label"],
                    }
                )
                continue
            if offsets is None:
                raise FormatError(
                    "document %r has token-index spans but no tokens to convert "
                    "them with" % self.doc_id
                )
            start = offsets[int(span["token_start"])][0]
            end = offsets[int(span["token_end"])][1]
            out.append({"start": start, "end": end, "label": span["label"]})
        return out

    def meta_value(self, name: str) -> Any:
        """Look up a metadata field in whichever place the format keeps it.

        ``advertiser_type`` and ``region`` live at the top level of a public
        record and under ``meta`` in a Prodigy one (where ``region`` is called
        ``advertiser_location_region``).
        """
        if self.fmt == PUBLIC:
            return self.raw.get(name)
        meta = self.raw.get("meta") or {}
        if name == "region":
            return meta.get("advertiser_location_region")
        return meta.get(name)


def document_from_record(record: Dict[str, Any], fmt: Optional[str] = None) -> Document:
    """Normalise one decoded JSON record into a :class:`Document`."""
    fmt = fmt or detect_format(record)
    text = record.get("text")
    if fmt == PRODIGY:
        doc_id = (record.get("meta") or {}).get("metadata_search_id")
        if doc_id is None:
            raise FormatError("Prodigy record without meta.metadata_search_id")
        raw_tokens = record.get("tokens")
        tokens: Optional[List[str]] = None
        offsets: Optional[List[Tuple[int, int]]] = None
        if raw_tokens:
            tokens = [token["text"] for token in raw_tokens]
            offsets = [(int(token["start"]), int(token["end"])) for token in raw_tokens]
        spans = [
            {
                "start": span.get("start"),
                "end": span.get("end"),
                "token_start": span.get("token_start"),
                "token_end": span.get("token_end"),
                "label": span["label"],
            }
            for span in record.get("spans") or []
        ]
        return Document(str(doc_id), text, tokens, offsets, spans, fmt, record)

    doc_id = record.get("id")
    if doc_id is None:
        raise FormatError("public record without an 'id' field")
    tokens = list(record.get("tokens") or []) or None
    offsets = None
    if tokens is not None:
        if text is None:
            raise FormatError(
                "public record %r has tokens but no text, so token character "
                "offsets cannot be derived" % doc_id
            )
        offsets = token_offsets_from_text(text, tokens)
    spans = [
        {
            "start": entity.get("start"),
            "end": entity.get("end"),
            "token_start": entity.get("token_start"),
            "token_end": entity.get("token_end"),
            "label": entity["label"],
        }
        for entity in record.get("entities") or []
    ]
    return Document(str(doc_id), text, tokens, offsets, spans, fmt, record)


def iter_documents(path: str) -> Iterator[Document]:
    """Yield :class:`Document` objects from a JSONL file, in file order."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise FormatError("%s line %d: %s" % (path, line_number, error)) from error
            yield document_from_record(record)


def load_documents(path: str) -> "Dict[str, Document]":
    """Load a JSONL file into an id-keyed dict, preserving file order.

    Raises :class:`FormatError` on a duplicate document id, which would
    otherwise silently drop a document from every downstream count.
    """
    docs: Dict[str, Document] = {}
    for doc in iter_documents(path):
        if doc.doc_id in docs:
            raise FormatError("duplicate document id %r in %s" % (doc.doc_id, path))
        docs[doc.doc_id] = doc
    if not docs:
        raise FormatError("no documents found in %s" % path)
    return docs


def load_document_list(path: str) -> List[Document]:
    """Load a JSONL file as a list, preserving file order (duplicates kept)."""
    return list(iter_documents(path))


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> int:
    """Write one compact JSON object per line; return the number written."""
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
            count += 1
    return count
