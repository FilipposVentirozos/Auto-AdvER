# -*- coding: utf-8 -*-
"""LLM generation backends for the Auto-AdvER NER baselines.

Three backends, selected by ``--backend``:

``openai``
    OpenAI chat completions, pinned to ``gpt-4o-2024-11-20`` (the version the
    paper reports).  ``temperature=0``, ``logprobs=True``, ``top_logprobs=0``,
    exactly as the original.  The log probability of each generated ``@@`` token
    is attached to the corresponding span and later used to break overlaps
    between labels.  **This backend costs money.**
``hf``
    A local Hugging Face ``transformers`` text-generation pipeline
    (``torch_dtype=bfloat16``, ``device_map="auto"``), used in the paper for
    ``Qwen/Qwen2.5-72B-Instruct`` and
    ``meta-llama/Meta-Llama-3.1-70B-Instruct``.  Generation settings are the
    originals: ``do_sample=True``, ``temperature=0.0001``,
    ``max_new_tokens=8192``.  That is *nearly* greedy but not greedy, so runs
    are not bit-reproducible; ``--seed`` now calls ``transformers.set_seed`` so
    a run is repeatable on identical hardware.
``vertex``
    Gemini 1.5 Flash.  **Not implemented** -- see :class:`VertexBackend`.

All heavy dependencies (``openai``, ``torch``, ``transformers``, ``textspan``)
are imported lazily, inside the backend that needs them, so this module imports
on a machine with none of them installed.  The original loaded a 72B checkpoint
and ``en_core_web_trf`` at import time, which meant even the GPT-4o path needed
a four-GPU box.

Marker parsing
--------------
The original duplicated the ``@@``/``##`` parser once per backend.  Here there
is one :func:`parse_marked_text` (pure, stdlib only) plus one
:func:`spans_from_generation` wrapper that performs the "was the advert copied
verbatim?" check and, when it was not, realigns the character offsets onto the
original advert with `pytextspan <https://pypi.org/project/pytextspan/>`_
(imported lazily).
"""

from __future__ import annotations

import copy
import os
import time
import warnings
from typing import List, Optional, Sequence, Tuple

try:  # works as `python src/llm/predict.py` and as `from src.llm import ...`
    from env import load_env
except ImportError:  # pragma: no cover
    from .env import load_env

__all__ = [
    "OPENAI_MODEL",
    "HF_MODELS",
    "BACKENDS",
    "parse_marked_text",
    "spans_from_generation",
    "get_backend",
    "Backend",
    "OpenAIBackend",
    "HuggingFaceBackend",
    "VertexBackend",
]

#: Pinned OpenAI snapshot.  The original used the floating alias ``gpt-4o``;
#: which snapshot that resolved to on 30 November 2024 is recorded nowhere in
#: the source repositories, so this is the paper's stated version, not a value
#: read back from the runs.
OPENAI_MODEL = "gpt-4o-2024-11-20"

#: The two open-weight checkpoints used in the paper.
HF_MODELS = (
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
)

BACKENDS = ("openai", "hf", "vertex")


# --------------------------------------------------------------------------- #
# Marker parsing
# --------------------------------------------------------------------------- #

def parse_marked_text(marked: str) -> Tuple[str, List[dict]]:
    """Strip ``@@``/``##`` markers and return ``(plain_text, spans)``.

    *spans* are dicts ``{'start', 'end', 'span'}`` with character offsets into
    the returned *plain_text* (``end`` exclusive).  They come out in the order
    the markers were consumed, i.e. **last-to-first**; callers sort by ``start``.

    The algorithm is the original one, kept intact: repeatedly take the *last*
    ``@@`` and the *last* ``##``, record the span between them, shift every
    already-collected offset back by four characters, and delete both markers.
    It behaves oddly on nested or crossed markers, but that is the behaviour the
    published predictions were produced with.

    The caller is responsible for the
    ``marked.count("@@") == marked.count("##")`` precondition; see
    :func:`spans_from_generation`.
    """
    temp = copy.copy(marked)
    spans: List[dict] = []
    while True:
        start = temp.rfind("@@")
        end = temp.rfind("##")
        if start == -1 or end == -1:
            break
        for span in spans:
            span["start"] = span["start"] - 4
            span["end"] = span["end"] - 4
        spans.append({"start": start, "end": end - 2, "span": temp[start + 2:end]})
        temp = temp[:end] + temp[end + 2:]
        temp = temp[:start] + temp[start + 2:]
    return temp, spans


def spans_from_generation(
    generated: str,
    prompt: str,
    logprobs: Optional[Sequence[float]] = None,
) -> Optional[List[dict]]:
    """Turn a marked generation into character spans over *prompt*.

    Parameters
    ----------
    generated:
        What the model produced: the advert text with ``@@``/``##`` inserted.
    prompt:
        The advert text that was sent to the model.  Offsets are returned
        relative to this string.
    logprobs:
        For the OpenAI backend, the log probabilities of the generated ``@@``
        tokens, in generation order.  When the count matches the number of
        spans, each span gets a ``logprob`` field.  Pass ``None`` for backends
        that expose no log probabilities (all local models).

    Returns
    -------
    A list of spans, or ``None`` when the generation could not be parsed.  An
    empty list means "parsed, but realignment found nothing" -- the original
    "Possible Hallucination" case.  Both are treated as "no spans" downstream.
    """
    if generated.count("@@") != generated.count("##"):
        warnings.warn("Model hasn't parsed correctly, number of @@ and ## deviate")
        return None

    temp, spans = parse_marked_text(generated)

    if temp != prompt:
        # The model did not copy the advert verbatim.  Realign the offsets onto
        # the original text with pytextspan, as the paper describes.
        spans = sorted(spans, key=lambda d: d["start"])
        pairs = [(span["start"], span["end"]) for span in spans]
        try:
            import textspan
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "the model did not reproduce the advert verbatim, so character "
                "offsets must be realigned with pytextspan. Install it with "
                "`pip install pytextspan`."
            ) from exc
        realigned = textspan.align_spans(pairs, temp, prompt)

        if not any(realigned):
            warnings.warn("Possible Hallucination")
            return []

        for span, updated in zip(spans, realigned):
            span["start"] = updated[0][0]
            span["end"] = updated[0][1]

        for span in spans:
            if span["span"] != prompt[span["start"]:span["end"]]:
                warnings.warn("Wrong match")
                return None

    spans = sorted(spans, key=lambda span: span["start"])
    if logprobs is not None:
        if len(spans) == len(logprobs):
            for index, logprob in enumerate(logprobs):
                spans[index]["logprob"] = logprob
        else:
            warnings.warn("Wrong generation or spurious characters in text.")
    return spans


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

class Backend:
    """Common interface: ``generate(chat_history, system_instruction, prompt)``.

    Returns ``(raw_generation, spans)`` where *spans* may be ``None`` (could not
    be parsed) or ``[]`` (nothing found).
    """

    name = "base"

    def generate(self, chat_history, system_instruction, prompt):
        raise NotImplementedError

    @staticmethod
    def _messages(chat_history, system_instruction, prompt) -> List[dict]:
        """Assemble system turn + in-context turns + the advert under test."""
        messages = [{"role": "system", "content": system_instruction}]
        for turn in chat_history:
            content = turn["content"] if "content" in turn else turn["text"]
            messages.append({"role": turn["role"], "content": content})
        messages.append({"role": "user", "content": prompt})
        return messages


class OpenAIBackend(Backend):
    """OpenAI chat completions: pinned snapshot, temperature 0, with logprobs.

    Reads ``OPENAI_API_KEY`` from the environment (``.env`` is loaded first).
    Retries indefinitely on rate limits with a ten-second pause, as the original
    did.
    """

    name = "openai"

    def __init__(self, model: str = OPENAI_MODEL, retry_seconds: int = 10):
        self.model = model
        self.retry_seconds = retry_seconds
        self._client = None

    @property
    def client(self):
        if self._client is None:
            load_env()
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Copy .env.example to .env at the "
                    "repository root and fill it in, or export the variable."
                )
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def generate(self, chat_history, system_instruction, prompt):
        from openai import RateLimitError

        messages = self._messages(chat_history, system_instruction, prompt)
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    logprobs=True,
                    top_logprobs=0,
                    temperature=0,
                    messages=messages,
                )
                break
            except RateLimitError as exc:
                print(exc)
                time.sleep(self.retry_seconds)

        dumped = response.model_dump()
        generated = dumped["choices"][0]["message"]["content"]
        logprobs = [
            token["logprob"]
            for token in ((dumped["choices"][0].get("logprobs") or {}).get("content") or [])
            if token["token"].strip() == "@@"
        ]
        return generated, spans_from_generation(generated, prompt, logprobs=logprobs)


class HuggingFaceBackend(Backend):
    """Local ``transformers`` text-generation pipeline (bf16, ``device_map="auto"``).

    The pipeline is built on first use, not at import, so this module stays
    importable without ``torch``.  A multi-GPU box is expected for the 70B/72B
    checkpoints: the paper used NVIDIA H100s and the original code assumed four
    visible devices.  Set ``CUDA_VISIBLE_DEVICES`` and ``HF_HOME`` in your shell
    -- the original hard-coded both; see ``src/llm/README.md``.

    ``HF_TOKEN`` is read from the environment by ``huggingface_hub`` itself once
    ``.env`` has been loaded; there is no explicit ``login()`` call and no token
    anywhere in the source.
    """

    name = "hf"

    def __init__(self, model: str = HF_MODELS[0], seed: Optional[int] = None,
                 temperature: float = 0.0001, max_new_tokens: int = 8192):
        self.model = model
        self.seed = seed
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            load_env()
            import torch
            import transformers

            if self.seed is not None:
                transformers.set_seed(self.seed)
            self._pipeline = transformers.pipeline(
                task="text-generation",
                model=self.model,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                framework="pt",
            )
        return self._pipeline

    def generate(self, chat_history, system_instruction, prompt):
        messages = self._messages(chat_history, system_instruction, prompt)
        outputs = self.pipeline(
            messages,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
        )
        generated = outputs[0]["generated_text"][-1]["content"]
        return generated, spans_from_generation(generated, prompt, logprobs=None)


class VertexBackend(Backend):
    """Gemini 1.5 Flash -- **not implemented**; the original code is lost.

    The Gemini-Flash column of the paper was produced by a Vertex AI code path
    that survives nowhere: not in the working tree and not in any of the 21
    commits of the source repository.  All that remains is its parameter file,
    shipped here as ``configs/gemini-1.5-flash-002.toml``
    (``temperature = 0.0``, ``max_output_tokens = 8192``; topK, topP and seed all
    commented out), plus a commented-out ``vertexai.init(project=..., location=...)``
    call in an archived backup.  How many in-context examples it received is
    unrecorded.

    The Gemini results are therefore **stored-predictions-only**: the generated
    outputs ship under ``predictions/gemini-1.5-flash/`` and can be re-scored,
    but not regenerated from this repository.
    """

    name = "vertex"

    MESSAGE = (
        "The Gemini/Vertex AI generation code for Auto-AdvER is lost and has not "
        "been reimplemented. Only its parameters survive, in "
        "configs/gemini-1.5-flash-002.toml (gemini-1.5-flash-002, temperature 0.0, "
        "max_output_tokens 8192); the number of in-context examples used is "
        "unrecorded. The Gemini-Flash predictions are shipped instead, under "
        "predictions/gemini-1.5-flash/, and can be re-scored with src/score.py. "
        "Set VERTEX_PROJECT and VERTEX_LOCATION in .env if you reimplement this "
        "backend."
    )

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(self.MESSAGE)

    def generate(self, chat_history, system_instruction, prompt):
        raise NotImplementedError(self.MESSAGE)


def get_backend(name: str, model: Optional[str] = None,
                seed: Optional[int] = None) -> Backend:
    """Instantiate a backend by name.  Nothing heavy is loaded yet."""
    if name == "openai":
        return OpenAIBackend(model=model or OPENAI_MODEL)
    if name == "hf":
        return HuggingFaceBackend(model=model or HF_MODELS[0], seed=seed)
    if name == "vertex":
        return VertexBackend()
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
