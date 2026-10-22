# Stored model predictions

One file per model configuration and fold, covering the 87 test
adverts of each fold. No model, API key or GPU is needed to reproduce the paper's tables from them.

The four generative configurations are the **exact stored outputs**
that were scored. The eight configurations that predict one label
per token — the four `crf*` variants and the four fine-tuned
encoders — ship their raw per-token labels as well, and their span
files are **built from those labels** by
`src/data_tools/tags_to_spans.py` (see *Notes on the token-label
files* below).

## Format

One JSON object per line, in the same advert order as the matching
`data/folds/f{n}/test.jsonl`:

```json
{"id": "0001", "entities": [{"start": 11, "end": 26, "label": "SALES_OPTIONS", "token_start": 4, "token_end": 6}]}
```

* `start` / `end` are character offsets into the advert `text`, with
  `end` **exclusive**, as in the data files.
* `token_start` / `token_end` are **inclusive** token indices into the
  advert's tokenisation. These are the indices the paper's scorer
  matches on, so they are the authoritative view of a prediction.
  Several configurations emitted only token indices; the character
  offsets for those were derived from the advert's tokenisation.
* In 34 entities across the four generative configurations, the stored
  character offsets cover a slightly different stretch of text than
  the stored token indices (typically one token more or less at an
  edge). Both are reproduced exactly as stored; prefer the token
  indices, which is what was scored.
* `gpt-4o` entities additionally carry `logprob`, the model's own
  confidence for that entity. It is model output, not annotation, and
  appears in `predictions/` only.

## Model configurations

| Directory | Model / configuration |
|---|---|
| `crf/` | `sklearn-crfsuite` CRF over spaCy `en_core_web_trf` features |
| `crf-pos/` | the same CRF, part-of-speech features only |
| `crf-wo-pos/` | the same CRF, without part-of-speech features |
| `crf-2-3/` | the same CRF with a wider context window (neighbour features at ±1, ±2 and ±3 tokens; "CRF 2n/3n" in the paper) |
| `bert-base/` | BERT base fine-tuned for token classification (the paper lists `bert-base-cased`; the training notes record uncased runs — see the repository README) |
| `bert-large/` | BERT large fine-tuned for token classification (cased/uncased as above; not in the paper's final table) |
| `deberta-v3-base/` | microsoft/deberta-v3-base |
| `deberta-v3-large/` | microsoft/deberta-v3-large |
| `gemini-1.5-flash/` | gemini-1.5-flash-002 (Vertex AI) |
| `gpt-4o/` | gpt-4o (OpenAI, unpinned alias at run time) |
| `llama-3.1-70b/` | meta-llama/Meta-Llama-3.1-70B-Instruct |
| `qwen2.5-72b/` | Qwen/Qwen2.5-72B-Instruct |

The four `crf*` directories are the paper's CRF configurations:
`crf` the full feature set, `crf-pos` part-of-speech features only,
`crf-wo-pos` without part-of-speech features, `crf-2-3` a wider
context window ("CRF 2n/3n" in the paper's table).

All eight token-label configurations ship
`raw_token_labels/f{n}_test.json` — the per-token label sequence as
decoded, one list per advert, positionally aligned with
`raw_token_labels/f{n}_test_ids.json` (and therefore with the fold's
test file). These are plain labels, not IOB2, so two adjacent
entities of the same label cannot be separated in that view; the
`f{n}_test.jsonl` entity files are the scored view.

The encoder rows were fine-tuned with the Hugging Face
`run_ner.py` example script. The checkpoint names above are those
recorded in the experiment log; note the paper's prose describes cased
BERT while the logged runs used the uncased checkpoints (see
*Differences from the paper* in the repository README).

## Notes on the token-label files

* **The CRF and encoder label lists count differently.** A CRF list
  has one label per token, whitespace-only tokens (`"\n"`, `" "`,
  `"\n\n"`) included, because the features were built over the
  advert's own tokenisation. `run_ner.py` instead writes one label per
  *word that produced at least one sub-word*, and a whitespace-only
  word produces none, so an encoder list is shorter than its token
  list by exactly the number of such tokens. `tags_to_spans.py` picks
  the alignment from the two lengths (`--alignment auto`) and refuses
  any length it cannot account for.
* **Two encoder documents are truncated.** BERT-base and BERT-large
  hit the 512 sub-word limit on adverts `0326` and `0340` of fold 3 (99 and 16 labels
  short). Their labelled prefix is used and the tail is left
  unlabelled; the converter warns about each.
* **Encoder labels for folds 1 and 2 are reconstructed.** Only fold 3's
  raw `run_ner.py` output survives. Folds 1 and 2 were rebuilt by
  reconstruction from the project's earlier span exports; it reproduces
  fold 3's stored output for 87/87 adverts in all four models, which
  is the evidence that it is faithful. Its one ambiguity — a span
  ending at token 86 in an advert longer than 87 non-whitespace
  tokens, which cannot be told apart from an artefact of those exports —
  affects 0 to 3 spans per model and fold and
  is resolved in favour of a genuine prediction.
* In-context-example ablations and duplicate export files from the
  original project tree are not part of this release; only the
  configurations reported in the paper are shipped.
