# Auto-AdvER

Auto-AdvER (**Auto**mobile **Adv**ertisement **E**ntity **R**ecognition) is a named-entity-recognition
schema and dataset for the free-text portion of UK car advertisements. It introduces three labels —
`CONDITION`, `HISTORIC` and `SALES_OPTIONS` — that disentangle a vehicle's present state, events from
its past, and the commercial offers attached to the sale. This repository ships the annotated corpus
of **605 adverts** (104,382 tokens) written by private sellers and motor traders, the annotation
guidelines, the paper's three cross-validation folds, an inter-annotator-agreement study in which
**three annotators** independently annotated the same 113 adverts, the stored predictions of every
model reported in the paper, and the code that scores them. The schema was developed as an
industry–academia collaboration, working from an industry partner's own advertisement data and
analytics needs, and is described in the paper *Auto-AdvER: Disentangling Condition, History, and
Commercial Offers in Automotive Advertisements*.

Everything the paper reports can be recomputed here from the shipped files with the Python standard
library alone — no model, API key or GPU.

## Contents

| Directory / file | What it holds |
|---|---|
| `data/` | the corpus, the character-level companion view, the three folds, the inter-annotator-agreement subset and per-file counts |
| `predictions/` | the outputs of all twelve model configurations on the three test folds, plus the raw per-token labels of the eight configurations that predict one label per token |
| `src/score.py` | the scorer: the paper's partial-match evaluation, for models and for annotator agreement |
| `src/data_tools/` | corpus statistics, fold construction, Hugging Face conversion, token-labels-to-spans conversion, the shared record reader |
| `src/crf/` | the CRF baselines: feature configurations, training/prediction, the top-features listing |
| `src/llm/` | the prompting code for the decoder models, and a `.env` loader |
| `src/encoders/` | a wrapper around Hugging Face's `run_ner.py` for the encoder baselines |
| `results/` | `make_tables.py` and the generated `tables.md`: every table in this release with the command that produced it, including the mention families that fall under all three labels and the two experts' label disagreements |
| `tests/` | the regression suite (standard library only) |
| `guidelines/` | the annotation guidelines, as LaTeX source and PDF |
| `configs/` | recorded model versions and decoding settings (documentation, not runtime configuration) |
| `.zenodo.json` | the metadata Zenodo uses when it archives a GitHub release (see *Persistent identifiers* under *Citation*) |
| `.gitattributes` | the Git LFS tracking rules for the files under `data/` and `predictions/` (see *Cloning* below) |

## Data

### Schema

`data/auto_adver.jsonl` — one JSON object per line, one advert per object:

```json
{"id": "0343",
 "text": "Very good condition only recently serviced nothing needed no expense spared",
 "tokens": ["Very", "good", "condition", "only", "recently", "serviced", "nothing", "needed", "no", "expense", "spared"],
 "labels": ["O", "O", "O", "O", "B-CONDITION", "I-CONDITION", "O", "O", "O", "O", "O"],
 "entities": [{"start": 25, "end": 42, "label": "CONDITION"}],
 "advertiser_type": "PRIVATE", "region": "NORTHERN IRELAND"}
```

- **`entities`** — character-offset spans `{start, end, label}`. This is the ground truth:
  it is what the annotators marked, it is lossless, and it is tokeniser-agnostic. `end` is
  **exclusive**, so `text[start:end]` is the surface form (`text[25:42] == "recently serviced"`).
- **`tokens` / `labels`** — a word-level **IOB2** view, ready to train or evaluate on. Tokenisation is
  the annotation tool's word-level tokenisation; no subword splitting is applied. The view is flat,
  so it cannot represent the single nested entity described below.
- **`advertiser_type`** — `PRIVATE` (314 adverts) or `TRADE` (291).
- **`region`** — the advertiser's UK region, or JSON `null` for the 25 adverts that carry none.

`data/auto_adver.char_iob2.jsonl` — a character-level IOB2 companion for the same adverts, one tag
per character (`len(char_labels) == len(text)`), keyed by the same `id`. Multi-word entities stay
contiguous: a space inside an entity is tagged `I-`, not `O`.

`data/COUNTS.md` gives adverts and per-label entity counts for **every** published file.

### Label set

| Label | Description | Entities |
|---|---|---|
| `CONDITION` | The vehicle's present material or documented state — mechanical, aesthetic, maintenance-related or a legal classification — positive or negative, as at the date the advert was posted. | 594 |
| `HISTORIC` | Events in the vehicle's past that affected its mechanical condition or bodywork: accidents, ownership, component changes, service history. | 810 |
| `SALES_OPTIONS` | Anything tangible, or any service, offered during or after the sale beyond the car itself, that enhances the sale proposition. | 2,174 |
| **Total** | | **3,578** |

26 adverts carry no entity. Full definitions,
the sub-entities detected under each label, the span-length rules, and the themes deliberately
**not** tracked (logistical costs, personal sentiment, reason for selling, car identifiables
restated in free text, contact and location details, optional and standard features, sales pitch)
are in `guidelines/Auto-AdvER_annotation_guidelines.pdf`.

### Folds

`data/folds/f1/`, `f2/`, `f3/` are the paper's three cross-validation folds: **426 train / 92
validation / 87 test** adverts each, with the three **test sets disjoint**, covering 261 distinct
adverts between them. Each fold directory holds `train.jsonl`, `val.jsonl` and `test.jsonl` in the
schema above, plus `train_hf.json`, `val_hf.json` and `test_hf.json` — the same adverts as
`[{"tokens": [...], "ner_tags": [...]}, ...]`, positionally aligned with their JSON Lines sibling.
The `ner_tags` are **IOB2**, matching `labels`.

The document partition is exactly the paper's. Four further folds
were cut but never used in the paper and are not shipped. Because the partition is a property of
scikit-learn's shuffling of the *input order*, and the released corpus is sorted by advert `id`,
`data/corpus_order.json` ships the 605 ids in their original annotation order so that
`src/data_tools/make_folds.py --order data/corpus_order.json` re-cuts the published partition
exactly (check it with `--verify-against data/folds`; this is the one tool in the package that
needs scikit-learn). For ordinary use, the shipped fold files *are* the paper's partition.

### Inter-annotator agreement

`data/iaa/annotator_1.jsonl`, `annotator_2.jsonl`, `annotator_3.jsonl` — the same **113 adverts**
(about 19 % of the corpus) annotated independently by **three annotators**, in the corpus schema and
sorted by `id`, so the three files can be zipped directly. Each carries the same `text` and `tokens`
as the corpus record for that `id`. Character-level companions (`annotator_N.char_iob2.jsonl`) are
shipped too. Entity totals: annotator 1 685 (125 / 159 / 401), annotator 2 713 (126 / 168 / 419),
annotator 3 677 (122 / 159 / 396), ordered CONDITION / HISTORIC / SALES_OPTIONS.

`annotator_2`'s pass became the gold corpus, so for those 113 adverts that file is
identical to `data/auto_adver.jsonl`. Anyone computing agreement should know that one of the three
"independent" files is the released gold. Annotator roles are given under
[Provenance and anonymisation](#provenance-and-anonymisation).

### The one nested entity

The corpus contains exactly **one** nested entity: in advert `0307`, a `SALES_OPTIONS` span sits
inside a longer `HISTORIC` span. `entities` keeps both. The flat views — `labels`, the `*_hf.json`
`ner_tags` and `char_labels` — can only carry one label per token, so they keep the longer span
only. Consumers who need every annotated span must read `entities`.

## Reproducing the paper

### Cloning

The files under `data/` and `predictions/` are stored with [Git LFS](https://git-lfs.com), so install
it once before cloning and the data arrives with the checkout:

```console
$ git lfs install
$ git clone https://github.com/FilipposVentirozos/Auto-AdvER.git
```

In a clone made without Git LFS those files are small pointer stubs instead of JSON; `git lfs install`
followed by `git lfs pull` in that clone replaces them with the real thing. The test suite says so
plainly if it finds a stub.

### Running

```console
$ python3 -m unittest discover tests        # 81 tests
$ python3 results/make_tables.py            # regenerates results/tables.md
```

`results/tables.md` prints every table beside the command that produced it. Scoring one model, for
example:

```console
$ python3 src/score.py models \
    --gold 'data/folds/{fold}/test.jsonl' \
    --pred 'CRF=predictions/crf/{fold}_test.jsonl' \
    --folds f1,f2,f3
```

`--pred` may be repeated, `--per-label` gives the per-label table and `--format json` the per-fold
counts. `score.py` also has `iaa`, `boundaries` and `families` subcommands; `--help` lists their
options. `python3 src/data_tools/stats.py data/auto_adver.jsonl` reproduces the corpus table.

### What the scorer does

`src/score.py` implements the paper's MUC-style partial-match scheme. Matching is on inclusive token
indices; character offsets are converted first. Each span pair is counted as COR (correct), INC
(right span, wrong label), PAR (partial overlap), MIS (missing) or SPU (spurious), and then:

```
possible = COR + INC + PAR + MIS          actual = COR + INC + PAR + SPU
exact    P = COR / actual                 R = COR / possible
partial  P = (COR + 0.5·PAR) / actual     R = (COR + 0.5·PAR) / possible
F1       = 2PR / (P + R)
```

Counts are pooled within a fold, the per-fold percentages are averaged over f1–f3, and the quoted
spread is a population standard deviation over the three folds. As in the paper, `--per-label`
filters only the side being iterated, matching is greedy first-match rather than one-to-one, and a
prediction that strictly contains a gold span counts as MIS but not as SPU.

## Re-running the models

Not needed for the tables above. The CRF, prompting and encoder code is in `src/crf/`, `src/llm/`
and `src/encoders/`, each with a short README. Dependencies are in `requirements.txt` (CRF) and
`requirements-llm.txt` (LLMs); credentials go in a `.env` file at the repository root, copied from
`.env.example`.

## Differences from the paper

- **Agreement figures.** Pairwise partial-match F1 from the shipped annotator files is 88.3
  (annotators 1–2), 72.7 (1–3) and 72.6 (2–3), mean 77.9, against 88 / 74 / 73 and about 78
  in the paper.
- **In-context examples.** The paper describes 100 random training examples. The runs took them in
  file order: the first 100 for Llama and Qwen, all 426 of the fold for GPT-4o.
- **BERT checkpoint.** The paper lists `bert-base-cased`; the recorded command lines use uncased
  checkpoints.
- **Two extra models.** BERT-large and DeBERTaV3-large were trained but are not in the paper's
  table. Their predictions are included and marked as such in `results/tables.md`.

## Provenance and anonymisation

The adverts were sourced from the industry partner's database of live car advertisements — half
written by motor traders, half by private individuals — and annotated in Prodigy with a customised
sequence-labelling interface.

- **Annotators are pseudonymised** as `annotator_1`, `annotator_2` and `annotator_3` (`A1`/`A2`/`A3`
  in the paper and the guidelines). `annotator_1` is a domain-expert data scientist from the industry
  team; `annotator_2` is a university researcher with automotive-data expertise who led the
  development of the annotation guidelines and produced the full corpus annotation; `annotator_3` is
  a university researcher who joined after the guidelines were finalised and has no automotive
  domain background. The mapping to individuals is not published.
- **Advert metadata was removed**, except `advertiser_type` (trade or private) and the advertiser's
  UK `region`. Everything else that the internal records carried — the live advert identifier, the
  dealer's legal name, vehicle derivative and registration details, editing and origin provenance,
  advertiser capabilities, the rendered specification tables that were shown to annotators for
  context, dataset and annotation-session bookkeeping — is not in this release.
- **Document ids were re-assigned** sequentially (`0001`–`0605`, ascending in the order of the
  original identifiers). The crosswalk back to the source adverts is not published.
- **Advert texts are as published on the marketplace, apart from personal details**, deliberately:
  the linguistic character of this domain is the point of the dataset, so the spelling, punctuation
  and capitalisation of the seller-written copy are untouched. In 112 adverts, telephone numbers,
  postcodes, registration plates, one vehicle identification number and the names of salespeople,
  a private seller and a previous owner were replaced by random values of the same length and format, so the
  text keeps the look of real adverts. Telephone numbers keep their area code or mobile prefix,
  postcodes their outward code (`KT17`), current-format plates their region letter and age
  digits, and the VIN its maker and model characters; the rest is random. Names became common
  names of the same length and, where the original is a known first name, the same gender. The
  same original always has the same replacement. Being random, a replacement may coincide with a
  real number or plate elsewhere; it is never the one in the original advert. Every replacement
  keeps its length, so character offsets, tokens and labels are exactly as annotated. Dealer
  names, dealer websites and dealers' business e-mail addresses are kept. Adverts also
  occasionally name the marketplace itself.
- **`logprob` fields were stripped from the gold data.** 85 gold spans in 27 adverts carried a
  large-language-model log probability, an artefact of the model-assisted annotation workflow the
  paper describes. Those fields are gone from the corpus and the folds.
  `logprob` survives only in `predictions/gpt-4o/`, where it is model output rather than annotation.
- **Ordering is stable and documented**: the corpus, the agreement files and their character-level
  companions are sorted by `id`; the fold files, the prediction files and the raw token-label files
  keep the original fold order, so positional alignment holds between each `*_hf.json`, its
  `.jsonl` sibling and the corresponding prediction file.
- **Not shipped**: the four unused folds, superseded annotation passes, the in-context-example
  ablation runs, rejected annotations, and the internal export by-products.

## Licence

- **Code** — `src/`, `tests/`, `results/make_tables.py`, `configs/` — is licensed under the
  [MIT Licence](LICENSE).
- **Data** — everything in `data/` — is licensed under
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](data/LICENSE).
- **Stored predictions** — everything in `predictions/` — are model outputs computed over that data,
  and are likewise licensed under [CC BY 4.0](predictions/LICENSE).
- The annotation guidelines in `guidelines/` are released with the data, under CC BY 4.0.

## Citation

If you use this dataset or code, please cite the paper (see also [`CITATION.cff`](CITATION.cff);
the full author list is in the paper):

```bibtex
@inproceedings{ventirozos2026autoadver,
  title     = {Auto-AdvER: Disentangling Condition, History, and Commercial Offers in
               Automotive Advertisements},
  author    = {Filippos Ventirozos and Ioanna Nteka and Tania Nandy and Jozef Baca and Charlie Roadhouse and Peter Appleby and Matthew Shardlow},
  booktitle = {Proceedings of the 5th Conference of the Asia-Pacific Chapter of the Association for
               Computational Linguistics and the 15th International Joint Conference on Natural
               Language Processing (AACL-IJCNLP 2026)},
  year      = {2026},
  note      = {To appear}
}
```

### Persistent identifiers

- **Zenodo — the DOI of record.** Every GitHub *release* of this repository is archived on
  Zenodo, which mints one DOI per version and a *concept DOI* that always resolves to the latest
  version. Cite the concept DOI when you cite the data: POST-RELEASE-TODO(zenodo-concept-doi).
  The archive is the complete repository at that release — data, predictions, code and
  documentation — with metadata from [`.zenodo.json`](.zenodo.json).
- **Manchester Metropolitan University e-space.** The university's research repository holds a
  record for this dataset that points to the same Zenodo DOI rather than a second copy:
  POST-RELEASE-TODO(e-space-record-url).
- **The paper** is the reference for the schema and the results; its DOI is added to the
  citation above once the proceedings appear.

### Acknowledgements

Auto-AdvER was developed with our industry partner, the Autotrader Research Group, Autotrader UK,
whose advertisement data, domain expertise and infrastructure engineers made the schema, the
guidelines and this dataset possible.
