# Counts — every file in this release

Adverts, entities per label and provenance for each published file.
Entity counts are taken from the `entities` field, which is the
lossless view (nested entities are counted; see the note
below). Internal file names are deliberately not given here; each
source is described by its role in the annotation campaign.

| File | Adverts | Entities per label | Total entities | Source |
|---|---|---|---|---|
| `data/auto_adver.jsonl` | 605 | CONDITION 594 / HISTORIC 810 / SALES_OPTIONS 2,174 | 3,578 | the gold annotation |
| `data/auto_adver.char_iob2.jsonl` | 605 | as above (derived) | 3,578 | derived from `data/auto_adver.jsonl` |
| `data/folds/f1/train.jsonl` | 426 | CONDITION 429 / HISTORIC 615 / SALES_OPTIONS 1,534 | 2,578 | the paper's f1 train partition |
| `data/folds/f1/train_hf.json` | 426 | as above (IOB2 `ner_tags`) | 2,578 | derived from `data/folds/f1/train.jsonl` |
| `data/folds/f1/val.jsonl` | 92 | CONDITION 96 / HISTORIC 114 / SALES_OPTIONS 335 | 545 | the paper's f1 val partition |
| `data/folds/f1/val_hf.json` | 92 | as above (IOB2 `ner_tags`) | 545 | derived from `data/folds/f1/val.jsonl` |
| `data/folds/f1/test.jsonl` | 87 | CONDITION 69 / HISTORIC 81 / SALES_OPTIONS 305 | 455 | the paper's f1 test partition |
| `data/folds/f1/test_hf.json` | 87 | as above (IOB2 `ner_tags`) | 455 | derived from `data/folds/f1/test.jsonl` |
| `data/folds/f2/train.jsonl` | 426 | CONDITION 450 / HISTORIC 606 / SALES_OPTIONS 1,512 | 2,568 | the paper's f2 train partition |
| `data/folds/f2/train_hf.json` | 426 | as above (IOB2 `ner_tags`) | 2,568 | derived from `data/folds/f2/train.jsonl` |
| `data/folds/f2/val.jsonl` | 92 | CONDITION 51 / HISTORIC 102 / SALES_OPTIONS 355 | 508 | the paper's f2 val partition |
| `data/folds/f2/val_hf.json` | 92 | as above (IOB2 `ner_tags`) | 508 | derived from `data/folds/f2/val.jsonl` |
| `data/folds/f2/test.jsonl` | 87 | CONDITION 93 / HISTORIC 102 / SALES_OPTIONS 307 | 502 | the paper's f2 test partition |
| `data/folds/f2/test_hf.json` | 87 | as above (IOB2 `ner_tags`) | 502 | derived from `data/folds/f2/test.jsonl` |
| `data/folds/f3/train.jsonl` | 426 | CONDITION 412 / HISTORIC 555 / SALES_OPTIONS 1,549 | 2,516 | the paper's f3 train partition |
| `data/folds/f3/train_hf.json` | 426 | as above (IOB2 `ner_tags`) | 2,516 | derived from `data/folds/f3/train.jsonl` |
| `data/folds/f3/val.jsonl` | 92 | CONDITION 95 / HISTORIC 155 / SALES_OPTIONS 378 | 628 | the paper's f3 val partition |
| `data/folds/f3/val_hf.json` | 92 | as above (IOB2 `ner_tags`) | 628 | derived from `data/folds/f3/val.jsonl` |
| `data/folds/f3/test.jsonl` | 87 | CONDITION 87 / HISTORIC 100 / SALES_OPTIONS 247 | 434 | the paper's f3 test partition |
| `data/folds/f3/test_hf.json` | 87 | as above (IOB2 `ner_tags`) | 434 | derived from `data/folds/f3/test.jsonl` |
| `data/iaa/annotator_1.jsonl` | 113 | CONDITION 125 / HISTORIC 159 / SALES_OPTIONS 401 | 685 | independent pass by annotator 1 over the 113 IAA adverts |
| `data/iaa/annotator_1.char_iob2.jsonl` | 113 | as above (derived) | 685 | derived from `data/iaa/annotator_1.jsonl` |
| `data/iaa/annotator_2.jsonl` | 113 | CONDITION 126 / HISTORIC 168 / SALES_OPTIONS 419 | 713 | independent pass by annotator 2 over the 113 IAA adverts |
| `data/iaa/annotator_2.char_iob2.jsonl` | 113 | as above (derived) | 713 | derived from `data/iaa/annotator_2.jsonl` |
| `data/iaa/annotator_3.jsonl` | 113 | CONDITION 122 / HISTORIC 159 / SALES_OPTIONS 396 | 677 | independent pass by annotator 3 over the 113 IAA adverts |
| `data/iaa/annotator_3.char_iob2.jsonl` | 113 | as above (derived) | 677 | derived from `data/iaa/annotator_3.jsonl` |
| `predictions/crf/f1_test.jsonl` | 87 | CONDITION 56 / HISTORIC 51 / SALES_OPTIONS 273 | 380 | crf output for fold f1 test, converted from the raw token labels |
| `predictions/crf/f2_test.jsonl` | 87 | CONDITION 50 / HISTORIC 69 / SALES_OPTIONS 272 | 391 | crf output for fold f2 test, converted from the raw token labels |
| `predictions/crf/f3_test.jsonl` | 87 | CONDITION 49 / HISTORIC 84 / SALES_OPTIONS 215 | 348 | crf output for fold f3 test, converted from the raw token labels |
| `predictions/crf-pos/f1_test.jsonl` | 87 | CONDITION 0 / HISTORIC 7 / SALES_OPTIONS 7 | 14 | crf-pos output for fold f1 test, converted from the raw token labels |
| `predictions/crf-pos/f2_test.jsonl` | 87 | CONDITION 2 / HISTORIC 7 / SALES_OPTIONS 27 | 36 | crf-pos output for fold f2 test, converted from the raw token labels |
| `predictions/crf-pos/f3_test.jsonl` | 87 | CONDITION 3 / HISTORIC 4 / SALES_OPTIONS 9 | 16 | crf-pos output for fold f3 test, converted from the raw token labels |
| `predictions/crf-wo-pos/f1_test.jsonl` | 87 | CONDITION 58 / HISTORIC 51 / SALES_OPTIONS 261 | 370 | crf-wo-pos output for fold f1 test, converted from the raw token labels |
| `predictions/crf-wo-pos/f2_test.jsonl` | 87 | CONDITION 43 / HISTORIC 64 / SALES_OPTIONS 265 | 372 | crf-wo-pos output for fold f2 test, converted from the raw token labels |
| `predictions/crf-wo-pos/f3_test.jsonl` | 87 | CONDITION 49 / HISTORIC 75 / SALES_OPTIONS 205 | 329 | crf-wo-pos output for fold f3 test, converted from the raw token labels |
| `predictions/crf-2-3/f1_test.jsonl` | 87 | CONDITION 47 / HISTORIC 58 / SALES_OPTIONS 257 | 362 | crf-2-3 output for fold f1 test, converted from the raw token labels |
| `predictions/crf-2-3/f2_test.jsonl` | 87 | CONDITION 58 / HISTORIC 75 / SALES_OPTIONS 281 | 414 | crf-2-3 output for fold f2 test, converted from the raw token labels |
| `predictions/crf-2-3/f3_test.jsonl` | 87 | CONDITION 46 / HISTORIC 85 / SALES_OPTIONS 215 | 346 | crf-2-3 output for fold f3 test, converted from the raw token labels |
| `predictions/bert-base/f1_test.jsonl` | 87 | CONDITION 95 / HISTORIC 106 / SALES_OPTIONS 344 | 545 | bert-base output for fold f1 test, converted from the raw token labels |
| `predictions/bert-base/f2_test.jsonl` | 87 | CONDITION 95 / HISTORIC 119 / SALES_OPTIONS 382 | 596 | bert-base output for fold f2 test, converted from the raw token labels |
| `predictions/bert-base/f3_test.jsonl` | 87 | CONDITION 94 / HISTORIC 156 / SALES_OPTIONS 321 | 571 | bert-base output for fold f3 test, converted from the raw token labels |
| `predictions/bert-large/f1_test.jsonl` | 87 | CONDITION 70 / HISTORIC 101 / SALES_OPTIONS 358 | 529 | bert-large output for fold f1 test, converted from the raw token labels |
| `predictions/bert-large/f2_test.jsonl` | 87 | CONDITION 86 / HISTORIC 124 / SALES_OPTIONS 451 | 661 | bert-large output for fold f2 test, converted from the raw token labels |
| `predictions/bert-large/f3_test.jsonl` | 87 | CONDITION 105 / HISTORIC 146 / SALES_OPTIONS 308 | 559 | bert-large output for fold f3 test, converted from the raw token labels |
| `predictions/deberta-v3-base/f1_test.jsonl` | 87 | CONDITION 82 / HISTORIC 100 / SALES_OPTIONS 317 | 499 | deberta-v3-base output for fold f1 test, converted from the raw token labels |
| `predictions/deberta-v3-base/f2_test.jsonl` | 87 | CONDITION 92 / HISTORIC 119 / SALES_OPTIONS 366 | 577 | deberta-v3-base output for fold f2 test, converted from the raw token labels |
| `predictions/deberta-v3-base/f3_test.jsonl` | 87 | CONDITION 90 / HISTORIC 133 / SALES_OPTIONS 303 | 526 | deberta-v3-base output for fold f3 test, converted from the raw token labels |
| `predictions/deberta-v3-large/f1_test.jsonl` | 87 | CONDITION 69 / HISTORIC 110 / SALES_OPTIONS 335 | 514 | deberta-v3-large output for fold f1 test, converted from the raw token labels |
| `predictions/deberta-v3-large/f2_test.jsonl` | 87 | CONDITION 110 / HISTORIC 112 / SALES_OPTIONS 355 | 577 | deberta-v3-large output for fold f2 test, converted from the raw token labels |
| `predictions/deberta-v3-large/f3_test.jsonl` | 87 | CONDITION 113 / HISTORIC 144 / SALES_OPTIONS 306 | 563 | deberta-v3-large output for fold f3 test, converted from the raw token labels |
| `predictions/gemini-1.5-flash/f1_test.jsonl` | 87 | CONDITION 46 / HISTORIC 54 / SALES_OPTIONS 196 | 296 | stored gemini-1.5-flash output for fold f1 test |
| `predictions/gemini-1.5-flash/f2_test.jsonl` | 87 | CONDITION 43 / HISTORIC 80 / SALES_OPTIONS 265 | 388 | stored gemini-1.5-flash output for fold f2 test |
| `predictions/gemini-1.5-flash/f3_test.jsonl` | 87 | CONDITION 27 / HISTORIC 69 / SALES_OPTIONS 196 | 292 | stored gemini-1.5-flash output for fold f3 test |
| `predictions/gpt-4o/f1_test.jsonl` | 87 | CONDITION 45 / HISTORIC 49 / SALES_OPTIONS 220 | 314 | stored gpt-4o output for fold f1 test |
| `predictions/gpt-4o/f2_test.jsonl` | 87 | CONDITION 74 / HISTORIC 72 / SALES_OPTIONS 246 | 392 | stored gpt-4o output for fold f2 test |
| `predictions/gpt-4o/f3_test.jsonl` | 87 | CONDITION 43 / HISTORIC 70 / SALES_OPTIONS 206 | 319 | stored gpt-4o output for fold f3 test |
| `predictions/llama-3.1-70b/f1_test.jsonl` | 87 | CONDITION 30 / HISTORIC 27 / SALES_OPTIONS 204 | 261 | stored llama-3.1-70b output for fold f1 test |
| `predictions/llama-3.1-70b/f2_test.jsonl` | 87 | CONDITION 35 / HISTORIC 48 / SALES_OPTIONS 235 | 318 | stored llama-3.1-70b output for fold f2 test |
| `predictions/llama-3.1-70b/f3_test.jsonl` | 87 | CONDITION 23 / HISTORIC 44 / SALES_OPTIONS 178 | 245 | stored llama-3.1-70b output for fold f3 test |
| `predictions/qwen2.5-72b/f1_test.jsonl` | 87 | CONDITION 20 / HISTORIC 38 / SALES_OPTIONS 149 | 207 | stored qwen2.5-72b output for fold f1 test |
| `predictions/qwen2.5-72b/f2_test.jsonl` | 87 | CONDITION 36 / HISTORIC 40 / SALES_OPTIONS 156 | 232 | stored qwen2.5-72b output for fold f2 test |
| `predictions/qwen2.5-72b/f3_test.jsonl` | 87 | CONDITION 16 / HISTORIC 45 / SALES_OPTIONS 128 | 189 | stored qwen2.5-72b output for fold f3 test |

## Notes

* The corpus is **605 adverts / 104,382 tokens**; 26 adverts carry no entity.
* The three folds share one partition of the corpus: 426 train / 92 validation / 87 test adverts each, and the three test sets are disjoint and cover 261 distinct adverts.
* Every advert therefore appears in several fold files; the corpus total of 605 adverts is the number of distinct adverts in the release.
* The 113 IAA adverts are a subset of the corpus, annotated independently by three annotators. `annotator_2` is the annotator whose pass became the gold corpus, so those 113 records match `data/auto_adver.jsonl` exactly.
* 1 nested entity: an entity contained inside an entity of a different label, in 1 advert. `entities` keeps both; the flat `labels` / `ner_tags` / `char_labels` views keep the longer entity only.
