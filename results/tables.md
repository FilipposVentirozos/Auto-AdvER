# Auto-AdvER results, regenerated

Every table below was produced by the command shown above it, run from the root of this repository, using only the files in `data/` and `predictions/`. Regenerate the whole file with:

```console
$ python3 results/make_tables.py
```

Scores use the paper's partial-match scheme: `P = (COR + 0.5·PAR) / (COR + INC + PAR + SPU)`, `R = (COR + 0.5·PAR) / (COR + INC + PAR + MIS)`. Counts are pooled within a fold, and the per-fold percentages are then averaged over folds f1–f3 with a population standard deviation. Gold spans are the corpus annotations carried by the fold files.

## Results table (paper matching, shipped predictions)

```console
$ python3 src/score.py models --gold 'data/folds/{fold}/test.jsonl' --pred 'BERT-base=predictions/bert-base/{fold}_test.jsonl' --pred 'BERT-large=predictions/bert-large/{fold}_test.jsonl' --pred 'DeBERTaV3-base=predictions/deberta-v3-base/{fold}_test.jsonl' --pred 'DeBERTaV3-large=predictions/deberta-v3-large/{fold}_test.jsonl' --pred 'GPT-4o=predictions/gpt-4o/{fold}_test.jsonl' --pred 'Gemini 1.5 Flash=predictions/gemini-1.5-flash/{fold}_test.jsonl' --pred 'Llama 3.1 70B it=predictions/llama-3.1-70b/{fold}_test.jsonl' --pred 'Qwen 2.5 72B it=predictions/qwen2.5-72b/{fold}_test.jsonl' --pred 'CRF=predictions/crf/{fold}_test.jsonl' --pred 'CRF PoS=predictions/crf-pos/{fold}_test.jsonl' --pred 'CRF w/o PoS=predictions/crf-wo-pos/{fold}_test.jsonl' --pred 'CRF 2n/3n=predictions/crf-2-3/{fold}_test.jsonl' --folds f1,f2,f3
```

| Model | Reproduced P / R / F1 | As published |
|---|---|---|
| BERT-base | 52.6 ± 1.7 / 63.5 ± 2.4 / 57.5 ± 1.2 | 52.6 ± 1.7 / 63.5 ± 2.4 / 57.5 ± 1.2 |
| BERT-large | 53.6 ± 2.5 / 65.0 ± 1.4 / 58.7 ± 2.1 | not in the paper's table |
| DeBERTaV3-base | 63.6 ± 1.2 / 72.4 ± 2.0 / 67.7 ± 0.8 | 63.6 ± 1.2 / 72.4 ± 2.0 / 67.7 ± 0.8 |
| DeBERTaV3-large | 63.9 ± 2.1 / 74.0 ± 2.6 / 68.5 ± 0.6 | not in the paper's table |
| GPT-4o | 73.8 ± 1.6 / 54.4 ± 1.3 / 62.6 ± 0.3 | 73.8 ± 1.6 / 54.4 ± 1.3 / 62.6 ± 0.3 |
| Gemini 1.5 Flash | 72.4 ± 2.1 / 49.9 ± 3.2 / 59.0 ± 2.0 | 72.4 ± 2.1 / 49.9 ± 3.2 / 59.0 ± 2.0 |
| Llama 3.1 70B it | 75.9 ± 1.1 / 44.5 ± 2.3 / 56.1 ± 2.1 | 75.9 ± 1.1 / 44.5 ± 2.3 / 56.1 ± 2.1 |
| Qwen 2.5 72B it | 76.8 ± 3.2 / 34.8 ± 1.1 / 47.9 ± 1.6 | 76.8 ± 3.2 / 34.8 ± 1.1 / 47.9 ± 1.6 |
| CRF | 72.5 ± 1.8 / 60.8 ± 1.5 / 66.1 ± 1.3 | 72.5 ± 1.8 / 60.8 ± 1.5 / 66.1 ± 1.3 |
| CRF PoS | 25.4 ± 8.6 / 1.1 ± 0.1 / 2.1 ± 0.2 | 25.4 ± 8.6 / 1.1 ± 0.1 / 2.1 ± 0.2 |
| CRF w/o PoS | 71.9 ± 0.9 / 58.5 ± 2.5 / 64.4 ± 1.7 | 71.9 ± 0.9 / 58.5 ± 2.5 / 64.4 ± 1.7 |
| CRF 2n/3n | 71.2 ± 0.8 / 60.1 ± 0.9 / 65.2 ± 0.8 | 71.2 ± 0.8 / 60.1 ± 0.9 / 65.2 ± 0.8 |

The paper reports exactly these figures, so the two columns agree by construction; the second column is typed into `results/make_tables.py` and therefore catches any drift between the paper and the release. The two large encoders were trained and then dropped from the paper's table, and are here for completeness.

The eight configurations that predict one label per token are built from `predictions/*/raw_token_labels/` with `src/data_tools/tags_to_spans.py`.

The scorer's own full output, including per-fold counts and the exact-match table, is what the command above prints; only the summary rows are reproduced here.

## Per-label results table

```console
$ python3 src/score.py models --gold 'data/folds/{fold}/test.jsonl' --pred 'DeBERTaV3-base=predictions/deberta-v3-base/{fold}_test.jsonl' --pred 'GPT-4o=predictions/gpt-4o/{fold}_test.jsonl' --pred 'Gemini 1.5 Flash=predictions/gemini-1.5-flash/{fold}_test.jsonl' --pred 'CRF=predictions/crf/{fold}_test.jsonl' --folds f1,f2,f3 --labels HISTORIC,SALES_OPTIONS,CONDITION --per-label
```

| Model | HISTORIC (reproduced / paper) | SALES_OPTIONS (reproduced / paper) | CONDITION (reproduced / paper) |
|---|---|---|---|
| DeBERTaV3-base | 58.1 ± 2.3 / 58.1 ± 2.3 | 74.0 ± 2.7 / 74.0 ± 2.7 | 57.8 ± 5.7 / 57.8 ± 5.7 |
| GPT-4o | 53.0 ± 2.2 / 53.0 ± 2.2 | 68.4 ± 1.9 / 68.4 ± 1.9 | 52.2 ± 5.2 / 52.2 ± 5.2 |
| Gemini 1.5 Flash | 51.6 ± 2.4 / 51.6 ± 2.4 | 66.7 ± 4.1 / 66.7 ± 4.1 | 38.5 ± 5.2 / 38.5 ± 5.2 |
| CRF | 59.5 ± 3.8 / 59.5 ± 3.8 | 73.2 ± 2.8 / 73.2 ± 2.8 | 46.5 ± 4.4 / 46.5 ± 4.4 |

These four are the paper's "top four performing models" by F1: DeBERTaV3-base, the CRF, GPT-4o and Gemini 1.5 Flash.

The label filter follows the paper's own definition: it applies to the side being iterated, so a gold CONDITION span may still match a differently labelled prediction and count as INC in the CONDITION pass. Filtering both sides instead moves these figures by up to 1.7 points.

## Where the partial matches are: CRF against GPT-4o

```console
$ python3 src/score.py models --gold 'data/folds/{fold}/test.jsonl' --pred 'CRF=predictions/crf/{fold}_test.jsonl' --pred 'GPT-4o=predictions/gpt-4o/{fold}_test.jsonl' --folds f1,f2,f3 --format json
```

| Model | COR | INC | PAR | MIS | SPU | Total |
|---|---|---|---|---|---|---|
| CRF | 714 | 63 | 261 | 353 | 128 | 1519 |
| GPT-4o | 648 | 58 | 218 | 467 | 104 | 1495 |

| Model | COR | INC | PAR | MIS | SPU |
|---|---|---|---|---|---|
| CRF | 47.0 | 4.1 | 17.2 | 23.2 | 8.4 |
| GPT-4o | 43.3 | 3.9 | 14.6 | 31.2 | 7.0 |

Counts are pooled over the three test folds, 261 adverts in all; the second table is each count as a percentage of that model's total. The CRF both finds more of the gold spans and gets more of their boundaries wrong: it carries the larger share of partial matches and the smaller share of misses, which is the trade the paper's discussion describes.

## Boundary direction of partial matches

```console
$ python3 src/score.py boundaries --gold 'data/folds/{fold}/test.jsonl' --pred 'DeBERTaV3-base=predictions/deberta-v3-base/{fold}_test.jsonl' --pred 'GPT-4o=predictions/gpt-4o/{fold}_test.jsonl' --pred 'Gemini 1.5 Flash=predictions/gemini-1.5-flash/{fold}_test.jsonl' --pred 'CRF=predictions/crf/{fold}_test.jsonl' --folds f1,f2,f3
```

| Model | PAR | Too wide | Too narrow | Staggered | In the paper's error analysis |
|---|---|---|---|---|---|
| DeBERTaV3-base | 340 | 143 (42.1%) | 179 (52.6%) | 18 (5.3%) | more often too narrow, 53% |
| GPT-4o | 218 | 127 (58.3%) | 83 (38.1%) | 8 (3.7%) | 58% too wide against 38% too narrow |
| Gemini 1.5 Flash | 169 | 83 (49.1%) | 80 (47.3%) | 6 (3.6%) | — |
| CRF | 261 | 152 (58.2%) | 87 (33.3%) | 22 (8.4%) | 58% too wide against 33% too narrow |

The paper's four top-performing models, with every partial match the scorer counts for them classified by which way its boundaries are wrong. A prediction is *too wide* when it reaches past the gold span on at least one side and stops short on neither, *too narrow* when it lies inside the gold span, and *staggered* when it begins and ends on the same side of the gold span. Shares are percentages of that model's own pooled PAR count, over the three test folds.

This is the table behind the paper's error analysis: “when the CRF or GPT-4o produce a partial match, the predicted span is more often too wide than too narrow (59% against 33% of the CRF’s partial matches, 58% against 38% of GPT-4o’s), whereas DeBERTaV3’s partial matches are more often too narrow (53%).” The shares here round to those integers. Gemini 1.5 Flash, which the paper does not quote, splits evenly.

A too-wide partial match always shares one boundary with its gold span, because the paper’s overlap test is one-sided: a prediction that strictly contains a gold span on *both* sides is not matched from the gold side at all and is counted MIS. The same quirk is documented on `Span.get_score` in `src/score.py` and pinned by `tests/test_matching.py`.

### Per fold

| Model | Fold | Docs | PAR | Too wide | Too narrow | Staggered |
|---|---|---|---|---|---|---|
| DeBERTaV3-base | f1 | 87 | 115 | 47 | 63 | 5 |
| DeBERTaV3-base | f2 | 87 | 137 | 59 | 70 | 8 |
| DeBERTaV3-base | f3 | 87 | 88 | 37 | 46 | 5 |
| GPT-4o | f1 | 87 | 63 | 28 | 32 | 3 |
| GPT-4o | f2 | 87 | 89 | 56 | 32 | 1 |
| GPT-4o | f3 | 87 | 66 | 43 | 19 | 4 |
| Gemini 1.5 Flash | f1 | 87 | 46 | 13 | 29 | 4 |
| Gemini 1.5 Flash | f2 | 87 | 75 | 42 | 32 | 1 |
| Gemini 1.5 Flash | f3 | 87 | 48 | 28 | 19 | 1 |
| CRF | f1 | 87 | 90 | 50 | 33 | 7 |
| CRF | f2 | 87 | 106 | 65 | 32 | 9 |
| CRF | f3 | 87 | 65 | 37 | 22 | 6 |

## Inter-annotator agreement

```console
$ python3 src/score.py iaa --annotators A1=data/iaa/annotator_1.jsonl A2=data/iaa/annotator_2.jsonl A3=data/iaa/annotator_3.jsonl
```

| Pair | Reproduced partial F1 | As published |
|---|---|---|
| A1–A2 | 88.27 | 88 |
| A1–A3 | 72.72 | 74 |
| A2–A3 | 72.63 | 73 |

A1 is the industry data scientist, A2 the guidelines lead and A3 the non-domain annotator; the files are the released annotation, which is what the paper's agreement section reports on. The published 74 for A1–A3 reproduces as 72.72; the other two agree to the printed precision.

### Per label, pooled over all six ordered annotator pairs

| Label | COR/INC/PAR/MIS/SPU | Exact P | Exact R | Exact F1 | Partial P | Partial R | Partial F1 |
|---|---|---|---|---|---|---|---|
| CONDITION | 564/53/90/39/39 | 75.60 | 75.60 | 75.60 | 81.64 | 81.64 | 81.64 |
| HISTORIC | 654/47/205/66/66 | 67.28 | 67.28 | 67.28 | 77.83 | 77.83 | 77.83 |
| SALES_OPTIONS | 1578/57/581/216/216 | 64.88 | 64.88 | 64.88 | 76.83 | 76.83 | 76.83 |

### Per label, share of each count type (%)

| Label | COR | INC | PAR | MIS | SPU |
|---|---|---|---|---|---|
| CONDITION | 71.8 | 6.8 | 11.5 | 5.0 | 5.0 |
| HISTORIC | 63.0 | 4.5 | 19.7 | 6.4 | 6.4 |
| SALES_OPTIONS | 59.6 | 2.2 | 21.9 | 8.2 | 8.2 |

## Mention families under the three labels

```console
$ python3 src/score.py families --corpus data/auto_adver.jsonl --gold 'data/folds/{fold}/test.jsonl' --pred 'DeBERTaV3-base=predictions/deberta-v3-base/{fold}_test.jsonl' --pred 'GPT-4o=predictions/gpt-4o/{fold}_test.jsonl' --pred 'Gemini 1.5 Flash=predictions/gemini-1.5-flash/{fold}_test.jsonl' --pred 'CRF=predictions/crf/{fold}_test.jsonl' --folds f1,f2,f3 --annotators A1=data/iaa/annotator_1.jsonl A2=data/iaa/annotator_2.jsonl A3=data/iaa/annotator_3.jsonl
```

### Gold spans of each mention family, by label (605 adverts)

| Family | Regex | CONDITION | HISTORIC | SALES_OPTIONS | Total |
|---|---|---|---|---|---|
| MOT | `\bmot\b` | 195 | 23 | 115 | 333 |
| service | `\bservic` | 42 | 139 | 388 | 569 |
| tyre | `\btyres?\b` | 34 | 62 | 5 | 101 |
| owner | `\bowners?\b` | 4 | 158 | 5 | 167 |
| mileage | `\bmile` | 103 | 39 | 37 | 179 |
| all other spans | — | 226 | 417 | 1630 | 2273 |
| every gold span | — | 594 | 810 | 2174 | 3578 |

44 of the 3578 gold spans match more than one family, so the family rows overlap; `all other spans` holds the spans that match none.

### Model behaviour on the families, folds f1, f2, f3 pooled (261 adverts)

| Model | Family | Gold spans | COR | INC | PAR | MIS | INC/(COR+INC+PAR) | MIS/(COR+INC+PAR+MIS) |
|---|---|---|---|---|---|---|---|---|
| DeBERTaV3-base | MOT | 133 | 75 | 20 | 37 | 1 | 15.2% | 0.8% |
| DeBERTaV3-base | service | 224 | 144 | 32 | 44 | 4 | 14.5% | 1.8% |
| DeBERTaV3-base | tyre | 42 | 17 | 6 | 14 | 5 | 16.2% | 11.9% |
| DeBERTaV3-base | owner | 59 | 47 | 2 | 10 | 0 | 3.4% | 0.0% |
| DeBERTaV3-base | mileage | 68 | 43 | 0 | 17 | 8 | 0.0% | 11.8% |
| DeBERTaV3-base | all other spans | 878 | 513 | 26 | 225 | 114 | 3.4% | 13.0% |
| DeBERTaV3-base | every gold span | 1391 | 835 | 84 | 340 | 132 | 6.7% | 9.5% |
| GPT-4o | MOT | 133 | 54 | 17 | 17 | 45 | 19.3% | 33.8% |
| GPT-4o | service | 224 | 115 | 20 | 23 | 66 | 12.7% | 29.5% |
| GPT-4o | tyre | 42 | 15 | 3 | 6 | 18 | 12.5% | 42.9% |
| GPT-4o | owner | 59 | 26 | 2 | 20 | 11 | 4.2% | 18.6% |
| GPT-4o | mileage | 68 | 37 | 1 | 10 | 20 | 2.1% | 29.4% |
| GPT-4o | all other spans | 878 | 404 | 17 | 143 | 314 | 3.0% | 35.8% |
| GPT-4o | every gold span | 1391 | 648 | 58 | 218 | 467 | 6.3% | 33.6% |
| Gemini 1.5 Flash | MOT | 133 | 44 | 23 | 18 | 48 | 27.1% | 36.1% |
| Gemini 1.5 Flash | service | 224 | 113 | 17 | 19 | 75 | 11.4% | 33.5% |
| Gemini 1.5 Flash | tyre | 42 | 8 | 3 | 6 | 25 | 17.6% | 59.5% |
| Gemini 1.5 Flash | owner | 59 | 40 | 0 | 5 | 14 | 0.0% | 23.7% |
| Gemini 1.5 Flash | mileage | 68 | 30 | 2 | 11 | 25 | 4.7% | 36.8% |
| Gemini 1.5 Flash | all other spans | 878 | 379 | 26 | 112 | 361 | 5.0% | 41.1% |
| Gemini 1.5 Flash | every gold span | 1391 | 612 | 69 | 169 | 541 | 8.1% | 38.9% |
| CRF | MOT | 133 | 67 | 22 | 26 | 18 | 19.1% | 13.5% |
| CRF | service | 224 | 129 | 20 | 35 | 40 | 10.9% | 17.9% |
| CRF | tyre | 42 | 19 | 1 | 7 | 15 | 3.7% | 35.7% |
| CRF | owner | 59 | 45 | 3 | 8 | 3 | 5.4% | 5.1% |
| CRF | mileage | 68 | 28 | 3 | 18 | 19 | 6.1% | 27.9% |
| CRF | all other spans | 878 | 430 | 19 | 169 | 260 | 3.1% | 29.6% |
| CRF | every gold span | 1391 | 714 | 63 | 261 | 353 | 6.1% | 25.4% |

`Gold spans` is COR + INC + PAR + MIS, the gold spans of that family the model was scored against; the `every gold span` row is the model's pooled count over every gold span and so agrees with the `score.py models` counts, SPU aside.

### Identically bounded spans two annotators labelled differently

| Pair | Adverts | Identically bounded spans | Same label | Different label | Share different |
|---|---|---|---|---|---|
| A1–A2 | 113 | 586 | 579 | 7 | 1.2% |
| A1–A3 | 113 | 423 | 406 | 17 | 4.0% |
| A2–A3 | 113 | 429 | 413 | 16 | 3.7% |

The first pair is the two experts. Their 7 disagreements, in corpus order:

| Span | Family | A1 | A2 |
|---|---|---|---|
| 12 month MOT | MOT | CONDITION | SALES_OPTIONS |
| 4x services | service | SALES_OPTIONS | HISTORIC |
| 93K | — | HISTORIC | CONDITION |
| 2 services | service | SALES_OPTIONS | HISTORIC |
| Recent service | service | HISTORIC | CONDITION |
| MOT for Hackney today | MOT | HISTORIC | CONDITION |
| 12 Months MOT | MOT | CONDITION | SALES_OPTIONS |

### What the models predict for the expert-disagreed spans in a test fold

| Span | Fold | Adjudicated gold | DeBERTaV3-base | GPT-4o | Gemini 1.5 Flash | CRF |
|---|---|---|---|---|---|---|
| 12 month MOT | f2 | SALES_OPTIONS | CONDITION (wide) | CONDITION (wide) | CONDITION (wide) | CONDITION (wide) |
| 4x services | f2 | HISTORIC | SALES_OPTIONS (exact) | — | SALES_OPTIONS (exact) | — |
| 2 services | f3 | HISTORIC | HISTORIC (exact) | HISTORIC (wide) | SALES_OPTIONS (exact) | HISTORIC (wide) |
| MOT for Hackney today | f3 | CONDITION | CONDITION (wide) | — | — | — |

The adjudicated gold is the label the fold file carries, so it shows which expert the gold sided with. A cell gives the label of the prediction the paper's greedy walk matches to the span, and whether its boundaries are `exact` or too wide, too narrow or staggered; `—` means no prediction overlapped the span.

### Candidate example snippets, 2–8 tokens, with their corpus count

| Family | Label | Candidate 1 | Candidate 2 | Candidate 3 |
|---|---|---|---|---|
| MOT | CONDITION | 12 months MOT (6) | 12 MONTHS MOT (5) | 12 month MOT (4) |
| MOT | HISTORIC | mot 0 advisory's (1) | recently had MOT (1) | recent clean MoT. (1) |
| MOT | SALES_OPTIONS | 12 Months MOT (6) | 12 months MOT (6) | 12 Month MOT (5) |
| service | CONDITION | just serviced (2) | Just been serviced (2) | Full service (1) |
| service | HISTORIC | Just been serviced (2) | serviced every year (2) | serviced every 6 months (2) |
| service | SALES_OPTIONS | full service history (42) | Full service history (33) | FULL SERVICE HISTORY (32) |
| tyre | CONDITION | good tyres (3) | Good tyres all round (2) | 4 Good Tyres (1) |
| tyre | HISTORIC | 4 new tyres (6) | New tyres (3) | new tyres (2) |
| tyre | SALES_OPTIONS | tyre changes (1) | Tyre Packages (1) | tyre repair kit (1) |
| owner | CONDITION | 1 owner (1) | ONE OWNER (1) | One previous owner (1) |
| owner | HISTORIC | 1 owner (21) | 1 OWNER (13) | 2 owners (11) |
| owner | SALES_OPTIONS | OWNERS MANUAL (1) | Owners Manual (1) | VW Owners manual (1) |
| mileage | CONDITION | 81446 miles (2) | 15,977 miles (2) | 45,977 miles (2) |
| mileage | HISTORIC | motorway miles (1) | MOTORWAY MILEAGE (1) | service at 47K miles (1) |
| mileage | SALES_OPTIONS | history and mileage checks (4) | mileage verification (2) | background and mileage checks (2) |

Candidates are gold spans of that family and label, ordered by descending corpus count and then by length and text, so the list is stable across runs; spans matching an email, URL, telephone, postcode or registration-plate pattern are dropped (0 in all). The paper's example table picks one snippet per cell from these.

Auto-AdvER's three labels are not three topics: they are three *readings* of a topic, and which reading applies is settled by the context around the mention. The five families above are blunt case-insensitive regular expressions over a span's text, chosen because they recur, and each of the five occurs under all three labels in the corpus — an MOT that the car holds now is CONDITION, one it passed last month is HISTORIC, and one the dealer will supply with the sale is SALES_OPTIONS. A span belongs to every family whose regex it matches, so the family rows overlap slightly and do not sum to the total; the counts are over the released corpus.

The model table is the paper's four top-performing models, scored exactly as the results table scores them, with each gold span's outcome attributed to its families. Two things show up. *Labelling a family is harder than labelling anything else*: every model's label-confusion rate on MOT (14.5% for DeBERTaV3-base, 19.5% for GPT-4o, 20.0% for the CRF, 27.7% for Gemini 1.5 Flash) is between four and seven times its rate on the spans in no family (3.1–5.3%), and on service (11.5–13.8%) two to four times. *Finding one is easier*, at least for the two models that find most spans: DeBERTaV3-base misses 0.8% of MOT spans against 12.7% of the others, and the CRF 12.9% against 29.3%. The exception is tyre, which is both confusable and easily missed — Gemini 1.5 Flash misses 59.5% of the 42 tyre spans — and whose label distribution inverts the corpus’s: 5 SALES_OPTIONS against 96 elsewhere, where the corpus as a whole is 61% SALES_OPTIONS.

The disagreement tables read the annotator files the agreement section uses. Restricting to spans the two annotators bounded *identically* separates the label decision from the boundary decision: the two experts placed 586 spans on exactly the same tokens and chose a different label for 7 of them, 1.2%, against 16 of 429 (3.7%) for A2 against the non-domain annotator A3. Six of the experts' seven are in the MOT and service families, and the three shapes are the contextual distinction itself — a warranted MOT read as the car's current state or as something the sale includes, a recent service read as a past event or as the car's present condition, a count of services read as a history or as a package on offer. Four of the seven adverts are in a test fold, and the models split the same way the experts did: all four read “12 month MOT” as CONDITION where the adjudicated gold says SALES_OPTIONS, and on “2 services” Gemini 1.5 Flash says SALES_OPTIONS where the other three and the gold say HISTORIC.

The snippet candidates are the raw material for the paper's example table, not the table itself: the ordering is fixed so the list does not move between runs, and a human picks one cell at a time. Everything printed is a gold span of the released corpus, 2–8 tokens long, with newlines flattened to spaces and anything matching an e-mail, URL, telephone, postcode or registration-plate pattern dropped.
