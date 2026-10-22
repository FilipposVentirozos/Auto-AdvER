# CRF baselines

The CRF is the best non-neural model in the paper's table and second overall behind DeBERTaV3-base.
It retrains in minutes on a laptop with no GPU. Four feature configurations correspond to four
rows of the results table.

| file | what it is |
|---|---|
| `features.py` | the four feature configurations; standard library only |
| `train_predict.py` | hyper-parameter search, training, and test predictions for one fold |
| `top_features.py` | the highest-weighted state features per label (the appendix table) |

## The four configurations

| `--config` | paper row | features |
|---|---|---|
| `all` | CRF | every token feature, neighbours at ±1 |
| `pos` | CRF PoS | `bias`, `postag`, `postag_detail`; the neighbours' `postag` only |
| `wo_pos` | CRF w/o PoS | `all` minus the coarse and fine PoS tags, token-level and neighbour |
| `2_3` | CRF 2n/3n | `all` plus the same neighbour block at ±2 and ±3 |

Token features in `all`: `bias`, lowercased form, last three and last two
characters, `is_upper`, `is_title`, `is_digit`, coarse PoS, fine PoS, shape,
`is_alpha`, `is_stop`. Neighbour features: lowercased form, coarse PoS,
`is_digit`. `BOS`/`EOS` mark a missing ±1 neighbour. `python src/crf/features.py`
prints all four.

## Running

```bash
python src/crf/train_predict.py --config all --seed 42 \
    --train data/folds/f1/train_hf.json \
    --val   data/folds/f1/val_hf.json \
    --test  data/folds/f1/test_hf.json \
    --out   work/crf/f1_pred.json \
    --save-params work/crf/f1_all_params.json \
    --save-model  work/crf/f1_all_model.pkl
```

Repeat for `--config {pos,wo_pos,2_3}` and folds 2 and 3, then convert the token
labels to character spans and score:

```bash
python src/data_tools/tags_to_spans.py \
    --labels work/crf/f1_pred.json \
    --gold   data/folds/f1/test.jsonl \
    --out    work/crf/f1_test.jsonl
python src/score.py models \
    --gold data/folds/{fold}/test.jsonl \
    --pred "CRF (re-run)=work/crf/{fold}_test.jsonl"
```

`--out` is a JSON array of per-document label lists, which `tags_to_spans.py`
reads. `--seed` fixes the search and `--save-params` records the winning `c1`
and `c2` with the library versions.

### Input text

The labels come from the Hugging Face-format folds, but the features come from
spaCy (`en_core_web_trf`) run over the raw advert text, read from the matching
`.jsonl` file (`--train-text`, `--val-text` and `--test-text` override it). When
spaCy's tokenisation of a document disagrees in length with its labels, the
document is dropped from training and validation with a warning; test documents
are kept so the output stays aligned with the gold fold. `--text-from tokens`
featurises `" ".join(tokens)` instead, which avoids most mismatches.

### Label scheme

The released folds carry IOB2 labels. `--label-scheme plain` (the default)
strips the `B-`/`I-` prefixes, as in the paper's runs, so two adjacent entities
of the same type merge. `--label-scheme iob2` keeps them separate.
`tags_to_spans.py` detects either.

## Hyper-parameter search

`sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100,
all_possible_transitions=True)`, searched by `RandomizedSearchCV` with
`c1 ~ expon(scale=0.5)`, `c2 ~ expon(scale=0.05)` and `n_iter=100`, scored by
weighted flat F1 over the non-`O` labels on a `PredefinedSplit` whose only
evaluation fold is the validation set. The final model is refitted on train and
validation together.

## Top features (the appendix)

The appendix table comes from a separate setting: `en_core_web_sm`,
`c1 = c2 = 0.1`, fold 1 only, features from `" ".join(tokens)`. These are
`top_features.py`'s defaults:

```bash
python src/crf/top_features.py --train data/folds/f1/train_hf.json    # the appendix
python src/crf/top_features.py --model work/crf/f1_all_model.pkl     # a searched model
```

## Dependencies

`features.py` needs only the standard library, and `--help` works on both
scripts without anything installed. Training needs:

```bash
pip install spacy sklearn-crfsuite scikit-learn scipy numpy
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
```
