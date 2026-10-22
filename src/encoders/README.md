# Encoder baselines (BERT-base, DeBERTaV3-base)

The encoder rows of the results table were produced with Hugging Face's
`run_ner.py` token-classification example. `run_encoder.sh` wraps it with the
paper's hyper-parameters and the fold paths; you supply `run_ner.py`.

## Getting `run_ner.py`

```bash
pip install "transformers==4.46.0" datasets evaluate seqeval accelerate torch
curl -O https://raw.githubusercontent.com/huggingface/transformers/v4.46.0/examples/pytorch/token-classification/run_ner.py
```

4.46.0 is the release nearest to the `transformers` checkout the paper's runs
used.

## Running one fold

```bash
export RUN_NER=./run_ner.py
src/encoders/run_encoder.sh --model deberta-v3-base --fold 1 \
    --data-dir data/folds --output-dir work/encoders
src/encoders/run_encoder.sh --model bert-base --fold 1 \
    --data-dir data/folds --output-dir work/encoders
```

One fold and one model per call; repeat for folds 2 and 3 and average.
`--dry-run` prints the command without running it, and anything after `--` is
passed through to `run_ner.py` (for example `-- --seed 42`). The script reads
`f{n}/{train,val,test}_hf.json` and writes `predictions.json`, one label list per
advert, into `<output-dir>/<model>-f<n>/`.

`--model bert-base` uses `google-bert/bert-base-cased`, as the paper lists; pass
`--model google-bert/bert-base-uncased` to follow the recorded command lines
instead.

## Converting `predictions.json` to spans

```bash
python src/data_tools/tags_to_spans.py \
    --labels work/encoders/deberta-v3-base-f1/predictions.json \
    --gold   data/folds/f1/test.jsonl \
    --out    work/encoders/deberta-v3-base-f1/f1_test.jsonl
```

Keep the predictions in the order of the gold fold. `run_ner.py` writes no label
for whitespace-only tokens (`"\n"`, `" "`), so its lists are shorter than the
advert's token list; `tags_to_spans.py` detects this from the lengths and skips
those tokens. An advert longer than `--max_seq_length` sub-words loses its tail,
and the converter warns about it; any other length mismatch is refused.

## Hyper-parameters

| flag | value |
|---|---|
| `--num_train_epochs` | 30 |
| `--per_device_train_batch_size` | 16 |
| `--per_device_eval_batch_size` | 16 |
| `--load_best_model_at_end` | True |
| `--metric_for_best_model` | eval_loss |
| `--greater_is_better` | False |
| `--warmup_steps` | 50 |
| `--eval_strategy` | epoch |
| `--save_strategy` | epoch |
| `--learning_rate` | 1e-5 |

Everything else is `run_ner.py`'s default.

## Label format

The released `*_hf.json` files carry IOB2 labels; the paper's runs trained on
plain labels (`CONDITION` on every token of an entity). To train in that setting,
regenerate the folds with:

```bash
python src/data_tools/to_hf.py --plain-labels \
    --input data/folds/f1/train.jsonl --out work/folds/f1_train_hf.json
```
