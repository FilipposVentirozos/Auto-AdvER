#!/usr/bin/env bash
# Fine-tune an encoder on one Auto-AdvER fold with Hugging Face's run_ner.py.
#
# The encoder results in the paper (BERT-base, DeBERTaV3-base) were produced with
# the upstream token-classification example script, not with code in this
# repository. This wrapper only supplies the recorded hyper-parameters and the
# fold paths; you supply run_ner.py.
#
# Usage:
#   src/encoders/run_encoder.sh --model deberta-v3-base --fold 1 \
#       --data-dir data/folds --output-dir work/encoders \
#       --run-ner /path/to/transformers/examples/pytorch/token-classification/run_ner.py
#
#   --model      deberta-v3-base | bert-base | any HF checkpoint id
#   --fold       1 | 2 | 3
#   --data-dir   directory holding f{n}_{train,val,test}_hf.json (or f{n}/ subdirs)
#   --output-dir where the run writes; predictions land in <output-dir>/<tag>/predictions.json
#   --run-ner    path to run_ner.py; or set RUN_NER in the environment
#   --dry-run    print the command and exit without running anything
#
# Everything after `--` is passed through to run_ner.py, so you can override any
# hyper-parameter, e.g. `-- --seed 42 --max_seq_length 512`.
#
# See src/encoders/README.md for which hyper-parameters are recorded and which
# are unknown, and for the pinned transformers version.

set -euo pipefail

MODEL="deberta-v3-base"
FOLD="1"
DATA_DIR="data/folds"
OUTPUT_DIR="work/encoders"
RUN_NER="${RUN_NER:-}"
DRY_RUN=0
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)      MODEL="$2"; shift 2 ;;
    --fold)       FOLD="$2"; shift 2 ;;
    --data-dir)   DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run-ner)    RUN_NER="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    --)           shift; PASSTHROUGH=("$@"); break ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

# Resolve the checkpoint. The paper's appendix lists `bert-base-cased` and
# `deberta-v3-base`; the recorded run commands used `microsoft/deberta-v3-base`.
case "$MODEL" in
  deberta-v3-base) CHECKPOINT="microsoft/deberta-v3-base" ;;
  bert-base)       CHECKPOINT="google-bert/bert-base-cased" ;;
  *)               CHECKPOINT="$MODEL" ;;
esac
TAG="$(basename "$MODEL")-f${FOLD}"

# Fold files. Two layouts are supported: flat (f1_train_hf.json, as in the source
# repositories) and per-fold directories (folds/f1/train_hf.json, the release
# layout).
if [[ -f "${DATA_DIR}/f${FOLD}_train_hf.json" ]]; then
  TRAIN="${DATA_DIR}/f${FOLD}_train_hf.json"
  VALID="${DATA_DIR}/f${FOLD}_val_hf.json"
  TEST="${DATA_DIR}/f${FOLD}_test_hf.json"
else
  TRAIN="${DATA_DIR}/f${FOLD}/train_hf.json"
  VALID="${DATA_DIR}/f${FOLD}/val_hf.json"
  TEST="${DATA_DIR}/f${FOLD}/test_hf.json"
fi

if [[ -z "$RUN_NER" ]]; then
  cat >&2 <<'MSG'
run_ner.py was not given. Auto-AdvER does not vendor it; get it with

  pip install "transformers==4.46.0" datasets evaluate seqeval accelerate torch
  curl -O https://raw.githubusercontent.com/huggingface/transformers/v4.46.0/examples/pytorch/token-classification/run_ner.py

then pass --run-ner ./run_ner.py (or export RUN_NER=./run_ner.py).
MSG
  exit 2
fi

# --- Recorded hyper-parameters -------------------------------------------------
# From the paper's appendix listing and the project notes (see README.md). These
# are the ONLY settings that are on record; every other run_ner.py default
# applies, and the ones that matter are listed as unknown in the README.
CMD=(python "$RUN_NER"
  --model_name_or_path "$CHECKPOINT"
  --train_file "$TRAIN"
  --validation_file "$VALID"
  --test_file "$TEST"
  --text_column_name tokens
  --label_column_name ner_tags
  --output_dir "${OUTPUT_DIR}/${TAG}"
  --do_train --do_eval --do_predict
  --num_train_epochs 30
  --per_device_train_batch_size 16
  --per_device_eval_batch_size 16
  --load_best_model_at_end True
  --metric_for_best_model eval_loss
  --greater_is_better False
  --warmup_steps 50
  --eval_strategy epoch
  --save_strategy epoch
  --learning_rate 1e-5
)
# NOTE: the recorded flag spelling was `--evaluation_strategy`, which transformers
# deprecated in 4.41 and removed in 4.46 in favour of `--eval_strategy`. The two
# are equivalent; `--eval_strategy` is used here so the script runs on the pinned
# version.

if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
  CMD+=("${PASSTHROUGH[@]}")
fi

printf '%q ' "${CMD[@]}"; echo
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "(--dry-run: nothing executed)"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}/${TAG}"
"${CMD[@]}"

echo
echo "predictions: ${OUTPUT_DIR}/${TAG}/predictions.json"
echo "convert to character spans with:"
echo "  python src/data_tools/tags_to_spans.py \\"
echo "      --labels ${OUTPUT_DIR}/${TAG}/predictions.json \\"
echo "      --gold   ${TEST%_hf.json}.jsonl \\"
echo "      --out    ${OUTPUT_DIR}/${TAG}/f${FOLD}_test.jsonl"
