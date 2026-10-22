# LLM baselines

Prompting code for the four decoder models in the paper: GPT-4o, Gemini 1.5
Flash, Llama 3.1 70B Instruct and Qwen 2.5 72B Instruct. The paper's numbers do
not need it: their outputs ship under `predictions/` and `src/score.py` scores
them.

| file | what it is |
|---|---|
| `icl.py` | the prompt: system instruction, per-label task turn, label definitions, `@@`/`##` span markers, and the alternating in-context chat built from a gold fold |
| `generate.py` | the backends and the `@@`/`##` marker parser |
| `predict.py` | the command-line driver: one fold, one model, one output file |
| `clear_test.py` | strips gold spans from a fold, so a model cannot see them |
| `env.py` | dependency-free `.env` loader |

## Credentials

Copy `.env.example` to `.env` at the repository root and fill in what you need.
Variables already set in the environment take precedence.

```
OPENAI_API_KEY=      # for --backend openai
HF_TOKEN=            # for gated checkpoints (Llama) with --backend hf
VERTEX_PROJECT=      # reserved for a Gemini backend
VERTEX_LOCATION=
```

## Running

```bash
# Llama or Qwen, as in the paper
python src/llm/predict.py \
    --gold-fold data/folds/f1/test.jsonl \
    --icl-from  data/folds/f1/train.jsonl \
    --backend hf --model Qwen/Qwen2.5-72B-Instruct \
    --n-icl 100 --icl-mode first --seed 42 \
    --out work/qwen2.5-72b/f1_test.jsonl

# GPT-4o, as in the paper
python src/llm/predict.py \
    --gold-fold data/folds/f1/test.jsonl \
    --icl-from  data/folds/f1/train.jsonl \
    --backend openai --n-icl 0 \
    --out work/gpt-4o/f1_test.jsonl

python src/score.py models \
    --gold data/folds/{fold}/test.jsonl \
    --pred "Qwen (re-run)=work/qwen2.5-72b/{fold}_test.jsonl"
```

- `--n-icl` caps the in-context examples: `100` with `--icl-mode first` is the
  Llama/Qwen setting, `0` (no cap) the GPT-4o setting. `--icl-mode random
  --seed S` samples them instead.
- `--labels` sets which labels are queried and in what order. The default,
  `SALES_OPTIONS,HISTORIC,CONDITION`, matters: when two labels claim the same
  start offset, the earlier label wins.
- `--out-format public` writes the release schema (`{id, entities}`, character
  offsets, `end` exclusive) instead of full Prodigy records.
- `--dry-run` prints the document count, the example count and the prompt size
  without contacting any service.
- `--backend openai` uses `gpt-4o-2024-11-20`. The `vertex` backend (Gemini) is
  not implemented; the Gemini predictions can be re-scored but not regenerated.

## GPUs

The 70B and 72B checkpoints need a multi-GPU machine (the paper used NVIDIA
H100s), loaded in bf16 with `device_map="auto"`. Set the cache and devices
yourself:

```bash
export HF_HOME=/path/with/space     # where checkpoints are cached
export CUDA_VISIBLE_DEVICES=...     # which GPUs to use, e.g. 0,1,2,3
```

## Dependencies

`icl.py`, `clear_test.py`, `env.py` and `predict.py --dry-run` need only the
standard library. Beyond that:

- `--backend openai`: `openai`
- `--backend hf`: `torch`, `transformers`, `accelerate`
- either backend, when a model does not copy the advert verbatim: `pytextspan`
- optional: `tqdm` for the progress bar
