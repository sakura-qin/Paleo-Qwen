# Paleo-Qwen

Role-separated multimodal framework for oracle bone script (OBS) and Houma covenant script processing, built on Qwen3.5-9B.

## Contents

```
samples/demo_records.jsonl              one training example per task
prompts/p0.py                           initial prompt p0
prompts/pstar.py                        optimized prompt p*
eval/ancient_four_dimension_scoring.py  LLM judge, four dimensions, 1–4
```

## Usage

```bash
python eval/ancient_four_dimension_scoring.py \
  --input inputs.json --output scored.json \
  --judge-model deepseek-v4.1-flash
```

Requires `JUDGE_API_KEY`; `JUDGE_BASE_URL` optional.

