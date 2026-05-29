# BFCL evaluation

Minimal BFCL v4 single-turn evaluation for Qwen3-8B checkpoints.

The entry scripts:

```bash
bash eval/bfcl/eval_base_qwen3_8b.sh
FORCE_MERGE=1 bash eval/bfcl/eval_grpo_step630.sh
FORCE_MERGE=1 bash eval/bfcl/eval_ucsdpo_emajf_vllm_step630.sh
```

Each entry script:

1. resolves a HF model path;
2. merges verl FSDP actor shards if needed;
3. starts a local vLLM OpenAI-compatible server;
4. runs `bfcl generate --test-category single_turn`;
5. runs `bfcl evaluate --partial-eval`;
6. writes `summary_step*.txt`, `result/`, `score/`, and `logs/` under `EVAL_ROOT`.

Important: before each run, port 8000 must be free. If it is not:

```bash
curl http://127.0.0.1:8000/v1/models
```

kill the stale vLLM process or run on a different port:

```bash
PORT=8001 FORCE_MERGE=1 bash eval/bfcl/eval_grpo_step630.sh
```

BFCL displays all three runs as `Qwen3-8B (Prompt)` because we intentionally use
the official local-inference registry name `Qwen/Qwen3-8B`. Check the vLLM log to
verify the actual checkpoint:

```bash
grep '\\[serve\\] MODEL_PATH' /cfs_turbo/jiaxicao/evals/bfcl/grpo/logs/vllm_step630.log
```

For BFCL single-turn, use `data_non_live.csv` and `data_live.csv`; do not use
`Overall Acc` from `data_overall.csv` as the main number when `--partial-eval` is used.
