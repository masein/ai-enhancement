#!/usr/bin/env bash
# Benchmark a set of HuggingFace models on a set of tasks, with settings held
# constant so the numbers are comparable to each other.
#
#   ./scripts/run_benchmarks.sh smoke     # ~10 min total, 20 items per task — do this FIRST
#   ./scripts/run_benchmarks.sh full      # the real run
#
# Resumable: a model whose results file already exists is skipped, so you can kill
# this and restart without losing hours.
#
# Everything below is deliberately explicit rather than defaulted, because every one
# of these values changes the score and therefore belongs in the record.

set -euo pipefail

MODE="${1:-smoke}"
OUT="results"
LOGS="logs"
SEED=1234

mkdir -p "$OUT" "$LOGS"

# ---------------------------------------------------------------------------
# what to run
# ---------------------------------------------------------------------------
# "hf_id|is_instruct"  — is_instruct decides whether the chat template is applied.
# Applying it to a base model, or omitting it on an instruct model, is the single
# biggest source of wrong-looking scores.
MODELS=(
  "google/gemma-3-270m|base"
  "Qwen/Qwen3-0.6B|instruct"
  "Qwen/Qwen3-1.7B|instruct"
  "google/gemma-3-4b-it|instruct"
  "Qwen/Qwen3-4B-Instruct-2507|instruct"
)

# Task list with the few-shot counts that make results comparable to published
# numbers. These are the Open LLM Leaderboard v1 conventions; if your team uses
# different ones, change them HERE, once, and re-run everything.
declare -A NFEWSHOT=(
  [mmlu]=5
  [hellaswag]=10
  [arc_challenge]=25
  [winogrande]=5
  [piqa]=0
)
TASK_ORDER=(mmlu hellaswag arc_challenge winogrande piqa)

# vLLM is 5-10x faster than the HF backend for this. Set BACKEND=hf if vLLM is not
# installed or misbehaves on your driver version.
BACKEND="${BACKEND:-vllm}"
GPU_UTIL="${GPU_UTIL:-0.85}"
TP="${TP:-1}"                     # tensor_parallel_size — set to your GPU count

if [[ "$MODE" == "smoke" ]]; then
  LIMIT_ARG="--limit 20"
  OUT="$OUT/smoke"
  echo "SMOKE MODE: 20 items per task. Validates the whole loop; the numbers are NOT reportable."
else
  LIMIT_ARG=""
  echo "FULL RUN: this will take hours. Run it under tmux."
fi
mkdir -p "$OUT"

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
TASKS=$(IFS=,; echo "${TASK_ORDER[*]}")
echo "backend=$BACKEND  tasks=$TASKS  seed=$SEED  out=$OUT"
echo

for entry in "${MODELS[@]}"; do
  MODEL="${entry%%|*}"
  KIND="${entry##*|}"
  SAFE="${MODEL//\//__}"
  MODEL_OUT="$OUT/$SAFE"

  if compgen -G "$MODEL_OUT/**/results*.json" > /dev/null 2>&1; then
    echo "SKIP  $MODEL  (results already in $MODEL_OUT)"
    continue
  fi

  # Chat template ONLY for instruct models. --fewshot_as_multiturn is auto-enabled
  # alongside it by the harness.
  CHAT_ARG=""
  [[ "$KIND" == "instruct" ]] && CHAT_ARG="--apply_chat_template"

  if [[ "$BACKEND" == "vllm" ]]; then
    MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$TP,gpu_memory_utilization=$GPU_UTIL,max_model_len=4096"
    DEVICE_ARG=""
  else
    MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16"
    DEVICE_ARG="--device cuda:0"
  fi

  echo "RUN   $MODEL  ($KIND, backend=$BACKEND)"
  START=$(date +%s)

  # --log_samples writes every prompt and completion. It costs disk and it is the
  # first thing you will want when a number looks wrong.
  lm_eval \
    --model "$BACKEND" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size auto \
    --seed "$SEED" \
    --output_path "$MODEL_OUT" \
    --log_samples \
    $CHAT_ARG $DEVICE_ARG $LIMIT_ARG \
    2>&1 | tee "$LOGS/$SAFE.log"

  echo "DONE  $MODEL in $(( ($(date +%s) - START) / 60 )) min"
  echo
done

echo "=========================================================="
echo "All runs finished. Build the report:"
echo "  python scripts/report_lm_eval.py $OUT -o artifacts/benchmark_report.html --csv artifacts/benchmark.csv"
echo "=========================================================="
