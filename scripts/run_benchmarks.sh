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

# ---------------------------------------------------------------------------
# preflight: this is a SHARED GPU
# ---------------------------------------------------------------------------
# Two other jobs can be holding most of the card. Starting a run that OOMs wastes
# your time; starting one that succeeds by squeezing the card can OOM someone
# else's training run, which is worse. So: check first, refuse if it is tight, and
# size our own allocation from FREE memory rather than total.
preflight_gpu() {
  command -v nvidia-smi >/dev/null || { echo "no nvidia-smi — is this the right box?"; exit 1; }
  local free total used util
  free=$(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  used=$(( total - free ))
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)

  echo "GPU: ${free} MiB free of ${total} MiB   (${used} MiB in use by others, ${util}% busy)"
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "Other compute processes on this GPU right now:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/    /'
    ps -o user:16,pid,etime,comm --no-headers \
       -p "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, -)" \
       2>/dev/null | sed 's/^/    /' || true
  fi

  if (( free < MIN_FREE_MIB )); then
    echo
    echo "REFUSING TO START: only ${free} MiB free, need at least ${MIN_FREE_MIB} MiB."
    echo "  - wait for the other job, or"
    echo "  - lower MIN_FREE_MIB and pick smaller models, or"
    echo "  - run:  ./scripts/run_benchmarks.sh $MODE --wait   to block until the card frees up"
    exit 1
  fi

  # vLLM's gpu_memory_utilization is a fraction of TOTAL memory, so on a shared card
  # you must derive it from what is actually free or vLLM will try to take memory
  # that belongs to someone else.
  GPU_UTIL=$(python3 -c "print(min(0.90, round(($free * 0.88) / $total, 2)))")
  echo "Sizing vLLM to gpu_memory_utilization=$GPU_UTIL (derived from free memory)"
  echo
}

wait_for_gpu() {
  echo "Waiting for ${MIN_FREE_MIB} MiB to free up (Ctrl-C to give up)..."
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    (( free >= MIN_FREE_MIB )) && { echo "GPU free: ${free} MiB. Starting."; return; }
    printf "\r  %s  free=%s MiB  (need %s)   " "$(date +%H:%M:%S)" "$free" "$MIN_FREE_MIB"
    sleep 60
  done
}

MODE="${1:-smoke}"
OUT="results"
LOGS="logs"
SEED=1234

mkdir -p "$OUT" "$LOGS"

# Model weights go wherever HF_HOME points. On a shared box the root filesystem is
# usually small — set this to a big volume BEFORE the first download or you will
# fill / and annoy everyone.
echo "HF cache: ${HF_HOME:-$HOME/.cache/huggingface}   ($(df -h "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | awk 'NR==2{print $4}' || echo '?') free)"

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
TP="${TP:-1}"                     # tensor_parallel_size — set to your GPU count

# Minimum free VRAM before we are willing to start, in MiB. 10 GiB comfortably fits
# a 4B model in bf16 (8 GB weights) plus a small KV cache. Raise it for bigger models:
# roughly 2 GB per billion parameters, plus 2-4 GB of headroom.
MIN_FREE_MIB="${MIN_FREE_MIB:-10000}"
GPU_UTIL=""                       # computed by preflight_gpu from FREE memory

[[ "${2:-}" == "--wait" ]] && WAIT_FOR_GPU=1 || WAIT_FOR_GPU=0
[[ "$WAIT_FOR_GPU" == "1" ]] && wait_for_gpu
preflight_gpu

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
