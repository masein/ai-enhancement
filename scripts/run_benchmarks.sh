#!/usr/bin/env bash
# Benchmark HuggingFace models on a shared GPU, with settings held constant so the
# numbers are comparable to each other.
#
#   ./scripts/run_benchmarks.sh smoke     # ~20 items/task — ALWAYS do this first
#   ./scripts/run_benchmarks.sh full      # the real run
#   ./scripts/run_benchmarks.sh full --wait   # block until the card frees up, then go
#
# TWO PROPERTIES THAT MATTER ON A SHARED BOX:
#
#   1. It runs whatever FITS RIGHT NOW. Each model carries an estimated VRAM cost;
#      anything that doesn't fit in currently-free memory is skipped with a reason.
#   2. It is RESUMABLE. A model whose results already exist is skipped.
#
# Together those mean you run the same command today with 3 GB free and get the
# small models, then run it again tomorrow with 30 GB free and it picks up exactly
# the ones it skipped. No editing, no bookkeeping.

set -euo pipefail

MODE="${1:-smoke}"
WAIT_FOR_GPU=0
[[ "${2:-}" == "--wait" ]] && WAIT_FOR_GPU=1

OUT_ROOT="${OUT_ROOT:-results}"
LOGS="${LOGS:-logs}"
SEED="${SEED:-1234}"

# ---------------------------------------------------------------------------
# what to run —  "hf_id | base|instruct | approx GB of VRAM needed"
# ---------------------------------------------------------------------------
# The GB figure is weights in bf16 (2 bytes/param) plus a rough allowance for
# activations and the KV cache. It only has to be good enough to sort models into
# "fits" and "doesn't"; the script prints what it skipped either way.
MODELS=(
  # ---- the Pythia scaling ladder: 14M -> 410M ---------------------------------
  # All trained on THE SAME DATA IN THE SAME ORDER, differing only in size. That
  # makes this a controlled experiment rather than a collection of models: the only
  # variable is parameter count, so the curve you get is a real scaling curve.
  # Apache 2.0, ungated, and the whole ladder fits in ~1.5 GB.
  "EleutherAI/pythia-14m|base|1.0"
  "EleutherAI/pythia-70m|base|1.0"
  "EleutherAI/pythia-160m|base|1.1"
  "EleutherAI/pythia-410m|base|1.5"

  # ---- modern small models, ungated -------------------------------------------
  "HuggingFaceTB/SmolLM2-135M|base|1.1"
  "HuggingFaceTB/SmolLM2-135M-Instruct|instruct|1.1"
  "HuggingFaceTB/SmolLM2-360M|base|1.4"
  "HuggingFaceTB/SmolLM2-360M-Instruct|instruct|1.4"

  # ---- gated: accept the licence at huggingface.co/google/gemma-3-270m first ---
  "google/gemma-3-270m|base|1.3"
  "google/gemma-3-270m-it|instruct|1.3"

  # ---- 600M: still sub-billion despite the "0.6B" name ------------------------
  "Qwen/Qwen3-0.6B|instruct|2.2"

  # ---- billions: commented out until the card frees up. Uncomment then; the
  #      script skips everything already finished, so nothing is repeated.
  # "EleutherAI/pythia-1b|base|3.5"
  # "EleutherAI/pythia-1.4b|base|4.5"
  # "HuggingFaceTB/SmolLM2-1.7B-Instruct|instruct|5"
  # "google/gemma-3-1b-it|instruct|4"
  # "Qwen/Qwen3-1.7B|instruct|5"
  # "google/gemma-3-4b-it|instruct|10"
  # "Qwen/Qwen3-4B-Instruct-2507|instruct|10"
)

# Few-shot counts pinned per task — the Open LLM Leaderboard v1 conventions, which
# is what makes your numbers comparable to published ones. Change them here, once,
# and re-run everything if you change them at all.
declare -A NFEWSHOT=([mmlu]=5 [hellaswag]=10 [arc_challenge]=25 [winogrande]=5 [piqa]=0)
TASK_ORDER=(${TASKS_OVERRIDE:-mmlu hellaswag arc_challenge winogrande piqa})

# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
# hf   — predictable memory, no engine startup cost. Right choice for small models
#        and for a card you are sharing.
# vllm — much faster on big models and long runs, but it grabs a large KV cache
#        block up front, which is antisocial when someone else is mid-training.
BACKEND="${BACKEND:-hf}"
TP="${TP:-1}"

# Fixed batch size, NOT `auto`. Auto-batching probes upward until something fails —
# on a shared card the thing that fails can be your colleague's job. Raise it when
# you have the card to yourself.
BATCH="${BATCH:-8}"

# Reduces allocator fragmentation, which matters when you are squeezing into a
# small slice of a busy card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Smallest model in the list decides whether it is worth starting at all.
MIN_FREE_MIB="${MIN_FREE_MIB:-1100}"

# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

wait_for_gpu() {
  echo "Waiting for ${MIN_FREE_MIB} MiB free (Ctrl-C to give up)..."
  while true; do
    local f; f=$(gpu_free_mib)
    (( f >= MIN_FREE_MIB )) && { echo; echo "GPU has ${f} MiB free. Starting."; return; }
    printf "\r  %s  free=%s MiB (need %s)   " "$(date +%H:%M:%S)" "$f" "$MIN_FREE_MIB"
    sleep 60
  done
}

command -v nvidia-smi >/dev/null || { echo "no nvidia-smi on this host"; exit 1; }
[[ "$WAIT_FOR_GPU" == "1" ]] && wait_for_gpu

FREE=$(gpu_free_mib)
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)

echo "=========================================================================="
echo "GPU     : ${FREE} MiB free of ${TOTAL} MiB   ($(( TOTAL - FREE )) MiB in use, ${UTIL}% busy)"
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | paste -sd, - || true)
if [[ -n "$PIDS" ]]; then
  echo "Sharing with:"
  ps -o user:16,pid,etime,comm --no-headers -p "$PIDS" 2>/dev/null | sed 's/^/          /' || true
fi
echo "HF cache: ${HF_HOME:-$HOME/.cache/huggingface} ($(df -h "${HF_HOME:-$HOME/.cache/huggingface}" 2>/dev/null | awk 'NR==2{print $4}' || echo '?') free)"
echo "Backend : $BACKEND   batch=$BATCH   seed=$SEED"

if (( FREE < MIN_FREE_MIB )); then
  echo
  echo "REFUSING TO START: ${FREE} MiB free, need ${MIN_FREE_MIB} for even the smallest model."
  echo "  ./scripts/run_benchmarks.sh $MODE --wait   # queue politely behind the running job"
  exit 1
fi

if [[ "$MODE" == "smoke" ]]; then
  LIMIT_ARG="--limit 20"; OUT="$OUT_ROOT/smoke"
  echo "Mode    : SMOKE (20 items/task — validates plumbing, numbers NOT reportable)"
else
  LIMIT_ARG=""; OUT="$OUT_ROOT/full"
  echo "Mode    : FULL — run this under tmux"
fi
mkdir -p "$OUT" "$LOGS"

# ---------------------------------------------------------------------------
# decide what fits
# ---------------------------------------------------------------------------
TASKS=$(IFS=,; echo "${TASK_ORDER[*]}")
RUNNABLE=(); TOO_BIG=(); ALREADY=()

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL KIND NEED_GB <<< "$entry"
  SAFE="${MODEL//\//__}"
  NEED_MIB=$(python3 -c "print(int($NEED_GB * 1024))")

  if compgen -G "$OUT/$SAFE/**/results*.json" > /dev/null 2>&1; then
    ALREADY+=("$MODEL"); continue
  fi
  if (( NEED_MIB > FREE )); then
    TOO_BIG+=("$MODEL (needs ~${NEED_GB} GB)"); continue
  fi
  RUNNABLE+=("$entry")
done

echo "Tasks   : $TASKS"
echo "--------------------------------------------------------------------------"
printf 'WILL RUN  (%d): %s\n' "${#RUNNABLE[@]}" "$(printf '%s ' "${RUNNABLE[@]%%|*}")"
(( ${#ALREADY[@]} )) && printf 'DONE      (%d): %s\n' "${#ALREADY[@]}" "${ALREADY[*]}"
if (( ${#TOO_BIG[@]} )); then
  printf 'TOO BIG   (%d): %s\n' "${#TOO_BIG[@]}" "$(printf '%s; ' "${TOO_BIG[@]}")"
  echo "            ^ re-run this exact command when the GPU frees up; the finished"
  echo "              models are skipped and these get picked up automatically."
fi
echo "=========================================================================="
echo

(( ${#RUNNABLE[@]} == 0 )) && { echo "Nothing to run right now."; exit 0; }

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
for entry in "${RUNNABLE[@]}"; do
  IFS='|' read -r MODEL KIND NEED_GB <<< "$entry"
  SAFE="${MODEL//\//__}"

  # Re-check free memory before each model: the neighbouring job can grow.
  NOW_FREE=$(gpu_free_mib)
  NEED_MIB=$(python3 -c "print(int($NEED_GB * 1024))")
  if (( NEED_MIB > NOW_FREE )); then
    echo "SKIP  $MODEL — free memory dropped to ${NOW_FREE} MiB (needs ~${NEED_MIB})"
    continue
  fi

  # Chat template ONLY on instruct models. Getting this wrong in either direction
  # moves scores by tens of points.
  CHAT_ARG=""; [[ "$KIND" == "instruct" ]] && CHAT_ARG="--apply_chat_template"

  if [[ "$BACKEND" == "vllm" ]]; then
    GPU_UTIL=$(python3 -c "print(min(0.90, round(($NOW_FREE * 0.85) / $TOTAL, 2)))")
    MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$TP,gpu_memory_utilization=$GPU_UTIL,max_model_len=4096"
    DEVICE_ARG=""
    echo "RUN   $MODEL  ($KIND, vllm, gpu_memory_utilization=$GPU_UTIL from ${NOW_FREE} MiB free)"
  else
    MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16"
    DEVICE_ARG="--device cuda:0"
    echo "RUN   $MODEL  ($KIND, hf, batch=$BATCH, ${NOW_FREE} MiB free)"
  fi

  START=$(date +%s)
  set +e
  lm_eval \
    --model "$BACKEND" \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH" \
    --seed "$SEED" \
    --output_path "$OUT/$SAFE" \
    --log_samples \
    $CHAT_ARG $DEVICE_ARG $LIMIT_ARG \
    2>&1 | tee "$LOGS/${SAFE}_${MODE}.log"
  STATUS=${PIPESTATUS[0]}
  set -e

  if (( STATUS != 0 )); then
    # One model failing must not kill the batch — you want the other results.
    echo "FAIL  $MODEL (exit $STATUS). See $LOGS/${SAFE}_${MODE}.log"
    grep -iE 'out of memory|no kernel image|CUDA error' "$LOGS/${SAFE}_${MODE}.log" | tail -3 || true
    echo
    continue
  fi
  echo "DONE  $MODEL in $(( ($(date +%s) - START) / 60 )) min"
  echo
done

echo "=========================================================================="
echo "Build the report:"
echo "  python scripts/report_lm_eval.py $OUT -o artifacts/benchmark_report.html --csv artifacts/benchmark.csv"
echo "=========================================================================="
