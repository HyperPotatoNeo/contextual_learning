#!/usr/bin/env bash
# =============================================================================
# GEPA Loop: Alternating RL Training + Prompt Optimization
#
# Runs N iterations of:
#   1. RL training (context distillation) for STEPS_PER_ITER steps
#   2. Merge LoRA weights into base model
#   3. Serve merged model with vLLM
#   4. Run GEPA prompt optimization
#   5. Use optimized prompt for next RL iteration
#
# All configuration variables can be overridden via environment variables.
#
# Usage:
#   bash experiments/context_distill/run_gepa_loop.sh
#   bash experiments/context_distill/run_gepa_loop.sh --dry-run
#   bash experiments/context_distill/run_gepa_loop.sh --start-iter 2  # resume from iter 2
#
# Parallel seeds (via env vars):
#   GEPA_SEED=42 OUTPUT_DIR=outputs/gepa_loop_s42 BASE_CONFIG=...s42.toml \
#     INFERENCE_GPU_IDS='[0]' TEACHER_GPU_IDS='[1]' TRAINER_GPU_IDS='[2,3]' \
#     GEPA_GPU_IDS=0,1,2,3 VLLM_PORT=8000 \
#     bash experiments/context_distill/run_gepa_loop.sh
# =============================================================================

set -euo pipefail

# Source .env if present (for WANDB_API_KEY, ANTHROPIC_API_KEY, etc.)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# =============================================================================
# Configuration — all overridable via environment variables
# =============================================================================

TOTAL_STEPS="${TOTAL_STEPS:-750}"
STEPS_PER_ITER="${STEPS_PER_ITER:-150}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gepa_loop}"
BASE_CONFIG="${BASE_CONFIG:-experiments/context_distill/rl_gepa_loop.toml}"
OVERLAY_GENERATOR="${OVERLAY_GENERATOR:-experiments/context_distill/generate_gepa_overlay.py}"
# INITIAL_PROMPT: env var > [gepa].initial_prompt in base config > default
if [ -z "${INITIAL_PROMPT:-}" ]; then
    INITIAL_PROMPT=$(python3 -c "
import sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open(sys.argv[1], 'rb') as f:
    c = tomllib.load(f)
print(c.get('gepa', {}).get('initial_prompt', ''))
" "$BASE_CONFIG" 2>/dev/null || true)
fi
INITIAL_PROMPT="${INITIAL_PROMPT:-dspy_gepa/teacher_prompt.txt}"

# Seed identifier (used in wandb names, logs)
GEPA_SEED="${GEPA_SEED:-}"

# GPU allocation for RL training
INFERENCE_GPU_IDS="${INFERENCE_GPU_IDS:-[0,1]}"
TEACHER_GPU_IDS="${TEACHER_GPU_IDS:-[2,3]}"
TRAINER_GPU_IDS="${TRAINER_GPU_IDS:-[4,5,6,7]}"

# GPU allocation for GEPA (vLLM serving merged model)
# GEPA_GPU_IDS: comma-separated GPU IDs for CUDA_VISIBLE_DEVICES (e.g., "0,1,2,3")
# If not set, vLLM uses all visible GPUs
GEPA_GPU_IDS="${GEPA_GPU_IDS:-}"
GEPA_TP="${GEPA_TP:-4}"

# GEPA configuration
REFLECTION_MODEL="${REFLECTION_MODEL:-claude-opus}"
GEPA_MAX_METRIC_CALLS="${GEPA_MAX_METRIC_CALLS:-10000}"
GEPA_TRAIN_SIZE="${GEPA_TRAIN_SIZE:-500}"
GEPA_VAL_SIZE="${GEPA_VAL_SIZE:-150}"
GEPA_NUM_THREADS="${GEPA_NUM_THREADS:-32}"
GEPA_MAX_TOKENS="${GEPA_MAX_TOKENS:-8192}"
GEPA_TEMPERATURE="${GEPA_TEMPERATURE:-1.0}"

# vLLM configuration for GEPA
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16884}"

# RL inference server port (teacher auto-assigned to port+1)
# Must differ between parallel runs to avoid port collisions
INFERENCE_PORT="${INFERENCE_PORT:-8000}"

# Model name (must match config)
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B-Instruct-2507}"

# =============================================================================
# Parse arguments
# =============================================================================

DRY_RUN=false
START_ITER=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --start-iter)
            START_ITER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run] [--start-iter N]"
            exit 1
            ;;
    esac
done

NUM_ITERS=$((TOTAL_STEPS / STEPS_PER_ITER))

# Log tag for parallel runs
LOG_TAG=""
if [ -n "$GEPA_SEED" ]; then
    LOG_TAG="[seed=$GEPA_SEED] "
fi

echo "${LOG_TAG}============================================================"
echo "${LOG_TAG}GEPA Loop Configuration"
echo "${LOG_TAG}============================================================"
echo "${LOG_TAG}  Seed:             ${GEPA_SEED:-default}"
echo "${LOG_TAG}  Total steps:      $TOTAL_STEPS"
echo "${LOG_TAG}  Steps per iter:   $STEPS_PER_ITER"
echo "${LOG_TAG}  Num iterations:   $NUM_ITERS"
echo "${LOG_TAG}  Start iteration:  $START_ITER"
echo "${LOG_TAG}  Output dir:       $OUTPUT_DIR"
echo "${LOG_TAG}  Base config:      $BASE_CONFIG"
echo "${LOG_TAG}  Initial prompt:   $INITIAL_PROMPT"
echo "${LOG_TAG}  Model:            $MODEL_NAME"
echo "${LOG_TAG}  Inference GPUs:   $INFERENCE_GPU_IDS"
echo "${LOG_TAG}  Teacher GPUs:     $TEACHER_GPU_IDS"
echo "${LOG_TAG}  Trainer GPUs:     $TRAINER_GPU_IDS"
echo "${LOG_TAG}  GEPA GPUs:        ${GEPA_GPU_IDS:-all visible}"
echo "${LOG_TAG}  GEPA TP:          $GEPA_TP"
echo "${LOG_TAG}  Inference port:   $INFERENCE_PORT (teacher: $((INFERENCE_PORT + 1)))"
echo "${LOG_TAG}  vLLM port:        $VLLM_PORT"
echo "${LOG_TAG}  Reflection model: $REFLECTION_MODEL"
echo "${LOG_TAG}  GEPA metric calls:$GEPA_MAX_METRIC_CALLS"
echo "${LOG_TAG}  Dry run:          $DRY_RUN"
echo "${LOG_TAG}============================================================"

# =============================================================================
# Helper functions
# =============================================================================

wait_for_vllm() {
    local port="$1"
    local max_wait=300  # 5 minutes
    local elapsed=0
    echo "${LOG_TAG}Waiting for vLLM server on port $port..."
    while ! curl -s "http://localhost:${port}/health" > /dev/null 2>&1; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "${LOG_TAG}ERROR: vLLM server did not start within ${max_wait}s"
            return 1
        fi
        echo "${LOG_TAG}  ... waiting ($elapsed/${max_wait}s)"
    done
    echo "${LOG_TAG}vLLM server is ready on port $port"
}

kill_vllm() {
    echo "${LOG_TAG}Stopping vLLM server..."
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        echo "${LOG_TAG}  vLLM server stopped (PID $VLLM_PID)"
    elif [ -n "${VLLM_PORT:-}" ]; then
        # Fallback: kill any vLLM on our port
        local pids
        pids=$(lsof -ti :"$VLLM_PORT" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "${LOG_TAG}  Killing processes on port $VLLM_PORT: $pids"
            echo "$pids" | xargs kill 2>/dev/null || true
            sleep 2
        fi
    fi
    unset VLLM_PID
}

# Cleanup on exit
trap kill_vllm EXIT

# =============================================================================
# Create output directory
# =============================================================================

mkdir -p "$OUTPUT_DIR"

# =============================================================================
# Main loop
# =============================================================================

for ((ITER=START_ITER; ITER<NUM_ITERS; ITER++)); do
    ITER_START=$((ITER * STEPS_PER_ITER))
    ITER_END=$(((ITER + 1) * STEPS_PER_ITER))

    echo ""
    echo "${LOG_TAG}============================================================"
    echo "${LOG_TAG}ITERATION $ITER: steps $ITER_START -> $ITER_END"
    echo "${LOG_TAG}============================================================"

    # ------------------------------------------------------------------
    # Determine teacher prompt for this iteration
    # ------------------------------------------------------------------
    if [ $ITER -eq 0 ]; then
        CURRENT_PROMPT="$INITIAL_PROMPT"
    else
        PREV_ITER=$((ITER - 1))
        CURRENT_PROMPT="${OUTPUT_DIR}/teacher_prompt_iter${ITER}.txt"
        if [ ! -f "$CURRENT_PROMPT" ] && [ "$DRY_RUN" = false ]; then
            echo "${LOG_TAG}ERROR: Expected prompt file not found: $CURRENT_PROMPT"
            echo "${LOG_TAG}  This should have been created by GEPA in iteration $PREV_ITER"
            exit 1
        fi
    fi
    echo "${LOG_TAG}Teacher prompt: $CURRENT_PROMPT"

    # ------------------------------------------------------------------
    # Step 1: Generate TOML overlay
    # ------------------------------------------------------------------
    OVERLAY_FILE="${OUTPUT_DIR}/overlay_iter${ITER}.toml"
    echo "${LOG_TAG}Generating overlay: $OVERLAY_FILE"

    # In dry-run mode, use initial prompt as placeholder if the real one doesn't exist yet
    OVERLAY_PROMPT="$CURRENT_PROMPT"
    if [ "$DRY_RUN" = true ] && [ ! -f "$CURRENT_PROMPT" ]; then
        OVERLAY_PROMPT="$INITIAL_PROMPT"
        echo "${LOG_TAG}  [DRY RUN] Using initial prompt as placeholder (real prompt not yet generated)"
    fi

    python "$OVERLAY_GENERATOR" \
        --iteration "$ITER" \
        --steps-per-iter "$STEPS_PER_ITER" \
        --prompt-file "$OVERLAY_PROMPT" \
        --output "$OVERLAY_FILE" \
        --base-config "$BASE_CONFIG"

    if [ "$DRY_RUN" = true ]; then
        echo "${LOG_TAG}[DRY RUN] Would run: rl @ $BASE_CONFIG @ $OVERLAY_FILE"
        echo "${LOG_TAG}[DRY RUN] Would merge LoRA: ${OUTPUT_DIR}/weights/step_${ITER_END} -> ${OUTPUT_DIR}/merged_iter${ITER}"
        echo "${LOG_TAG}[DRY RUN] Would start vLLM with ${OUTPUT_DIR}/merged_iter${ITER} (GEPA_GPU_IDS=${GEPA_GPU_IDS:-all})"
        echo "${LOG_TAG}[DRY RUN] Would run GEPA with prompt from $CURRENT_PROMPT"
        echo "${LOG_TAG}[DRY RUN] Would save optimized prompt to ${OUTPUT_DIR}/teacher_prompt_iter$((ITER+1)).txt"
        continue
    fi

    # ------------------------------------------------------------------
    # Step 2: Run RL training
    # ------------------------------------------------------------------
    echo ""
    echo "${LOG_TAG}--- RL Training (steps $ITER_START -> $ITER_END) ---"

    # Unset VLLM_PORT during RL training to prevent vLLM distributed init port
    # collisions between inference and teacher servers. VLLM_PORT is only needed
    # for the GEPA vLLM step later.
    VLLM_PORT_SAVED="$VLLM_PORT"
    unset VLLM_PORT || true

    # Build wandb ID args for run continuity (iterations > 0 reuse iter 0's runs)
    WANDB_ID_ARGS=()
    if [ $ITER -gt 0 ] && [ -f "$OUTPUT_DIR/wandb_ids.txt" ]; then
        read TRAINER_WANDB_ID ORCH_WANDB_ID < "$OUTPUT_DIR/wandb_ids.txt"
        echo "${LOG_TAG}Resuming W&B runs: trainer=$TRAINER_WANDB_ID orchestrator=$ORCH_WANDB_ID"
        WANDB_ID_ARGS=(
            --trainer.wandb.id "$TRAINER_WANDB_ID"
            --orchestrator.wandb.id "$ORCH_WANDB_ID"
        )
    fi

    rl \
        @ "$BASE_CONFIG" \
        @ "$OVERLAY_FILE" \
        --output-dir "$OUTPUT_DIR" \
        --inference-gpu-ids "$INFERENCE_GPU_IDS" \
        --teacher-gpu-ids "$TEACHER_GPU_IDS" \
        --trainer-gpu-ids "$TRAINER_GPU_IDS" \
        --inference.server.port "$INFERENCE_PORT" \
        ${WANDB_ID_ARGS[@]+"${WANDB_ID_ARGS[@]}"}

    # Restore VLLM_PORT for GEPA step
    export VLLM_PORT="$VLLM_PORT_SAVED"

    # Extract and save wandb run IDs after the first iteration we run (for run continuity)
    if [ ! -f "$OUTPUT_DIR/wandb_ids.txt" ]; then
        TRAINER_WANDB_ID=$(ls -td "$OUTPUT_DIR"/wandb/run-* 2>/dev/null | head -1 | grep -oP '[a-z0-9]{8}$' || true)
        ORCH_WANDB_ID=$(ls -td "$OUTPUT_DIR"/run_default/wandb/run-* 2>/dev/null | head -1 | grep -oP '[a-z0-9]{8}$' || true)
        if [ -n "$TRAINER_WANDB_ID" ] && [ -n "$ORCH_WANDB_ID" ]; then
            echo "$TRAINER_WANDB_ID $ORCH_WANDB_ID" > "$OUTPUT_DIR/wandb_ids.txt"
            echo "${LOG_TAG}Saved W&B run IDs: trainer=$TRAINER_WANDB_ID orchestrator=$ORCH_WANDB_ID"
        else
            echo "${LOG_TAG}WARNING: Could not extract W&B run IDs (trainer='$TRAINER_WANDB_ID' orch='$ORCH_WANDB_ID')"
            echo "${LOG_TAG}  Subsequent iterations will create new W&B runs"
        fi
    fi

    # Verify checkpoint exists
    STEP_DIR="${OUTPUT_DIR}/weights/step_${ITER_END}"
    if [ ! -d "$STEP_DIR" ]; then
        echo "${LOG_TAG}ERROR: Expected checkpoint not found: $STEP_DIR"
        exit 1
    fi
    echo "${LOG_TAG}Checkpoint verified: $STEP_DIR"

    # ------------------------------------------------------------------
    # Step 3: Merge LoRA weights
    # ------------------------------------------------------------------
    MERGED_DIR="${OUTPUT_DIR}/merged_iter${ITER}"
    echo ""
    echo "${LOG_TAG}--- Merging LoRA weights ---"
    echo "${LOG_TAG}  Source: $STEP_DIR"
    echo "${LOG_TAG}  Output: $MERGED_DIR"

    python dspy_gepa/merge_lora_weights.py "$STEP_DIR" "$MERGED_DIR"

    if [ ! -f "${MERGED_DIR}/config.json" ]; then
        echo "${LOG_TAG}ERROR: Merged model missing config.json: $MERGED_DIR"
        exit 1
    fi
    echo "${LOG_TAG}Merged model verified: $MERGED_DIR"

    # ------------------------------------------------------------------
    # Step 4: Start vLLM server with merged model
    # ------------------------------------------------------------------
    echo ""
    echo "${LOG_TAG}--- Starting vLLM server ---"
    echo "${LOG_TAG}  Model: $MERGED_DIR"
    echo "${LOG_TAG}  Port: $VLLM_PORT"
    echo "${LOG_TAG}  TP: $GEPA_TP"
    echo "${LOG_TAG}  GPUs: ${GEPA_GPU_IDS:-all visible}"

    # Restrict vLLM to specific GPUs if GEPA_GPU_IDS is set
    VLLM_ENV_PREFIX=""
    if [ -n "$GEPA_GPU_IDS" ]; then
        VLLM_ENV_PREFIX="CUDA_VISIBLE_DEVICES=$GEPA_GPU_IDS"
    fi

    env $VLLM_ENV_PREFIX python -m vllm.entrypoints.openai.api_server \
        --model "$MERGED_DIR" \
        --served-model-name "$MODEL_NAME" \
        --tensor-parallel-size "$GEPA_TP" \
        --port "$VLLM_PORT" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --disable-log-requests \
        > "${OUTPUT_DIR}/vllm_iter${ITER}.log" 2>&1 &
    VLLM_PID=$!
    echo "${LOG_TAG}  vLLM PID: $VLLM_PID"

    # ------------------------------------------------------------------
    # Step 5: Wait for vLLM readiness
    # ------------------------------------------------------------------
    wait_for_vllm "$VLLM_PORT"

    # ------------------------------------------------------------------
    # Step 6: Clear DSPy cache
    # ------------------------------------------------------------------
    echo "${LOG_TAG}Clearing DSPy cache..."
    rm -rf "$HOME/.dspy_cache"
    echo "${LOG_TAG}  DSPy cache cleared"

    # ------------------------------------------------------------------
    # Step 7: Save current prompt and run GEPA
    # ------------------------------------------------------------------
    # Save the prompt used for this iteration (for provenance)
    cp "$CURRENT_PROMPT" "${OUTPUT_DIR}/teacher_prompt_iter${ITER}.txt" 2>/dev/null || true

    GEPA_OUTPUT_DIR="${OUTPUT_DIR}/gepa_iter${ITER}"
    echo ""
    echo "${LOG_TAG}--- Running GEPA optimization ---"
    echo "${LOG_TAG}  Teacher prompt: $CURRENT_PROMPT"
    echo "${LOG_TAG}  Output dir: $GEPA_OUTPUT_DIR"
    echo "${LOG_TAG}  Reflection model: $REFLECTION_MODEL"

    python dspy_gepa/sokoban_gepa.py \
        --model "$MODEL_NAME" \
        --api-base "http://localhost:${VLLM_PORT}/v1" \
        --reflection-model "$REFLECTION_MODEL" \
        --teacher-prompt-file "$CURRENT_PROMPT" \
        --max-metric-calls "$GEPA_MAX_METRIC_CALLS" \
        --train-size "$GEPA_TRAIN_SIZE" \
        --val-size "$GEPA_VAL_SIZE" \
        --num-threads "$GEPA_NUM_THREADS" \
        --max-tokens "$GEPA_MAX_TOKENS" \
        --temperature "$GEPA_TEMPERATURE" \
        --output-dir "$GEPA_OUTPUT_DIR" \
        --skip-baseline

    # ------------------------------------------------------------------
    # Step 8: Kill vLLM server
    # ------------------------------------------------------------------
    kill_vllm

    # ------------------------------------------------------------------
    # Step 9: Extract optimized prompt for next iteration
    # ------------------------------------------------------------------
    # GEPA saves the optimized prompt as system_prompt.txt inside its output dir.
    # The output dir structure is: $GEPA_OUTPUT_DIR/gepa_<timestamp>/system_prompt.txt
    # Find the most recent one.
    GEPA_PROMPT=$(find "$GEPA_OUTPUT_DIR" -name "system_prompt.txt" -type f | sort | tail -n 1)

    if [ -z "$GEPA_PROMPT" ]; then
        echo "${LOG_TAG}ERROR: GEPA did not produce system_prompt.txt in $GEPA_OUTPUT_DIR"
        exit 1
    fi

    NEXT_ITER=$((ITER + 1))
    NEXT_PROMPT="${OUTPUT_DIR}/teacher_prompt_iter${NEXT_ITER}.txt"
    cp "$GEPA_PROMPT" "$NEXT_PROMPT"
    echo "${LOG_TAG}Optimized prompt saved: $NEXT_PROMPT"
    echo "${LOG_TAG}  Source: $GEPA_PROMPT"
    echo "${LOG_TAG}  Size: $(wc -c < "$NEXT_PROMPT") bytes"

    echo ""
    echo "${LOG_TAG}============================================================"
    echo "${LOG_TAG}ITERATION $ITER COMPLETE"
    echo "${LOG_TAG}============================================================"
done

echo ""
echo "${LOG_TAG}============================================================"
echo "${LOG_TAG}GEPA LOOP COMPLETE"
echo "${LOG_TAG}  Total iterations: $NUM_ITERS (starting from $START_ITER)"
echo "${LOG_TAG}  Final checkpoint: ${OUTPUT_DIR}/weights/step_${TOTAL_STEPS}"
echo "${LOG_TAG}  Final prompt: ${OUTPUT_DIR}/teacher_prompt_iter${NUM_ITERS}.txt"
echo "${LOG_TAG}============================================================"
