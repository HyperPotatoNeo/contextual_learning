#!/bin/bash
# Inner script: runs inside the container
# Starts vLLM server, waits for it, then runs GEPA

set -e

cd /pscratch/sd/s/siddart2/dspy_gepa
source .venv/bin/activate
export WANDB_API_KEY=${WANDB_API_KEY:?Set WANDB_API_KEY env var}
unset NCCL_SOCKET_IFNAME
export NCCL_DEBUG=WARN

echo "============================================"
echo "GEPA Run 8 — Full Run with All Fixes"
echo "Started: $(date)"
echo "============================================"

# 1. Start vLLM server in background
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --dtype auto \
    --port 8000 \
    --trust-remote-code \
    --disable-log-requests \
    > /pscratch/sd/s/siddart2/dspy_gepa/vllm_run8.log 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# 2. Wait for vLLM to be ready
echo "Waiting for vLLM server to be ready..."
for i in $(seq 1 120); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "  vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  ERROR: vLLM process died. Check vllm_run8.log"
        cat /pscratch/sd/s/siddart2/dspy_gepa/vllm_run8.log | tail -30
        exit 1
    fi
    sleep 1
done

# Verify vLLM is actually running
if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "  ERROR: vLLM not ready after 120s. Check vllm_run8.log"
    tail -30 /pscratch/sd/s/siddart2/dspy_gepa/vllm_run8.log
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "vLLM server is running."
echo ""

# 3. Clear DSPy disk cache
echo "Clearing DSPy disk cache..."
rm -rf "$HOME/.dspy_cache"
mkdir -p "$HOME/.dspy_cache"
echo "  Cache cleared: $HOME/.dspy_cache"
echo ""

# 4. Run GEPA
echo "Starting GEPA optimization..."
echo "============================================"
python -u sokoban_gepa.py \
    --reflection-model claude-opus \
    --train-size 500 --val-size 200 --test-size 500 \
    --max-metric-calls 10000 \
    --reflection-minibatch-size 5 \
    --num-threads 32 \
    --temperature 1.0 \
    --max-tokens 8192

GEPA_EXIT=$?
echo ""
echo "============================================"
echo "GEPA finished with exit code: $GEPA_EXIT"
echo "Finished: $(date)"
echo "============================================"

# 5. Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "vLLM server stopped."
