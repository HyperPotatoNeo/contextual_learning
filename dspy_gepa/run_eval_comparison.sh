#!/bin/bash
# Run standalone eval comparison: standard vs GEPA teacher prompt
# 512 examples, seed 22, temp 1.0, max_tokens 8192
set -e

cd /pscratch/sd/s/siddart2/dspy_gepa
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
export NCCL_DEBUG=WARN

echo "============================================"
echo "Standalone Eval Comparison"
echo "Started: $(date)"
echo "============================================"

# 1. Check if vLLM is already running
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "vLLM already running."
else
    echo "Starting vLLM server..."
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --tensor-parallel-size 4 \
        --max-model-len 16384 \
        --dtype auto \
        --port 8000 \
        --trust-remote-code \
        --disable-log-requests \
        > /pscratch/sd/s/siddart2/dspy_gepa/vllm_eval.log 2>&1 &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID"

    echo "Waiting for vLLM server to be ready..."
    for i in $(seq 1 180); do
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "  vLLM ready after ${i}s"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "  ERROR: vLLM process died."
            tail -30 /pscratch/sd/s/siddart2/dspy_gepa/vllm_eval.log
            exit 1
        fi
        sleep 1
    done

    if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "  ERROR: vLLM not ready after 180s."
        tail -30 /pscratch/sd/s/siddart2/dspy_gepa/vllm_eval.log
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
fi

echo ""
echo "Running eval: standard (prime_rl_student) vs GEPA teacher (prime_rl_teacher)"
echo "  512 examples, seed 22, temp 1.0, max_tokens 8192"
echo "============================================"

python -u eval_prompt_standalone.py \
    --prompt-file /pscratch/sd/s/siddart2/dspy_gepa/best_prompt_run8.txt \
    --num-examples 512 \
    --seed 22 \
    --max-tokens 8192 \
    --temperature 1.0 \
    --concurrency 32 \
    --methods prime_rl_student prime_rl_teacher

echo ""
echo "============================================"
echo "Eval finished: $(date)"
echo "============================================"
