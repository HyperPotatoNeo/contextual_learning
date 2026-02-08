#!/bin/bash
# Self-contained eval script - all output to one log
set -e
exec > /pscratch/sd/s/siddart2/dspy_gepa/eval_v2.log 2>&1

cd /pscratch/sd/s/siddart2/dspy_gepa
source .venv/bin/activate
unset NCCL_SOCKET_IFNAME
export NCCL_DEBUG=WARN

echo "============================================"
echo "Eval Comparison v2"
echo "Started: $(date)"
echo "============================================"

# Start vLLM in background
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --dtype auto \
    --port 8000 \
    --trust-remote-code \
    --disable-log-requests \
    > /dev/null 2>&1 &
VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

echo "Waiting for vLLM..."
for i in $(seq 1 180); do
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "  vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  ERROR: vLLM died"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "  ERROR: vLLM not ready after 180s"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

echo "vLLM running. Starting eval..."
echo "============================================"

python -u eval_prompt_standalone.py \
    --prompt-file /pscratch/sd/s/siddart2/dspy_gepa/best_prompt_run8.txt \
    --num-examples 512 \
    --seed 22 \
    --max-tokens 8192 \
    --temperature 1.0 \
    --concurrency 32 \
    --methods prime_rl_student prime_rl_teacher

EVAL_EXIT=$?
echo ""
echo "============================================"
echo "Eval exit code: $EVAL_EXIT"
echo "Finished: $(date)"
echo "============================================"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "DONE"
