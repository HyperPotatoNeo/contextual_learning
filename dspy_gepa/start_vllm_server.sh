#!/bin/bash
# Start vLLM server for GEPA optimization
#
# Usage:
#   ./start_vllm_server.sh                           # Default model
#   ./start_vllm_server.sh Qwen/Qwen3-4B-Instruct   # Custom model
#   MODEL=meta-llama/Llama-3.1-8B-Instruct ./start_vllm_server.sh
#
# Environment variables:
#   MODEL              - Model name (default: Qwen/Qwen3-4B-Instruct-2507)
#   TENSOR_PARALLEL    - Number of GPUs (default: 4)
#   PORT               - Server port (default: 8000)
#   MAX_MODEL_LEN      - Max context length (default: 8192)
#   DTYPE              - Data type (default: auto)

set -e

# Configuration with defaults
MODEL="${MODEL:-${1:-Qwen/Qwen3-4B-Instruct-2507}}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-4}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
DTYPE="${DTYPE:-auto}"

echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="
echo "Model: $MODEL"
echo "Tensor Parallel: $TENSOR_PARALLEL GPUs"
echo "Port: $PORT"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Dtype: $DTYPE"
echo "=========================================="

# Unset NCCL_SOCKET_IFNAME to avoid issues on some HPC systems
unset NCCL_SOCKET_IFNAME
export NCCL_DEBUG=WARN

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --port "$PORT" \
    --trust-remote-code \
    --disable-log-requests
