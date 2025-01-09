HOST=localhost
PORT=8787
MODEL_PATH="/path/to/Qwen/Qwen2.5-3B-Instruct"
CMD="
    vllm serve \
        ${MODEL_PATH} \
        --host ${HOST} \
        --port ${PORT} \
        --api-key token-abc123
        "
echo "Starting VLLM server on ${HOST}:${PORT}..."
bash -c "${CMD}" &

sleep 60 # Wait for the server to start