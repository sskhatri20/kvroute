#!/bin/bash
set -e

redis-server --daemonize yes

poetry run vllm serve "$VLLM_MODEL" --port 8001 --enable-prefix-caching &
poetry run vllm serve "$VLLM_MODEL" --port 8002 --enable-prefix-caching &

for port in 8001 8002; do
    until curl -sf "http://localhost:${port}/health" > /dev/null; do
        sleep 2
    done
done

poetry run uvicorn app:app --host 0.0.0.0 --port 8000 &

wait -n
