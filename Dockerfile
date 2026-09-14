FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl redis-server \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3.13 -
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry env use python3.13 && poetry install --extras vllm

COPY . .

ENV KVROUTE_STRATEGY=round_robin
ENV REDIS_URL=redis://localhost:6379/0
ENV VLLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct

EXPOSE 8000 8001 8002 6379

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
