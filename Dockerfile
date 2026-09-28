# App image only. Ollama runs natively on the host: Docker on macOS can't use the M1 GPU,
# so a containerized Ollama would be CPU-only and miss the latency target (design.md 8.4).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# Dependencies first: this layer is cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev
COPY config.toml config-whole.toml ./

EXPOSE 8000
CMD ["notematch", "serve", "--host", "0.0.0.0", "--prepare"]
