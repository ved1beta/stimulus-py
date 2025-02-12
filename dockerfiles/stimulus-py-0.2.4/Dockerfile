# syntax=docker/dockerfile:1.9

FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set uv environment variables
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_SYSTEM_PYTHON=1

# Install stimulus-py (ray will be installed as a dependency)
RUN uv pip install --system --no-build-isolation --compile --no-cache stimulus-py==0.2.4

# Create final image
FROM python:3.12-slim

RUN apt-get update && apt-get install -y procps && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/stimulus-* /usr/local/bin
COPY --from=builder /usr/local/bin/ray /usr/local/bin/ray

# Create non-root user
RUN useradd -m -s /bin/bash app
USER app

# Set working directory
WORKDIR /app

CMD ["python3"]
