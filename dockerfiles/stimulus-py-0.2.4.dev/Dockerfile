# syntax=docker/dockerfile:1.9

############################
# Stage 1: Builder         #
############################
FROM python:3.12-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set uv environment variables for optimized build
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_SYSTEM_PYTHON=1

# Set working directory for building
WORKDIR /src

# Copy the entire repository (source code, pyproject.toml, etc.) into the container
COPY . .

# Install build-system requirements for pdm backend
RUN uv pip install pdm-backend pdm

# Set fallback version for SCM
ENV PDM_BUILD_SCM_VERSION=0.2.4.dev

# Build and install stimulus-py from local source using uv pip install
RUN uv pip install --system --no-build-isolation --compile --no-cache .

############################
# Stage 2: Final Image     #
############################
FROM python:3.12-slim

# Install minimal runtime dependency
RUN apt-get update && apt-get install -y procps \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copy stimulus CLI commands (stimulus-*) and the ray command
COPY --from=builder /usr/local/bin/stimulus-* /usr/local/bin/
COPY --from=builder /usr/local/bin/ray /usr/local/bin/ray

# Create a non-root user for running the app
RUN useradd -m -s /bin/bash app
USER app

# Set working directory
WORKDIR /app

# Default command
CMD ["python3"] 