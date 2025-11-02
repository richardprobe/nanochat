# Multi-stage build for NanoChat web application
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for building rustbpe
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install uv for fast Python package management
RUN pip install uv

# Create virtual environment and install dependencies
RUN uv venv && \
    . .venv/bin/activate && \
    uv sync --extra gpu

# Build the rustbpe tokenizer
RUN . .venv/bin/activate && \
    maturin develop --release --manifest-path rustbpe/Cargo.toml

# Final stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the application and virtual environment from builder
COPY --from=builder /app /app

# Copy model artifacts from the host (will be mounted or copied during build)
# These paths will be customized based on your NANOCHAT_BASE_DIR
ARG NANOCHAT_BASE_DIR=/Users/richardhsu/Desktop/nanochat-artifacts
ENV NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR}

# Create directories for model artifacts
RUN mkdir -p ${NANOCHAT_BASE_DIR}/sft \
    ${NANOCHAT_BASE_DIR}/tokenizer \
    ${NANOCHAT_BASE_DIR}/report

# Set environment variables
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PORT=8080

# Expose the port Cloud Run expects
EXPOSE 8080

# Run the web application
CMD ["python", "-m", "scripts.chat_web", "--host", "0.0.0.0", "--port", "8080"]