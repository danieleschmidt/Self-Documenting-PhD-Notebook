# Multi-stage build for Self-Documenting PhD Notebook
FROM python:3.9-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[all]"

# Production stage
FROM python:3.9-slim as production

# Create non-root user
RUN groupadd -r researcher && useradd -r -g researcher researcher

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src

# Create directories for research data
RUN mkdir -p /data/research && \
    chown -R researcher:researcher /data

# Switch to non-root user
USER researcher

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1
ENV RESEARCH_DATA_DIR=/data/research

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import phd_notebook; print('OK')" || exit 1

# Default command
CMD ["sdpn", "--help"]

# Labels for metadata
LABEL maintainer="Daniel Schmidt"
LABEL description="Self-Documenting PhD Notebook - AI-powered research automation"
LABEL version="0.1.0"
LABEL org.opencontainers.image.source="https://github.com/danieleschmidt/Self-Documenting-PhD-Notebook"

# Development stage (optional)
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov black isort flake8 mypy pre-commit

# Install additional tools for development
RUN apt-get update && apt-get install -y \
    vim \
    curl \
    wget \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch back to researcher user
USER researcher

# Default to bash for development
CMD ["/bin/bash"]