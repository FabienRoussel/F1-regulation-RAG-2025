FROM python:3.12-slim-bookworm

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install dependencies using uv
RUN uv sync --frozen --no-cache

# Copy application code
COPY . .

# Set Python path
ENV PATH="/app/.venv/bin:$PATH"

# Default command
CMD ["python", "app.py"]
