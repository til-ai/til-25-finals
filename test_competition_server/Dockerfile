FROM debian:12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install gcc curl -y

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set venv for Python things
ENV VIRTUAL_ENV=/workspace/.venv
# Add uv and venv binaries to PATH
ENV PATH=${VIRTUAL_ENV}/bin:/root/.local/bin:${PATH}

# Install project
COPY .python-version .
COPY pyproject.toml .
COPY uv.lock .
COPY til-25-environment til-25-environment
RUN uv sync

# Copy source files
COPY src src
