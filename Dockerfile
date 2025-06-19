FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl bzip2 git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy installer script
COPY Miniforge3-Linux-x86_64.sh /workspace/

# Add to PATH after install
ENV PATH="/opt/miniforge/bin:$PATH"

# Install mamba after miniforge (done via postCreateCommand in devcontainer.json)

CMD [ "bash" ]
