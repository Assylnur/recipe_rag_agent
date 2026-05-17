FROM nvidia/cuda:12.5.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Python + compiler toolchain (needed for Triton JIT)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential gcc g++ make \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

# deps
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt

# project code (data excluded by .dockerignore)
COPY . /workspace

CMD ["bash"]
