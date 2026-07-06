# CUDA 12.8 toolkit + cuDNN 9 + nvcc, matching the host driver 570.x / CUDA 12.8.
# Provides the `nvcc` your current runner is missing so `cuda-preprocess` (cudarc) builds.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG RUNNER_VERSION=2.335.1
ARG DEBIAN_FRONTEND=noninteractive

# Build prerequisites for the cargo build plus what actions/runner needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git build-essential pkg-config libssl-dev \
      sudo jq unzip \
    && rm -rf /var/lib/apt/lists/*

# Fetch the runner and its OS dependencies (installdependencies.sh needs root).
RUN mkdir -p /opt/runner && cd /opt/runner \
    && curl -fsSLo runner.tar.gz \
       https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf runner.tar.gz && rm runner.tar.gz \
    && ./bin/installdependencies.sh

# The runner refuses to configure as root, so it runs as an unprivileged user.
RUN useradd -m -s /bin/bash runner \
    && echo 'runner ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/runner \
    && chown -R runner:runner /opt/runner
USER runner
WORKDIR /home/runner

# Rust stable with the components the GPU workflow drives: clippy (lint step) and
# llvm-tools-preview (cargo-llvm-cov instrumentation, which the workflow toolchain omits).
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable \
        --component clippy --component llvm-tools-preview

# nvcc (CUDA toolkit) and cargo on PATH for every job step.
ENV PATH="/home/runner/.cargo/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

COPY --chown=runner:runner entrypoint.sh /home/runner/entrypoint.sh
# chmod here so the image does not depend on the host file's mode bit.
RUN chmod +x /home/runner/entrypoint.sh
ENTRYPOINT ["/home/runner/entrypoint.sh"]
