# Dockerfile.pytorch
# Flexible Dockerfile to install PyTorch (CPU or CUDA wheels) and Jupyter Lab.
# Usage examples are in README_DOCKER_PYTORCH.md
from nvcr.io/nvidia/pytorch:25.09-py3

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
setuptools \
wheel \
geopandas



# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"

# # Install wget to fetch Miniconda
# RUN apt-get update && \
#     apt-get install -y wget && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Install Miniconda on x86 or ARM platforms
# RUN arch=$(uname -m) && \
#     if [ "$arch" = "x86_64" ]; then \
#     MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
#     elif [ "$arch" = "aarch64" ]; then \
#     MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
#     else \
#     echo "Unsupported architecture: $arch"; \
#     exit 1; \
#     fi && \
#     wget $MINICONDA_URL -O miniconda.sh && \
#     mkdir -p /root/.conda && \
#     bash miniconda.sh -b -p /root/miniconda3 && \
#     rm -f miniconda.sh

# RUN conda --version










# ARG PYTHON_VERSION=3.11
# FROM python:${PYTHON_VERSION}-slim

# LABEL org.opencontainers.image.title="pytorch-jupyter"
# LABEL org.opencontainers.image.description="Lightweight image with PyTorch and Jupyter Lab (configurable for CPU or CUDA wheels)."

# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONUNBUFFERED=1 \
#     PIP_NO_CACHE_DIR=1

# # Choose installation target at build-time. Set TORCH_CUDA=cpu (default) or e.g. cu118
# ARG TORCH_CUDA=cpu

# WORKDIR /workspace

# # Install system deps
# RUN apt-get update \
#  && apt-get install -y --no-install-recommends \
#     build-essential \
#     git \
#     curl \
#     ca-certificates \
#     libglib2.0-0 \
#     libsm6 \
#     libxext6 \
#     libxrender1 \
#  && rm -rf /var/lib/apt/lists/*

# # Upgrade pip and tooling
# RUN python -m pip install --upgrade pip setuptools wheel

# # Install PyTorch wheels. Default: CPU wheels. To build CUDA-enabled wheels, pass --build-arg TORCH_CUDA=cu118 (or desired tag)
# RUN set -eux; \
#     if [ "${TORCH_CUDA}" = "cpu" ]; then \
#         pip install --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
#     else \
#         pip install --no-deps torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${TORCH_CUDA}; \
#     fi

# # Install Jupyter and common extras
# RUN pip install --upgrade jupyterlab notebook ipywidgets jupyter-server-proxy

# # Create a non-root user for safety (optional) and set permissions
# ARG NB_USER=developer
# ARG NB_UID=1000
# RUN useradd -m -s /bin/bash -u ${NB_UID} ${NB_USER} || true \
#  && chown -R ${NB_USER}:${NB_USER} /workspace

# USER ${NB_USER}

# EXPOSE 8888

# VOLUME ["/workspace"]

# # Default command: start JupyterLab without token (for local dev). For production, override the CMD and add a token/password.
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--ServerApp.token=", "--ServerApp.allow_origin='*'", "--ServerApp.allow_remote_access=True"]
