# Use the same base as our Singularity definition
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install system dependencies
# openssh-server is CRITICAL for Vast.ai if the image doesn't have it
RUN apt-get update && apt-get install -y \
    git \
    openssh-server \
    sudo \
    curl \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Setup SSH
# Vast.ai mounts the public key to /root/.ssh/authorized_keys
# We just need to ensure the directory exists and permissions are right (optional but good practice)
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh

# Install Python dependencies matching our requirements.txt
# We install them into the system python
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    scipy \
    scikit-learn \
    einops \
    ftfy \
    regex \
    Pillow \
    umap-learn \
    matplotlib \
    seaborn \
    plotly \
    streamlit \
    opencv-python \
    jsonlines \
    tqdm \
    pyyaml \
    types-PyYAML

# Set environment variables
ENV LC_ALL=C.UTF-8
ENV HF_HOME=/tmp/huggingface

# Standard Vast.ai entrypoint usually involves starting sshd
# But Vast often overrides this. We'll set a sensible default.
CMD ["/usr/sbin/sshd", "-D"]
