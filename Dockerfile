FROM nvcr.io/nvidia/pytorch:25.11-py3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      vim \
      git \
      tmux \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY lerobot /opt/lerobot
RUN python -m pip install --no-cache-dir /opt/lerobot

#COPY pyproject.toml /workspace/pyproject.toml
#RUN python -m pip install --no-cache-dir /workspace

RUN rm -rf /opt/lerobot /workspace/*

WORKDIR /workspace
RUN mkdir -p /workspace/.cache/xdg /workspace/.cache/hf /workspace/.cache/hf_datasets \
 && chmod -R 775 /workspace/.cache

# install codex in Dockerfile
RUN curl -sSf https://get.volta.sh | bash -s -- --skip-setup && \
    export VOLTA_HOME=/root/.volta && export PATH=$VOLTA_HOME/bin:$PATH && \
    volta install node@22 @openai/codex && \
    mkdir -p /root/.codex && \
    echo "export VOLTA_HOME=/root/.volta && export PATH=\$VOLTA_HOME/bin:\$PATH" >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-lc"]
CMD ["bash"]