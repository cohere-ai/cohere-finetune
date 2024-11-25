FROM docker.io/library/ubuntu:22.04@sha256:a6d2b38300ce017add71440577d5b0a90460d0e57fd7aec21dd0d1b0761bbfb2 as cuda-base

ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"
ENV NV_CUDA_CUDART_VERSION 12.4.127-1
ENV NV_CUDA_CUBLAS_VERSION 12.4.5.8-1
ENV CUDA_VERSION 12.4.1
ENV NV_CUDNN_VERSION 9.1.0.70-1
ENV NV_NCCL_VER 2.21.5-1+cuda12.4

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

COPY ./docker/cuda.repo /etc/yum.repos.d/cuda.repo

RUN apt-get update && apt-get install -y --no-install-recommends \
        gnupg2=2.2.27-3ubuntu2.1 \
        curl=7.81.0-1ubuntu1.19 \
        ca-certificates=20240203~22.04.1 \
    && curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
        make \
        cuda-compat-12-4 \
        cuda-nvcc-12-4 \
        cuda-profiler-api-12-4 \
        cuda-nvtx-12-4 \
        cuda-nvrtc-12-4 \
        libcudnn9-cuda-12=${NV_CUDNN_VERSION} \
        libcudnn9-dev-cuda-12=${NV_CUDNN_VERSION} \
        cuda-cudart-12-4=${NV_CUDA_CUDART_VERSION} \
        libcublas-12-4=${NV_CUDA_CUBLAS_VERSION} \
        libcurand-12-4 \
        libcusparse-12-4 \
        libcusolver-12-4 \
        libcufft-12-4 \
        libnccl2=${NV_NCCL_VER} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/* \
    && echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/lib:/usr/local/nvidia/lib64:/usr/local/cuda-12.4/lib64:/usr/local/cuda/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NCCL_LAUNCH_MODE GROUP

FROM cuda-base as base

ARG DEBIAN_FRONTEND=noninteractive

ENV PYENV_ROOT="/root/.pyenv"
ENV PATH /root/.pyenv/versions/3.10.12/bin:$PYENV_ROOT/bin:$PYENV_ROOT/shims:/usr/local/cuda-12.4/bin:/usr/local/cmake/bin:/usr/lib64/openmpi/bin:/opt/program:${PATH}
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

RUN apt-get update -y \
    && apt-get install -y build-essential \
    && apt-get update && apt-get install -y git git-lfs openssh-client make libssl-dev zlib1g-dev libbz2-dev \
        libreadline-dev wget curl llvm libncurses5-dev xz-utils tk-dev libffi-dev \
        liblzma-dev ca-certificates gettext vim libopenmpi-dev libsqlite3-dev \
        && apt-get clean

# install python 3.10 through pyenv
RUN git clone --depth=1 https://github.com/pyenv/pyenv.git /root/.pyenv
RUN pyenv install 3.10.12 && pyenv global 3.10.12

# adds github.com to known hosts to allow ssh dependencies download
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

ENV PYTHONUNBUFFERED 1

WORKDIR /opt/program
ENV PATH="/opt/program:${PATH}"

# pin numpy version to avoid failure due to numpy>=2.0
RUN pip install --no-cache-dir \
    pip==24.0 \
    setuptools==69.1.0 \
    flask==3.0.2 \
    numpy==1.26.3 \
    tritonclient[grpc]==2.31.0 \
    torch==2.4.0 \
    cuda-python==12.4.0 \
    safetensors==0.4.2 \
    mpi4py==3.1.5 \
    StrEnum==0.4.15 \
    datasets==2.19.0 \
    h5py==3.10.0 \
    pulp==2.8.0 \
    janus==1.0.0 \
    transformers==4.44.2 \
    liquidpy==0.8.2

# nvidia-ammo needs to be installed from NVIDIA's private pypi registry
RUN pip install --no-cache-dir --no-deps nvidia-ammo==0.7.0 torchprofile==0.0.4 scipy==1.13.0 --extra-index-url https://pypi.nvidia.com

# pin nvidia-nccl-cu12 to 2.21.5 to prevent nccl related failure: https://github.com/pytorch/pytorch/issues/113245
RUN pip install --no-cache-dir -U nvidia-nccl-cu12==2.21.5

# Peft build
FROM base as peft

ENV FINETUNE_BACKEND=peft

RUN apt-get update && \
    apt-get install -y software-properties-common libaio-dev ffmpeg libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN git lfs install

RUN pip install --no-cache-dir \
    bitsandbytes==0.43.3 \
    auto-gptq==0.7.1 \
    librosa==0.10.2.post1 \
    soundfile==0.12.1 \
    accelerate==0.34.0 \
    peft[test]==0.12.0 \
    aqlm[gpu]==1.0.2 \
    hqq==0.2.1.post1 \
    deepspeed==0.15.1 \
    trl==0.10.1 \
    wandb==0.17.8 \
    zstandard==0.23.0 \
    wheel==0.44.0 \
    optimum==1.17.1

ENV CPATH=/root/.pyenv/versions/3.10.12/lib/python3.10/site-packages/nvidia/cusparse/include:${CPATH}
ENV CPATH=/root/.pyenv/versions/3.10.12/lib/python3.10/site-packages/nvidia/cublas/include:${CPATH}
ENV CPATH=/root/.pyenv/versions/3.10.12/lib/python3.10/site-packages/nvidia/cusolver/include:${CPATH}

# Add autoawq for quantization testing
# We ignore their requirements for nvidia-cudnn-cu12-8.9.2.26, torch-2.3.1, triton-2.3.1
RUN pip install --no-cache-dir --no-deps \
    autoawq==0.2.6 \
    autoawq-kernels==0.0.7 \
    pynvml==11.5.3

RUN pip install --no-cache-dir git+https://github.com/NetEase-FuXi/EETQ.git

RUN pip install --no-cache-dir --no-build-isolation flash-attn==2.6.3

# Revert nvidia-nccl-cu12 back to 2.21.5, as packages above change it to 2.20.5
RUN pip install --no-cache-dir -U nvidia-nccl-cu12==2.21.5

# DeepSpeed's CPU offloading requires DeepSpeed's CPU_ADAM, which further requires the "libcurand.so" to be found
RUN ln -s /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcurand.so.10 /usr/local/cuda-12.4/targets/x86_64-linux/lib/libcurand.so

FROM peft as peft-dev

RUN apt-get update && \
    apt-get install -y tmux nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

RUN pip install --no-cache-dir jupyter==1.1.1

# Install gsutil, which can be used to upload / download files to / from GCS
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz && \
    tar -xf google-cloud-cli-linux-x86_64.tar.gz && \
    rm google-cloud-cli-linux-x86_64.tar.gz && \
    ./google-cloud-sdk/install.sh --rc-path /root/.bashrc --quiet && \
    source /root/.bashrc

FROM peft as peft-prod

COPY ./src/cohere_finetune /opt/program/cohere_finetune/
COPY "./entrypoint.sh" /opt/program/

RUN chmod +x /opt/program/entrypoint.sh

WORKDIR /opt/program

ENTRYPOINT ["./entrypoint.sh"]
