FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 AS builder

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test -y \
    && apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    gcc-12 \
    g++-12 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Rust for wheels
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Python deps
RUN pip3 install --break-system-packages \
    langchain langchain-community streamlit python-dotenv requests alpaca-py discord.py

WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp

WORKDIR /app/llama.cpp
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_F16=ON \
      -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_SERVER=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_TTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_C_COMPILER=/usr/bin/gcc-12 \
      -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12 \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build . -j$(nproc) --config Release

# Runtime
FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages/
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages/
COPY --from=builder /root/.cargo /root/.cargo
COPY --from=builder /app/llama.cpp/build/bin /usr/local/bin

WORKDIR /app
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]