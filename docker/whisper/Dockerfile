# whisper.cpp server built with the Vulkan backend, so speech-to-text runs on
# the Intel iGPU (the same way the Ollama container uses it).
ARG WHISPER_CPP_VERSION=v1.9.5

FROM debian:trixie-slim AS build
ARG WHISPER_CPP_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libvulkan-dev glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${WHISPER_CPP_VERSION} https://github.com/ggml-org/whisper.cpp /src
WORKDIR /src
# Not -march=native: the image may be built on a different machine than the NAS.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
        -DBUILD_SHARED_LIBS=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_SDL2=OFF \
    && cmake --build build -j"$(nproc)" --target whisper-server

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers libgomp1 curl ca-certificates vulkan-tools \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/whisper-server /usr/local/bin/whisper-server
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["/entrypoint.sh"]
