#!/bin/sh
# Download the models on first start (into the mounted /models volume), then
# start whisper-server. Model files are reused across restarts.
set -eu

MODELS_DIR="${MODELS_DIR:-/models}"
WHISPER_MODEL="${WHISPER_MODEL:-ggml-large-v3-q5_0.bin}"
VAD_MODEL="${VAD_MODEL:-ggml-silero-v5.1.2.bin}"
HF="https://huggingface.co"

mkdir -p "$MODELS_DIR"

fetch() { # fetch <file> <url>
    if [ ! -s "$MODELS_DIR/$1" ]; then
        echo "[whisper] downloading $1 ..."
        curl -fL --retry 3 -o "$MODELS_DIR/$1.part" "$2"
        mv "$MODELS_DIR/$1.part" "$MODELS_DIR/$1"
    fi
}

# Official ggml models; a custom/Thai fine-tuned ggml-*.bin can just be copied into ./models.
case "$WHISPER_MODEL" in
    ggml-large-v3*.bin|ggml-medium*.bin|ggml-small*.bin|ggml-base*.bin|ggml-tiny*.bin)
        fetch "$WHISPER_MODEL" "$HF/ggerganov/whisper.cpp/resolve/main/$WHISPER_MODEL" ;;
esac
fetch "$VAD_MODEL" "$HF/ggml-org/whisper-vad/resolve/main/$VAD_MODEL"

if [ ! -s "$MODELS_DIR/$WHISPER_MODEL" ]; then
    echo "[whisper] model $MODELS_DIR/$WHISPER_MODEL not found" >&2
    exit 1
fi

echo "[whisper] Vulkan devices:"
vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" || echo "  (vulkaninfo found no GPU - whisper will run on CPU)"

exec whisper-server \
    --host 0.0.0.0 --port 8080 \
    -m "$MODELS_DIR/$WHISPER_MODEL" \
    -l th \
    -t "${WHISPER_THREADS:-4}" \
    --vad -vm "$MODELS_DIR/$VAD_MODEL" \
    -nlp \
    ${WHISPER_EXTRA_ARGS:-}
