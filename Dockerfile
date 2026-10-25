# Web app + FFmpeg with Intel VAAPI (iHD driver) for iGPU rendering.
FROM python:3.14-slim

# non-free is needed for intel-media-va-driver-non-free (iHD with full H.264 encode support).
RUN sed -i 's/^Components: main$/Components: main non-free non-free-firmware/' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg intel-media-va-driver-non-free vainfo libva-drm2 fontconfig \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LIBVA_DRIVER_NAME=iHD \
    DATA_DIR=/data \
    MODELS_DIR=/models

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
    # warm PyThaiNLP's dictionary so the first request isn't slow
    && python -c "from pythainlp.tokenize import word_tokenize; word_tokenize('ทดสอบ')"

COPY app ./app
COPY fonts ./fonts

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/jobs', timeout=4)" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "30"]
