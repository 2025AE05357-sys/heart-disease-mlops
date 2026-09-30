# syntax=docker/dockerfile:1.7
# Heart-disease risk API: a small, non-root runtime image with the champion model baked in.
# The image tag and OCI labels carry the git commit and model version (traceability, D0).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ---- build stage: wheels for the serving lock only; nothing from here but /wheels ships ----
FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements-serve.txt .
RUN python -m pip wheel --wheel-dir /wheels -r requirements-serve.txt

# ---- runtime stage ----
FROM ${PYTHON_IMAGE} AS runtime
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
ARG MODEL_VERSION=unknown
ARG MODEL_RUN_ID=unknown
LABEL org.opencontainers.image.title="heartrisk-api" \
      org.opencontainers.image.description="Heart-disease risk classifier API. Educational project, not a medical device." \
      org.opencontainers.image.source="https://github.com/2025AE05357-sys/heart-disease-mlops" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      io.heartrisk.model.version="${MODEL_VERSION}" \
      io.heartrisk.model.run_id="${MODEL_RUN_ID}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    MODEL_DIR=/app/models/champion \
    LOG_LEVEL=INFO \
    AUDIT_LOG_INPUTS=false

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
# Bind-mount the wheels instead of COPYing them: a COPY would keep ~150 MB of wheel files in an
# image layer even after deleting them in a later step.
RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    python -m pip install --no-index --find-links=/wheels /wheels/*.whl

# Code and model stay root-owned and read-only for the service user.
WORKDIR /app
COPY src/heartrisk /app/src/heartrisk
COPY models/champion /app/models/champion
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]

# One worker per container: scale with replicas, which keeps Prometheus metrics per process simple.
CMD ["uvicorn", "heartrisk.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
