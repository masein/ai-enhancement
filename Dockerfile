# The benchmark service, containerized. See SERVICE.md § Docker for run commands.
#
# Base: official PyTorch runtime with CUDA 12.8 — the RTX 5090 is Blackwell
# (sm_120) and needs cu128+ kernels; torch 2.11.0 matches the version already
# validated on the target server. (Tag existence verified against Docker Hub.)
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

# lm-eval + service deps; torch comes from the base image.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

WORKDIR /app
COPY scripts/ scripts/
COPY service/ service/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# BENCH_ROOT (results/, eval_tasks/, logs/, service.sqlite3) and HF_HOME are
# expected as path-identical volume mounts — see docker-compose.yml. nvidia-smi
# is injected by the NVIDIA container toolkit at run time; it is not in the image.
EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8899/healthz', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8899"]
