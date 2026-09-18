# The benchmark service, containerized. See SERVICE.md § Docker for run commands.
#
# Base: official PyTorch runtime with CUDA 12.8 — the RTX 5090 is Blackwell
# (sm_120) and needs cu128+ kernels; torch 2.11.0 matches the version already
# validated on the target server. (Tag existence verified against Docker Hub.)
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

# lm-eval + service deps; torch comes from the base image.
#
# `python -m pip`, not bare `pip`: it targets the exact interpreter that CMD runs,
# which is the one the base image installed torch into. `--break-system-packages`:
# these images use Ubuntu's system Python 3.12, which is PEP 668 "externally
# managed" and rejects bare pip installs — inside a single-purpose container that
# protection protects nothing, and it's how the base image got torch in there too.
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# Fail the BUILD, not the first submission, if the env is incoherent (e.g. deps
# landed in a different interpreter than torch).
RUN python -c "import torch, lm_eval, transformers, accelerate, datasets, fastapi, uvicorn; \
print('image env OK — torch', torch.__version__, '| built for CUDA', torch.version.cuda, \
'| lm_eval', lm_eval.__version__)"

# An unprivileged account for evaluating uploads that carry their own model code
# (EVAL_USER). The service itself still runs as root — it needs to write the
# shared results tree the CLI also writes — but a job executing someone's
# modeling_*.py drops to this user, which owns nothing.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin benchjob

WORKDIR /app
COPY scripts/ scripts/
COPY service/ service/
# served over HTTP by the app: FRIENDS.md at /guide, bench_client.py at /client
# (.dockerignore carries the FRIENDS.md exception)
COPY clients/ clients/
# the permutation control's task yaml + utils.py: a suite=control run passes
# this directory to lm_eval --include_path (.dockerignore carries the exception)
COPY eval_tasks/mmlu_perm/ eval_tasks/mmlu_perm/
COPY FRIENDS.md ./

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# BENCH_ROOT (results/, eval_tasks/, logs/, service.sqlite3) and HF_HOME are
# expected as path-identical volume mounts — see docker-compose.yml. nvidia-smi
# is injected by the NVIDIA container toolkit at run time; it is not in the image.
EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8899/healthz', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8899"]
