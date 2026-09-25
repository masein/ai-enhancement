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

# The unit suite also runs inside the running container, on this image's
# Python and packages (HANDOFF.md § Checks, deploy step 3). That needs pytest,
# and httpx for FastAPI's test client, and nothing else: the tests are not in
# the image — the deploy step streams the commit in with `git archive`,
# because they read files the image leaves out (the Dockerfile, the docs).
RUN python -m pip install --no-cache-dir --break-system-packages "pytest>=8" httpx

# 12h.1: no vLLM in this image. IFEval, MMLU-Pro and MATH-500 run on hf. vLLM
# 0.26.0 has no CUDA 12.8 build (this image's torch is cu128), and it would
# move FastAPI below 0.137, away from the version the check tests the service
# on. If the trials say hf is too slow, vLLM gets a container of its own, in
# its own brief — not this image.
# 12h.1: the Mamba2 hybrids (Granite-4.0-H, Nemotron-3-Nano) are far faster
# with these kernels, which may need a build this runtime image cannot do.
# Tried, never required. WITH_MAMBA=0 skips it.
ARG WITH_MAMBA=1
RUN if [ "$WITH_MAMBA" = "1" ]; then \
      timeout 900 python -m pip install --no-cache-dir --break-system-packages \
        --no-build-isolation "causal-conv1d==1.7.0" "mamba-ssm==2.3.2.post1" \
        && echo "Mamba kernels installed" \
        || echo "Mamba kernels did not build: the Mamba2 hybrids run on the slower path"; \
    fi

# Fail the BUILD, not the first submission, if the env is incoherent (e.g. deps
# landed in a different interpreter than torch).
RUN python -c "import torch, lm_eval, transformers, accelerate, datasets, fastapi, uvicorn, pytest, httpx; \
import math_verify, langdetect, nltk, immutabledict; \
import importlib.util as u; \
print('image env OK — torch', torch.__version__, '| built for CUDA', torch.version.cuda, \
'| lm_eval', lm_eval.__version__, '| transformers', transformers.__version__, \
'| fastapi', fastapi.__version__, \
'| Mamba kernels', 'yes' if u.find_spec('mamba_ssm') else 'no')"

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
# the free-response side the service reads at run time: the task template
# exam_build.build() writes each exam task from, the per-topic rubrics and
# criteria files the judge grades against, the canary scripts, the delivered
# banks and the skill-suite seeds. service/app.py refuses to start without
# them, so a missing one fails at `up` rather than on someone's first click
COPY eval_tasks/fr/ eval_tasks/fr/
# 12a.3: the Everyday bank (333 questions) and its task template; the page
# shows the questions and everyday.build_task() writes the task from them
COPY eval_tasks/everyday/ eval_tasks/everyday/
COPY FRIENDS.md ./

# the same check the service runs at startup, at BUILD time: an image missing
# a file the service reads is a broken image, and this is where that is cheap
# to find out
RUN python -c "import sys; sys.path.insert(0, '/app'); \
from service.startup import missing_repo_files; m = missing_repo_files(); \
print('image files OK') if not m else sys.exit('image is missing: ' + ', '.join(m))"

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# BENCH_ROOT (results/, eval_tasks/, logs/, service.sqlite3) and HF_HOME are
# expected as path-identical volume mounts — see docker-compose.yml. nvidia-smi
# is injected by the NVIDIA container toolkit at run time; it is not in the image.
EXPOSE 8899

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8899/healthz', timeout=4)" || exit 1

# The git sha this image was built from, for the page's "the dashboard was
# updated" bar. Optional: without it the page's own hash still tells an old
# page from a new one. Last, so a new sha never invalidates the layers above.
ARG EVALBOARD_BUILD=""
ENV EVALBOARD_BUILD=${EVALBOARD_BUILD}

CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8899"]
