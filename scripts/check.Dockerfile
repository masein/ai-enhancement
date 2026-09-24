# The image scripts/check.sh runs the pre-merge check in: Python 3.12, the
# test and lint tooling, and Playwright's Chromium — never the host's Python
# (HANDOFF.md § 5b). Built from a context of requirements-dev.txt alone; the
# repo is mounted at run time, so a new commit needs no rebuild.
#
# git: check.sh names the commit it checked, and one test lists the tracked
# files. nodejs: two tests syntax-check the page's JavaScript with node, as
# the CI runner did.
FROM python:3.12-slim

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# retries: a slow or flaky link drops a download now and then (this image was
# first built at 30 KB/s)
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80retries \
    && apt-get update \
    && apt-get install -y --no-install-recommends git nodejs \
    && rm -rf /var/lib/apt/lists/* \
    # the repo is mounted and owned by the host's user, not this one
    && git config --system --add safe.directory '*'

# --only-shell: the headless shell is all a headless run launches, at about
# half the download of the full browser
COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --retries 10 --timeout 120 -r /tmp/requirements-dev.txt \
    && playwright install --with-deps --only-shell chromium \
    && chmod -R a+rX "$PLAYWRIGHT_BROWSERS_PATH"

# the host's user, by name, uid and gid: the check runs as it (nothing in the
# mounted repo ends up owned by root), and a test that asks who is running —
# getpass.getuser(), as the service does for EVAL_USER — gets an answer. Last,
# and cheap: another user on another host rebuilds only this layer
ARG HOST_UID=1000
ARG HOST_GID=1000
ARG HOST_USER=checker
RUN if [ "$HOST_UID" != 0 ]; then \
      (getent group "$HOST_GID" >/dev/null || groupadd -g "$HOST_GID" hostgroup) \
      && (id -u "$HOST_USER" >/dev/null 2>&1 || useradd -o -M -u "$HOST_UID" -g "$HOST_GID" \
            -d /tmp -s /bin/bash "$HOST_USER"); \
    fi
