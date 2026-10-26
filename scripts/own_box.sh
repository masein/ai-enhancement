#!/usr/bin/env bash
# 29: our own boxes on a DGX Station — one GPU each (D3.0 … D3.3: station
# D3's GPUs 0 … 3), each in a container of the runner image's V100 tag, run as
# the board's account on the station: in the docker group, no sudo, every file
# in its work folder on /raid. The station is colleagues' too. This script
# never uses sudo, never reboots, never touches the driver, Docker's own
# service or NetBird, installs nothing, and never signals a process that isn't
# in its own containers (evalboard-<label>) — it reads nvidia-smi and waits.
#
# It comes from the image itself — nothing is installed on the station:
#
#     docker run --rm ghcr.io/masein/evalboard-runner:v100 cat scripts/own_box.sh > own_box.sh
#
#     bash own_box.sh check D3.0          # GPU 0: free, or who uses it
#     bash own_box.sh start D3.0          # its container (the image pulled once)
#     read -rs HF_TOKEN && export HF_TOKEN
#     bash own_box.sh pilot D3.0 -- --as served/… --gguf hf://… --server-dir /opt/llama
#     bash own_box.sh run D3.0 -- --as <Q4_K_M's> … --then --as <Q8_0's> …
#     bash own_box.sh log D3.0            # its line's output, followed
#     bash own_box.sh serve D3.3 18090 -- --as served/… --gguf hf://… --server-dir /opt/llama
#     bash own_box.sh stop D3.0           # its container, and nothing else
#
# `run` and `pilot` give scripts/frontier_box.py the label and the rest of the
# line as typed (`--then` starts the next line, on the same GPU, after it);
# `serve` gives scripts/remote_gguf.py --serve its port, published on the
# station's loopback alone, for an SSH tunnel from the board's server
# (scripts/own_tunnel.sh). HF_TOKEN goes in by name only (docker exec -e
# HF_TOKEN), never on a command line. OWN_WORK is the work folder
# (/raid/<account> unless set), OWN_IMAGE the image.
set -euo pipefail

WORK="${OWN_WORK:-/raid/$(id -un)}"
IMAGE="${OWN_IMAGE:-ghcr.io/masein/evalboard-runner:v100}"
IDLE_MIB="${OWN_IDLE_MIB:-256}"

die() { echo "own_box: $*" >&2; exit 1; }

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed -e '$d' -e 's/^# \{0,1\}//' >&2
  exit 2
}

# D3.0 → its GPU (0) and its container's name
label_of() {
  [[ "${1:-}" =~ ^[A-Z][0-9]{1,2}\.([0-9])$ ]] || die "${1:-}: a box's label, D3.0 (station D3's GPU 0)"
  LABEL="$1"; GPU="${BASH_REMATCH[1]}"; NAME="evalboard-$1"
}

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null || true)" = "true" ]; }

# who uses GPU $GPU, as the station's own nvidia-smi says — every process on
# it but those of this box's containers; '' when none. Memory in use with no
# process to show for it counts too, while this box runs nothing there
busy() {
  local uuid apps ours used out="" pid name mib who
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" 2>/dev/null)" \
    || { echo "GPU $GPU couldn't be read (nvidia-smi)"; return; }
  ours=" "
  for c in "$NAME" "$NAME-serve"; do
    if running "$c"; then ours+="$(docker top "$c" -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ')"; fi
  done
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader,nounits 2>/dev/null | awk -F', *' -v u="$uuid" '$1 == u')"
  while IFS=, read -r _ pid name mib; do
    pid="${pid// /}"; [ -n "$pid" ] || continue
    case "$ours" in *" $pid "*) continue ;; esac
    who="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    out+="${out:+; }pid $pid${who:+, $who}, $(basename "${name# }"), ${mib// /} MiB"
  done <<< "$apps"
  if [ -z "$out" ] && [ "$ours" = " " ]; then
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')"
    [ "${used:-0}" -le "$IDLE_MIB" ] || out="$used MiB in use, by no process this account can see"
  fi
  echo "$out"
}

check() {
  local who
  who="$(busy)"
  if [ -n "$who" ]; then
    echo "GPU $GPU of ${LABEL%.*} is busy: $who — nothing is started while it is (nothing of theirs is touched)"
    return 1
  fi
  echo "GPU $GPU of ${LABEL%.*} is free of anyone else's work"
}

start() {
  running "$NAME" && die "$NAME is running already (bash own_box.sh stop $LABEL first)"
  docker inspect "$NAME" >/dev/null 2>&1 && docker rm "$NAME" >/dev/null
  check || exit 1
  mkdir -p "$WORK/workspace/home" "$WORK/workspace/logs"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
  # its own GPU alone; the station's processes visible (nvidia-smi names them,
  # gpu_free.py reads whose they are) but out of reach: this account's user,
  # no capabilities, no new privileges
  docker run -d --name "$NAME" --gpus "device=$GPU" --pid=host \
    --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
    -e HOME=/workspace/home -e EVALBOARD_OWN="$LABEL" \
    -v "$WORK/workspace:/workspace" "$IMAGE" sleep infinity >/dev/null
  echo "$NAME started on GPU $GPU ($IMAGE), its work in $WORK/workspace"
}

# the line(s) after --, each --then starting the next, as one shell command
lines() {
  local cmd="" part=() a
  for a in "$@" "--then"; do
    if [ "$a" = "--then" ]; then
      [ "${#part[@]}" -gt 0 ] || die "an empty line before or after --then"
      cmd+="${cmd:+; }python scripts/frontier_box.py $(printf '%q ' "$LABEL" "${part[@]}")"
      part=()
    else
      part+=("$a")
    fi
  done
  echo "$cmd"
}

run() {
  running "$NAME" || die "$NAME isn't running (bash own_box.sh start $LABEL)"
  check || exit 1
  local cmd log="/workspace/logs/$LABEL.log"
  cmd="$(lines "$@")"
  docker exec -d ${HF_TOKEN:+-e HF_TOKEN} -w /app "$NAME" bash -c "{ $cmd; } >> $log 2>&1"
  echo "$LABEL: started — bash own_box.sh log $LABEL to follow it"
}

pilot() {
  running "$NAME" || die "$NAME isn't running (bash own_box.sh start $LABEL)"
  check || exit 1
  docker exec ${HF_TOKEN:+-e HF_TOKEN} -w /app "$NAME" bash -c \
    "python scripts/frontier_box.py $(printf '%q ' "$LABEL" --pilot "$@") 2>&1 | tee -a /workspace/logs/$LABEL-pilot.log"
}

serve() {
  local port="${1:-}"; shift || true
  [[ "$port" =~ ^[0-9]{4,5}$ ]] || die "serve $LABEL <port> -- …: the port the tunnel reaches"
  [ "${1:-}" = "--" ] && shift
  running "$NAME" && die "$NAME is running a line on GPU $GPU: serve on another GPU, or stop it"
  running "$NAME-serve" && die "$NAME-serve is serving already (bash own_box.sh stop $LABEL)"
  docker inspect "$NAME-serve" >/dev/null 2>&1 && docker rm "$NAME-serve" >/dev/null
  check || exit 1
  mkdir -p "$WORK/workspace/home" "$WORK/workspace/logs"
  docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"
  # the port on the station's loopback alone: the board reaches it through
  # an SSH tunnel as this account, nobody else on the network does
  docker run -d --name "$NAME-serve" --gpus "device=$GPU" --pid=host \
    --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges \
    -e HOME=/workspace/home -e EVALBOARD_OWN="$LABEL" ${HF_TOKEN:+-e HF_TOKEN} \
    -p "127.0.0.1:$port:$port" -v "$WORK/workspace:/workspace" -w /app "$IMAGE" \
    bash -c "python scripts/remote_gguf.py $(printf '%q ' --serve "$port" --own --label "$LABEL" \
      --files /workspace/files --out "/workspace/serve-$LABEL" "$@") >> /workspace/logs/$LABEL-serve.log 2>&1" \
    >/dev/null
  echo "$NAME-serve: serving on the station's 127.0.0.1:$port — bash own_box.sh log $LABEL-serve to follow it"
}

log() {
  local f="$WORK/workspace/logs/$1.log"
  [ -f "$f" ] || die "no log $f yet"
  tail -n 200 -f "$f"
}

stop() {
  local c stopped=""
  for c in "$NAME" "$NAME-serve"; do
    if docker inspect "$c" >/dev/null 2>&1; then
      docker rm -f "$c" >/dev/null && stopped+="${stopped:+, }$c"
    fi
  done
  echo "${stopped:-no container of $LABEL} ${stopped:+stopped}"
}

what="${1:-}"; shift || true
case "$what" in
  check|start|run|pilot|serve|stop) ;;
  log) [[ "${1:-}" =~ ^[A-Z][0-9]{1,2}\.[0-9](-serve|-pilot)?$ ]] || die "log D3.0 (or D3.0-serve, D3.0-pilot)"
       log "$1"; exit ;;
  *) usage ;;
esac
label_of "${1:-}"; shift
[ "$what" = run ] || [ "$what" = pilot ] && { [ "${1:-}" = "--" ] && shift; [ $# -gt 0 ] || die "$what $LABEL -- <the line's arguments>"; }
"$what" "$@"
