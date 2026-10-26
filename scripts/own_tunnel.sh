#!/usr/bin/env bash
# 29: the board's server reaching a model one of our own boxes serves (bash
# own_box.sh serve D3.3 18090 …, on the station) — an SSH tunnel as the
# board's account, from this server's docker bridge to the station's
# loopback, so the bench container asks it at http://host.docker.internal:<port>
# and nothing else on either network can. The station's address is the
# server's own config, never the repo's: EVALBOARD_OWN_<station> in the
# environment or the board's .env beside this checkout, as
# user@address:ssh-port:work-folder (frontier_fetch.py --own reads the same).
#
#     bash scripts/own_tunnel.sh D3 18090 --key ~/.ssh/id_ed25519
#
# It runs until stopped (Ctrl-C), in the foreground: tmux keeps it. OWN_BIND
# is the address it listens on here (the docker bridge's, docker0's, unless
# set).
set -euo pipefail

die() { echo "own_tunnel: $*" >&2; exit 1; }
station="${1:-}"; port="${2:-}"; shift 2 || die "own_tunnel.sh <station, e.g. D3> <port> --key <key>"
[[ "$station" =~ ^[A-Z][0-9]{1,2}$ ]] || die "$station: a station's label, e.g. D3"
[[ "$port" =~ ^[0-9]{4,5}$ ]] || die "$port: the port own_box.sh serve was given"
key=""
while [ $# -gt 0 ]; do
  case "$1" in
    --key) key="${2:-}"; shift 2 ;;
    *) die "$1: only --key" ;;
  esac
done
[ -n "$key" ] || die "--key <your SSH key>"

var="EVALBOARD_OWN_$station"
val="${!var:-}"
here="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "$val" ] && [ -f "$here/.env" ]; then
  val="$(grep -E "^$var=" "$here/.env" | tail -1 | cut -d= -f2- | tr -d "\"'")"
fi
[ -n "$val" ] || die "$var isn't set here or in $here/.env (user@address:ssh-port:work-folder)"
[[ "$val" =~ ^([A-Za-z0-9._-]+)@([A-Za-z0-9.:-]+):([0-9]+):(/.+)$ ]] \
  || die "$var: user@address:ssh-port:work-folder"
who="${BASH_REMATCH[1]}@${BASH_REMATCH[2]}"; ssh_port="${BASH_REMATCH[3]}"

bind="${OWN_BIND:-$(ip -4 -o addr show docker0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"
[ -n "$bind" ] || die "docker0 has no address here: give OWN_BIND"
echo "the bench container reaches $station's port $port at http://host.docker.internal:$port (this side listens on $bind:$port)"
exec ssh -N -i "$key" -p "$ssh_port" -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o LogLevel=ERROR \
  -L "$bind:$port:127.0.0.1:$port" "$who"
