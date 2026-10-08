"""18: the agent on the host (docs/AGENT-RUNS-design.md § 2, decision 1) — a
Harbor "external" agent that runs mini-swe-agent's own loop beside the task's
container and sends each command into it through Harbor's exec. The task's
container has no network, no address of the model and no key: the agent
reaches the model through the relay (scripts/agent_relay.py), on the host.

What it keeps of mini-swe-agent, so a run is the benchmark's agent:
- its config file (SWE-bench's `benchmarks/swebench.yaml` for SWE-bench
  Multilingual; the default `mini.yaml` for DeepSWE, as its board runs it):
  prompts, step and cost limits, the observation and error templates;
- a command's shell (`bash -c` for the SWE-bench config's docker
  environment, `sh -c` for the default config's local one), its working
  folder, its time limit (60 s / 30 s; the whole command group killed when
  it passes, as mini-swe-agent's local environment kills it), stderr merged
  into stdout, and the submission line;
- its trajectory file, written after every step — in the trial's own
  agent-host/ folder, which no container mounts, never through a link
  (18b point 1).

Before the model is asked anything, the container is checked from inside:
nothing answers on the targets the runner gives (the host's other ports, the
tailnet, the LAN, the internet), no Docker socket, a process limit. After
the agent stops — however it stops — every process it left is killed and
everything in the folders Harbor mounts (/logs/agent, /logs/artifacts,
/logs/verifier) is removed, before the verifier runs: no reward file of the
model's, no link, FIFO or device for a host process to open. The clean-up
says what it left, with an exit code kept in meta.json; one that didn't run
to its end keeps the task from being verified (18c point 3).

Harbor and mini-swe-agent are imported only on the server, in the agent
venv; the adapter's logic is tested without them."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import shlex
import sys
import threading
import time
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_bench as ab  # noqa: E402

try:                                    # the agent venv; never on the Mac or in CI
    from harbor.agents.base import BaseAgent
except ImportError:                     # pragma: no cover — tests use the adapter alone
    BaseAgent = object                  # type: ignore[misc,assignment]

SUBMIT = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
TIMEOUT_DEFAULT = {"docker": 60, "local": 30}


try:                                    # mini-swe-agent's own, so its loop ends as it ends
    from minisweagent.exceptions import Submitted
except ImportError:                     # pragma: no cover — the adapter's tests
    class Submitted(Exception):         # type: ignore[no-redef]
        def __init__(self, *messages):
            super().__init__(*messages)
            self.messages = messages


class Stopped(Exception):
    """Harbor's time limit for the agent passed: the loop stops"""


class ReachRefused(Exception):
    """the task's container reached something it must not: not run"""


class ServerDown(Exception):
    """the model's server failed the relay's health check: an error of ours,
    asked again — one bad reply is the model's"""


class CleanupFailed(Exception):
    """the clean-up after the agent didn't run to its end: the task is not
    verified (18c point 3)"""


DOWN = "relay_server_down"              # the relay's word for it, in its 503
SENT_WAIT_S = 120                       # a command sent just before the stop: its 60 s and more
MOUNTED = "/logs/agent /logs/artifacts /logs/verifier"
# what the container holds once the agent stops: no process but its own
# sleep, nothing in the folders Harbor mounts — and it says so, with an exit
# code: 0 only when both are true (a zombie is dead, and counts for nothing)
CLEAN = "\n".join([
    "kill -9 -1 2>/dev/null; sleep 1; kill -9 -1 2>/dev/null",
    f"for d in {MOUNTED}; do find \"$d\" -mindepth 1 -delete 2>/dev/null; "
    "rm -rf \"$d\"/* \"$d\"/.[!.]* 2>/dev/null; done",
    # what is left, and what can't be read, counts: it fails closed
    f"left=$(for d in {MOUNTED}; do [ -e \"$d\" ] && find \"$d\" -mindepth 1 2>&1; done | wc -l)",
    "n=0; for p in /proc/[0-9]*; do i=${p#/proc/}; [ \"$i\" = 1 ] || [ \"$i\" = $$ ] && continue",
    "  st=; while read -r k v _; do [ \"$k\" = State: ] && st=$v; done 2>/dev/null < \"$p/status\"",
    "  [ -n \"$st\" ] && [ \"$st\" != Z ] && n=$((n+1)); done",
    "echo \"CLEAN files=$left processes=$n\"",
    "[ \"$left\" = 0 ] && [ \"$n\" = 0 ]"])


def clean_verdict(code, output: str) -> str:
    """'' when the clean-up ran to its end and left nothing; else what"""
    said = next((x.strip() for x in reversed((output or "").splitlines())
                 if x.strip().startswith("CLEAN ")), "")
    if code == 0 and said == "CLEAN files=0 processes=0":
        return ""
    return f"exit {code}" + (f", {said[6:]}" if said else ", no word from it")


# names the container must not resolve: DNS answering is a way out
REACH_NAMES = ("pypi.org", "github.com")
# the capabilities a task's container keeps: what Harbor's own steps need on
# the folders it mounts (agent_run.OVERRIDE_YAML)
CAP_NAMES = {0: "CHOWN", 1: "DAC_OVERRIDE", 2: "DAC_READ_SEARCH", 3: "FOWNER", 4: "FSETID",
             5: "KILL", 6: "SETGID", 7: "SETUID", 8: "SETPCAP", 10: "NET_BIND_SERVICE",
             12: "NET_ADMIN", 13: "NET_RAW", 18: "SYS_CHROOT", 19: "SYS_PTRACE",
             21: "SYS_ADMIN", 27: "MKNOD", 29: "AUDIT_WRITE", 31: "SETFCAP"}
CAPS_KEPT = (1 << 0) | (1 << 1) | (1 << 3)


def reach_script(targets: list[str]) -> str:
    """a check run inside the container, which fails closed (18b point 8):
    it says first that it can try a connection at all (bash's /dev/tcp
    refused by the container's own loopback, `sleep` there to time it), then
    one line a target that answered (IPv4 or IPv6, by address or name), a
    name that resolved, the Docker socket, the network interfaces and the
    process limit, and DONE last — a check that didn't get that far refuses"""
    lines = [
        "command -v sleep >/dev/null 2>&1 || echo 'PROBE missing sleep'",
        "o=$(bash -c 'exec 3<>/dev/tcp/127.0.0.1/1' 2>&1)",
        'case "$o" in *[Rr]efused*) echo "PROBE tcp-ok";; *) echo "PROBE no /dev/tcp: $o";; esac',
        # a try, given 3 s, without relying on `timeout`
        'try() { ( exec 3<>"/dev/tcp/$1/$2" ) 2>/dev/null & local p=$! i=0',
        '  while kill -0 $p 2>/dev/null; do i=$((i+1)); [ $i -gt 3 ] && '
        '{ kill -9 $p 2>/dev/null; return 124; }; sleep 1; done; wait $p; }',
        "for t in " + " ".join(shlex.quote(t) for t in targets) + "; do",
        '  h="${t%:*}"; p="${t##*:}"',
        '  if try "$h" "$p"; then echo "REACHED $t"; fi',
        "done",
        "for n in " + " ".join(REACH_NAMES) + "; do",
        '  getent hosts "$n" >/dev/null 2>&1 && echo "REACHED dns $n"',
        '  try "$n" 443 && echo "REACHED $n:443"',
        "done",
        "[ -e /var/run/docker.sock ] && echo 'REACHED docker.sock'",
        "echo \"NET $(ls /sys/class/net 2>/dev/null | tr '\\n' ' ')\"",
        # its capabilities and no-new-privileges, as the kernel has them
        "while read -r k v _; do case $k in CapEff:) echo \"CAPS $v\";; "
        "NoNewPrivs:) echo \"NNP $v\";; esac; done < /proc/self/status",
        "p=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max "
        "2>/dev/null); echo \"PIDS ${p:-unknown}\"",
        "echo DONE"]
    return "\n".join(lines)


def reached(output: str) -> list[str]:
    """what answered, as the check says it"""
    return [x.strip().split(" ", 1)[1] for x in output.splitlines()
            if x.strip().startswith("REACHED ")]


def reach_verdict(output: str, what: str = "container") -> str:
    """'' only when the check ran to its end, could try a connection, nothing
    answered, the only interface is the loopback and processes are limited;
    else why not. what="build": a check run in an image's build (18c point
    2), where only what answered counts"""
    lines = [x.strip() for x in output.splitlines()]
    if "DONE" not in lines:
        return f"the {what}'s check didn't run to its end: " + (output.strip()[-200:] or
                                                               "no output")
    probe = [x.split(" ", 1)[1] for x in lines if x.startswith("PROBE ") and x != "PROBE tcp-ok"]
    if probe or "PROBE tcp-ok" not in lines:
        return f"the {what}'s check couldn't try a connection: " + ("; ".join(probe)
                                                                   or "no word")
    if reached(output):
        return f"the task's {what} reached " + ", ".join(reached(output))
    if what == "build":
        return ""
    net = next((x[4:].split() for x in lines if x.startswith("NET")), None)
    if net is None or not net:
        return "the container's network interfaces couldn't be read"
    if set(net) != {"lo"}:
        return "the task's container has a network interface: " + ", ".join(
            x for x in net if x != "lo")
    pids = next((x.split(" ", 1)[1] for x in lines if x.startswith("PIDS ")), "")
    if pids in ("", "max", "unknown"):
        return "the task's container has no process limit"
    caps = next((x.split(" ", 1)[1] for x in lines if x.startswith("CAPS ")), "")
    try:
        extra = int(caps, 16) & ~CAPS_KEPT
    except ValueError:
        return "the container's capabilities couldn't be read"
    if extra:
        names = [n for b, n in CAP_NAMES.items() if extra & (1 << b)]
        return "the task's container keeps capabilities: " + ", ".join(
            names + ([f"{extra & ~sum(1 << b for b in CAP_NAMES):#x}"]
                     if extra & ~sum(1 << b for b in CAP_NAMES) else []))
    if next((x.split(" ", 1)[1] for x in lines if x.startswith("NNP ")), "") != "1":
        return "the task's container can gain privileges (no-new-privileges isn't set)"
    return ""


class HarborEnv:
    """mini-swe-agent's environment, over Harbor's async exec — called from
    the agent's thread, run on Harbor's event loop"""

    def __init__(self, exec_async, loop, *, cwd: str = "", timeout: int = 30,
                 interpreter: tuple[str, ...] = ("sh", "-c"), env: dict | None = None,
                 stop: threading.Event | None = None, kind: str = "local"):
        self._exec = exec_async
        self.loop = loop
        self.cwd = cwd
        self.timeout = timeout
        self.interpreter = tuple(interpreter)
        self.env = dict(env or {})
        self.stop = stop or threading.Event()
        self.kind = kind
        # 18c point 11: the commands sent and not yet back; closed as one with
        # the stop, so none starts after the clean-up's kill
        self._lock = threading.Lock()
        self.inflight: set = set()

    def close(self) -> list:
        """no command is sent from now on; those already sent, to wait for"""
        with self._lock:
            self.stop.set()
            return list(self.inflight)

    def command(self, command: str, timeout: int) -> str:
        """the command as the container runs it: its shell, its time limit for
        the whole group, stderr into stdout"""
        return (f"timeout -s KILL {int(timeout)} {' '.join(map(shlex.quote, self.interpreter))} "
                f"{shlex.quote(command)} 2>&1")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        command = action.get("command", "")
        t = int(timeout or self.timeout)
        t0 = time.time()
        with self._lock:
            if self.stop.is_set():
                raise Stopped("the agent's time is up")
            fut = asyncio.run_coroutine_threadsafe(
                self._exec(self.command(command, t), cwd=(cwd or self.cwd or None),
                           env=self.env or None, timeout_sec=t + 30), self.loop)
            self.inflight.add(fut)
        fut.add_done_callback(self.inflight.discard)
        try:
            r = fut.result()
            out = (getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or "")
            rc = int(getattr(r, "return_code", 0))
            if rc == 137 and time.time() - t0 >= t - 0.5:
                # the limit passed: as mini-swe-agent says it
                output = {"output": out, "returncode": -1,
                          "exception_info": f"An error occurred while executing the command: "
                                            f"Command '{command}' timed out after {t} seconds",
                          "extra": {"exception_type": "TimeoutExpired"}}
            else:
                output = {"output": out, "returncode": rc, "exception_info": ""}
        except Stopped:
            raise
        except Exception as e:                          # noqa: BLE001 — said to the model, as mini says it
            output = {"output": "", "returncode": -1,
                      "exception_info": f"An error occurred while executing the command: {e}",
                      "extra": {"exception_type": type(e).__name__, "exception": str(e)}}
        self.check_finished(output)
        return output

    @staticmethod
    def check_finished(output: dict) -> None:
        """the submission line, as mini-swe-agent reads it"""
        lines = (output.get("output") or "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT and output.get("returncode") == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission,
                             "extra": {"exit_status": "Submitted", "submission": submission}})

    def get_template_vars(self, **kwargs) -> dict:
        # as mini-swe-agent's own environments give them: their config and
        # the uname of the machine the agent runs on (the host, as with its
        # docker environment)
        return {"cwd": self.cwd, "timeout": self.timeout, "env": self.env,
                **platform.uname()._asdict(), **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {
            "environment": {"cwd": self.cwd, "timeout": self.timeout,
                            "interpreter": list(self.interpreter), "env": self.env,
                            "as": f"mini-swe-agent's {self.kind} environment, run by Harbor's exec "
                                  "from the host"},
            "environment_type": "agent_host_mini.HarborEnv"}}}


def load_config(name: str) -> dict:
    """a config file of mini-swe-agent's, as installed (the run records its hash)"""
    import yaml
    from importlib.resources import files
    return yaml.safe_load((files("minisweagent.config") / name).read_text())


def prompt_sha(cfg: dict) -> str:
    """the agent's prompt: its system and instance templates"""
    a = cfg.get("agent") or {}
    return hashlib.sha256((str(a.get("system_template")) + "\n"
                           + str(a.get("instance_template"))).encode()).hexdigest()


def usage_of(messages: list[dict]) -> tuple[int, int, int]:
    """(tokens in, tokens out, steps) from the trajectory's replies — tokens
    in adds up every step's whole prompt"""
    tin = tout = steps = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        u = _usage(m)
        tin += int(u.get("prompt_tokens") or 0)
        tout += int(u.get("completion_tokens") or 0)
        steps += 1
    return tin, tout, steps


def _usage(m: dict) -> dict:
    resp = (m.get("extra") or {}).get("response") or {}
    return (resp.get("usage") or {}) if isinstance(resp, dict) else {}


def last_prompt(messages: list[dict]) -> int | None:
    """the last step's prompt, in tokens: how big the conversation grew"""
    for m in reversed(messages):
        if m.get("role") == "assistant" and _usage(m).get("prompt_tokens"):
            return int(_usage(m)["prompt_tokens"])
    return None


def abort_request(relay: str, trial: str) -> None:
    """cut this trial's request in flight at the relay (18b point 11): the
    agent's time is up, and the server stops a reply nobody waits for"""
    if not relay or not trial:
        return
    import urllib.request
    url = relay.rstrip("/").removesuffix("/v1") + "/v1/agent/abort"
    req = urllib.request.Request(url, data=b"", method="POST", headers={"X-Agent-Trial": trial})
    try:
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:                   # noqa: BLE001 — the relay may be gone; nothing to cut
        pass


def mini_classes():
    """mini-swe-agent's loop and model, as installed in the agent venv"""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models.litellm_model import LitellmModel
    return DefaultAgent, LitellmModel


class HostMini(BaseAgent):  # type: ignore[misc,valid-type]
    """`-a agent_host_mini:HostMini`, with `--ak config=…`, `--ak cwd=…`,
    `--ak cost_limit=…`, `--ak relay=http://127.0.0.1:<port>/v1` and
    `--ak reach=<host:port>,…`"""
    OURS = ("config", "cwd", "cost_limit", "relay", "reach")

    def __init__(self, *args, **kwargs):
        self.ours = {k: kwargs.pop(k) for k in list(kwargs) if k in self.OURS}
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "host-mini-swe-agent"

    def version(self) -> str | None:
        try:
            import minisweagent
            return str(minisweagent.__version__)
        except ImportError:
            return None

    async def setup(self, environment) -> None:
        return None                     # nothing is installed in the task's container

    def host_dir(self) -> Path:
        """the agent's own files: in the trial's folder, never mounted"""
        return Path(self.logs_dir).parent / ab.HOST_DIR

    async def run(self, instruction: str, environment, context) -> None:
        DefaultAgent, LitellmModel = mini_classes()
        cfg = load_config(str(self.ours.get("config") or "mini.yaml"))
        host = self.host_dir()
        stop = threading.Event()
        held: dict = {"agent": None, "ours": "", "clean": None, "env": None}

        def fill() -> None:
            """the counts, to Harbor and to our own meta.json — never read
            back from a folder the container can write"""
            agent = held["agent"]
            msgs = list(agent.messages) if agent is not None else []
            tin, tout, steps = usage_of(msgs)
            last = (msgs[-1].get("extra") or {}) if msgs else {}
            meta = {"exit_status": str(last.get("exit_status") or ""), "steps": steps,
                    "tokens_in": tin, "tokens_out": tout, "last_prompt": last_prompt(msgs),
                    "config": self.ours.get("config"), "prompt_sha256": prompt_sha(cfg),
                    "submission_chars": len(str(last.get("submission") or "")),
                    "ours": held["ours"], "clean": held["clean"]}
            context.n_input_tokens, context.n_output_tokens = tin, tout
            # set, so Harbor never reads a usage file from the mounted folder
            context.model_usage = {}
            context.metadata = meta
            ab.safe_write(host, ab.META, json.dumps(meta))

        failed: BaseException | None = None
        try:
            targets = [t for t in str(self.ours.get("reach") or "").split(",") if t]
            probe = await environment.exec(reach_script(targets), timeout_sec=300)
            why = reach_verdict((probe.stdout or "") + (probe.stderr or ""))
            if why:
                held["ours"] = f"ReachRefused: {why}"
                raise ReachRefused(why)
            ecfg = dict(cfg.get("environment") or {})
            kind = "docker" if ecfg.get("environment_class") == "docker" else "local"
            env = HarborEnv(environment.exec, asyncio.get_running_loop(),
                            cwd=str(self.ours.get("cwd") or ecfg.get("cwd") or ""),
                            timeout=int(ecfg.get("timeout") or TIMEOUT_DEFAULT[kind]),
                            interpreter=tuple(ecfg.get("interpreter") or
                                              (("bash", "-c") if kind == "docker"
                                               else ("sh", "-c"))),
                            env=ecfg.get("env") or {}, stop=stop, kind=kind)
            held["env"] = env
            mcfg = dict(cfg.get("model") or {})
            mcfg.pop("model_class", None)
            mcfg["model_name"] = self.model_name
            mcfg["cost_tracking"] = "ignore_errors"
            mcfg["model_kwargs"] = {**(mcfg.get("model_kwargs") or {}),
                                    "api_base": str(self.ours.get("relay")), "api_key": "relay",
                                    "timeout": 3600,
                                    "extra_headers": {"X-Agent-Trial": str(self.session_id or "")}}

            class Model(LitellmModel):
                # mini-swe-agent asks again on most errors; never once the
                # time is up, nor when the server is down
                abort_exceptions = [*LitellmModel.abort_exceptions, Stopped, ServerDown]

                def _query(self, messages, **kw):
                    if stop.is_set():
                        raise Stopped("the agent's time is up")
                    try:
                        return super()._query(messages, **kw)
                    except Exception as e:      # noqa: BLE001 — only the relay's word is ours
                        if stop.is_set():       # cut by the abort: never asked again
                            raise Stopped("the agent's time is up") from e
                        if DOWN in str(e):
                            raise ServerDown(str(e)[:300]) from e
                        raise

            class Agent(DefaultAgent):
                def save(self, path, *extra_dicts):     # never through a link
                    data = self.serialize(*extra_dicts)
                    if path:
                        ab.safe_write(Path(path).parent, Path(path).name,
                                      json.dumps(data, indent=2))
                    return data
            acfg = dict(cfg.get("agent") or {})
            acfg.pop("mode", None)
            if "cost_limit" in self.ours:
                acfg["cost_limit"] = float(self.ours["cost_limit"])
            acfg["output_path"] = host / ab.TRAJECTORY
            agent = held["agent"] = Agent(Model(**mcfg), env, **acfg)
            try:
                await asyncio.to_thread(agent.run, instruction)
            except asyncio.CancelledError:
                stop.set()              # Harbor's time limit: the loop stops at its next step
                raise
            except ServerDown as e:
                held["ours"] = f"ServerDown: {e}"
                raise
            last = (agent.messages[-1].get("extra") or {}) if agent.messages else {}
            ab.safe_write(host, ab.PATCH, str(last.get("submission") or ""))
        except BaseException as e:
            failed = e
            raise
        finally:
            stop.set()
            # a reply still being written for this trial is cut
            await asyncio.to_thread(abort_request, str(self.ours.get("relay") or ""),
                                    str(self.session_id or ""))
            # however it stopped: nothing the model left runs on, and nothing
            # it put in the mounted folders is there when the verifier runs
            # or a host process looks. Harbor 0.24.0's exec is exec(command,
            # cwd=None, env=None, timeout_sec=None, user=None)
            # (environments/base.py): root, whoever the task's agent runs as
            # 18c point 11: the exec path is closed first, and a command sent
            # just before the stop is waited for — none starts after the kill
            sent = held["env"].close() if held["env"] is not None else []
            late = []
            if sent:
                _, late = await asyncio.wait([asyncio.wrap_future(f) for f in sent],
                                             timeout=SENT_WAIT_S)
            try:
                r = await environment.exec(CLEAN, timeout_sec=180, user="root")
                code = getattr(r, "return_code", None)
                why = clean_verdict(code, (getattr(r, "stdout", "") or "")
                                    + (getattr(r, "stderr", "") or ""))
            except Exception as e:      # noqa: BLE001 — recorded, and the task isn't verified
                code, why = None, f"{type(e).__name__}: {str(e)[:200]}"
            if late and not why:
                why = (f"{len(late)} command{'s' if len(late) > 1 else ''} sent before the stop "
                       f"hadn't come back after {SENT_WAIT_S} s")
            held["clean"] = {"exit": code, "ok": not why, "why": why}
            try:
                fill()
            except Exception:           # noqa: BLE001 — result.json keeps the counts too
                pass
            # 18c point 3: a task whose clean-up didn't run is never verified
            # — Harbor verifies after a clean end or the agent's time limit
            # (the cancel), never after another exception, which stays the
            # one said
            if why and (failed is None or isinstance(failed, asyncio.CancelledError)):
                raise CleanupFailed(f"the clean-up after the agent didn't run to its end "
                                    f"({why}): not verified")

    def populate_context_post_run(self, context) -> None:
        return None
