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
- its trajectory file, written after every step.

Before the model is asked anything, the container is checked from inside:
nothing answers on the targets the runner gives (the host's other ports, the
tailnet, the LAN, the internet), no Docker socket, a process limit. After
the agent stops, every process it left is killed and any reward file it
wrote is removed, before the verifier runs.

Harbor and mini-swe-agent are imported only on the server, in the agent
venv; the adapter's logic is tested without them."""

from __future__ import annotations

import asyncio
import hashlib
import platform
import shlex
import threading
import time
from pathlib import Path

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


def reach_script(targets: list[str]) -> str:
    """a check run inside the container: one line a target that answered,
    then the Docker socket and the process limit"""
    lines = ["for t in " + " ".join(shlex.quote(t) for t in targets) + "; do",
             '  h="${t%:*}"; p="${t##*:}"',
             '  if timeout 3 bash -c "exec 3<>/dev/tcp/$h/$p" 2>/dev/null; then echo "REACHED $t"; fi',
             "done",
             "[ -e /var/run/docker.sock ] && echo 'REACHED docker.sock'",
             "p=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || cat /sys/fs/cgroup/pids/pids.max "
             "2>/dev/null); echo \"PIDS ${p:-unknown}\""]
    return "\n".join(lines)


def reach_verdict(output: str) -> str:
    """'' when nothing answered and processes are limited; else why not"""
    reached = [x.split(" ", 1)[1] for x in output.splitlines() if x.startswith("REACHED ")]
    if reached:
        return "the task's container reached " + ", ".join(reached)
    pids = next((x.split(" ", 1)[1] for x in output.splitlines() if x.startswith("PIDS ")), "")
    if pids in ("", "max", "unknown"):
        return "the task's container has no process limit"
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

    def command(self, command: str, timeout: int) -> str:
        """the command as the container runs it: its shell, its time limit for
        the whole group, stderr into stdout"""
        return (f"timeout -s KILL {int(timeout)} {' '.join(map(shlex.quote, self.interpreter))} "
                f"{shlex.quote(command)} 2>&1")

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        if self.stop.is_set():
            raise Stopped("the agent's time is up")
        command = action.get("command", "")
        t = int(timeout or self.timeout)
        t0 = time.time()
        fut = asyncio.run_coroutine_threadsafe(
            self._exec(self.command(command, t), cwd=(cwd or self.cwd or None),
                       env=self.env or None, timeout_sec=t + 30), self.loop)
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
    """(tokens in, tokens out, steps) from the trajectory's replies"""
    tin = tout = steps = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        resp = (m.get("extra") or {}).get("response") or {}
        u = (resp.get("usage") or {}) if isinstance(resp, dict) else {}
        tin += int(u.get("prompt_tokens") or 0)
        tout += int(u.get("completion_tokens") or 0)
        steps += 1
    return tin, tout, steps


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

    async def run(self, instruction: str, environment, context) -> None:
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.models.litellm_model import LitellmModel

        targets = [t for t in str(self.ours.get("reach") or "").split(",") if t]
        probe = await environment.exec(reach_script(targets), timeout_sec=120)
        why = reach_verdict((probe.stdout or "") + (probe.stderr or ""))
        if why:
            raise ReachRefused(why)
        cfg = load_config(str(self.ours.get("config") or "mini.yaml"))
        ecfg = dict(cfg.get("environment") or {})
        kind = "docker" if ecfg.get("environment_class") == "docker" else "local"
        stop = threading.Event()
        env = HarborEnv(environment.exec, asyncio.get_running_loop(),
                        cwd=str(self.ours.get("cwd") or ecfg.get("cwd") or ""),
                        timeout=int(ecfg.get("timeout") or TIMEOUT_DEFAULT[kind]),
                        interpreter=tuple(ecfg.get("interpreter") or
                                          (("bash", "-c") if kind == "docker" else ("sh", "-c"))),
                        env=ecfg.get("env") or {}, stop=stop, kind=kind)
        mcfg = dict(cfg.get("model") or {})
        mcfg.pop("model_class", None)
        mcfg["model_name"] = self.model_name
        mcfg["cost_tracking"] = "ignore_errors"
        mcfg["model_kwargs"] = {**(mcfg.get("model_kwargs") or {}),
                                "api_base": str(self.ours.get("relay")), "api_key": "relay",
                                "timeout": 3600,
                                "extra_headers": {"X-Agent-Trial": str(self.session_id or "")}}
        class Model(LitellmModel):
            def query(self, messages, **kw):        # Harbor's time limit: no more asking
                if stop.is_set():
                    raise Stopped("the agent's time is up")
                return super().query(messages, **kw)
        acfg = dict(cfg.get("agent") or {})
        acfg.pop("mode", None)
        if "cost_limit" in self.ours:
            acfg["cost_limit"] = float(self.ours["cost_limit"])
        traj = Path(self.logs_dir) / "mini-swe-agent.trajectory.json"
        acfg["output_path"] = traj
        agent = DefaultAgent(Model(**mcfg), env, **acfg)

        def fill() -> None:
            tin, tout, steps = usage_of(agent.messages)
            last = (agent.messages[-1].get("extra") or {}) if agent.messages else {}
            context.n_input_tokens, context.n_output_tokens = tin, tout
            context.metadata = {"exit_status": str(last.get("exit_status") or ""),
                                "steps": steps, "config": self.ours.get("config"),
                                "prompt_sha256": prompt_sha(cfg),
                                "submission_chars": len(str(last.get("submission") or ""))}
        try:
            await asyncio.to_thread(agent.run, instruction)
        except asyncio.CancelledError:
            stop.set()                  # Harbor's time limit: the loop stops at its next command
            fill()
            raise
        except Exception:               # noqa: BLE001 — Harbor records it; the counts are kept
            fill()
            raise
        fill()
        last = (agent.messages[-1].get("extra") or {}) if agent.messages else {}
        (Path(self.logs_dir) / "patch.diff").write_text(str(last.get("submission") or ""))
        # nothing the model left runs on, and nothing it wrote is a reward
        await environment.exec("kill -9 -1 2>/dev/null; rm -f /logs/verifier/reward.txt "
                               "/logs/verifier/reward.json; true", timeout_sec=60)

    def populate_context_post_run(self, context) -> None:
        return None
