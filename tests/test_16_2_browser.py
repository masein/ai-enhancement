"""16.2 on the page: GPU memory said one way everywhere. The status dot's
panel: "GPU memory: X of 32 GB in use · Y GB free" and what holds it. The
Playground's picker: each model's state (Ready, Loads on the first message, On
the CPU, slower, Not now and why), the reason in view; a model that can't
answer now holds Send back and says why, and Send comes back on its own when
the GPU frees. A load that runs out: one line, the message kept, Try again.
Test a model: "Needs about 9.5 GB · 12.4 GB free now · starts now". A run
waiting: "Waiting for GPU memory: needs 9.5 GB, 6.2 GB is free." Fakes only:
nvidia-smi's answers, free memory, a loader that runs out. No GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import open_submit, pg_choose, set_name
from service import chat, config, db, gpu
from test_14_3_browser import no_sideways, steady_shot

pytestmark = pytest.mark.dashboard
SCREENS = Path(__file__).resolve().parent / "_screens" / "phase16_2"
SMALL, BIG = "fx/below-135m-it", "fx/chat-1.7b-it"
MIB = 1024 ** 2


def shot(part, name):
    SCREENS.mkdir(parents=True, exist_ok=True)
    steady_shot(part, SCREENS / name)


def fake_smi(monkeypatch, card: str, apps: str, cmdlines: dict[int, str]):
    """nvidia-smi's two answers, and what each process runs"""
    monkeypatch.setattr(gpu, "_smi", lambda args, timeout=10.0:
                        card + "\n" if args[0].startswith("--query-gpu") else apps)
    monkeypatch.setattr(gpu, "_cmdline", lambda pid: cmdlines.get(pid, ""))
    monkeypatch.setattr(gpu, "_parent", lambda pid: 1)
    gpu._cache.clear()


VLLM = "python -m vllm.entrypoints.openai.api_server --model gemma"
SERVER = "/home/m/llama-server -m x.gguf --port 8090 -ngl 99"


@pytest.fixture(autouse=True)
def fake(live, monkeypatch):
    monkeypatch.setattr(config, "CHAT_BACKEND", "fake")
    monkeypatch.setattr(chat.FakeBackend, "delay_s", 0.005)
    monkeypatch.setattr(chat.FakeBackend, "responder", None)
    monkeypatch.setattr(chat.FakeBackend, "oom", None)
    monkeypatch.setattr(chat, "ENGINE", chat.Engine())
    from service import hfmeta
    monkeypatch.setattr(hfmeta, "preflight", lambda hf_id, kind="auto", **k: {
        "remote_code": False, "revision": None, "archinfo": {}, "need_gb": 9.0})
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 40 * 10 ** 9)
    d = config.OUT_DIR / BIG.replace("/", "__")
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_meta.json").write_text(json.dumps({
        "model": BIG, "kind": "instruct", "params": 1_700_000_000, "ctx": 4096,
        "tmpl_sha": "abc"}), encoding="utf-8")
    # the judge, this service (the Playground) and something the board can't name
    fake_smi(monkeypatch, "32607, 19866, 12741",
             "\n".join([f"{os.getpid()}, 3100", "41, 13200", "63, 400"]), {41: VLLM})
    import service.app as appmod
    appmod._NEED.clear()
    yield


def pg(page, base, width=1400):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(base + "/#tab=home")
    set_name(page, "masein")
    page.goto("about:blank")
    page.goto(base + "/#tab=playground")
    page.wait_for_selector("[data-pg-main]")


@pytest.mark.parametrize("width", [1400, 375])
def test_the_status_dot_says_gpu_memory_and_what_holds_it(live, page, width):
    page.set_viewport_size({"width": width, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    page.locator("#warnings [data-warn-summary]").click()
    pop = page.locator("#pop-checks")
    line = pop.locator("#checks-gpu [data-gpu-line='1']")
    line.wait_for()
    assert line.inner_text() == "GPU memory: 19.4 of 32 GB in use · 12.4 GB free"
    held = pop.locator("[data-gpu-holder]").all_inner_texts()
    assert held == ["the Playground 3.0 GB", "the judge 12.9 GB", "other: 0.4 GB"]
    no_sideways(page)
    shot(pop, f"status-dot-gpu-{width}.png")
    assert page.errors == []


def test_a_card_that_cant_be_read_is_one_line_in_the_panel(live, page, monkeypatch):
    def smi(args, timeout=10.0):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(gpu, "_smi", smi)
    gpu._cache.clear()
    page.set_viewport_size({"width": 1400, "height": 900})
    page.goto(live["base"] + "/#tab=home")
    page.locator("#warnings [data-warn-summary]").click()
    line = page.locator("#checks-gpu [data-gpu-line='0']")
    line.wait_for()
    assert line.inner_text() == "GPU memory can't be read on this server."
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_the_picker_says_each_models_state_and_send_waits_for_not_now(live, page, monkeypatch,
                                                                       width):
    # the GPU nearly full: a served model's server and the judge hold it
    fake_smi(monkeypatch, "32607, 30700, 1907", "41, 13200\n52, 14000\n63, 3400",
             {41: VLLM, 52: SERVER})
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 1907 * MIB)
    pg(page, live["base"], width)
    page.locator("[data-pg-model]").click()
    pop = page.locator("#pop-pg-model-a")
    big = pop.locator(f"[data-pg-option='{BIG}']")
    big.wait_for()
    assert "notnow" in big.get_attribute("class")
    assert big.locator("[data-pg-state]").inner_text() == (
        "Not now: chat-1.7b-it needs about 5.2 GB of GPU memory and 1.9 GB is free now.")
    small = pop.locator(f"[data-pg-option='{SMALL}'] [data-pg-state]")
    assert small.inner_text() == "On the CPU, slower"
    no_sideways(page)
    shot(pop, f"picker-states-{width}.png")
    page.keyboard.press("Escape")
    pg_choose(page, BIG)
    # Send is held back, the reason beside it; nothing is sent
    send = page.locator("[data-pg-send]")
    assert send.is_disabled()
    why = page.locator(f"[data-pg-notnow='{BIG}']")
    assert why.inner_text() == ("Not now: chat-1.7b-it needs about 5.2 GB of GPU memory and "
                                "1.9 GB is free now.")
    # the bar: the state's word, and one quiet line on the GPU — what holds it on hover
    tag = page.locator(f"[data-pg-state-of='{BIG}']")
    assert tag.inner_text() == "Not now" and tag.get_attribute("title").startswith("chat-1.7b-it")
    gl = page.locator("[data-pg-gpu]")
    assert gl.inner_text() == "GPU memory: 30.0 of 32 GB in use · 1.9 GB free"
    assert gl.get_attribute("title") == ("In use by: the judge 12.9 GB · llama-server on port "
                                         "8090 13.7 GB · other: 3.3 GB")
    page.locator("[data-pg-input]").fill("hello")
    page.locator("[data-pg-input]").press("Enter")
    assert page.locator("[data-pg-reply]").count() == 0
    shot(page.locator("[data-pg-main]"), f"send-held-back-{width}.png")
    # the server stops: the state is asked again every few seconds and Send comes back
    fake_smi(monkeypatch, "32607, 17000, 15607", "41, 13200\n63, 3400", {41: VLLM})
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 15607 * MIB)
    page.wait_for_selector("[data-pg-send]:not([disabled])", timeout=20000)
    assert page.locator("[data-pg-notnow]").count() == 0
    assert page.locator(f"[data-pg-state-of='{BIG}']").first.inner_text() == \
        "Loads on the first message"
    assert page.errors == []


def test_running_out_while_loading_keeps_the_message_and_try_again_answers(live, page,
                                                                          monkeypatch):
    monkeypatch.setattr(chat, "gpu_free_bytes", lambda: 6 * 10 ** 9)
    monkeypatch.setattr(chat.FakeBackend, "oom", "load")
    pg(page, live["base"])
    pg_choose(page, BIG)
    page.locator("[data-pg-input]").fill("hello")
    page.locator("[data-pg-input]").press("Enter")
    line = ("Ran out of GPU memory while loading chat-1.7b-it (it needed about 5.2 GB; 5.6 GB "
            "was free). Your message is kept.")
    said = page.locator("[data-pg-noreply], [data-pg-refused]").filter(has_text="Ran out")
    said.first.wait_for()
    assert said.first.inner_text().startswith(line)
    assert "hello" in page.locator("[data-pg-main]").inner_text()        # kept
    again = page.locator("[data-pg-try-again]")
    assert again.inner_text() == "Try again"
    shot(page.locator("[data-pg-main]"), "oom-try-again-1400.png")
    monkeypatch.setattr(chat.FakeBackend, "oom", None)
    again.click()
    page.wait_for_selector("[data-pg-reply] [data-pg-stats]")
    assert page.locator("[data-pg-noreply]").filter(has_text="Ran out").count() == 0
    assert page.errors == []


@pytest.mark.parametrize("width", [1400, 375])
def test_test_a_model_says_what_it_needs_whats_free_and_when(live, page, monkeypatch, width):
    page.set_viewport_size({"width": width, "height": 1000})
    open_submit(page, live["base"], "masein")
    box = page.locator("[data-dialog='test'] [data-ms='submit'] input")
    box.fill("Qwen/Qwen3.5-2B")
    need = page.locator("[data-gpu-need='starts now']")
    need.wait_for()
    assert need.inner_text() == "Needs about 9.5 GB · 12.4 GB free now · starts now"
    page.locator("#test-title").click()                    # the suggestions close
    page.wait_for_selector("#ms-list-submit", state="hidden")
    no_sideways(page)
    shot(page.locator("[data-dialog='test'] [data-submit-form]"), f"test-a-model-gpu-{width}.png")
    # less free than it needs: it says it waits
    fake_smi(monkeypatch, "32607, 26259, 6348", "41, 13200", {41: VLLM})
    box.fill("Qwen/Qwen3.5-9B")
    waits = page.locator("[data-gpu-need='waits for GPU memory']")
    waits.wait_for()
    assert waits.inner_text() == "Needs about 9.5 GB · 6.2 GB free now · waits for GPU memory"
    assert page.errors == []


def test_a_run_waiting_says_so_in_gb(live, page):
    sid = db.add("Qwen/Qwen3.5-9B", "instruct", "full", "masein", "", status="waiting_gpu")
    db.update(sid, progress=gpu.waiting_line(9.5 * gpu.GB, 6.2 * gpu.GB))
    try:
        page.set_viewport_size({"width": 1400, "height": 1000})
        page.goto(live["base"] + "/#tab=home")
        set_name(page, "masein")
        page.goto(live["base"] + "/#tab=runs")
        row = page.locator(f"[data-queue-row='{sid}']")
        row.wait_for()
        assert row.locator("[data-stage]").inner_text() == "waiting for GPU memory"
        assert row.locator(f"[data-watch='q|{sid}|progress']").inner_text().startswith(
            "Waiting for GPU memory: needs 9.5 GB, 6.2 GB is free.")
        shot(row, "run-waiting-1400.png")
        assert page.errors == []
    finally:
        db.update(sid, status="canceled", progress="")
