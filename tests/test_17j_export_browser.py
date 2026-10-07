"""17j on the page, part 4, point 23: a model marked public says CHECK until
its page gives the public file it is; given and checked, it says "the same
file as …, checked". Hugging Face is stood in for; nothing is sent anywhere."""

from __future__ import annotations

import json

import pytest

from conftest import set_name

pytestmark = pytest.mark.dashboard
SID = "served/Qwen3.6-35B-A3B-BF16-17j"
SHA = "ab" * 32


def hf(monkeypatch, files: dict, private=()):
    from service import public_files as pf

    def ask(url, method="HEAD"):
        rest = url.removeprefix(pf.HF + "/")
        repo, _, path = rest.partition("/resolve/main/")
        if repo in private:
            raise pf.NotPublic("Hugging Face asks to sign in for it: it is private or gated")
        if f"{repo}/{path}" not in files:
            raise pf.NotPublic("Hugging Face has no such file there")
        return {"X-Linked-Etag": json.dumps(files[f"{repo}/{path}"])}, b""
    monkeypatch.setattr(pf, "_ask", ask)


def test_23_the_page_says_check_until_the_public_file_is_given_and_checked(live, page,
                                                                         monkeypatch):
    import service.app as appmod
    from service import db, served
    db.served_put({"id": SID, "name": "Qwen3.6 BF16 17j", "base_url": "", "key": "",
                   "how": "unsloth's BF16 file", "based_on": "Qwen/Qwen3.6-35B-A3B",
                   "thinking": "auto", "pin": {"file": "x.gguf"}, "rented_only": True,
                   "by": "masein", "at": 0,
                   "file_sha256": {"sha256": SHA, "name": "x.gguf",
                                   "source": "hf://teamacct/evalboard-private@main/x.gguf"}})
    served.write_meta(db.served_get(SID))
    db.public_set(SID, True, "masein")
    db.public_file_set(SID, None, "masein")
    hf(monkeypatch, {"unsloth/Q/BF16/x.gguf": SHA, "teamacct/evalboard-private/x.gguf": SHA},
       private={"teamacct/evalboard-private"})
    appmod._cache.update(key=None, payload=None, at=0.0)
    page.set_viewport_size({"width": 1400, "height": 1000})
    page.goto("about:blank")
    page.goto(live["base"] + f"/#model={SID}")
    set_name(page, "masein")
    line = page.locator("[data-public-file]")
    line.wait_for()                                         # 0abb757: nothing asks
    assert line.get_attribute("data-public-file") == "check"
    assert line.inner_text().startswith("CHECK: not shown to be a public file")

    def give(repo, path):
        page.locator("[data-public-file-edit]").click()
        page.locator("[data-public-file-input='repo']").fill(repo)
        page.locator("[data-public-file-input='path']").fill(path)
        page.locator("[data-public-file-check]").click()
    # the team's private repository: still CHECK, and why
    give("teamacct/evalboard-private", "x.gguf")
    page.wait_for_function("(document.querySelector('[data-public-file]') || {}).textContent"
                           "?.includes('sign in')")
    assert line.get_attribute("data-public-file") == "check"
    # the public file: checked
    give("unsloth/Q", "BF16/x.gguf")
    page.wait_for_selector("[data-public-file='checked']")
    words = page.locator("[data-public-file]").inner_text()
    assert words.startswith("The same file as unsloth/Q/BF16/x.gguf, checked on "), words
    assert db.public_files_all()[SID]["same"] is True
    assert page.errors == []
