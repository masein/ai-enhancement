"""34: plan P's check on a box, before a question — the server stops the
thinking at the budget a request sends (thinking_budget_tokens: its
reasoning-budget sampler forces the end of thinking), and a reply follows.
One question that thinks long, asked with a budget of BUDGET tokens and room
for a short reply: it passes when the thinking came back within the budget
and the forced end (frontier.APP_ROOM), with a reply after it. A server that
ignores the budget thinks to the request's limit and gives no reply; the box
then stops, and nothing is asked"""
import path_check

BUDGET = 64
REPLY_ROOM = 400
QUESTION = ("Work out, showing every step, the sum of the squares of the whole numbers from 1 "
            "to 40, then check the total a second way before you answer.")


def check(base: str, path: int, room: int) -> dict:
    body = {"messages": [{"role": "user", "content": QUESTION}], "max_tokens": BUDGET + REPLY_ROOM,
            "expert_path": path, "thinking_budget_tokens": BUDGET}
    got = path_check._post(base, "/v1/chat/completions", body)
    ch = (got.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    think, reply = msg.get("reasoning_content") or "", msg.get("content") or ""
    n = len(path_check._post(base, "/tokenize", {"content": think, "add_special": False,
                                                 "parse_special": False}).get("tokens") or []) \
        if think else 0
    out = {"budget": BUDGET, "max_tokens": body["max_tokens"], "path": path,
           "thinking_tokens": n, "reply_chars": len(reply.strip()),
           "finish": ch.get("finish_reason")}
    why = []
    if not think:
        why.append("no thinking came back apart from the reply")
    if n > BUDGET + room:
        why.append(f"the thinking ran to {n} tokens, past the budget of {BUDGET}")
    if not reply.strip():
        why.append("no reply followed the thinking")
    out["pass"] = not why
    out["why"] = "; ".join(why)
    return out


def words(chk: dict) -> str:
    if chk["pass"]:
        return (f"the thinking budget holds on expert path k{chk['path']}: asked with "
                f"{chk['budget']}, the thinking came back at {chk['thinking_tokens']} tokens and a "
                f"reply of {chk['reply_chars']} characters followed")
    return f"the thinking budget doesn't hold on expert path k{chk['path']}: {chk['why']}"
