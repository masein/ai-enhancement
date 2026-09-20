# Brief for Claude Code — the generator writes for the wrong reader (phase 8f)

Do this after 8e. One PR, small.

## What the two demo runs of 2026-09-20 showed

Both topics ran end to end and the datasets passed the gate. But read the
documents. Medicine produced a clinical case ("Mr. Peterson, an 80-year-old
admitted for persistent leg swelling… serial troponin monitoring, urgent
ECG"); law produced guidance for lawyers ("in drafting advisory memoranda,
the professional framing should be…"). The exam is members of the public
asking about their own symptoms and their own disputes; the spec said the
model fails to answer *them*. A model fine-tuned on clinician notes and
associate memos learns a register it is never asked to produce.

The cause is structural, not the model's taste: `proposals.py` hands the
generator the approved spec and nothing else — by design, so no question
text leaks. The audience was never in the spec because the proposal step
never had it either.

## The fix

**Audience and register travel with the topic, as labels.** From the bank's
`meta`, per topic, compute a short audience line at proposal time and pass
it through to generation:

```
Audience: members of the public asking about their own situation
(styles: conversational 78%, context_rich 12%, telegraphic 10%;
subjects: self 70%, child 22%, parent 6%, …).
Register for documents: guidance a layperson can read and act on — what to
do, what to watch for, when to escalate — not clinical notes, case files,
legal memoranda or textbook exposition. Second person is fine.
```

- The counts come from `meta.style` and `meta.subject` (and `intent`) over
  the topic's **whole** bank — labels and percentages only. No prompt text,
  no qids. The recorded-request-body tests must show that nothing else
  from the bank reaches either request.
- The register sentence is per topic and lives in the criteria file as an
  optional `"audience"` string, so the author can set it; when absent, the
  line above is the default for any topic whose items carry `style` or
  `subject`, and nothing is added for topics that carry neither (the
  legacy `other` items).
- It goes into the **proposal** request (so the spec is written for the
  right reader) and into the **generation** request (so the documents are).
  Record it in provenance under `audience` beside the spec.
- Add one line to the generation prompt's format instructions: the
  document should read as something a person with that question would be
  helped by reading — an explainer, a guide, a worked "what to do if…" —
  and should never be shaped as a question followed by an answer (that
  rule already exists; keep it next to the new one).

## The token cap, per role

The law run lost 1 of 20 documents to `LOCAL_MAX_TOKENS=1024` with the
count already at one document per request; there is no smaller request to
make. Replace the single cap with per-role caps: `LOCAL_MAX_TOKENS_LLM`
(generation and proposals, default **1536**), `LOCAL_MAX_TOKENS_JUDGE`
(default 1024 — a 23-key reply is ~350 tokens), `LOCAL_MAX_TOKENS_EXAM`
(default 1024). `LOCAL_MAX_TOKENS`, if set, is the fallback for any role
without its own. Document in `.env.example` and `SERVICE.md` with the
shared-card reasoning: only generation needs the headroom, and 1536 on two
concurrent requests is well inside what the card has left with vLLM and
Ollama loaded. The truncation counter and the demo's note stay.


## Also in this PR: the author's revised law bank

Dr. Hossein returned `eval_tasks/fr/law_v2.json`: the same 100
prompts and metadata as v1, plus **his own per-question `difficulty`**
(20 / 33 / 31 / 13 / 3 across levels 1–5; 55 items differ from the
id-range mapping v1 carried) and **`jurisdiction_required`** on every item
(85 true, 15 false). Prompts are byte-identical, so qids and the split do
not move.

- Delete `law_v1.json` (git keeps it); point `DEMO.md`, the
  demo's printed live-import line and any fixture at v2.
- On a bank that already holds v1, `import` of v2 must **update the
  `meta` of the 100 existing records** (same qid) rather than skip them —
  the prompt is the identity, the metadata is the author's to revise.
  Record `source: law_v2` and the new `accepted_at`; keep
  `accepted_by`. Print "100 updated, 0 imported, 0 skipped". Add the
  test. (v1 → v2 medicine was 50 skipped + 50 new because nothing on the
  existing 50 changed; this is the other case.)
- The judge must see `jurisdiction_required`. The reference line built
  from metadata gains `Jurisdiction required: yes.` / `no.` after
  `Difficulty:` when the field is present — the `jurisdiction_awareness`
  criterion in his file is defined against it. A reference-line change
  changes what the judge reads, so the demo's law numbers from
  2026-09-20 are not comparable to the next run; say so in the PR.
- `jurisdiction_required` joins the default breakdown list when items
  carry it, so the page and the demo show law's means for the 85 vs the
  15 — that table is the direct test of whether the model asks for the
  jurisdiction when it should.

## Tests

Audience line built from a fixture bank with known style/subject counts;
absent when the topic carries neither field; present in the proposal and
generation request bodies (fake backend) and in provenance; author's
`audience` string overrides the default; the request-body leak tests
unchanged and passing; per-role caps resolve with and without the
fallback; v2 law import updates 100 metas in place; the reference line
carries `Jurisdiction required:`; the breakdown table appears for law and
not for medicine.

## Definition of done

Re-run the law demo. The printed document reads as guidance to a member
of the public with a landlord, an employer or a summons — not as a
memorandum — and 20 of 20 documents survive the cap.
