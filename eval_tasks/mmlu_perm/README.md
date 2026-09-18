# mmlu_perm — the permutation control

MMLU, a fixed subset of twelve subjects, with each item's answer options
rotated left by `doc_id mod 4`. The correct answer therefore lands on A, B, C
and D equally often, whatever the published key does. Option text is untouched;
the few-shot examples are rotated the same way; there is no randomness and no
seed. `utils.py` is the whole transform.

It is a **control, not a leaderboard task**: it never enters the official
average (`CONTROL_TASKS` in `scripts/report_lm_eval.py`, tested), and the
Diagnose section of a model that has both `mmlu` and `mmlu_perm` shows the two
scores side by side with one derived sentence — *the format was hiding
measurable knowledge* or *the knowledge is not there to hide*.

Run it through the service, which takes the same lock and queue as every
evaluation:

```bash
python clients/bench_client.py --base http://<tailscale-ip>:8899 \
    submit HuggingFaceTB/SmolLM2-360M --suite control --submitter you
```

The subjects: anatomy, astronomy, business_ethics, clinical_knowledge,
college_computer_science, econometrics, high_school_geography,
high_school_macroeconomics, high_school_mathematics, philosophy,
us_foreign_policy, world_religions. Adding one is a six-line yaml copied from
any of the others; `tests/test_mmlu_perm.py` checks the set stays consistent.
