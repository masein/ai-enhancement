"""Benchmark-as-a-service: friends submit a Hugging Face model id, a single worker
runs it through the same lm-eval pipeline as scripts/run_benchmarks.sh, and the
interactive dashboard serves live results.

Design constraints this package answers to (see SERVICE.md for the runbook):

  * ONE shared GPU. A single worker thread runs one submission at a time, waits
    for free VRAM before each model, and takes the same results/.run.lock as the
    CLI script — the service and a manual run can never race each other.
  * Friends-scale. FastAPI + SQLite + one process. No Kafka, no Postgres, no
    accounts: the tailnet is the auth boundary. Every piece here is swappable for
    the heavier thing later without changing the API surface.
  * Same results tree. Service runs write into results/full exactly like the CLI,
    so everything stays comparable, resumable, and in one leaderboard.
"""
