"""A stand-in for lm_eval 0.4.12, as far as choosing tasks goes, for
tests/test_task_discovery.py: the real one is not installed where the unit
suite runs. It follows the harness's rules for --tasks (lm_eval/config/
evaluate_config.py, process_tasks): one value that names a folder in the
working directory is read as a folder of task yaml files; anything else is
looked up by name in the index, and a name not there is "Tasks not found"."""

__version__ = "0.4.12-fake"

from lm_eval.evaluator import simple_evaluate  # noqa: E402,F401
