import re
from pathlib import Path

# what the harness ships, of what the board runs
BUILT_IN = {"mmlu", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "piqa",
            "truthfulqa_mc2", "gsm8k"}


def names_in(yaml_text: str) -> set[str]:
    """a yaml's task name, or its group's"""
    got = set(re.findall(r"^group:\s*(\S+)\s*$", yaml_text, re.M))
    got |= set(re.findall(r"^task:\s*([^\s\[\]]+)\s*$", yaml_text, re.M))
    return got


class TaskManager:
    def __init__(self, include_path=None, metadata=None):
        self.index = set(BUILT_IN)
        for p in ([include_path] if isinstance(include_path, str) else include_path or []):
            for y in Path(p).rglob("*.yaml"):
                self.index |= names_in(y.read_text(encoding="utf-8"))

    def match_tasks(self, task_list):
        return [t for t in task_list if t in self.index]
