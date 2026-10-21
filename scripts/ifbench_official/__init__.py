# 25: vendored from allenai/IFBench at 1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d, ifbench/__init__.py
# 25: (Apache-2.0, LICENSE here), with these changes, each marked on its line: the
# 25: package's imports are this folder's; nothing is downloaded or made on import —
# 25: the image has NLTK's data (scripts/nltk_data.py). tests/test_25_ifbench.py undoes
# 25: each and checks every file against the commit's, byte for byte.
"""IFBench: verifiable instruction-following benchmark.

Exposes the verifier classes (`instructions`, `classic_instructions`),
the shared registry (`instructions_registry.INSTRUCTION_DICT`), and a
helper for locating bundled data files.
"""

from importlib import resources
from pathlib import Path

from . import classic_instructions, instructions, instructions_registry, instructions_util  # 25: was: from ifbench import classic_instructions, instructions, instructions_registry, instructions_util  # noqa: F401


def data_path(name: str = "IFBench_test.jsonl") -> Path:
    """Return the filesystem path to a bundled data file."""
    return Path(resources.files(__name__) / "data" / name)
