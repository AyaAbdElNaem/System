"""
pipeline_utils.py
==================
Tiny helper so the numerically-prefixed pipeline stage files
(01_documents.py, 02_preprocessing.py, ...) can import one another.

Python module names can't start with a digit, so a plain
``import 01_documents`` is invalid syntax. ``importlib.import_module``
works around this fine since it resolves by file lookup rather than
identifier rules, as long as the project root is on ``sys.path`` (true by
default for the currently running script's own directory).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_stage(module_name: str) -> ModuleType:
    """
    Import and return one of the numbered pipeline stage modules, e.g.
    ``load_stage("01_documents")``.
    """
    return importlib.import_module(module_name)
