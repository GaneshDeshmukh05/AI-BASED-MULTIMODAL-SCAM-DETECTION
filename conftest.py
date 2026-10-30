"""conftest.py
Global pytest configuration and root path resolution for the test suite.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure project root is always in sys.path during pytest execution
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure numba shim is active across test runs
importlib.import_module("adapters.numba_compat")

# Support legacy import naming for domain intelligence module
try:
    import domain_intelligence
    sys.modules.setdefault("domain_intelligence", domain_intelligence)
except ImportError:
    pass
