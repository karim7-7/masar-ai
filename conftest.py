"""
conftest.py
────────────
Root-level pytest configuration.

Why this file exists:
  pytest collects tests from the `tests/` subfolder but `main.py` lives at the
  project root. Without adding the root to sys.path, `from main import app`
  inside the fixture raises an ImportError (or a PytestUnraisableExceptionWarning
  on some pytest versions because the import happens inside a function body
  after the module collection phase).

  This conftest.py runs before any test module is imported, so sys.path is
  ready before the fixture even tries to import `main`.
"""

import sys
import os

# ── Add project root to sys.path ───────────────────────────────────────────────
# This lets `from main import app` and `from app.xxx import yyy` work from
# anywhere pytest is invoked (project root, tests/ folder, CI).
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)