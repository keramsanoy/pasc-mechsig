"""Make ``pasc`` importable when the package is not installed.

Every script in this directory starts with ``import _bootstrap``. If ``pasc`` is
installed (``pip install -e .``) this is a no-op; otherwise the repository's
``src/`` directory is put on ``sys.path`` so the scripts run from a plain clone.
"""
import os
import sys

try:
    import pasc  # noqa: F401
except ImportError:  # pragma: no cover - only when the package is not installed
    _SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    import pasc  # noqa: F401
