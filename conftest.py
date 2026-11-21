"""Repository-level pytest configuration.

``tests/test_v3_edge_cases.py`` and ``tests/test_v3_integration.py`` are manual
end-to-end validation scripts: they execute at import time and require the
git-ignored ``Datasets/`` directory and the V3 checkpoint. They are excluded from
automatic collection and can be run directly once those artifacts are supplied:

    python tests/test_v3_edge_cases.py
    python tests/test_v3_integration.py
"""

collect_ignore = [
    "tests/test_v3_edge_cases.py",
    "tests/test_v3_integration.py",
]
