import os
import shutil
from pathlib import Path

import pytest

BRIDGE = Path(__file__).resolve().parents[2] / "bridge"


def pytest_collection_modifyitems(config, items):
    has_tsx = (BRIDGE / "node_modules" / "tsx" / "dist" / "cli.mjs").exists()
    has_node = bool(os.environ.get("OPENFRONT_NODE") or shutil.which("node"))
    if has_tsx and has_node:
        return
    skip = pytest.mark.skip(reason="falta Node 24 o `npm ci` en bridge/: estas pruebas usan el motor real")
    for item in items:
        item.add_marker(skip)
