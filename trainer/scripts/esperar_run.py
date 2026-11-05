"""Espera a que un run termine (status.json con state=finished) o quede abandonado (sin actualizar en 15 min)."""
import json
import sys
import time
from pathlib import Path

st = Path(sys.argv[1]) / "status.json"
while True:
    try:
        s = json.loads(st.read_text(encoding="utf8"))
        if s.get("state") not in ("running", "paused", "starting"):
            break
        if time.time() - float(s.get("updated_at", 0)) > 900 and s.get("state") == "running":
            break
    except (OSError, ValueError):
        pass
    time.sleep(20)
print("run terminado:", s.get("state"))
