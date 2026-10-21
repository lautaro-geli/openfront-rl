"""Estadísticas y registro por worker (docs/specs/dashboard.md §5b)."""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

from . import runctl

RECENT = 10


class WorkerStats:
    """Acumula, por worker, lo ocurrido en sus partidas terminadas."""

    def __init__(self, n: int):
        self.n = n
        self.w: list[dict[str, Any]] = [self._new() for _ in range(n)]

    @staticmethod
    def _new() -> dict[str, Any]:
        return {"games": 0, "wins": 0, "eliminated": 0, "sum_max": 0.0, "sum_final": 0.0, "sum_ticks": 0,
                "conquered_by_tribe": 0, "conquered_by_nation": 0, "tribes_conquered": 0, "nations_conquered": 0,
                "recent": []}

    def end_game(self, i: int, *, won: bool, eliminated: bool, max_territory: float, final_territory: float, ticks: int,
                 events: dict[str, float]) -> dict[str, Any]:
        """Anota una partida terminada y devuelve su resumen (el que va al registro)."""
        s = self.w[i]
        s["games"] += 1
        s["wins"] += int(won)
        s["eliminated"] += int(eliminated)
        s["sum_max"] += max_territory
        s["sum_final"] += final_territory
        s["sum_ticks"] += ticks
        for k in ("conquered_by_tribe", "conquered_by_nation", "tribes_conquered", "nations_conquered"):
            s[k] += int(events.get(k, 0))
        by = "tribu" if events.get("conquered_by_tribe", 0) else "nación" if events.get("conquered_by_nation", 0) else None
        game = {"n": s["games"], "result": "victoria" if won else "eliminado" if eliminated else "tiempo",
                "max_territory": round(max_territory, 4), "final_territory": round(final_territory, 4), "ticks": ticks,
                "tribes_conquered": int(events.get("tribes_conquered", 0)), "nations_conquered": int(events.get("nations_conquered", 0)),
                "conquered_by": by}
        s["recent"] = (s["recent"] + [game])[-RECENT:]
        return game

    def summary(self, i: int) -> dict[str, Any]:
        s = self.w[i]
        g = max(1, s["games"])
        return {"games": s["games"], "wins": s["wins"], "eliminated": s["eliminated"],
                "avg_max_territory": s["sum_max"] / g if s["games"] else None,
                "avg_final_territory": s["sum_final"] / g if s["games"] else None,
                "avg_length": s["sum_ticks"] / g if s["games"] else None,
                **{k: s[k] for k in ("conquered_by_tribe", "conquered_by_nation", "tribes_conquered", "nations_conquered")},
                "recent": s["recent"]}

    def save(self, path: Path) -> None:
        runctl.write_json(path, self.w)

    def load(self, path: Path) -> None:
        data = runctl.read_json(path)
        if isinstance(data, list):
            for i, d in enumerate(data[: self.n]):
                self.w[i] = {**self._new(), **d}


class WorkerLogs:
    """Un archivo de texto por worker en `runs/<run>/workers/`."""

    def __init__(self, run: Path, n: int):
        self.dir = run / "workers"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = n

    def write(self, i: int, msg: str) -> None:
        with open(self.dir / f"worker-{i}.log", "a", encoding="utf8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def game_line(g: dict[str, Any]) -> str:
    return (f"partida {g['n']}: {g['result']} | territorio máx {100 * g['max_territory']:.1f}% final {100 * g['final_territory']:.1f}% "
            f"| {g['ticks']} ticks | tribus conquistadas {g['tribes_conquered']}, naciones {g['nations_conquered']} "
            f"| conquistado por {g['conquered_by'] or '–'}")
