"""Varias partidas en paralelo, una por proceso Node."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from .client import NodeGame, Result


def default_workers() -> int:
    """Núcleos − 1 (mínimo 1): un núcleo se deja libre para Python y el entrenamiento."""
    return max(1, (os.cpu_count() or 2) - 1)


class VecGame:
    """N partidas con los mismos ajustes. `step` envía a todas y luego lee de todas, así los procesos Node corren a la vez.

    Con `auto_reset`, una partida terminada se reinicia sola con una semilla nueva; el `Result` que se devuelve
    ya es el del episodio nuevo y `final` guarda el último resultado del episodio que acabó.
    """

    def __init__(self, n: int | None = None, config: dict[str, Any] | None = None, base_seed: str = "run",
                 auto_reset: bool = True, profile: bool = False, **kw: Any):
        self.n = n or default_workers()
        self.base_seed = base_seed
        self.auto_reset = auto_reset
        self.games: list[NodeGame] = []
        try:
            for _ in range(self.n):
                self.games.append(NodeGame(config, **kw))
        except Exception:
            self.close()
            raise
        self.manifest = self.games[0].manifest
        self._episode = [0] * self.n
        self.final: list[Result | None] = [None] * self.n
        self._last: list[Result] = []
        # Con `profile`, cada recv corre en su hilo y se mide cuánto tardó cada worker (docs/adr/0010): `lat[i]` = segundos desde
        # que se enviaron las acciones hasta que llegó su resultado (0 si no estaba activo).
        self._pool = ThreadPoolExecutor(self.n) if profile else None
        self.lat = [0.0] * self.n

    def _seed(self, i: int) -> str:
        return f"{self.base_seed}-{i}-{self._episode[i]}"

    def reset(self) -> list[Result]:
        self._episode = [0] * self.n
        for i, g in enumerate(self.games):
            g.send_reset(self._seed(i))
        self._last = [g.recv_result() for g in self.games]
        return list(self._last)

    def step(self, actions: np.ndarray, active: np.ndarray | None = None) -> list[Result]:
        """`actions`: [N, agentes, 6]. Las partidas con `active[i]` falso no avanzan: su proceso queda en espera y se
        devuelve su último resultado (docs/specs/dashboard.md §5)."""
        act = [True] * self.n if active is None else [bool(x) for x in active]
        t0 = time.perf_counter()
        for i, (g, a) in enumerate(zip(self.games, actions)):
            if act[i]:
                g.send_step(a)
        if self._pool is None:
            results = [g.recv_result() if act[i] else self._last[i] for i, g in enumerate(self.games)]
        else:
            def recv(g: NodeGame) -> tuple[Result, float]:
                r = g.recv_result()
                return r, time.perf_counter() - t0
            futs = [self._pool.submit(recv, g) if act[i] else None for i, g in enumerate(self.games)]
            got = [f.result() if f is not None else None for f in futs]
            results = [got[i][0] if got[i] else self._last[i] for i in range(self.n)]
            self.lat = [got[i][1] if got[i] else 0.0 for i in range(self.n)]
        self.final = [None] * self.n
        if self.auto_reset:
            for i, r in enumerate(results):
                if act[i] and r.done:
                    self.final[i] = r
                    self._episode[i] += 1
                    self.games[i].send_reset(self._seed(i))
            for i, r in enumerate(results):
                if act[i] and r.done:
                    results[i] = self.games[i].recv_result()
                    if self._pool is not None:
                        self.lat[i] = time.perf_counter() - t0  # el reinicio de la partida también es trabajo del worker
        self._last = list(results)
        return results

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        for g in self.games:
            g.close()

    def __enter__(self) -> "VecGame":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
