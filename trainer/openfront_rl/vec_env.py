"""Varias partidas en paralelo, una por proceso Node."""
from __future__ import annotations

import os
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
                 auto_reset: bool = True, **kw: Any):
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

    def _seed(self, i: int) -> str:
        return f"{self.base_seed}-{i}-{self._episode[i]}"

    def reset(self) -> list[Result]:
        self._episode = [0] * self.n
        for i, g in enumerate(self.games):
            g.send_reset(self._seed(i))
        return [g.recv_result() for g in self.games]

    def step(self, actions: np.ndarray) -> list[Result]:
        """`actions`: [N, agentes, 6]."""
        for g, a in zip(self.games, actions):
            g.send_step(a)
        results = [g.recv_result() for g in self.games]
        self.final = [None] * self.n
        if self.auto_reset:
            for i, r in enumerate(results):
                if r.done:
                    self.final[i] = r
                    self._episode[i] += 1
                    self.games[i].send_reset(self._seed(i))
            for i, r in enumerate(results):
                if r.done:
                    results[i] = self.games[i].recv_result()
        return results

    def close(self) -> None:
        for g in self.games:
            g.close()

    def __enter__(self) -> "VecGame":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
