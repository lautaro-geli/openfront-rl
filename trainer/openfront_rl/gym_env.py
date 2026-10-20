"""Envoltorio Gymnasium de un agente. Para self-play con varios agentes se usa `NodeGame`/`VecGame` directamente."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import NodeGame


class OpenFrontEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None, **kw: Any):
        cfg = {**(config or {}), "agents": 1}
        self.game = NodeGame(cfg, **kw)
        m = self.game.manifest
        assert m is not None
        self.manifest = m
        sp = m.space
        self.observation_space = spaces.Dict(
            {n: spaces.Box(-1.0, 1.0, b.shape, np.float32) for n, b in m.obs.items()}
        )
        cells = m.grid[0] * m.grid[1]
        self.action_space = spaces.MultiDiscrete(
            [len(sp["kinds"]), sp["unitRows"], cells, len(sp["fractions"]), max(len(sp["buildTypes"]), len(sp["nukeKinds"])), sp["emojis"]]
        )
        self._episode = 0
        self._seed = "gym"

    def _out(self, r: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs = {n: np.array(a[0]) for n, a in r.obs.items()}
        info = {"action_mask": {n: a[0] for n, a in r.mask.items()}, "invalid": bool(r.invalid[0]),
                "won": bool(r.won[0]), "eliminated": bool(r.eliminated[0]), "tick": r.tick}
        return obs, info

    def reset(self, *, seed: int | str | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed if isinstance(seed, int) else None)
        if seed is not None:
            self._seed, self._episode = str(seed), 0
        r = self.game.reset(f"{self._seed}-{self._episode}")
        self._episode += 1
        return self._out(r)

    def step(self, action: np.ndarray):
        r = self.game.step(np.asarray(action, dtype=np.int32).reshape(1, -1))
        obs, info = self._out(r)
        terminated = bool(r.terminated[0])
        truncated = bool(r.truncated) and not terminated
        return obs, float(r.reward[0]), terminated, truncated, info

    def close(self) -> None:
        self.game.close()
