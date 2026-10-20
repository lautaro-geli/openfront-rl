"""Muestreo de acciones válidas a partir de la máscara. Sirve para políticas aleatorias y tests (docs/specs/actions.md)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .manifest import Manifest


def _pick(rng: np.random.Generator, valid: np.ndarray) -> int | None:
    idx = np.flatnonzero(valid)
    return None if idx.size == 0 else int(rng.choice(idx))


def sample_valid_action(m: Manifest, mask: dict[str, np.ndarray], rng: np.random.Generator,
                        agent: int = 0, allow: set[str] | None = None) -> np.ndarray:
    """Un vector `[kind, row, cell, fraction, type, aux]` válido según la máscara del agente `agent`.

    `allow` restringe los tipos de acción que se pueden elegir (por nombre). Si no hay ninguno válido, devuelve noop.
    """
    sp: dict[str, Any] = m.space
    kinds: list[str] = sp["kinds"]
    kmask = mask["kind"][agent].copy()
    if allow is not None:
        for i, k in enumerate(kinds):
            if k not in allow:
                kmask[i] = False
    k = _pick(rng, kmask)
    v = np.zeros(m.action_dims, dtype=np.int32)
    if k is None:
        return v
    name = kinds[k]
    v[0] = k
    n_cells = mask["cell"].shape[-1]

    def player_row(kind: str) -> int:
        i = sp["playerKinds"].index(kind)
        return _pick(rng, mask["player"][agent][i]) or 0

    def unit_row(kind: str) -> int:
        i = sp["unitKinds"].index(kind)
        return _pick(rng, mask["unit"][agent][i]) or 0

    def cell(kind: str) -> int:
        i = sp["cellKinds"].index(kind)
        return _pick(rng, mask["cell"][agent][i]) or 0

    frac = int(rng.integers(len(sp["fractions"])))
    if name == "spawn":
        v[2] = cell("spawn")
    elif name == "expand":
        v[3] = frac
    elif name == "attack":
        v[1] = 1 + (_pick(rng, mask["attack_slots"][agent]) or 0)
        v[3] = frac
    elif name == "boat_attack":
        v[2], v[3] = cell("boat_attack"), frac
    elif name == "cancel_attack":
        v[1] = _pick(rng, mask["attack_row"][agent]) or 0
    elif name in ("cancel_boat", "upgrade", "delete_unit"):
        v[1] = unit_row(name)
    elif name == "move_warship":
        v[1], v[2] = unit_row(name), cell("move_warship")
    elif name == "build":
        t = _pick(rng, mask["build_type"][agent]) or 0
        v[4] = t
        v[2] = cell("build_ship" if sp["buildTypes"][t] == "warship" else "build")
    elif name == "nuke":
        v[4] = _pick(rng, mask["nuke_type"][agent]) or 0
        v[2] = cell("nuke")
    elif name in sp["playerKinds"]:
        v[1] = player_row(name)
        if name == "emoji":
            v[5] = int(rng.integers(sp["emojis"]))
        if name in ("donate_gold", "donate_troops"):
            v[3] = frac
    return v
