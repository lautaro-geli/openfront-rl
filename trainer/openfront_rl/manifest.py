"""Manifiesto que el servidor Node envía al iniciar (docs/specs/protocol.md §3)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

SUPPORTED_VERSION = 2
# Bloques que este cliente conoce. Si el servidor cambia el contrato, falla al arrancar, no a mitad del entrenamiento.
KNOWN_OBS = {
    "grid", "globals", "econ_table", "econ_valid", "econ_scalars", "mil_units", "mil_units_valid",
    "mil_attacks", "mil_attacks_valid", "mil_scalars", "players", "players_valid", "cmp_scalars",
}
KNOWN_MASK = {
    "spawn", "expand", "attack_slots", "kind", "player", "unit", "attack_row", "build_type",
    "nuke_type", "cell",
}


@dataclass(frozen=True)
class Block:
    name: str
    shape: tuple[int, ...]
    offset: int

    @property
    def size(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class Manifest:
    raw: dict[str, Any]
    agents: int
    obs_floats: int
    mask_bytes: int
    agent_bytes: int
    header_bytes: int
    agent_header_bytes: int
    reward_parts: tuple[str, ...]
    action_dims: int
    obs: dict[str, Block]
    mask: dict[str, Block]
    space: dict[str, Any]

    @staticmethod
    def parse(raw: dict[str, Any]) -> "Manifest":
        if raw.get("version") != SUPPORTED_VERSION:
            raise ValueError(f"versión del protocolo {raw.get('version')} no soportada (se esperaba {SUPPORTED_VERSION})")

        def blocks(section: dict[str, Any], known: set[str]) -> dict[str, Block]:
            out = {b["name"]: Block(b["name"], tuple(b["shape"]), b["offset"]) for b in section["blocks"]}
            unknown = set(out) - known
            missing = known - set(out)
            if unknown or missing:
                raise ValueError(f"el manifiesto no coincide con el cliente: desconocidos={sorted(unknown)} faltan={sorted(missing)}")
            return out

        return Manifest(
            raw=raw,
            agents=raw["agents"],
            obs_floats=raw["obs"]["floats"],
            mask_bytes=raw["mask"]["bytes"],
            agent_bytes=raw["result"]["agentBytes"],
            header_bytes=raw["result"]["headerBytes"],
            agent_header_bytes=raw["result"]["agentHeaderBytes"],
            reward_parts=tuple(raw["result"]["rewardParts"]),
            action_dims=raw["actionDims"],
            obs=blocks(raw["obs"], KNOWN_OBS),
            mask=blocks(raw["mask"], KNOWN_MASK),
            space=raw["actionSpace"],
        )

    @property
    def grid(self) -> tuple[int, int]:
        return self.raw["grid"]["h"], self.raw["grid"]["w"]
