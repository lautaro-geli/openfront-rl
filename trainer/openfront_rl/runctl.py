"""Archivos de control y estado de un run (docs/specs/dashboard.md): el panel y `train.py` se hablan por aquí."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch.nn as nn

# Grupos de la arquitectura, por prefijo del nombre del módulo (docs/adr/0007-red-por-ramas-y-ppo.md).
GROUPS: list[tuple[str, str, str]] = [
    ("enc.conv", "map", "Rama mapa"), ("enc.map_fc", "map", "Rama mapa"), ("enc.key_cell", "map", "Rama mapa"),
    ("enc.bias_cell", "map", "Rama mapa"),
    ("enc.econ", "econ", "Rama economía"), ("enc.key_econ", "econ", "Rama economía"),
    ("enc.mil", "mil", "Rama defensa/ataque"), ("enc.att", "mil", "Rama defensa/ataque"),
    ("enc.key_mil", "mil", "Rama defensa/ataque"), ("enc.key_att", "mil", "Rama defensa/ataque"),
    ("enc.cmp", "cmp", "Rama comparación"), ("enc.key_player", "cmp", "Rama comparación"),
    ("enc.core", "core", "Núcleo"),
    ("kind_head", "heads", "Cabezas de acción"), ("kind_emb", "heads", "Cabezas de acción"),
    ("act", "heads", "Cabezas de acción"), ("q_", "heads", "Cabezas de acción"), ("type_emb", "heads", "Cabezas de acción"),
    ("frac_head", "heads", "Cabezas de acción"), ("btype_head", "heads", "Cabezas de acción"),
    ("ntype_head", "heads", "Cabezas de acción"), ("aux_head", "heads", "Cabezas de acción"),
    ("value", "value", "Cabeza de valor"),
]


def _group(name: str) -> tuple[str, str]:
    for prefix, key, label in GROUPS:
        if name.startswith(prefix):
            return key, label
    return "other", "Otros"


def _shape(m: nn.Module) -> str | None:
    if isinstance(m, nn.Linear):
        return f"{m.in_features} → {m.out_features}"
    if isinstance(m, nn.Conv2d):
        return f"{m.in_channels} → {m.out_channels}, k{m.kernel_size[0]}, paso {m.stride[0]}"
    if isinstance(m, nn.Embedding):
        return f"{m.num_embeddings} × {m.embedding_dim}"
    return None


def architecture(policy: nn.Module) -> dict[str, Any]:
    """Módulos con parámetros reales, agrupados por rama. Se compone solo de módulos hoja con parámetros propios."""
    mods: list[dict[str, Any]] = []
    for name, m in policy.named_modules():
        own = sum(p.numel() for p in m.parameters(recurse=False))
        if own == 0:
            continue
        key, label = _group(name)
        mods.append({"name": name, "type": type(m).__name__, "params": own, "shape": _shape(m), "group": key, "label": label})
    # Parámetros sueltos (no son un módulo hoja), p. ej. `nn.Parameter` directos del Encoder.
    total = sum(p.numel() for p in policy.parameters())
    counted = sum(m["params"] for m in mods)
    groups: dict[str, dict[str, Any]] = {}
    for m in mods:
        g = groups.setdefault(m["group"], {"key": m["group"], "label": m["label"], "params": 0})
        g["params"] += m["params"]
    return {"total": total, "unlisted": total - counted, "groups": list(groups.values()), "modules": mods}


def write_json(path: Path, obj: Any) -> None:
    """Escritura atómica: el panel nunca lee un archivo a medias."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return default


def read_command(run: Path) -> str:
    cmd = (read_json(run / "control.json", {}) or {}).get("command", "run")
    return cmd if cmd in ("run", "pause", "stop") else "run"


def read_control(run: Path) -> tuple[str, list[int]]:
    """(comando, workers pausados)."""
    c = read_json(run / "control.json", {}) or {}
    cmd = c.get("command", "run")
    paused = c.get("paused_workers", [])
    ok = [int(i) for i in paused if isinstance(i, int) and not isinstance(i, bool) and i >= 0] if isinstance(paused, list) else []
    return (cmd if cmd in ("run", "pause", "stop") else "run"), sorted(set(ok))


def write_control(run: Path, command: str | None = None, paused_workers: list[int] | None = None) -> None:
    """Cambia solo lo que se pasa y conserva el resto."""
    cmd, paused = read_control(run)
    write_json(run / "control.json", {"command": command or cmd, "paused_workers": paused if paused_workers is None else sorted(set(paused_workers))})


def write_status(run: Path, state: str, **fields: Any) -> None:
    write_json(run / "status.json", {"pid": os.getpid(), "state": state, "updated_at": time.time(), **fields})
