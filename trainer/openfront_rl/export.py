"""Exporta un checkpoint de entrenamiento a un modelo de solo pesos y lo carga (docs/adr/0009-modelos-en-git.md).

    python -m openfront_rl.export --ckpt runs/Day2/latest.pt --out ../models/tribus-day2 --note "..." --results "24/50 victorias"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from .manifest import Manifest
from .model import ModelConfig, Policy


def export_model(ckpt: str | Path, out: str | Path, note: str = "", results: str = "", source: str = "") -> Path:
    """Guarda `<out>.pt` (solo pesos) y `<out>.json` (origen, configuración y resultados). Devuelve la ruta del `.pt`."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pt = out.with_suffix(".pt")
    torch.save({"model": ck["model"], "protocol_version": ck["protocol_version"], "config_model": ck["config"].get("model", {}),
                "torch_version": ck.get("torch_version")}, pt)
    meta: dict[str, Any] = {
        "source": source or str(ckpt), "update": ck["update"], "episodes_done": ck.get("episodes_done"), "steps": ck.get("steps"),
        "protocol_version": ck["protocol_version"], "scenario": ck["config"].get("env"), "config": ck["config"],
        "results": results, "note": note, "exported": time.strftime("%Y-%m-%d"), "size_bytes": pt.stat().st_size,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8")
    return pt


def load_policy(path: str | Path, manifest: Manifest, device: str = "cpu") -> Policy:
    """Carga un modelo exportado. Falla si la versión del protocolo no coincide con la del manifiesto."""
    data = torch.load(path, map_location=device, weights_only=False)
    if data["protocol_version"] != manifest.raw["version"]:
        raise ValueError(f"el modelo es del protocolo v{data['protocol_version']} y el entorno usa v{manifest.raw['version']}")
    policy = Policy(manifest, ModelConfig.from_dict(data.get("config_model", {}))).to(device)
    policy.load_state_dict(data["model"])
    policy.eval()
    return policy


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="ruta sin extensión: se escriben .pt y .json")
    ap.add_argument("--note", default="")
    ap.add_argument("--results", default="")
    ap.add_argument("--source", default="", help="texto que identifica el origen (p. ej. runs/Day2/latest.pt)")
    a = ap.parse_args(argv)
    print(export_model(a.ckpt, a.out, a.note, a.results, a.source))


if __name__ == "__main__":
    main()
