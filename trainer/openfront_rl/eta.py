"""Estima cuánto falta para que termine un run, con los tiempos medidos en el propio run (trainer/README.md).

    python -m openfront_rl.eta --run runs/Day19 [--final-eval 80] [--final-workers 4]

Usa solo `metrics.jsonl`, `status.json` y `config.yaml` del run:
- duración de una actualización: mediana de las diferencias de `time` entre actualizaciones sin evaluación (últimas 15);
- costo de una evaluación dentro del run: lo que la actualización siguiente a una evaluación tardó de más (la evaluación corre
  después de escribir la fila con `eval_won`, así que su tiempo cae en la diferencia siguiente), convertido a segundos por partida
  y por worker; sin evaluaciones medidas todavía, estima con la duración media de una partida de entrenamiento;
- partidas por actualización: media de las últimas 15 (cambia con las victorias: las ganadas son más cortas).
Devuelve un rango (percentiles 25 y 75), no un número único. La evaluación final del .bat (`evalckpt`) se suma con `--final-eval`.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf8").splitlines():
        if line.strip():
            d = json.loads(line)
            rows.setdefault(int(d["update"]), d)  # sin duplicados por `update` (dos lanzamientos sobre la misma carpeta)
    return [rows[k] for k in sorted(rows)]


def estimate(rows: list[dict[str, Any]], status: dict[str, Any], cfg: dict[str, Any], final_eval: int = 0,
             final_workers: int | None = None, window: int = 15) -> dict[str, Any]:
    """Segundos restantes (p25, mediana, p75) y el detalle de las tasas medidas."""
    workers = int(cfg.get("workers", 4))
    ev = cfg.get("eval", {}) or {}
    every, ev_games = int(ev.get("every", 0) or 0), int(ev.get("episodes", 0) or 0)
    dts, extra_per_game = [], []
    for a, b in pairwise(rows):
        dt = float(b["time"]) - float(a["time"])
        if "eval_won" in a:
            extra_per_game.append((dt, int(a.get("eval_episodes", ev_games) or ev_games)))
        elif "eval_won" not in b:
            dts.append(dt)
    dts = dts[-window:]
    if not dts:
        raise ValueError("hacen falta al menos dos actualizaciones sin evaluación")
    upd = np.percentile(dts, [25, 50, 75])
    eps = [float(r.get("episodes", 0)) for r in rows[-window:]]
    gpu = max(0.5, float(np.mean(eps)))
    # Costo por partida de evaluación y por worker (segundos·worker / partida).
    # Una partida de entrenamiento: actualización × workers / partidas por actualización (incluye el paso PPO, así que sobreestima un poco).
    g_train = float(upd[1]) * workers / gpu
    if extra_per_game:
        per_game = [max(0.0, dt - float(upd[1])) * workers / g for dt, g in extra_per_game if g > 0]
        n = len(per_game)
        if n < 3:  # con pocas evaluaciones una sola lenta (p. ej. la primera) domina: se mezcla con la estimación por entrenamiento
            per_game = per_game + [g_train]
        eval_pg = np.percentile(per_game, [25, 50, 75])
        eval_src = f"medido en {n} evaluaciones" + (" y mezclado con la duración de las partidas de entrenamiento" if n < 3 else "")
    else:
        eval_pg = np.array([0.8 * g_train, g_train, 1.25 * g_train])
        eval_src = "estimado con la duración de las partidas de entrenamiento (sin evaluaciones medidas)"
    u = int(status.get("update", rows[-1]["update"]))
    left_eps = max(0.0, float(status.get("episodes_total") or 0) - float(status.get("episodes_done") or 0))
    left_upd = math.ceil(left_eps / gpu) if left_eps else 0
    n_evals = sum(1 for k in range(u + 1, u + left_upd + 1) if every and k % every == 0)
    fw = final_workers or workers
    out = {}
    for i, name in enumerate(("p25", "mediana", "p75")):
        train_s = left_upd * float(upd[i])
        evals_s = n_evals * ev_games * float(eval_pg[i]) / workers
        final_s = final_eval * float(eval_pg[i]) / fw
        out[name] = train_s + evals_s + final_s
    return {"segundos": out, "actualizacion_s": [round(float(x)) for x in upd], "partidas_por_actualizacion": round(gpu, 1),
            "partidas_restantes": int(left_eps), "actualizaciones_restantes": left_upd, "evaluaciones_restantes": n_evals,
            "eval_s_por_partida_y_worker": [round(float(x)) for x in eval_pg], "eval_fuente": eval_src}


def fmt(s: float) -> str:
    m = round(s / 60)
    return f"{m // 60} h {m % 60:02d} min" if m >= 60 else f"{m} min"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="carpeta del run (con metrics.jsonl, status.json y config.yaml)")
    ap.add_argument("--final-eval", type=int, default=0, help="partidas de la evaluación final del .bat (evalckpt)")
    ap.add_argument("--final-workers", type=int, default=None)
    a = ap.parse_args(argv)
    run = Path(a.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf8"))
    status = json.loads((run / "status.json").read_text(encoding="utf8"))
    r = estimate(load_rows(run / "metrics.jsonl"), status, cfg, a.final_eval, a.final_workers)
    s = r.pop("segundos")
    print(json.dumps(r, ensure_ascii=False))
    print(f"Falta: {fmt(s['mediana'])} (entre {fmt(s['p25'])} y {fmt(s['p75'])})")


if __name__ == "__main__":
    main()
