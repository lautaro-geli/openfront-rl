"""Mide pasos/s y cuánto cuesta el canal Node ⇄ Python. Uso: python scripts/bench.py --workers 1 5 --map World --bots 400"""
import argparse
import os
import time

import numpy as np

from openfront_rl import VecGame, sample_valid_action


def run(workers: int, cfg: dict, steps: int) -> dict:
    env = {**os.environ, "OPENFRONT_TIMINGS": "1"}
    os.environ.update(env)
    rng = np.random.default_rng(0)
    with VecGame(workers, cfg, base_seed="bench", auto_reset=True) as v:
        m = v.manifest
        rs = v.reset()
        # Calentamiento: pasar la fase de spawn.
        t_py = 0.0
        t0 = time.perf_counter()
        for _ in range(steps):
            a0 = time.perf_counter()
            acts = np.stack([np.stack([sample_valid_action(m, r.mask, rng, agent=i) for i in range(m.agents)]) for r in rs])
            t_py += time.perf_counter() - a0
            rs = v.step(acts)
        wall = time.perf_counter() - t0
        games = v.games
    time.sleep(0.5)  # deja que el hilo de stderr lea las líneas de tiempos que Node imprime al cerrar
    tails = [g.stderr_tail() for g in games]
    n = steps * workers
    return {"workers": workers, "steps": steps, "wall_s": wall, "steps_per_s": n / wall,
            "ms_per_step_per_worker": 1000 * wall / steps, "python_sampling_ms": 1000 * t_py / steps, "stderr": tails}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--map", default="Britannia")
    ap.add_argument("--bots", type=int, default=20)
    ap.add_argument("--agents", type=int, default=1)
    a = ap.parse_args()
    cfg = {"map": a.map, "bots": a.bots, "agents": a.agents, "maxTicks": 6000}
    for w in a.workers:
        r = run(w, cfg, a.steps)
        print(f"workers={r['workers']}: {r['steps_per_s']:.1f} pasos/s en total | {r['ms_per_step_per_worker']:.1f} ms por paso y worker"
              f" (de ellos, {r['python_sampling_ms']:.1f} ms muestreando acciones en Python)")
        for line in r["stderr"][:1]:
            print("   ", [l for l in line.splitlines() if l.startswith("timings")][-1:])
