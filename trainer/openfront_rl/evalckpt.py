"""Evalúa un checkpoint sin entrenar: N partidas con cuota fija por worker (misma función que usa train.py).

    python -m openfront_rl.evalckpt --ckpt runs/Day8/ckpt_000700.pt --episodes 40 --workers 5 [--env difficulty=Easy] [--seed e1]
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from .model import ModelConfig, Policy
from .train import evaluate, parse_env, resolve_device
from .vec_env import VecGame


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", default="evalckpt")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--env", action="append", default=[], help="clave=valor, como en train (p. ej. difficulty=Easy)")
    a = ap.parse_args(argv)
    device = resolve_device(a.device)
    ck = torch.load(a.ckpt, map_location=device, weights_only=False)
    env: dict[str, Any] = {**ck["config"]["env"], **parse_env(a.env)}
    with VecGame(1, {**env, "agents": 1}) as v:
        manifest = v.manifest
    policy = Policy(manifest, ModelConfig.from_dict(ck["config"].get("model", {}))).to(device)
    policy.load_state_dict(ck["model"])
    policy.eval()
    out = evaluate(policy, env, a.workers, a.episodes, a.seed, device)
    n, p = out["eval_episodes"], out["eval_won"]
    out["ci95_won"] = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / max(n, 1))
    print(json.dumps({"ckpt": a.ckpt, "update": ck["update"], "env": {k: env[k] for k in ("map", "bots", "nations", "difficulty") if k in env}, **{k: round(v, 3) for k, v in out.items() if k in ("eval_episodes", "eval_won", "eval_eliminated", "eval_return", "ci95_won", "eval_ev_tribes_conquered", "eval_ev_nations_conquered", "eval_ev_built_port", "eval_ev_nukes_launched")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
