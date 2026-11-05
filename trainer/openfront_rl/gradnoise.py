"""Mide cuánta señal tiene el gradiente de la política en un checkpoint, sin entrenar (diagnóstico, trainer/README.md).

    python -m openfront_rl.gradnoise --ckpt runs/Day8/latest.pt --batches 40 --workers 5 [--rollout-steps 128]

Juega `batches` lotes con la política congelada (como `train.py`: `rollout_steps` × workers muestras, GAE y ventajas normalizadas por
lote como `ppo_update`) y calcula el gradiente de la pérdida de política (sin crítico ni entropía) de cada lote. Con esos gradientes:

- `coseno_medio`: coseno medio entre gradientes de lotes distintos. Cerca de 0 = cada lote empuja hacia un lado distinto (ruido).
- `b_noise`: escala de ruido del gradiente, B_simple = tr(Σ)/|G|² (McCandlish et al. 2018, arXiv 1812.06162, apéndice A), estimada con
  los lotes chicos (b muestras) y su media (K·b). Es el tamaño de lote a partir del cual agrandarlo deja de ayudar. En muestras y en
  partidas terminadas equivalentes. Si es mucho mayor que el lote de entrenamiento (640), cada actualización es casi ruido.

Los lotes consecutivos salen de las mismas partidas en curso: no son independientes del todo, así que `b_noise` es aproximado.
"""
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch

from .model import ModelConfig, Policy
from .ppo import PPOConfig, RolloutBuffer, RunningStd
from .train import Runner, resolve_device
from .vec_env import VecGame


def policy_params(policy: Policy) -> list[torch.nn.Parameter]:
    """Parámetros que mueve la pérdida de política (todo menos la cabeza y el codificador del crítico)."""
    return [p for n, p in policy.named_parameters() if not n.startswith(("value.", "enc_v.")) and p.requires_grad]


def batch_gradient(policy: Policy, buf: RolloutBuffer, device: torch.device, chunk: int = 128) -> torch.Tensor:
    """Gradiente de la pérdida de política media del lote (ratio = 1: la política no cambió desde que jugó)."""
    params = policy_params(policy)
    for p in params:
        p.grad = None
    adv_all = buf.adv[buf.valid]
    mean, std = float(adv_all.mean()), float(adv_all.std() + 1e-8)
    idx_all = np.flatnonzero(buf.valid.reshape(-1))
    n = len(idx_all)
    for s in range(0, n, chunk):
        idx = idx_all[s : s + chunk]
        ob, mk, m = buf.gather(idx, device)
        _, logp, _, _ = policy(ob, mk, action=m["action"])
        adv = (m["adv"] - mean) / std
        loss = -(adv * (logp - m["logp"]).exp()).sum() / n
        loss.backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params]).double().cpu()


def noise_scale(grads: list[torch.Tensor], b: float) -> dict[str, float]:
    """B_simple con dos tamaños de lote: b (cada lote) y K·b (la media de los K)."""
    k = len(grads)
    g_small = float(np.mean([float(g.dot(g)) for g in grads]))
    gm = torch.stack(grads).mean(0)
    g_big = float(gm.dot(gm))
    big = k * b
    g2 = (big * g_big - b * g_small) / (big - b)
    s = (g_small - g_big) / (1 / b - 1 / big)
    return {"g2": g2, "traza": s, "b_noise": s / g2 if g2 > 0 else float("inf")}


def mean_cosine(grads: list[torch.Tensor]) -> float:
    g = torch.stack(grads)
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-30)
    c = g @ g.T
    k = len(grads)
    return float((c.sum() - c.diag().sum()) / (k * (k - 1)))


def measure(ckpt_path: str, batches: int, workers: int, rollout_steps: int | None, seed: str, device: torch.device,
            conquest_w: float = 0.0, log: bool = True) -> dict[str, Any]:
    torch.manual_seed(0)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    pcfg = PPOConfig.from_dict(cfg["ppo"])
    T = int(rollout_steps or cfg["rollout_steps"])
    with VecGame(workers, {**cfg["env"], "agents": 1}, base_seed=f"gradnoise-{seed}") as vec:
        m = vec.manifest
        policy = Policy(m, ModelConfig.from_dict(cfg.get("model", {}))).to(device)
        policy.load_state_dict(ck["model"])
        rstd = RunningStd(pcfg.gamma)
        rstd.load(ck["reward_std"])
        runner = Runner(vec, policy, device)
        runner.conquest_w = conquest_w
        buf = RolloutBuffer(T, workers, {k: b.shape for k, b in m.obs.items()}, {k: b.shape for k, b in m.mask.items()}, m.action_dims)
        grads: list[torch.Tensor] = []
        ended, won, samples = 0, 0.0, 0
        for i in range(batches):
            with torch.no_grad():
                episodes, _ = runner.collect(buf)
            buf.compute_gae(runner.last_value, pcfg.gamma, pcfg.gae_lambda, rstd.std)
            grads.append(batch_gradient(policy, buf, device))
            ended += len(episodes)
            won += sum(float(e["won"]) for e in episodes)
            samples += int(buf.valid.sum())
            if log:
                print(f"lote {i + 1}/{batches}: partidas terminadas {len(episodes)} (total {ended})", flush=True)
    b = samples / batches
    ns = noise_scale(grads, b)
    eps_per_batch = ended / batches
    half = batches // 2
    out = {
        "ckpt": ckpt_path, "update": ck["update"], "lotes": batches, "muestras_por_lote": b,
        "partidas_terminadas_por_lote": eps_per_batch, "victorias": won / max(1, ended),
        "coseno_medio": mean_cosine(grads), "b_noise_muestras": ns["b_noise"],
        "b_noise_partidas": ns["b_noise"] / b * eps_per_batch if np.isfinite(ns["b_noise"]) else float("inf"),
        "b_noise_mitad1": noise_scale(grads[:half], b)["b_noise"] if half >= 2 else None,
        "b_noise_mitad2": noise_scale(grads[half:], b)["b_noise"] if batches - half >= 2 else None,
    }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--rollout-steps", type=int, default=None, help="por defecto, el del checkpoint")
    ap.add_argument("--seed", default="g1")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    device = resolve_device(a.device)
    out = measure(a.ckpt, a.batches, a.workers, a.rollout_steps, a.seed, device)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
