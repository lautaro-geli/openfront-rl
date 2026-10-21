"""PPO con máscaras (docs/adr/0007-red-por-ramas-y-ppo.md): buffer, GAE y actualización."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor

from .model import MASK_KEYS, OBS_KEYS, Policy


@dataclass
class PPOConfig:
    gamma: float = 0.997
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_clip: float = 0.2
    epochs: int = 3
    minibatch: int = 128
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float = 0.001
    target_kl: float = 0.05

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "PPOConfig":
        return PPOConfig(**(d or {}))


class RunningStd:
    """Desviación típica del retorno descontado, para escalar las recompensas (sin restar la media)."""

    def __init__(self, gamma: float):
        self.gamma, self.ret = gamma, None
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        if self.ret is None:
            self.ret = np.zeros(rewards.shape[1], np.float64)
        for r, d in zip(rewards, dones):
            self.ret = self.ret * self.gamma + r
            for x in self.ret:
                self.n += 1
                dlt = x - self.mean
                self.mean += dlt / self.n
                self.m2 += dlt * (x - self.mean)
            self.ret = self.ret * (1.0 - d)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.m2 / self.n)) if self.n > 1 and self.m2 > 0 else 1.0

    def state(self) -> dict[str, Any]:
        return {"n": self.n, "mean": self.mean, "m2": self.m2}

    def load(self, s: dict[str, Any]) -> None:
        self.n, self.mean, self.m2 = s["n"], s["mean"], s["m2"]


class RolloutBuffer:
    """T pasos × N entornos. El grid se guarda en float16 (sus valores están en [0,1])."""

    def __init__(self, T: int, N: int, obs_shapes: dict[str, tuple[int, ...]], mask_shapes: dict[str, tuple[int, ...]], action_dims: int):
        self.T, self.N = T, N
        self.obs = {k: np.zeros((T, N, *obs_shapes[k]), np.float16 if k == "grid" else np.float32) for k in OBS_KEYS}
        self.mask = {k: np.zeros((T, N, *mask_shapes[k]), bool) for k in MASK_KEYS}
        self.action = np.zeros((T, N, action_dims), np.int64)
        self.logp = np.zeros((T, N), np.float32)
        self.value = np.zeros((T, N), np.float32)
        self.reward = np.zeros((T, N), np.float32)
        self.done = np.zeros((T, N), np.float32)  # el episodio terminó tras este paso
        self.boot = np.zeros((T, N), np.float32)  # valor del estado siguiente si el episodio se truncó (si no, 0)
        self.adv = np.zeros((T, N), np.float32)
        self.ret = np.zeros((T, N), np.float32)

    def add_obs(self, t: int, obs: dict[str, np.ndarray], mask: dict[str, np.ndarray]) -> None:
        for k in OBS_KEYS:
            self.obs[k][t] = obs[k][:, ...]
        for k in MASK_KEYS:
            self.mask[k][t] = mask[k]

    def compute_gae(self, last_value: np.ndarray, gamma: float, lam: float, reward_scale: float) -> None:
        r = self.reward / reward_scale
        boot = self.boot / 1.0
        last = np.zeros(self.N, np.float32)
        for t in reversed(range(self.T)):
            nxt = last_value if t == self.T - 1 else self.value[t + 1]
            cont = 1.0 - self.done[t]
            nv = np.where(self.done[t] > 0, boot[t], nxt)
            delta = r[t] + gamma * nv - self.value[t]
            last = delta + gamma * lam * cont * last
            self.adv[t] = last
        self.ret = self.adv + self.value

    def minibatches(self, size: int, rng: np.random.Generator) -> Iterator[np.ndarray]:
        idx = rng.permutation(self.T * self.N)
        for s in range(0, len(idx), size):
            yield idx[s : s + size]

    def gather(self, idx: np.ndarray, device: torch.device) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        def flat(a: np.ndarray) -> np.ndarray:
            return a.reshape(self.T * self.N, *a.shape[2:])[idx]

        ob = {k: torch.from_numpy(flat(self.obs[k]).astype(np.float32)).to(device) for k in OBS_KEYS}
        mk = {k: torch.from_numpy(flat(self.mask[k])).to(device) for k in MASK_KEYS}
        misc = {
            "action": torch.from_numpy(flat(self.action)).to(device),
            "logp": torch.from_numpy(flat(self.logp)).to(device),
            "value": torch.from_numpy(flat(self.value)).to(device),
            "adv": torch.from_numpy(flat(self.adv)).to(device),
            "ret": torch.from_numpy(flat(self.ret)).to(device),
        }
        return ob, mk, misc


def ppo_update(policy: Policy, opt: torch.optim.Optimizer, buf: RolloutBuffer, cfg: PPOConfig, ent_coef: float,
               rng: np.random.Generator, device: torch.device) -> dict[str, float]:
    """Una actualización completa (varias épocas). Devuelve métricas medias."""
    stats: dict[str, list[float]] = {k: [] for k in ("loss", "pg", "vf", "ent", "kl", "clipfrac", "expl_var")}
    adv_all = buf.adv.reshape(-1)
    mean, std = float(adv_all.mean()), float(adv_all.std() + 1e-8)
    stop = False
    for _ in range(cfg.epochs):
        for idx in buf.minibatches(cfg.minibatch, rng):
            ob, mk, m = buf.gather(idx, device)
            _, logp, ent, value = policy(ob, mk, action=m["action"])
            adv = (m["adv"] - mean) / std
            log_ratio = logp - m["logp"]
            ratio = log_ratio.exp()
            pg = torch.max(-adv * ratio, -adv * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
            v_clip = m["value"] + (value - m["value"]).clamp(-cfg.value_clip, cfg.value_clip)
            vf = 0.5 * torch.max((value - m["ret"]) ** 2, (v_clip - m["ret"]) ** 2).mean()
            entropy = ent.mean()
            loss = pg + cfg.vf_coef * vf - ent_coef * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean().item()
                var = m["ret"].var()
                stats["loss"].append(loss.item())
                stats["pg"].append(pg.item())
                stats["vf"].append(vf.item())
                stats["ent"].append(entropy.item())
                stats["kl"].append(kl)
                stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                stats["expl_var"].append(float(1 - ((m["ret"] - value) ** 2).mean() / (var + 1e-8)))
            if kl > 1.5 * cfg.target_kl:
                stop = True
                break
        if stop:
            break
    return {k: float(np.mean(v)) for k, v in stats.items()}
