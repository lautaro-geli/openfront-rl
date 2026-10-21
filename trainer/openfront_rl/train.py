"""Entrenamiento PPO contra bots (docs/adr/0007-red-por-ramas-y-ppo.md).

    python -m openfront_rl.train --config configs/ppo_base.yaml

Cada ejecución crea `runs/<nombre>-<semilla>/` con `config.yaml`, `metrics.jsonl` y los checkpoints (que llevan la
configuración, la versión del protocolo y las semillas dentro).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from .manifest import Manifest
from .model import MASK_KEYS, OBS_KEYS, ModelConfig, Policy, to_tensors
from . import runctl
from .ppo import PPOConfig, RolloutBuffer, RunningStd, ppo_update
from .vec_env import VecGame, default_workers

DEFAULTS: dict[str, Any] = {
    "name": "ppo",
    "seed": 0,
    "device": "cpu",
    "workers": None,  # None = núcleos − 1
    "torch_threads": None,  # None = núcleos físicos − 2 (mín. 1): deja dos núcleos libres a la PC
    "low_priority": True,  # entrenamiento y procesos Node con prioridad por debajo de lo normal: la PC sigue usable
    "rollout_steps": 128,
    "updates": 1000,
    "checkpoint_every": 25,
    "eval": {"every": 50, "episodes": 5},
    "env": {},
    "model": {},
    "ppo": {},
}


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf8") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    _merge(cfg, overrides or {})
    return cfg


def _merge(a: dict[str, Any], b: dict[str, Any]) -> None:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v


def stack_obs(results: list[Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Un agente por partida: [N, ...]."""
    obs = {k: np.stack([r.obs[k][0] for r in results]) for k in OBS_KEYS}
    mask = {k: np.stack([r.mask[k][0] for r in results]) for k in MASK_KEYS}
    return obs, mask


ABBR = {"territory": "terr", "attack": "atq", "economy": "eco", "defense": "def", "terminal": "fin"}


# (etiqueta de consola, evento): lo que se muestra por episodio; el resto queda en metrics.jsonl como ep_ev_*.
EV_SHOW = [("atq", "attack_orders"), ("a_trib", "attack_tribe_orders"), ("tribus", "tribes_conquered"), ("nuke", "nuke_orders")]


class Runner:
    """Juega N partidas a la vez con la política y llena el buffer."""

    def __init__(self, vec: VecGame, policy: Policy, device: torch.device):
        self.vec, self.policy, self.device = vec, policy, device
        self.m: Manifest = vec.manifest
        self.parts = list(self.m.reward_parts)
        self.results = vec.reset()
        self.obs, self.mask = stack_obs(self.results)
        n = vec.n
        self.ep_parts = np.zeros((n, len(self.parts)))
        self.ep_events = np.zeros((n, len(self.m.event_names)))
        self.ep_len = np.zeros(n, int)
        self.ep_invalid = np.zeros(n, int)
        self.kind_counts = np.zeros(len(self.m.space["kinds"]))

    @torch.no_grad()
    def _act(self, obs: dict[str, np.ndarray], mask: dict[str, np.ndarray], greedy: bool = False):
        ob, mk = to_tensors(obs, mask, self.device)
        return self.policy(ob, mk, greedy=greedy)

    @torch.no_grad()
    def collect(self, buf: RolloutBuffer, greedy: bool = False) -> tuple[list[dict[str, Any]], dict[str, float]]:
        episodes: list[dict[str, Any]] = []
        n = self.vec.n
        invalid = 0
        self.kind_counts[:] = 0  # el histograma de acciones es por actualización, no acumulado
        for t in range(buf.T):
            buf.add_obs(t, self.obs, self.mask)
            act, logp, _, val = self._act(self.obs, self.mask, greedy)
            a = act.cpu().numpy().astype(np.int32)
            buf.action[t], buf.logp[t], buf.value[t] = a, logp.cpu().numpy(), val.cpu().numpy()
            np.add.at(self.kind_counts, a[:, 0], 1)
            results = self.vec.step(a.reshape(n, 1, -1))
            for i in range(n):
                fin = self.vec.final[i]
                r = fin if fin is not None else results[i]
                buf.reward[t, i] = r.reward[0]
                self.ep_parts[i] += r.reward_parts[0]
                self.ep_events[i] += r.events[0]
                self.ep_len[i] += 1
                invalid += int(r.invalid[0])
                self.ep_invalid[i] += int(r.invalid[0])
                if fin is not None:
                    buf.done[t, i] = 1.0
                    if fin.truncated and not fin.terminated[0]:
                        o, m = {k: v[0:1] for k, v in fin.obs.items()}, {k: v[0:1] for k, v in fin.mask.items()}
                        buf.boot[t, i] = self.policy.values(to_tensors(o, m, self.device)[0]).item()
                    p = dict(zip(self.parts, self.ep_parts[i].tolist()))
                    episodes.append({
                        "return": float(sum(p.values())), "length": int(self.ep_len[i]), "won": bool(fin.won[0]),
                        "eliminated": bool(fin.eliminated[0]), "truncated": bool(fin.truncated),
                        "invalid": int(self.ep_invalid[i]), **{f"part_{k}": v for k, v in p.items()},
                        **{f"ev_{k}": v for k, v in zip(self.m.event_names, self.ep_events[i].tolist())},
                    })
                    self.ep_parts[i] = 0
                    self.ep_events[i] = 0
                    self.ep_len[i] = 0
                    self.ep_invalid[i] = 0
            self.results = results
            self.obs, self.mask = stack_obs(results)
        _, _, _, last = self._act(self.obs, self.mask, greedy)
        info = {"invalid_rate": invalid / (buf.T * n)}
        self.last_value = last.cpu().numpy()
        return episodes, info


def evaluate(policy: Policy, env_cfg: dict[str, Any], workers: int, episodes: int, seed: str, device: torch.device) -> dict[str, float]:
    """Partidas con semillas propias (distintas de las del entrenamiento). Devuelve medias."""
    n = max(1, min(workers, episodes))
    done: list[dict[str, float]] = []
    with VecGame(n, {**env_cfg, "agents": 1}, base_seed=f"eval-{seed}") as vec:
        res = vec.reset()
        acc = np.zeros((n, len(vec.manifest.reward_parts)))  # suma de componentes del episodio en curso, por entorno
        evs = np.zeros((n, len(vec.manifest.event_names)))
        guard = 0
        while len(done) < episodes and guard < 20000:
            guard += 1
            obs, mask = stack_obs(res)
            with torch.no_grad():
                act, *_ = policy(*to_tensors(obs, mask, device))
            res = vec.step(act.cpu().numpy().astype(np.int32).reshape(n, 1, -1))
            for i in range(n):
                fin = vec.final[i]
                acc[i] += (fin if fin is not None else res[i]).reward_parts[0]
                evs[i] += (fin if fin is not None else res[i]).events[0]
                if fin is not None:
                    if len(done) < episodes:
                        p = dict(zip(vec.manifest.reward_parts, acc[i].tolist()))
                        done.append({"won": float(fin.won[0]), "eliminated": float(fin.eliminated[0]),
                                     "return": float(sum(p.values())), **{f"part_{k}": v for k, v in p.items()},
                                     **{f"ev_{k}": v for k, v in zip(vec.manifest.event_names, evs[i].tolist())}})
                    acc[i] = 0
                    evs[i] = 0
    keys = done[0].keys() if done else []
    return {f"eval_{k}": float(np.mean([d[k] for d in done])) for k in keys} | {"eval_episodes": float(len(done))}


def save_checkpoint(path: Path, policy: Policy, opt: torch.optim.Optimizer, update: int, cfg: dict[str, Any],
                    manifest: Manifest, rstd: RunningStd, steps: int) -> None:
    torch.save({
        "model": policy.state_dict(), "optimizer": opt.state_dict(), "update": update, "steps": steps, "config": cfg,
        "protocol_version": manifest.raw["version"], "reward_std": rstd.state(),
        "torch_version": torch.__version__, "rng": {"torch": torch.get_rng_state()},
    }, path)


def apply_resources(cfg: dict[str, Any]) -> None:
    """Limita los hilos de PyTorch y baja la prioridad (los procesos Node que se lancen después la heredan)."""
    threads = cfg.get("torch_threads") or max(1, (psutil.cpu_count(logical=False) or 1) - 2)
    torch.set_num_threads(int(threads))
    if cfg.get("low_priority"):
        try:
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS") else 10)
        except (psutil.Error, OSError):
            pass


def train(cfg: dict[str, Any], run_dir: str | Path | None = None, resume: str | Path | None = None, log: bool = True) -> Path:
    seed = int(cfg["seed"])
    apply_resources(cfg)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(cfg["device"])
    run = Path(run_dir or f"runs/{cfg['name']}-{seed}")
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf8")

    workers = cfg["workers"] or default_workers()
    env_cfg = {**cfg["env"], "agents": 1}
    pcfg = PPOConfig.from_dict(cfg["ppo"])
    vec = VecGame(workers, env_cfg, base_seed=f"{cfg['name']}-{seed}")
    try:
        m = vec.manifest
        policy = Policy(m, ModelConfig.from_dict(cfg["model"])).to(device)
        opt = torch.optim.Adam(policy.parameters(), lr=pcfg.lr, eps=1e-5)
        rstd = RunningStd(pcfg.gamma)
        update0, steps = 0, 0
        if resume is not None:
            ck = torch.load(resume, map_location=device, weights_only=False)
            policy.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            rstd.load(ck["reward_std"])
            update0, steps = ck["update"], ck["steps"]
        runner = Runner(vec, policy, device)
        T = int(cfg["rollout_steps"])
        buf = RolloutBuffer(T, workers, {k: b.shape for k, b in m.obs.items()}, {k: b.shape for k, b in m.mask.items()}, m.action_dims)
        n_params = sum(p.numel() for p in policy.parameters())
        runctl.write_json(run / "architecture.json", runctl.architecture(policy))
        runctl.write_json(run / "control.json", {"command": "run"})
        if log:
            print(f"run={run} workers={workers} params={n_params:,} T={T} protocolo=v{m.raw['version']}", flush=True)
        total = int(cfg["updates"])
        t0 = time.time()
        runctl.write_status(run, "running", update=update0, total=total, steps=steps, time=0.0, steps_per_s=0.0, workers=[])
        with open(run / "metrics.jsonl", "a", encoding="utf8") as mf:
            for u in range(update0 + 1, total + 1):
                frac = 1 - (u - 1) / max(1, total)
                for g in opt.param_groups:
                    g["lr"] = pcfg.lr * max(0.1, frac)
                ent_coef = pcfg.ent_coef_final + (pcfg.ent_coef - pcfg.ent_coef_final) * frac
                tr = time.time()
                episodes, info = runner.collect(buf)
                t_roll = time.time() - tr
                rstd.update(buf.reward, buf.done)
                buf.compute_gae(runner.last_value, pcfg.gamma, pcfg.gae_lambda, rstd.std)
                tu = time.time()
                stats = ppo_update(policy, opt, buf, pcfg, ent_coef, rng, device)
                steps += T * workers
                row: dict[str, Any] = {
                    "update": u, "steps": steps, "time": round(time.time() - t0, 1),
                    "t_rollout": round(t_roll, 2), "t_update": round(time.time() - tu, 2),
                    "reward_scale": rstd.std, "ent_coef": ent_coef, **stats, **info,
                    "noop_frac": float(runner.kind_counts[0] / max(1, runner.kind_counts.sum())),
                    "kind_frac": (runner.kind_counts / max(1, runner.kind_counts.sum())).round(4).tolist(),
                    "step_reward_mean": float(buf.reward.mean()),
                }
                if episodes:
                    for k in episodes[0]:
                        row[f"ep_{k}"] = float(np.mean([e[k] for e in episodes]))
                    row["episodes"] = len(episodes)
                ev = cfg["eval"]
                if ev["every"] and u % ev["every"] == 0:
                    row |= evaluate(policy, env_cfg, workers, ev["episodes"], f"{cfg['name']}-{seed}-{u}", device)
                mf.write(json.dumps(row) + "\n")
                mf.flush()
                if log:
                    pr = " ".join(f"{ABBR.get(k, k[:4])}={row[f'ep_part_{k}']:+.3f}" for k in runner.parts if f"ep_part_{k}" in row)
                    ev = " ".join(f"{a}={row[f'ep_ev_{k}']:.1f}" for a, k in EV_SHOW if f"ep_ev_{k}" in row)
                    print(f"[{u}/{total}] steps={steps} r/step={row['step_reward_mean']:+.4f} "
                          f"ep_ret={row.get('ep_return', float('nan')):+.3f} ({pr}) ent={stats['ent']:.2f} kl={stats['kl']:.4f} "
                          f"[{ev}] inv={info['invalid_rate']:.2f} noop={row['noop_frac']:.2f} "
                          f"roll={t_roll:.1f}s upd={row['t_update']:.1f}s", flush=True)
                saved = False
                if u % cfg["checkpoint_every"] == 0 or u == total:
                    save_checkpoint(run / f"ckpt_{u:06d}.pt", policy, opt, u, cfg, m, rstd, steps)
                    save_checkpoint(run / "latest.pt", policy, opt, u, cfg, m, rstd, steps)
                    saved = True
                sps = T * workers / max(1e-6, t_roll + row["t_update"])

                def status(state: str) -> None:
                    runctl.write_status(
                        run, state, update=u, total=total, steps=steps, time=round(time.time() - t0, 1), steps_per_s=round(sps, 1),
                        workers=[{"i": i, "tick": int(runner.results[i].tick), "ep_len": int(runner.ep_len[i]),
                                  "events": dict(zip(m.event_names, runner.ep_events[i].tolist()))} for i in range(workers)])

                # Control cooperativo del panel (docs/specs/dashboard.md): se atiende entre actualizaciones.
                cmd = runctl.read_command(run)
                if cmd in ("pause", "stop") and not saved:
                    save_checkpoint(run / "latest.pt", policy, opt, u, cfg, m, rstd, steps)
                while cmd == "pause":
                    status("paused")
                    time.sleep(1.0)
                    cmd = runctl.read_command(run)
                    if cmd == "stop":
                        save_checkpoint(run / "latest.pt", policy, opt, u, cfg, m, rstd, steps)
                if cmd == "stop":
                    status("stopped")
                    break
                status("finished" if u == total else "running")
    finally:
        vec.close()
    return run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args(argv)
    over: dict[str, Any] = {k: v for k, v in (("updates", a.updates), ("workers", a.workers), ("seed", a.seed)) if v is not None}
    train(load_config(a.config, over), a.run_dir, a.resume)


if __name__ == "__main__":
    main()
