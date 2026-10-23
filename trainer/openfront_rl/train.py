"""Entrenamiento PPO contra bots (docs/adr/0007-red-por-ramas-y-ppo.md).

    python -m openfront_rl.train --config configs/v1_ppo_base.yaml

Cada ejecución crea `runs/<nombre>-<semilla>/` con `config.yaml`, `metrics.jsonl` y los checkpoints (que llevan la
configuración, la versión del protocolo y las semillas dentro).
"""
from __future__ import annotations

import argparse
import copy
import json
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import yaml

from .client import Result
from .manifest import Manifest
from .model import MASK_KEYS, OBS_KEYS, ModelConfig, Policy, to_tensors
from . import runctl
from .ppo import PPOConfig, RolloutBuffer, RunningStd, ppo_update
from .vec_env import VecGame, default_workers
from .workers import WorkerLogs, WorkerStats, game_line

DEFAULTS: dict[str, Any] = {
    "name": "ppo",
    "seed": 0,
    "device": "auto",  # auto = CUDA si hay GPU, si no CPU
    "workers": None,  # None = núcleos − 1
    "torch_threads": None,  # None = núcleos físicos − 2 (mín. 1): deja dos núcleos libres a la PC
    "low_priority": True,  # entrenamiento y procesos Node con prioridad por debajo de lo normal: la PC sigue usable
    "rollout_steps": 128,
    "updates": 1000,  # límite en actualizaciones; se ignora si `episodes` está definido
    "rollout_mode": "sync",  # sync = todos los workers al mismo paso; async = cada uno a su ritmo (docs/adr/0010)
    "rollout_cap": 2,  # async: tope de pasos por worker = rollout_cap × rollout_steps
    "episodes": None,  # duración del run en partidas de entrenamiento terminadas (None = usar `updates`)
    "checkpoint_every": 25,
    "eval": {"every": 50, "episodes": 5},
    "env": {},
    "conquest": {"budget": 0.20, "until": 0.5},  # bonus por tribu conquistada que decae a 0 (docs/specs/reward.md §9)
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


def conquest_weight(cfg: dict[str, Any], progress: float) -> float:
    """Recompensa por tribu conquistada: lineal de `budget / N_tribus` a 0 en `until` (docs/specs/reward.md §9)."""
    cq = cfg.get("conquest") or {}
    decay = max(0.0, 1 - progress / max(1e-9, float(cq.get("until", 0.5))))
    return float(cq.get("budget", 0)) * decay / max(1, int(cfg["env"].get("bots", 1)))


def stack_obs(results: list[Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Un agente por partida: [N, ...]."""
    obs = {k: np.stack([r.obs[k][0] for r in results]) for k in OBS_KEYS}
    mask = {k: np.stack([r.mask[k][0] for r in results]) for k in MASK_KEYS}
    return obs, mask


ABBR = {"conquest": "conq", "territory": "terr", "attack": "atq", "economy": "eco", "defense": "def", "milestones": "hito", "troops": "trop", "terminal": "fin"}


# (etiqueta de consola, evento): lo que se muestra por episodio; el resto queda en metrics.jsonl como ep_ev_*.
EV_SHOW = [("atq", "attack_orders"), ("a_trib", "attack_tribe_orders"), ("tribus", "tribes_conquered"), ("nuke", "nukes_launched"),
           ("ciu", "built_city"), ("pue", "built_port"), ("fab", "built_factory"), ("pst", "built_post")]


class Runner:
    """Juega N partidas a la vez con la política y llena el buffer."""

    def __init__(self, vec: VecGame, policy: Policy, device: torch.device, wstats: WorkerStats | None = None,
                 wlogs: WorkerLogs | None = None):
        self.vec, self.policy, self.device = vec, policy, device
        self.wstats, self.wlogs = wstats or WorkerStats(vec.n), wlogs
        self.m: Manifest = vec.manifest
        self.parts = list(self.m.reward_parts)
        self.results = vec.reset()
        self.obs, self.mask = stack_obs(self.results)
        n = vec.n
        self.ep_parts = np.zeros((n, len(self.parts)))
        self.ep_events = np.zeros((n, len(self.m.event_names)))
        self.conquest_w = 0.0  # recompensa por tribu conquistada en este momento del entrenamiento
        self.ep_conquest = np.zeros(n)
        self.tribes_idx = self.m.event_names.index("tribes_conquered")
        self.ep_len = np.zeros(n, int)
        self.ep_invalid = np.zeros(n, int)
        self.kind_counts = np.zeros(len(self.m.space["kinds"]))
        self.timing = {"net": 0.0, "env": 0.0, "busy": 0.0, "max": 0.0}  # por rollout: red, motor, suma de lat y suma de max(lat)
        self.land_idx = self.m.global_features.index("land_share")
        self.ep_max = np.array([float(r.obs["globals"][0][self.land_idx]) for r in self.results])  # mayor territorio de la partida en curso

    @torch.no_grad()
    def _act(self, obs: dict[str, np.ndarray], mask: dict[str, np.ndarray], greedy: bool = False):
        ob, mk = to_tensors(obs, mask, self.device)
        return self.policy(ob, mk, greedy=greedy)

    @torch.no_grad()
    def collect(self, buf: RolloutBuffer, greedy: bool = False, active: np.ndarray | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
        episodes: list[dict[str, Any]] = []
        n = self.vec.n
        act_mask = np.ones(n, bool) if active is None else np.asarray(active, bool)
        buf.valid[:] = act_mask[None, :]  # los workers pausados no aportan muestras
        invalid = 0
        self.timing = {"net": 0.0, "env": 0.0, "busy": 0.0, "max": 0.0}
        self.kind_counts[:] = 0  # el histograma de acciones es por actualización, no acumulado
        for t in range(buf.T):
            buf.add_obs(t, self.obs, self.mask)
            t_a = time.perf_counter()
            act, logp, _, val = self._act(self.obs, self.mask, greedy)
            a = act.cpu().numpy().astype(np.int32)
            buf.action[t], buf.logp[t], buf.value[t] = a, logp.cpu().numpy(), val.cpu().numpy()
            np.add.at(self.kind_counts, a[act_mask, 0], 1)
            t_b = time.perf_counter()
            results = self.vec.step(a.reshape(n, 1, -1), act_mask)
            self.timing["net"] += t_b - t_a
            self.timing["env"] += time.perf_counter() - t_b
            lat = [self.vec.lat[i] for i in range(n) if act_mask[i]]
            if lat and max(lat) > 0:
                self.timing["busy"] += sum(lat) / len(lat)
                self.timing["max"] += max(lat)
            for i in range(n):
                if act_mask[i]:
                    invalid += self._after_step(buf, t, i, results[i], self.vec.final[i], episodes)
            self.results = results
            self.obs, self.mask = stack_obs(results)
        _, _, _, last = self._act(self.obs, self.mask, greedy)
        info = {"invalid_rate": invalid / (buf.T * max(1, int(act_mask.sum())))}
        self.last_value = last.cpu().numpy()
        return episodes, info

    def _after_step(self, buf: RolloutBuffer, t: int, i: int, cur: Result, fin: Result | None, episodes: list[dict[str, Any]]) -> int:
        """Registra en la fila `t` del buffer el resultado del paso del worker `i` y cierra el episodio si terminó.
        `cur` es el resultado vigente (ya de la partida nueva si hubo reinicio) y `fin` el del episodio que acabó."""
        r = fin if fin is not None else cur
        bonus = self.conquest_w * float(r.events[0][self.tribes_idx])
        buf.reward[t, i] = r.reward[0] + bonus
        self.ep_conquest[i] += bonus
        self.ep_parts[i] += r.reward_parts[0]
        self.ep_events[i] += r.events[0]
        self.ep_len[i] += 1
        share = float(r.obs["globals"][0][self.land_idx])
        self.ep_max[i] = max(self.ep_max[i], share)
        invalid = int(r.invalid[0])
        self.ep_invalid[i] += invalid
        if fin is not None:
            buf.done[t, i] = 1.0
            if fin.truncated and not fin.terminated[0]:
                o, m = {k: v[0:1] for k, v in fin.obs.items()}, {k: v[0:1] for k, v in fin.mask.items()}
                buf.boot[t, i] = self.policy.values(to_tensors(o, m, self.device)[0]).item()
            p = {**dict(zip(self.parts, self.ep_parts[i].tolist())), "conquest": float(self.ep_conquest[i])}
            episodes.append({
                "return": float(sum(p.values())), "length": int(self.ep_len[i]), "won": bool(fin.won[0]),
                "eliminated": bool(fin.eliminated[0]), "truncated": bool(fin.truncated),
                "invalid": int(self.ep_invalid[i]), **{f"part_{k}": v for k, v in p.items()},
                **{f"ev_{k}": v for k, v in zip(self.m.event_names, self.ep_events[i].tolist())},
                "max_territory": float(self.ep_max[i]), "final_territory": share,
            })
            g = self.wstats.end_game(i, won=bool(fin.won[0]), eliminated=bool(fin.eliminated[0]), truncated=bool(fin.truncated), max_territory=float(self.ep_max[i]),
                                     final_territory=share, ticks=int(fin.tick),
                                     events=dict(zip(self.m.event_names, self.ep_events[i].tolist())))
            if self.wlogs is not None:
                self.wlogs.write(i, game_line(g))
            self.ep_max[i] = float(cur.obs["globals"][0][self.land_idx])
            self.ep_parts[i] = 0
            self.ep_conquest[i] = 0
            self.ep_events[i] = 0
            self.ep_len[i] = 0
            self.ep_invalid[i] = 0
        return invalid

    @torch.no_grad()
    def collect_async(self, buf: RolloutBuffer, greedy: bool = False, active: np.ndarray | None = None,
                      samples: int | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Rollout asíncrono (docs/adr/0010): cada worker avanza apenas la red le contesta, sin esperar a los demás.
        Se reúnen `samples` pasos en total (por defecto T por worker activo); `buf.T` es el tope por worker. La política no cambia
        durante la recolección, así que sigue siendo PPO en política. Cada worker llena su columna del buffer hasta `buf.lengths[i]`."""
        episodes: list[dict[str, Any]] = []
        n = self.vec.n
        act_mask = np.ones(n, bool) if active is None else np.asarray(active, bool)
        idx_active = [i for i in range(n) if act_mask[i]]
        target = samples if samples is not None else buf.T * len(idx_active)
        buf.valid[:] = False
        buf.lengths[:] = 0
        self.kind_counts[:] = 0
        invalid = 0
        net = wait = 0.0
        busy = np.zeros(n)
        q: queue.Queue = queue.Queue()
        pool = ThreadPoolExecutor(max(1, len(idx_active)))

        def work(i: int, a: np.ndarray) -> None:
            t0 = time.perf_counter()
            try:
                g = self.vec.games[i]
                g.send_step(a)
                r = g.recv_result()
                fin = None
                if r.done and self.vec.auto_reset:
                    fin = r
                    self.vec._episode[i] += 1
                    g.send_reset(self.vec._seed(i))
                    r = g.recv_result()
                q.put((i, r, fin, time.perf_counter() - t0, None))
            except BaseException as e:  # se relanza en el hilo principal
                q.put((i, None, None, 0.0, e))

        def launch(ready: list[int]) -> None:
            nonlocal net
            t_a = time.perf_counter()
            obs = {k: np.concatenate([self.results[i].obs[k][0:1] for i in ready]) for k in OBS_KEYS}
            mask = {k: np.concatenate([self.results[i].mask[k][0:1] for i in ready]) for k in MASK_KEYS}
            act, logp, _, val = self._act(obs, mask, greedy)
            a = act.cpu().numpy().astype(np.int32)
            lp, vl = logp.cpu().numpy(), val.cpu().numpy()
            for j, i in enumerate(ready):
                row = int(buf.lengths[i])
                for k in OBS_KEYS:
                    buf.obs[k][row, i] = obs[k][j]
                for k in MASK_KEYS:
                    buf.mask[k][row, i] = mask[k][j]
                buf.action[row, i], buf.logp[row, i], buf.value[row, i] = a[j], lp[j], vl[j]
                self.kind_counts[a[j, 0]] += 1
            net += time.perf_counter() - t_a
            for j, i in enumerate(ready):
                pool.submit(work, i, a[j].reshape(1, -1))

        t_start = time.perf_counter()
        try:
            launch(idx_active)
            inflight, recorded = len(idx_active), 0
            while inflight:
                t_w = time.perf_counter()
                items = [q.get()]
                wait += time.perf_counter() - t_w
                while True:
                    try:
                        items.append(q.get_nowait())
                    except queue.Empty:
                        break
                ready: list[int] = []
                for i, r, fin, dt, err in items:
                    if err is not None:
                        raise err
                    inflight -= 1
                    busy[i] += dt
                    row = int(buf.lengths[i])
                    invalid += self._after_step(buf, row, i, r, fin, episodes)
                    buf.valid[row, i] = True
                    buf.lengths[i] += 1
                    recorded += 1
                    self.results[i] = r
                    if recorded + inflight + len(ready) < target and buf.lengths[i] < buf.T:
                        ready.append(i)
                if ready:
                    launch(ready)
                    inflight += len(ready)
        finally:
            pool.shutdown(wait=True)
        wall = time.perf_counter() - t_start
        self.obs, self.mask = stack_obs(self.results)
        _, _, _, last = self._act(self.obs, self.mask, greedy)
        info = {"invalid_rate": invalid / max(1, int(buf.valid.sum()))}
        self.last_value = last.cpu().numpy()
        for i in range(n):
            if not act_mask[i]:
                buf.lengths[i] = buf.T  # sin muestras válidas: no importa
        self.timing = {"net": net, "env": wait, "busy": float(busy.sum()), "max": wall * max(1, len(idx_active))}
        return episodes, info


def evaluate(policy: Policy, env_cfg: dict[str, Any], workers: int, episodes: int, seed: str, device: torch.device) -> dict[str, float]:
    """Partidas con semillas propias (distintas de las del entrenamiento). Devuelve medias.

    Cada worker juega una cuota fija de partidas y se detiene al cumplirla: así se promedian todas las partidas pedidas y no las
    primeras en terminar (esas son las más cortas y sesgaban la evaluación hacia eliminaciones y victorias rápidas).
    """
    n = max(1, min(workers, episodes))
    quota = [episodes // n + (1 if i < episodes % n else 0) for i in range(n)]
    played = [0] * n
    done: list[dict[str, float]] = []
    with VecGame(n, {**env_cfg, "agents": 1}, base_seed=f"eval-{seed}") as vec:
        res = vec.reset()
        active = np.ones(n, bool)
        acc = np.zeros((n, len(vec.manifest.reward_parts)))  # suma de componentes del episodio en curso, por entorno
        evs = np.zeros((n, len(vec.manifest.event_names)))
        guard = 0
        while active.any() and guard < 20000:
            guard += 1
            obs, mask = stack_obs(res)
            with torch.no_grad():
                act, *_ = policy(*to_tensors(obs, mask, device))
            res = vec.step(act.cpu().numpy().astype(np.int32).reshape(n, 1, -1), active)
            for i in range(n):
                if not active[i]:
                    continue
                fin = vec.final[i]
                acc[i] += (fin if fin is not None else res[i]).reward_parts[0]
                evs[i] += (fin if fin is not None else res[i]).events[0]
                if fin is not None:
                    p = dict(zip(vec.manifest.reward_parts, acc[i].tolist()))
                    done.append({"won": float(fin.won[0]), "eliminated": float(fin.eliminated[0]),
                                 "return": float(sum(p.values())), **{f"part_{k}": v for k, v in p.items()},
                                 **{f"ev_{k}": v for k, v in zip(vec.manifest.event_names, evs[i].tolist())}})
                    acc[i] = 0
                    evs[i] = 0
                    played[i] += 1
                    if played[i] >= quota[i]:
                        active[i] = False
    keys = done[0].keys() if done else []
    return {f"eval_{k}": float(np.mean([d[k] for d in done])) for k in keys} | {"eval_episodes": float(len(done))}


def resolve_device(name: str) -> torch.device:
    """`auto` usa la GPU (CUDA) si existe y, si no, la CPU; `cuda` sin GPU falla con un mensaje claro."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device=cuda pero PyTorch no ve ninguna GPU (¿versión solo CPU?). Usa device: cpu o instala PyTorch con CUDA.")
    return torch.device(name)


def save_checkpoint(path: Path, policy: Policy, opt: torch.optim.Optimizer, update: int, cfg: dict[str, Any],
                    manifest: Manifest, rstd: RunningStd, steps: int, episodes_done: int = 0) -> None:
    torch.save({
        "model": {k: v.cpu() for k, v in policy.state_dict().items()}, "optimizer": opt.state_dict(), "update": update, "steps": steps, "episodes_done": episodes_done, "config": cfg,
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


RUN_FILES = ("metrics.jsonl", "workers.json", "status.json", "architecture.json", "control.json", "config.yaml", "workers")


def archive_previous(run: Path) -> Path | None:
    """Un run nuevo en una carpeta con datos no mezcla filas: lo anterior pasa a `anterior-<fecha>/` (dashboard.md §4c)."""
    old = [p for p in (*(run / n for n in RUN_FILES), *run.glob("*.pt")) if p.exists()]
    if not (run / "metrics.jsonl").exists():
        return None
    dest = run / f"anterior-{time.strftime('%Y%m%d-%H%M%S')}"
    dest.mkdir()
    for p in old:
        p.rename(dest / p.name)
    return dest


def _float_or_text(v: str) -> float | str:
    try:
        return float(v)
    except ValueError:
        return v


def parse_env(items: list[str]) -> dict[str, Any]:
    """`--env bots=30 --env nations=disabled`: valores numéricos como número, el resto como texto."""
    out: dict[str, Any] = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k:
            raise SystemExit(f"--env espera clave=valor, recibí {it!r}")
        out[k] = int(v) if v.lstrip("-").isdigit() else _float_or_text(v)
    return out


def scenario_text(env: dict[str, Any]) -> str:
    n = env.get("nations", "default")
    nations = "sin naciones" if n == "disabled" else f"{n} naciones"
    return (f"mapa {env.get('map', '?')} | {env.get('bots', '?')} tribus | {nations} | nivel de las naciones {env.get('difficulty', '?')}")


def train(cfg: dict[str, Any], run_dir: str | Path | None = None, resume: str | Path | None = None, log: bool = True) -> Path:
    seed = int(cfg["seed"])
    apply_resources(cfg)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(str(cfg["device"]))
    run = Path(run_dir or f"runs/{cfg['name']}-{seed}")
    run.mkdir(parents=True, exist_ok=True)
    if resume is None:
        archive_previous(run)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf8")

    workers = cfg["workers"] or default_workers()
    env_cfg = {**cfg["env"], "agents": 1}
    pcfg = PPOConfig.from_dict(cfg["ppo"])
    vec = VecGame(workers, env_cfg, base_seed=f"{cfg['name']}-{seed}", profile=True)
    try:
        m = vec.manifest
        policy = Policy(m, ModelConfig.from_dict(cfg["model"])).to(device)
        opt = torch.optim.Adam(policy.parameters(), lr=pcfg.lr, eps=1e-5)
        rstd = RunningStd(pcfg.gamma)
        update0, steps, eps0 = 0, 0, 0
        if resume is not None:
            ck = torch.load(resume, map_location=device, weights_only=False)
            policy.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            rstd.load(ck["reward_std"])
            update0, steps, eps0 = ck["update"], ck["steps"], ck.get("episodes_done", 0)
        wstats, wlogs = WorkerStats(workers), WorkerLogs(run, workers)
        if resume is not None:
            wstats.load(run / "workers.json")
        runner = Runner(vec, policy, device, wstats, wlogs)
        T = int(cfg["rollout_steps"])
        async_mode = str(cfg.get("rollout_mode", "sync")) == "async"
        if str(cfg.get("rollout_mode", "sync")) not in ("sync", "async"):
            raise ValueError("rollout_mode debe ser 'sync' o 'async'")
        buf = RolloutBuffer(T * int(cfg.get("rollout_cap", 2)) if async_mode else T, workers, {k: b.shape for k, b in m.obs.items()}, {k: b.shape for k, b in m.mask.items()}, m.action_dims)
        n_params = sum(p.numel() for p in policy.parameters())
        runctl.write_json(run / "architecture.json", runctl.architecture(policy))
        runctl.write_json(run / "control.json", {"command": "run"})
        if log:
            print(f"escenario: {scenario_text(cfg['env'])}", flush=True)
            print(f"run={run} rollout={'asíncrono' if async_mode else 'sincrónico'} workers={workers} params={n_params:,} T={T} protocolo=v{m.raw['version']} dispositivo={device}"
                  + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""), flush=True)
        total = int(cfg["updates"])
        goal = int(cfg["episodes"]) if cfg.get("episodes") else None  # duración en partidas (docs/specs/dashboard.md §5)
        t0 = time.time()
        st: dict[str, Any] = {"u": update0, "eps": eps0, "sps": 0.0, "saved_u": update0}

        def status(state: str) -> None:
            runctl.write_status(
                run, state, update=st["u"], total=None if goal else total, episodes_done=st["eps"], episodes_total=goal, steps=steps,
                time=round(time.time() - t0, 1), steps_per_s=round(st["sps"], 1),
                workers=[{"i": i, "paused": i in st.get("paused", ()), "stats": wstats.summary(i), "tick": int(runner.results[i].tick), "ep_len": int(runner.ep_len[i]),
                          "events": dict(zip(m.event_names, runner.ep_events[i].tolist()))} for i in range(workers)])

        def save(name: str | None) -> None:
            if name:
                save_checkpoint(run / name, policy, opt, st["u"], cfg, m, rstd, steps, st["eps"])
            save_checkpoint(run / "latest.pt", policy, opt, st["u"], cfg, m, rstd, steps, st["eps"])
            st["saved_u"] = st["u"]

        status("running")
        with open(run / "metrics.jsonl", "a", encoding="utf8") as mf:
            while True:
                # Control cooperativo del panel (docs/specs/dashboard.md): se atiende al empezar cada actualización.
                while True:
                    cmd, paused = runctl.read_control(run)
                    st["paused"] = set(paused)
                    active = np.array([i not in st["paused"] for i in range(workers)])
                    if cmd != "run" and st["saved_u"] != st["u"]:
                        save(None)
                    if cmd == "stop" or (cmd == "run" and active.any()):
                        break
                    status("paused")
                    time.sleep(1.0)
                if cmd == "stop":
                    status("stopped")
                    break
                for i in range(workers):  # pausas y reanudaciones al registro del worker
                    if bool(active[i]) != st.get("was_active", [True] * workers)[i]:
                        wlogs.write(i, "reanudado" if active[i] else "en pausa")
                st["was_active"] = [bool(x) for x in active]
                u = st["u"] + 1
                progress = st["eps"] / goal if goal else (u - 1) / max(1, total)
                frac = max(0.0, 1 - progress)
                for g in opt.param_groups:
                    g["lr"] = pcfg.lr * max(pcfg.lr_floor, frac)
                ent_coef = pcfg.ent_coef_final + (pcfg.ent_coef - pcfg.ent_coef_final) * frac
                runner.conquest_w = conquest_weight(cfg, progress)
                tr = time.time()
                if async_mode:
                    episodes, info = runner.collect_async(buf, active=active, samples=T * int(active.sum()))
                else:
                    episodes, info = runner.collect(buf, active=active)
                t_roll = time.time() - tr
                rstd.update(buf.reward, buf.done, buf.valid)
                buf.compute_gae(runner.last_value, pcfg.gamma, pcfg.gae_lambda, rstd.std, use_lengths=async_mode)
                tu = time.time()
                stats = ppo_update(policy, opt, buf, pcfg, ent_coef, rng, device)
                steps += int(buf.valid.sum()) if async_mode else T * int(active.sum())
                st["u"], st["eps"] = u, st["eps"] + len(episodes)
                for i, g in enumerate(vec.games):
                    for line in g.drain_stderr():
                        wlogs.write(i, f"[motor] {line}")
                wstats.save(run / "workers.json")
                done = st["eps"] >= goal if goal else u >= total
                row: dict[str, Any] = {
                    "update": u, "steps": steps, "episodes_done": st["eps"], "active_workers": int(active.sum()),
                    "time": round(time.time() - t0, 1),
                    "t_rollout": round(t_roll, 2), "t_update": round(time.time() - tu, 2),
                    "samples": int(buf.valid.sum()), "t_net": round(runner.timing["net"], 2), "t_env": round(runner.timing["env"], 2),
                    "worker_util": round(runner.timing["busy"] / runner.timing["max"], 3) if runner.timing["max"] else None,
                    "reward_scale": rstd.std, "ent_coef": ent_coef, **stats, **info,
                    "noop_frac": float(runner.kind_counts[0] / max(1, runner.kind_counts.sum())),
                    "kind_frac": (runner.kind_counts / max(1, runner.kind_counts.sum())).round(4).tolist(),
                    "step_reward_mean": float(buf.reward[buf.valid].mean()),
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
                    pr = " ".join(f"{ABBR.get(k, k[:4])}={row[f'ep_part_{k}']:+.3f}" for k in [*runner.parts, "conquest"] if f"ep_part_{k}" in row)
                    evs = " ".join(f"{a}={row[f'ep_ev_{k}']:.1f}" for a, k in EV_SHOW if f"ep_ev_{k}" in row)
                    prog = f"{st['eps']}/{goal} partidas" if goal else f"{u}/{total}"
                    print(f"[{u}|{prog}] steps={steps} r/step={row['step_reward_mean']:+.4f} "
                          f"ep_ret={row.get('ep_return', float('nan')):+.3f} ({pr}) ent={stats['ent']:.2f} kl={stats['kl']:.4f} "
                          f"[{evs}] inv={info['invalid_rate']:.2f} noop={row['noop_frac']:.2f} "
                          f"roll={t_roll:.1f}s (red {row['t_net']:.1f} motor {row['t_env']:.1f} util {row['worker_util'] or 0:.0%}) upd={row['t_update']:.1f}s", flush=True)
                if u % cfg["checkpoint_every"] == 0 or done:
                    save(f"ckpt_{u:06d}.pt")
                st["sps"] = (int(buf.valid.sum()) if async_mode else T * int(active.sum())) / max(1e-6, t_roll + row["t_update"])
                if done:
                    status("finished")
                    break
                status("running")
    finally:
        vec.close()
    return run


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--updates", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--env", action="append", default=[], metavar="CLAVE=VALOR", help="cambia el escenario (map, bots, nations, difficulty); se puede repetir")
    ap.add_argument("--conquest-budget", type=float, default=None, help="bonus máximo por tribus conquistadas (reward.md §9)")
    ap.add_argument("--conquest-until", type=float, default=None, help="progreso del run en que el bonus llega a 0")
    a = ap.parse_args(argv)
    over: dict[str, Any] = {k: v for k, v in (("updates", a.updates), ("episodes", a.episodes), ("workers", a.workers), ("seed", a.seed)) if v is not None}
    if a.env:
        over["env"] = parse_env(a.env)
    cq = {k: v for k, v in (("budget", a.conquest_budget), ("until", a.conquest_until)) if v is not None}
    if cq:
        over["conquest"] = cq
    train(load_config(a.config, over), a.run_dir, a.resume)


if __name__ == "__main__":
    main()
