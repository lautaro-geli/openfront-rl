"""Red, política autorregresiva y PPO contra el motor real (docs/adr/0007)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openfront_rl import NodeGame  # noqa: E402
from openfront_rl.model import Policy, ModelConfig, to_tensors  # noqa: E402
from openfront_rl.ppo import PPOConfig, RolloutBuffer, ppo_update  # noqa: E402
from openfront_rl.train import load_config, stack_obs, train  # noqa: E402

CFG = {"agents": 1, "bots": 8, "maxTicks": 1500, "rewardAttackWeight": 0.12, "rewardEconomyWeight": 0.1, "rewardDefenseWeight": 0.05}


def play(policy, steps, seed="pol-1"):
    """Juega con la política y devuelve lo visto: observaciones, máscaras, acciones, log-probs y valores."""
    rows = []
    with NodeGame(CFG) as g:
        r = g.reset(seed)
        for _ in range(steps):
            ob, mk = to_tensors(r.obs, r.mask)
            with torch.no_grad():
                act, logp, ent, val = policy(ob, mk)
            rows.append((r.obs, r.mask, act.numpy(), logp.numpy(), val.numpy(), r.invalid.copy()))
            r = g.step(act.numpy().astype(np.int32))
            rows[-1] += (r.invalid.copy(), r.reward_parts.copy())
            if r.done:
                break
    return rows, g.manifest


@pytest.fixture(scope="module")
def policy_and_rows():
    with NodeGame(CFG) as g:
        m = g.manifest
    torch.manual_seed(0)
    p = Policy(m)
    rows, m = play(p, 120)
    return p, rows, m


def test_ninguna_accion_viola_la_mascara(policy_and_rows):
    p, rows, m = policy_and_rows
    sp = m.space
    kinds = sp["kinds"]
    for obs, mask, act, *_ in rows:
        k, row, cell, frac, typ, aux = (int(x) for x in act[0])
        assert mask["kind"][0][k], kinds[k]
        name = kinds[k]
        if name in sp["playerKinds"]:
            assert mask["player"][0][sp["playerKinds"].index(name)][row]
        if name in sp["unitKinds"]:
            assert mask["unit"][0][sp["unitKinds"].index(name)][row]
        if name == "cancel_attack":
            assert mask["attack_row"][0][row]
        if name == "spawn":
            assert mask["cell"][0][sp["cellKinds"].index("spawn")][cell]
        if name == "build":
            assert mask["build_type"][0][typ]
            ck = "build_ship" if sp["buildTypes"][typ] == "warship" else "build"
            assert mask["cell"][0][sp["cellKinds"].index(ck)][cell]
        if name == "nuke":
            assert mask["nuke_type"][0][typ]
            assert mask["cell"][0][sp["cellKinds"].index("nuke")][cell]
        if name in ("boat_attack", "move_warship"):
            assert mask["cell"][0][sp["cellKinds"].index(name)][cell]
        # Los campos que el tipo no usa van a 0.
        if name == "noop":
            assert not act[0][1:].any()
    invalid = np.mean([bool(r[6][0]) for r in rows])
    assert invalid < 0.35, f"demasiadas acciones inválidas con una red sin entrenar: {invalid:.2f}"


def test_logprob_recalculado_coincide(policy_and_rows):
    p, rows, _ = policy_and_rows
    obs = {k: np.concatenate([r[0][k] for r in rows[:32]]) for k in rows[0][0]}
    mask = {k: np.concatenate([r[1][k] for r in rows[:32]]) for k in rows[0][1]}
    act = np.concatenate([r[2] for r in rows[:32]])
    logp = np.concatenate([r[3] for r in rows[:32]])
    val = np.concatenate([r[4] for r in rows[:32]])
    ob, mk = to_tensors(obs, mask)
    with torch.no_grad():
        a2, lp2, ent, v2 = p(ob, mk, action=torch.as_tensor(act))
    assert (a2.numpy() == act).all()
    np.testing.assert_allclose(lp2.numpy(), logp, atol=1e-4)
    np.testing.assert_allclose(v2.numpy(), val, atol=1e-4)
    assert torch.isfinite(ent).all() and (ent >= 0).all()


def test_ppo_baja_la_perdida_en_un_lote_fijo(policy_and_rows):
    _, rows, m = policy_and_rows
    torch.manual_seed(1)
    p = Policy(m)
    T = 16
    buf = RolloutBuffer(T, 1, {k: b.shape for k, b in m.obs.items()}, {k: b.shape for k, b in m.mask.items()}, m.action_dims)
    obs = {k: np.concatenate([r[0][k] for r in rows[:T]]) for k in rows[0][0]}
    mask = {k: np.concatenate([r[1][k] for r in rows[:T]]) for k in rows[0][1]}
    for t in range(T):
        buf.add_obs(t, {k: v[t : t + 1] for k, v in obs.items()}, {k: v[t : t + 1] for k, v in mask.items()})
    ob, mk = to_tensors(obs, mask)
    with torch.no_grad():
        act, logp, _, val = p(ob, mk)
    buf.action[:, 0], buf.logp[:, 0], buf.value[:, 0] = act.numpy(), logp.numpy(), val.numpy()
    buf.reward[:, 0] = np.linspace(0, 1, T)
    buf.compute_gae(np.zeros(1, np.float32), 0.99, 0.95, 1.0)
    cfg = PPOConfig(epochs=8, minibatch=8, lr=1e-3, target_kl=10.0)
    opt = torch.optim.Adam(p.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(0)
    first = ppo_update(p, opt, buf, cfg, 0.0, rng, torch.device("cpu"))
    for _ in range(3):
        last = ppo_update(p, opt, buf, cfg, 0.0, rng, torch.device("cpu"))
    assert last["vf"] < first["vf"]
    assert all(np.isfinite(v) for v in last.values())


def test_entrenamiento_corto_guarda_config_y_checkpoint(tmp_path):
    cfg = load_config(None, {
        "name": "humo", "seed": 3, "workers": 2, "rollout_steps": 8, "updates": 2, "checkpoint_every": 1,
        "eval": {"every": 0, "episodes": 1}, "ppo": {"minibatch": 8, "epochs": 1},
        "env": {"bots": 5, "maxTicks": 800, "difficulty": "Easy", "nations": "disabled",
                "rewardAttackWeight": 0.12, "rewardEconomyWeight": 0.1, "rewardDefenseWeight": 0.05},
    })
    run = train(cfg, tmp_path / "run", log=False)
    assert (run / "config.yaml").exists() and (run / "metrics.jsonl").exists() and (run / "latest.pt").exists()
    ck = torch.load(run / "latest.pt", weights_only=False)
    assert ck["config"]["seed"] == 3 and ck["protocol_version"] == 3 and ck["update"] == 2
    lines = (run / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    import json
    row = json.loads(lines[-1])
    assert np.isfinite(row["loss"]) and np.isfinite(row["ent"])


def test_buffer_ignora_muestras_de_workers_pausados():
    from openfront_rl.model import MASK_KEYS, OBS_KEYS
    from openfront_rl.ppo import RolloutBuffer, RunningStd

    buf = RolloutBuffer(4, 3, {k: (1,) for k in OBS_KEYS}, {k: (1,) for k in MASK_KEYS}, 6)
    buf.valid[:, 1] = False
    idx = np.concatenate(list(buf.minibatches(5, np.random.default_rng(0))))
    assert len(idx) == 8 and set(idx % 3) == {0, 2}  # índice plano = t * N + worker
    r = RunningStd(0.99)
    rew = np.ones((4, 3))
    r.update(rew, np.zeros((4, 3)), buf.valid)
    assert r.n == 8 and r.ret[1] == 0.0  # el worker pausado no avanza su retorno descontado


def test_entrenamiento_por_partidas_termina_y_lo_guarda(tmp_path):
    import json

    cfg = load_config(None, {
        "name": "partidas", "seed": 5, "workers": 2, "rollout_steps": 40, "episodes": 3, "updates": 999, "checkpoint_every": 1000,
        "eval": {"every": 0, "episodes": 1}, "ppo": {"minibatch": 16, "epochs": 1},
        "env": {"bots": 5, "maxTicks": 400, "difficulty": "Easy", "nations": "disabled"},
    })
    run = train(cfg, tmp_path / "run", log=False)
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["episodes_done"] >= 3 and rows[-2]["episodes_done"] < 3 and len(rows) < 10
    st = json.loads((run / "status.json").read_text())
    assert st["state"] == "finished" and st["episodes_total"] == 3 and st["episodes_done"] == rows[-1]["episodes_done"]
    assert torch.load(run / "latest.pt", weights_only=False)["episodes_done"] == rows[-1]["episodes_done"]
    ws = json.loads((run / "workers.json").read_text())
    assert sum(w["games"] for w in ws) == rows[-1]["episodes_done"]  # cada partida terminada se anota en su worker
    assert all(0 <= w["sum_max"] / max(1, w["games"]) <= 1 for w in ws)
    assert any("partida 1:" in x.read_text() for x in (run / "workers").glob("worker-*.log"))


def test_resolve_device():
    from openfront_rl.train import resolve_device
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="GPU"):
            resolve_device("cuda")


def test_vecgame_con_perfil_mide_latencias_y_no_cambia_resultados():
    from openfront_rl.vec_env import VecGame
    acts = np.zeros((2, 1, 6), np.int32)
    out = {}
    for prof in (False, True):
        with VecGame(2, CFG, base_seed="perfil", profile=prof) as v:
            v.reset()
            rs = [v.step(acts) for _ in range(5)]
            out[prof] = ([r.reward.tolist() for step in rs for r in step], list(v.lat))
    assert out[False][0] == out[True][0]  # mismos resultados con y sin hilos
    assert all(x > 0 for x in out[True][1]) and out[False][1] == [0.0, 0.0]


def test_gae_con_largos_distintos_equivale_a_buffers_independientes():
    rng = np.random.default_rng(3)
    T, N = 12, 3
    shapes = {"x": (1,)}
    def make(t):
        b = RolloutBuffer.__new__(RolloutBuffer)
        b.T, b.N = t, 1
        b.reward = rng.normal(size=(t, 1)).astype(np.float32); b.value = rng.normal(size=(t, 1)).astype(np.float32)
        b.done = np.zeros((t, 1), np.float32); b.done[min(4, t - 1), 0] = 1.0 if t > 5 else 0.0
        b.boot = np.zeros((t, 1), np.float32); b.adv = np.zeros((t, 1), np.float32); b.ret = np.zeros((t, 1), np.float32)
        b.lengths = np.full(1, t)
        return b
    lens = [12, 7, 3]
    parts = [make(L) for L in lens]
    lastv = rng.normal(size=3).astype(np.float32)
    big = RolloutBuffer.__new__(RolloutBuffer)
    big.T, big.N = T, N
    for name in ("reward", "value", "done", "boot", "adv", "ret"):
        arr = np.zeros((T, N), np.float32)
        for j, b in enumerate(parts):
            arr[: lens[j], j] = getattr(b, name)[:, 0]
        setattr(big, name, arr)
    big.lengths = np.array(lens)
    big.compute_gae(lastv, 0.99, 0.9, 2.0, use_lengths=True)
    for j, b in enumerate(parts):
        b.compute_gae(lastv[j: j + 1], 0.99, 0.9, 2.0)
        assert np.allclose(big.adv[: lens[j], j], b.adv[:, 0], atol=1e-5)
        assert np.allclose(big.ret[: lens[j], j], b.ret[:, 0], atol=1e-5)


def test_rollout_asincrono_llena_el_objetivo_y_respeta_el_tope(tmp_path):
    from openfront_rl.vec_env import VecGame
    from openfront_rl.train import Runner
    T, N = 10, 2
    with VecGame(N, {**CFG, "agents": 1}, base_seed="asinc", profile=True) as vec:
        m = vec.manifest
        pol = Policy(m, ModelConfig.from_dict({}))
        runner = Runner(vec, pol, torch.device("cpu"))
        buf = RolloutBuffer(2 * T, N, {k: b.shape for k, b in m.obs.items()}, {k: b.shape for k, b in m.mask.items()}, m.action_dims)
        eps, info = runner.collect_async(buf, samples=T * N)
        assert int(buf.valid.sum()) == T * N
        assert (buf.lengths >= 1).all() and (buf.lengths <= 2 * T).all()
        for i in range(N):  # cada columna es contigua desde 0 hasta su largo
            assert buf.valid[: buf.lengths[i], i].all() and not buf.valid[buf.lengths[i]:, i].any()
        assert runner.last_value.shape == (N,)
        buf.compute_gae(runner.last_value, 0.997, 0.95, 0.05, use_lengths=True)
        assert np.isfinite(buf.adv[buf.valid]).all()


def test_entrenar_con_rollout_asincrono(tmp_path):
    cfg = load_config(None, {"name": "asinc", "workers": 2, "updates": 3, "rollout_steps": 16, "rollout_mode": "async", "device": "cpu",
                             "eval": {"every": 0, "episodes": 1}, "env": CFG})
    run = train(cfg, tmp_path / "r", log=False)
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and all(r["samples"] == 32 and 0 < r["worker_util"] <= 1.0 for r in rows)


def test_evaluate_juega_exactamente_las_partidas_pedidas():
    from openfront_rl.train import evaluate
    with NodeGame(CFG) as g:
        m = g.manifest
    pol = Policy(m, ModelConfig.from_dict({})).eval()
    out = evaluate(pol, CFG, workers=2, episodes=3, seed="ev", device=torch.device("cpu"))
    assert out["eval_episodes"] == 3.0 and 0.0 <= out["eval_won"] <= 1.0


def test_critico_separado_no_toca_el_codificador_de_la_politica(policy_and_rows):
    rows = policy_and_rows[1]
    obs, mask = rows[0][0], rows[0][1]
    with NodeGame(CFG) as g:
        man = g.manifest
    torch.manual_seed(1)
    p = Policy(man, ModelConfig(separate_critic=True))
    assert p.enc_v is not None
    ob, mk = to_tensors(obs, mask)
    _, _, _, val = p(ob, mk)
    val.sum().backward()
    assert all(q.grad is None or float(q.grad.abs().sum()) == 0.0 for q in p.enc.parameters())
    assert any(q.grad is not None and float(q.grad.abs().sum()) > 0 for q in p.enc_v.parameters())
    assert torch.allclose(p.values(ob), p(ob, mk)[3])


def test_reanudar_con_critico_separado_copia_el_codificador(tmp_path):
    base = {
        "name": "humo", "seed": 3, "workers": 2, "rollout_steps": 8, "updates": 1, "checkpoint_every": 1,
        "eval": {"every": 0, "episodes": 1}, "ppo": {"minibatch": 8, "epochs": 1},
        "env": {"bots": 5, "maxTicks": 800, "difficulty": "Easy", "nations": "disabled"},
    }
    run1 = train(load_config(None, base), tmp_path / "a", log=False)
    cfg2 = load_config(None, {**base, "updates": 2, "model": {"separate_critic": True}})
    run2 = train(cfg2, tmp_path / "b", resume=run1 / "latest.pt", log=False)
    sd = torch.load(run2 / "latest.pt", weights_only=False)["model"]
    assert any(k.startswith("enc_v.") for k in sd)


def test_should_stop_alarmas():
    from openfront_rl.train import should_stop
    sp = {"clipfrac": 0.03, "window": 3, "min_updates": 2, "win_below": 0.3, "win_window": 4}
    assert should_stop(None, 100, [0.9] * 50, [0.0] * 50) is None  # sin configuración nunca para
    assert should_stop(sp, 1, [0.5] * 10, []) is None  # antes de min_updates
    assert "clipfrac" in should_stop(sp, 5, [0.01, 0.04, 0.04, 0.04], [])
    assert should_stop(sp, 5, [0.01] * 5, [1.0, 1.0, 1.0, 1.0]) is None
    assert "victorias" in should_stop(sp, 5, [0.01] * 5, [1.0, 0.0, 0.0, 0.0])
    assert should_stop(sp, 5, [0.01] * 5, [0.0, 0.0]) is None  # pocas partidas para juzgar
    se = {"eval_below": 0.45, "eval_count": 2, "min_updates": 2}
    assert should_stop(se, 5, [], [], [0.6, 0.3]) is None  # una sola evaluación baja no para
    assert should_stop(se, 5, [], [], [0.3]) is None  # pocas evaluaciones
    assert should_stop(se, 1, [], [], [0.3, 0.3]) is None  # antes de min_updates
    assert "evaluaciones" in should_stop(se, 5, [], [], [0.6, 0.4, 0.3])
    assert should_stop(se, 5, [], [], None) is None  # sin evaluaciones (every: 0)


def test_lr_propio_del_critico_y_parada_en_un_run_corto(tmp_path):
    base = {
        "name": "humo", "seed": 3, "workers": 2, "rollout_steps": 8, "updates": 3, "checkpoint_every": 1,
        "eval": {"every": 0, "episodes": 1}, "ppo": {"minibatch": 8, "epochs": 1},
        "env": {"bots": 5, "maxTicks": 800, "difficulty": "Easy", "nations": "disabled"},
    }
    run1 = train(load_config(None, {**base, "updates": 1}), tmp_path / "a", log=False)
    cfg2 = load_config(None, {**base, "updates": 3, "ppo": {"minibatch": 8, "epochs": 1, "critic_lr_mult": 4.0},
                              "rollout_steps": 90, "stop": {"win_below": 2.0, "win_window": 1, "min_updates": 1}})
    run2 = train(cfg2, tmp_path / "b", resume=run1 / "latest.pt", log=False)  # el optimizador no encaja: se reinicia
    ck = torch.load(run2 / "latest.pt", weights_only=False)
    mults = sorted(g.get("mult", 1.0) for g in ck["optimizer"]["param_groups"])
    assert mults == [1.0, 4.0]
    assert (run2 / "alarma.txt").exists() and ck["update"] == 2  # reanudó en 1 y paró tras 1 actualización nueva
    grp = {g["mult"]: g["lr"] for g in ck["optimizer"]["param_groups"]}
    assert abs(grp[4.0] / grp[1.0] - 4.0) < 1e-6


def test_bonus_de_conquista_relativo():
    from openfront_rl.train import conquest_weight
    cfg = {"conquest": {"budget": 0.2, "until": 0.5, "relative": True}, "env": {"bots": 20}}
    assert conquest_weight(cfg, 0.69) == 0.0  # progreso absoluto al reanudar: ya apagado
    assert conquest_weight(cfg, 0.0) > 0.0    # progreso relativo al inicio de este run


def test_parada_por_evaluacion_en_un_run_corto(tmp_path):
    cfg = load_config(None, {
        "name": "humo", "seed": 3, "workers": 2, "rollout_steps": 8, "updates": 4, "checkpoint_every": 1,
        "eval": {"every": 1, "episodes": 2}, "ppo": {"minibatch": 8, "epochs": 1},
        "env": {"bots": 5, "maxTicks": 800, "difficulty": "Easy", "nations": "disabled"},
        "stop": {"eval_below": 2.0, "eval_count": 1, "min_updates": 1},
    })
    run = train(cfg, tmp_path / "e", log=False)
    rows = [json.loads(x) for x in (run / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and "eval_won" in rows[0]  # evaluó en la actualización 1 y paró
    assert "evaluaciones" in (run / "alarma.txt").read_text(encoding="utf8")
