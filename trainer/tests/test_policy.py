"""Red, política autorregresiva y PPO contra el motor real (docs/adr/0007)."""
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
