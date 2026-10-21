"""Pruebas del cliente contra el servidor Node real (motor de OpenFront incluido)."""
import numpy as np
import pytest

from openfront_rl import NodeGame, OpenFrontEnv, ProtocolError, VecGame, sample_valid_action

CFG = {"agents": 2, "bots": 5, "maxTicks": 1200}


def spawn_actions(m, r):
    """Cada agente aparece en una celda de tierra distinta."""
    cells = np.flatnonzero(r.mask["cell"][0][m.space["cellKinds"].index("spawn")])
    a = np.zeros((m.agents, m.action_dims), np.int32)
    for i in range(m.agents):
        a[i, 0] = m.space["kinds"].index("spawn")
        a[i, 2] = cells[(i + 1) * len(cells) // (m.agents + 1)]
    return a


def test_manifest_y_formas():
    with NodeGame(CFG) as g:
        m = g.manifest
        r = g.reset("py-1")
        assert m.agents == 2 and m.action_dims == 6 and len(m.space["kinds"]) == 24
        assert r.obs["grid"].shape == (2, 20, 32, 64)
        assert r.obs["players"].shape == (2, 16, 46)
        assert r.mask["kind"].shape == (2, 24) and r.mask["cell"].shape == (2, 6, 2048)
        assert r.mask["kind"][:, 0].all()  # noop siempre
        assert r.mask["spawn"].all()  # empieza en fase de spawn
        assert r.obs["grid"].dtype == np.float32 and r.mask["kind"].dtype == bool
        assert not r.done and r.reward.shape == (2,)
        assert r.events.shape == (2, len(m.event_names)) and "tribes_conquered" in m.event_names and (r.events == 0).all()


def test_determinismo_y_hash():
    def play(seed):
        with NodeGame(CFG) as g:
            r = g.reset(seed)
            m = g.manifest
            r = g.step(spawn_actions(m, r))
            for _ in range(10):
                a = np.zeros((2, 6), np.int32)
                a[:, 0] = m.space["kinds"].index("expand")
                a[:, 3] = 2
                r = g.step(a)
            return r.obs["grid"].copy(), r.reward.copy(), g.state_hash()

    a, b = play("py-det"), play("py-det")
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all() and a[2] == b[2]
    assert play("py-otra")[2] != a[2]


def test_accion_invalida_y_errores():
    with NodeGame({"agents": 1, "bots": 3, "maxTicks": 600}) as g:
        with pytest.raises(ValueError):
            g.send_step(np.zeros((2, 6), np.int32))  # forma incorrecta
        g.reset("py-err")
        bad = np.zeros((1, 6), np.int32)
        bad[0, 0] = 99
        assert g.step(bad).invalid[0]
        # Un nuke sin silo está enmascarado: llega como inválido y el motor no cambia.
        nuke = np.zeros((1, 6), np.int32)
        nuke[0, 0] = g.manifest.space["kinds"].index("nuke")
        assert g.step(nuke).invalid[0]


def test_proceso_muerto_da_error_claro():
    g = NodeGame({"agents": 1, "bots": 3})
    g._proc.kill()
    with pytest.raises(ProtocolError):
        g.reset("x")
    g.close()


def test_acciones_validas_aleatorias_no_son_invalidas():
    rng = np.random.default_rng(0)
    with NodeGame({"agents": 1, "bots": 8, "maxTicks": 3000}) as g:
        m = g.manifest
        r = g.reset("py-rand")
        invalid = steps = 0
        kinds_seen = set()
        while not r.done and steps < 250:
            v = sample_valid_action(m, r.mask, rng)
            kinds_seen.add(m.space["kinds"][v[0]])
            r = g.step(v.reshape(1, -1))
            invalid += int(r.invalid[0])
            steps += 1
        assert steps > 50
        # La resolución exacta puede rechazar alguna construcción o desembarco, pero debe ser la excepción.
        assert invalid <= steps * 0.15, f"{invalid} inválidas en {steps} pasos"
        assert len(kinds_seen) >= 6


def test_vec_igual_a_serie():
    def run_one(seed):
        with NodeGame({"agents": 1, "bots": 4, "maxTicks": 800}) as g:
            r = g.reset(seed)
            for _ in range(20):
                a = np.zeros((1, 6), np.int32)
                a[0, 0] = 1 if r.mask["spawn"][0] else 2
                if r.mask["spawn"][0]:
                    a[0, 2] = np.flatnonzero(r.mask["cell"][0][0])[100]
                r = g.step(a)
            return g.state_hash()

    serial = [run_one(f"vec-0-{0}"), run_one("vec-1-0")]
    with VecGame(2, {"agents": 1, "bots": 4, "maxTicks": 800}, base_seed="vec", auto_reset=False) as v:
        rs = v.reset()
        for _ in range(20):
            acts = np.zeros((2, 1, 6), np.int32)
            for i, r in enumerate(rs):
                acts[i, 0, 0] = 1 if r.mask["spawn"][0] else 2
                if r.mask["spawn"][0]:
                    acts[i, 0, 2] = np.flatnonzero(r.mask["cell"][0][0])[100]
            rs = v.step(acts)
        assert [g.state_hash() for g in v.games] == serial


def test_vec_auto_reset():
    with VecGame(2, {"agents": 1, "bots": 2, "maxTicks": 30}, base_seed="auto") as v:
        v.reset()
        saw_final = False
        for _ in range(40):
            rs = v.step(np.zeros((2, 1, 6), np.int32))
            saw_final |= any(f is not None for f in v.final)
        # maxTicks=30 con la fase de spawn: los episodios terminan y se reinician sin colgarse.
        assert saw_final or all(not r.done for r in rs)


def test_gymnasium():
    env = OpenFrontEnv({"bots": 4, "maxTicks": 600})
    obs, info = env.reset(seed="gym-1")
    assert set(obs) == set(env.observation_space.spaces)
    assert env.observation_space["grid"].shape == (20, 32, 64)
    assert env.action_space.shape == (6,)
    rng = np.random.default_rng(1)
    total = 0.0
    for _ in range(40):
        v = sample_valid_action(env.manifest, {k: x[None] for k, x in info["action_mask"].items()}, rng)
        obs, reward, terminated, truncated, info = env.step(v)
        total += reward
        if terminated or truncated:
            break
    assert np.isfinite(total)
    env.close()
