"""Panel de control (docs/specs/dashboard.md): lanzar, pausar, reanudar y detener un entrenamiento real y evaluar metas."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest
import yaml

from openfront_rl.dashboard import Panel, make_server

CFG = {
    "name": "panel", "seed": 1, "rollout_steps": 8, "updates": 400, "checkpoint_every": 1000, "workers": 1,
    "eval": {"every": 0, "episodes": 1}, "ppo": {"minibatch": 8, "epochs": 1},
    "env": {"bots": 5, "maxTicks": 800, "difficulty": "Easy", "nations": "disabled"},
}
GOALS = {"goals": [
    {"id": "g1", "title": "ataca", "metric": "ep_ev_attack_orders", "op": ">", "value": 0},
    {"id": "g2", "title": "vive", "metric": "ep_eliminated", "op": "<", "value": 0.5},
]}


@pytest.fixture()
def panel(tmp_path):
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    (cfgs / "mini.yaml").write_text(yaml.safe_dump(CFG), encoding="utf8")
    (cfgs / "duo.yaml").write_text(yaml.safe_dump({**CFG, "workers": 2}), encoding="utf8")
    (cfgs / "goals.yaml").write_text(yaml.safe_dump(GOALS), encoding="utf8")
    p = Panel(tmp_path / "runs", cfgs)
    srv = make_server(p, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    p.base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield p
    srv.shutdown()
    for r in p.list_runs():
        try:
            p.kill(r["name"])
        except Exception:
            pass


def call(p, path, body=None):
    req = urllib.request.Request(p.base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait(fn, what, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        v = fn()
        if v:
            return v
        time.sleep(0.5)
    raise AssertionError(f"tiempo agotado esperando: {what}")


def test_rechaza_peticiones_fuera_de_la_lista(panel):
    assert call(panel, "/api/runs", {"config": "../../etc/passwd"})[0] == 400
    assert call(panel, "/api/runs", {"config": "mini.yaml", "cmd": "rm -rf /"})[0] == 400
    assert call(panel, "/api/runs", {"config": "mini.yaml", "name": "../fuera"})[0] == 400
    assert call(panel, "/api/runs", {"config": "mini.yaml", "workers": "3"})[0] == 400
    assert call(panel, "/api/runs/../x/status")[0] in (400, 404)
    assert call(panel, "/api/runs/noexiste/status")[0] == 404
    assert call(panel, "/api/configs") == (200, ["duo.yaml", "mini.yaml"])


def test_ciclo_completo_lanzar_pausar_reanudar_detener(panel):
    code, r = call(panel, "/api/runs", {"config": "mini.yaml", "name": "t1"})
    assert code == 201 and r["name"] == "t1"
    d = panel.runs_dir / "t1"

    def st():
        return call(panel, "/api/runs/t1/status")[1]

    wait(lambda: (st()["status"] or {}).get("update", 0) >= 2 and st()["state"] == "running", "2 actualizaciones")
    assert call(panel, "/api/runs/t1/architecture")[1]["total"] > 1_000_000
    assert call(panel, "/api/runs")[1][0]["name"] == "t1"

    assert call(panel, "/api/runs/t1/control", {"command": "pause"})[0] == 200
    wait(lambda: st()["state"] == "paused", "pausa")
    u = st()["status"]["update"]
    time.sleep(3)
    assert st()["status"]["update"] == u  # pausado de verdad: no avanza
    assert (d / "latest.pt").exists()

    assert call(panel, "/api/runs/t1/control", {"command": "run"})[0] == 200
    wait(lambda: st()["state"] == "running" and st()["status"]["update"] > u, "reanudar")

    rows = call(panel, "/api/runs/t1/metrics?since=0")[1]
    assert rows and "loss" in rows[-1]
    goals = call(panel, "/api/runs/t1/goals")[1]
    assert [g["id"] for g in goals] == ["g1", "g2"] and all("done" in g for g in goals)

    assert call(panel, "/api/runs/t1/control", {"command": "stop"})[0] == 200
    wait(lambda: st()["state"] == "stopped", "detención")
    assert (d / "latest.pt").exists() and (d / "metrics.jsonl").exists()
    wait(lambda: panel._proc(st()["status"]) is None, "que el proceso termine solo")
    assert call(panel, "/api/runs/t1/kill", {})[0] == 409  # ya no está en ejecución
    assert call(panel, "/api/runs", {"config": "mini.yaml", "name": "t1", "resume": True})[0] == 201
    wait(lambda: st()["state"] == "running", "reanudado desde el checkpoint")
    call(panel, "/api/runs/t1/kill", {})


def test_metas_se_evaluan_contra_metrics(panel):
    d = panel.runs_dir / "m1"
    d.mkdir()
    rows = [{"update": 1, "ep_ev_attack_orders": 0.0, "ep_eliminated": 1.0}, {"update": 2, "ep_ev_attack_orders": 3.0, "ep_eliminated": 0.0}]
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf8")
    g = {x["id"]: x for x in panel.goals("m1")}
    assert g["g1"]["done"] and g["g1"]["best"] == 3.0
    assert g["g2"]["done"] and g["g2"]["best"] == 0.0


def test_pausar_y_reanudar_un_worker(panel):
    assert call(panel, "/api/runs", {"config": "duo.yaml", "name": "w1", "episodes": 500})[0] == 201

    def st():
        return call(panel, "/api/runs/w1/status")[1]

    wait(lambda: len((st()["status"] or {}).get("workers") or []) == 2 and st()["status"]["update"] >= 2, "2 workers activos")
    assert call(panel, "/api/runs/w1/workers", {"worker": 5, "paused": True})[0] == 400
    assert call(panel, "/api/runs/w1/workers", {"worker": 1, "paused": "si"})[0] == 400
    assert call(panel, "/api/runs/w1/workers", {"worker": 1, "paused": True}) == (200, {"paused_workers": [1]})
    assert st()["paused_workers"] == [1]

    wait(lambda: st()["status"]["workers"][1]["paused"], "worker 1 en pausa")
    ticks = [w["tick"] for w in st()["status"]["workers"]]
    u0 = st()["status"]["update"]
    wait(lambda: st()["status"]["update"] >= u0 + 2, "dos actualizaciones más con un solo worker")
    now = st()["status"]["workers"]
    assert now[1]["tick"] == ticks[1] and now[0]["tick"] != ticks[0]  # el pausado no avanza, el otro sí
    rows = call(panel, "/api/runs/w1/metrics?since=0")[1]
    assert rows[-1]["active_workers"] == 1 and rows[-1]["episodes_done"] >= 0

    assert call(panel, "/api/runs/w1/workers", {"worker": 1, "paused": False}) == (200, {"paused_workers": []})
    u1 = st()["status"]["update"]
    wait(lambda: st()["status"]["update"] >= u1 + 2 and not st()["status"]["workers"][1]["paused"], "worker 1 reanudado")
    assert call(panel, "/api/runs/w1/metrics?since=0")[1][-1]["active_workers"] == 2
    call(panel, "/api/runs/w1/kill", {})
