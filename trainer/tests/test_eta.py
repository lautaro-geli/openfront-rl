from openfront_rl.eta import estimate


def _rows(n_upd: int, dt: float, eps: int, eval_every: int, eval_extra: float, eval_games: int):
    rows, t = [], 0.0
    for u in range(1, n_upd + 1):
        prev_eval = rows and "eval_won" in rows[-1]
        t += dt + (eval_extra if prev_eval else 0.0)
        r = {"update": u, "time": t, "episodes": eps}
        if u % eval_every == 0:
            r |= {"eval_won": 0.8, "eval_episodes": eval_games}
        rows.append(r)
    return rows


def test_estimacion_con_tasas_medidas():
    # 140 s por actualización, 12 partidas por actualización, evaluación de 40 partidas que suma 400 s, 4 workers.
    rows = _rows(30, 140.0, 12, 10, 400.0, 40)
    cfg = {"workers": 4, "eval": {"every": 10, "episodes": 40}}
    status = {"update": 30, "episodes_done": 360, "episodes_total": 600}
    r = estimate(rows, status, cfg, final_eval=80)
    assert r["actualizaciones_restantes"] == 20 and r["evaluaciones_restantes"] == 2
    assert r["eval_s_por_partida_y_worker"][1] == 40  # 400 s × 4 workers / 40 partidas
    # 20 × 140 + 2 × 400 + evaluación final de 80 partidas (800 s) = 4400 s
    assert abs(r["segundos"]["mediana"] - 4400) < 1


def test_sin_evaluaciones_medidas_usa_las_partidas_de_entrenamiento():
    rows = _rows(5, 120.0, 10, 1000, 0.0, 40)
    r = estimate(rows, {"update": 5, "episodes_done": 50, "episodes_total": 150}, {"workers": 4, "eval": {"every": 10, "episodes": 40}})
    assert "sin evaluaciones medidas" in r["eval_fuente"] and r["evaluaciones_restantes"] == 1
