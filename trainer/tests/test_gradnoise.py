import json

import numpy as np
import torch

from openfront_rl.gradnoise import mean_cosine, measure, noise_scale
from openfront_rl.train import load_config, train


def test_escala_de_ruido_sintetica():
    # Cada lote de b muestras: G + ruido con varianza sigma²/b por dimensión → B_simple = d·sigma²/|G|².
    rng = np.random.default_rng(0)
    d, b, sigma = 200, 64.0, 3.0
    g = torch.ones(d, dtype=torch.float64) * 0.5
    grads = [g + torch.from_numpy(rng.normal(0, sigma / np.sqrt(b), d)) for _ in range(400)]
    ns = noise_scale(grads, b)
    esperado = d * sigma**2 / float(g.dot(g))
    assert abs(ns["b_noise"] / esperado - 1) < 0.15
    assert mean_cosine([g, g, g]) > 0.999


def test_medir_en_un_checkpoint_chico(tmp_path):
    cfg = load_config(None, {
        "name": "humo", "seed": 3, "workers": 2, "rollout_steps": 8, "updates": 1, "checkpoint_every": 1,
        "eval": {"every": 0, "episodes": 1}, "ppo": {"minibatch": 8, "epochs": 1},
        "env": {"bots": 5, "maxTicks": 800, "difficulty": "Easy", "nations": "disabled"},
    })
    run = train(cfg, tmp_path / "g", log=False)
    out = measure(str(run / "latest.pt"), batches=4, workers=2, rollout_steps=8, seed="t", device=torch.device("cpu"), log=False)
    assert out["lotes"] == 4 and out["muestras_por_lote"] == 16
    assert -1.0 <= out["coseno_medio"] <= 1.0
    json.dumps(out)
