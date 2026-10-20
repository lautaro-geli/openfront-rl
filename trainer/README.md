# trainer

Cliente Python del bridge, entornos y (próximamente) PPO y self-play.

**Estado: Fase 2.** Hecho: cliente del protocolo (`openfront_rl/client.py`), entorno en paralelo (`vec_env.py`), envoltorio Gymnasium de un agente (`gym_env.py`) y muestreo de acciones válidas (`masks.py`). Falta: la red por ramas y PPO (Fase 3).

## Requisitos
- Python 3.11+ con `numpy` y `gymnasium` (`pip install -e .[dev]`).
- Node 24 y `npm ci` hechos en `vendor/openfront` (con `--ignore-scripts`) y en `bridge/`. Si `node` no es el 24, define `OPENFRONT_NODE` con su ruta.

## Uso
```python
import numpy as np
from openfront_rl import NodeGame, VecGame, OpenFrontEnv, sample_valid_action

# Una partida con dos agentes (self-play)
with NodeGame({"agents": 2, "bots": 20}) as g:
    r = g.reset("semilla")             # r.obs["grid"]: [2, 20, 32, 64]; r.mask["kind"]: [2, 24]
    r = g.step(np.zeros((2, 6), np.int32))

# Varias partidas en paralelo (por defecto núcleos - 1)
with VecGame(5, {"agents": 1, "bots": 20}) as v:
    results = v.reset()

# Gymnasium, un agente
env = OpenFrontEnv({"bots": 20})
obs, info = env.reset(seed="x")        # info["action_mask"]
```

## Pruebas y medición
```bash
pytest                                    # usa el motor real; se salta si falta Node o `npm ci`
python scripts/bench.py --workers 1 5 --map World --bots 400
```
