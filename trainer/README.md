# trainer

Cliente Python del bridge, entornos y (próximamente) PPO y self-play.

**Estado: Fase 3.** Hecho: cliente del protocolo (`client.py`), entorno en paralelo (`vec_env.py`), Gymnasium (`gym_env.py`), muestreo de acciones válidas (`masks.py`), la red por ramas con cabezas autorregresivas (`model.py`), PPO (`ppo.py`) y el entrenamiento (`train.py`). Falta: entrenar en el PC y ajustar.

## Requisitos
- Python 3.11+ con `numpy`, `gymnasium`, `torch` y `pyyaml` (`pip install -e .[dev]`; en Windows sin GPU basta el torch de CPU).
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

## Entrenar
```
python -m openfront_rl.train --config configs/ppo_base.yaml
```
Crea `runs/<nombre>-<semilla>/` con `config.yaml`, `metrics.jsonl` (una línea por actualización: recompensa por componente, entropía, KL, acciones inválidas, fracción de `noop`, evaluación cada N) y `ckpt_*.pt` / `latest.pt` (modelo, optimizador, config, versión del protocolo). Se reanuda con `--resume runs/.../latest.pt`. En la nube (2 núcleos) una actualización de 256 muestras tarda ~10 s; en un i5-9400 con 5 partidas, 640 muestras por actualización.

## Panel de control
```
pip install -e ".[dev]"          # instala también psutil
python -m openfront_rl.dashboard # abre http://127.0.0.1:8765
```
El run se mide en **partidas** (`episodes` en la configuración, `--episodes N` o el campo "Partidas" del panel) y cada worker se puede pausar y reanudar por separado desde la pestaña Run en vivo. Desde el navegador se lanzan, pausan, reanudan, detienen y matan entrenamientos, y se ven las métricas en vivo, los workers, la arquitectura del agente y las metas (`configs/goals.yaml`). Pausar y Detener guardan un checkpoint al terminar la actualización en curso; los entrenamientos siguen corriendo aunque se cierre el panel. El formulario de lanzamiento muestra y permite cambiar el escenario (mapa, tribus, naciones y su nivel), y cada worker cuenta las construcciones que coloca el agente. Escucha solo en 127.0.0.1. Ver `docs/specs/dashboard.md`.
