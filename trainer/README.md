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
python -m openfront_rl.train --config configs/v1_ppo_base.yaml
```
Crea `runs/<nombre>-<semilla>/` con `config.yaml`, `metrics.jsonl` (una línea por actualización: recompensa por componente, entropía, KL, acciones inválidas, fracción de `noop`, evaluación cada N) y `ckpt_*.pt` / `latest.pt` (modelo, optimizador, config, versión del protocolo). Se reanuda con `--resume runs/.../latest.pt`. En la nube (2 núcleos) una actualización de 256 muestras tarda ~10 s; en un i5-9400 con 5 partidas, 640 muestras por actualización.

## Panel de control
```
pip install -e ".[dev]"          # instala también psutil
python -m openfront_rl.dashboard # abre http://127.0.0.1:8765
```
El run se mide en **partidas** (`episodes` en la configuración, `--episodes N` o el campo "Partidas" del panel) y cada worker se puede pausar y reanudar por separado desde la pestaña Run en vivo. Desde el navegador se lanzan, pausan, reanudan, detienen y matan entrenamientos, y se ven las métricas en vivo, los workers, la arquitectura del agente y las metas (`configs/goals.yaml`). Pausar y Detener guardan un checkpoint al terminar la actualización en curso; los entrenamientos siguen corriendo aunque se cierre el panel. El formulario de lanzamiento muestra y permite cambiar el escenario (mapa, tribus, naciones y su nivel), y cada worker cuenta las construcciones que coloca el agente. Escucha solo en 127.0.0.1. Ver `docs/specs/dashboard.md`.

## Dispositivo (CPU / GPU)
`device` en la config: `auto` (por defecto) usa la GPU CUDA si PyTorch la ve y, si no, la CPU; `cpu` o `cuda` fuerzan uno (`cuda` sin GPU falla con un mensaje claro). La primera línea del log dice `dispositivo=...`. Los checkpoints se guardan siempre en CPU y se pueden reanudar en cualquier dispositivo. Comprobar CUDA: `python -c "import torch; print(torch.cuda.is_available())"`; si da `False` hay que instalar PyTorch con CUDA (https://pytorch.org/get-started/locally/). En el i5-9400 la actualización PPO en CPU era el 56 % del tiempo (7,5 s de 13,3 s por actualización).

## Rollout sincrónico o asíncrono
`rollout_mode: sync` (por defecto) avanza los workers al mismo paso; `async` deja que cada uno avance a su ritmo y reúne `rollout_steps × workers` muestras en total, con un tope por worker de `rollout_cap × rollout_steps` (ADR 0010, experimental). `metrics.jsonl` y el log traen `t_net`, `t_env` y `worker_util` (el log: `roll=5.9s (red 0.8 motor 5.0 util 62%)`). `ppo.lr_floor` (0,1 por defecto) fija el mínimo del lr al decaer con el progreso.

## Versiones de las configs
Los archivos de `configs/` llevan prefijo de versión (`v1_ppo_base`, `v2_ppo_naciones`, `v3_ppo_naciones_estable`, `v4_ppo_critico`, `v5_ppo_prueba`, `v6`…`v8`, `v9_ppo_largo_rapido`, `v10_ppo_critico_separado`, `v11_ppo_lote_grande`, `v12_ppo_kl_estricto`, `v13_ppo_conquista`, `v14_ppo_critico_lr`, `v15_ppo_desde_day5`); la siguiente prueba usa `v16_…`. El campo `name:` dentro del YAML no lleva el prefijo (da el nombre del run por defecto y la semilla de las partidas, así que cambiarlo cambiaría las partidas de un run reproducido).

## Evaluación sin sesgo
`evaluate` (train.py) reparte las partidas pedidas en cuotas fijas por worker y cada worker se detiene al cumplirla. Antes se contaban las primeras N partidas en terminar, que son las más cortas (eliminaciones y victorias rápidas), y las evaluaciones de runs anteriores (Day1–Day7) tienen ese sesgo: sobrestiman las eliminaciones. No son comparables con las nuevas.


## Crítico separado
`model: {separate_critic: true}` da al crítico su propio codificador (`enc_v`), de modo que el gradiente del valor no toca la política. Cuesta una segunda pasada del codificador por paso (más tiempo de red). Al reanudar desde un checkpoint con codificador compartido, `enc_v` arranca copiando `enc` y el optimizador se reinicia. `ppo.gae_lambda` también es configurable (v10 usa 0,97).

## Diagnóstico: señal del gradiente (`gradnoise`)
`python -m openfront_rl.gradnoise --ckpt runs/Day8/latest.pt --batches 40 --workers 5` juega 40 lotes con la política congelada (como el entrenamiento: `rollout_steps` × workers muestras, GAE y ventajas normalizadas por lote) y calcula el gradiente de la pérdida de política de cada lote. Imprime:
- `coseno_medio`: coseno medio entre los gradientes de lotes distintos (≈ 0: cada lote apunta a otro lado, es ruido).
- `b_noise_muestras` / `b_noise_partidas`: escala de ruido del gradiente B_simple = tr(Σ)/|G|² (McCandlish et al. 2018, arXiv 1812.06162). Es el lote a partir del cual agrandarlo deja de ayudar. Si es ≫ 640 muestras, cada actualización es casi ruido y hay que agrandar el lote (hipótesis #1 de `claude/03-hipotesis-de-fallo.md`).
- `b_noise_mitad1/2`: la misma estimación con cada mitad de los lotes (estabilidad).
Los lotes consecutivos salen de las mismas partidas en curso, así que es una estimación aproximada. No modifica el checkpoint.

## Opciones nuevas (v11–v14)
- `ppo.critic_lr_mult` (por defecto 1,0): el crítico (`value.*` y, con `separate_critic`, `enc_v.*`) usa `lr × critic_lr_mult`. Al reanudar desde un checkpoint con otros grupos de parámetros, el optimizador se reinicia (se avisa por consola).
- `ppo.target_kl` (por defecto 0,05): un minibatch corta las épocas si su KL supera `1,5 × target_kl`. Con 0,05 nunca actuó (KL real ≤ 0,007 en Day5–Day10); v12 usa 0,003.
- `conquest.relative: true`: el bonus por tribu conquistada decae con el progreso de **este** run y no con el total absoluto de `episodes` (`docs/specs/reward.md` §9). Sin esto, al reanudar en 340/490 el bonus ya vale 0.
- `stop: {clipfrac, window, min_updates, win_below, win_window, eval_below, eval_count}`: parada automática por alarma de colapso. Para si la media de `clipfrac` de las últimas `window` actualizaciones supera `clipfrac`, o si las victorias de las últimas `win_window` partidas de entrenamiento caen bajo `win_below` (con al menos `min_updates` actualizaciones del run). Guarda un checkpoint, escribe `alarma.txt` en el run y termina con estado `finished`. Con `eval_below` > 0 (y `eval.every` > 0) también para si las últimas `eval_count` evaluaciones sin sesgo dentro del run (`eval_won`) quedan todas bajo `eval_below`: es la alarma que sí ve la degradación real, porque las victorias de entrenamiento están infladas (v15: 0,45 con 2 seguidas; con 40 partidas y p real 0,55 un falso positivo seguido es ~0,6 %, con p real 0,40 se detecta en ~47 %). Valores de v11–v14: 0,03 / 20 / 30 / 0,3 / 30; calibrados con Day5–Day10 (en esos datos no habría parado Day5 ni Day8, y sí Day7, Day9 y Day10). Son 5 runs: es una calibración dentro de la muestra.
- Medir la política heurística en el escenario de entrenamiento: `cd bridge && npm run random -- --policy heuristic --bots 20 --nations 1 --difficulty Easy --max-ticks 18000 --seed heur40 --episodes 40` (imprime un RESUMEN al final).
