# Spec: panel de control (ADR 0008)

## 1. Archivos por run (`runs/<run>/`)
| Archivo | Lo escribe | Contenido |
|---|---|---|
| `config.yaml`, `metrics.jsonl`, `*.pt` | `train.py` | ya existentes |
| `status.json` | `train.py`, tras cada actualización y mientras está en pausa | `{pid, state: "running"\|"paused"\|"stopping"\|"finished", update, total, steps, time, steps_per_s, updated_at, workers: [{i, tick, ep_len, events: {nombre: n}}]}` |
| `architecture.json` | `train.py`, al arrancar | `{params, total, modules: [{name, params, shape?}], branches: [...]}` |
| `control.json` | el panel | `{command: "run"\|"pause"\|"stop"}`; ausente = `run` |
| `train.log` | el panel (salida del proceso) | texto |

`updated_at` es epoch en segundos. Un run está **vivo** si su `pid` existe y su proceso es de Python; si el estado dice `running` pero el proceso no existe, se muestra `caído`.

## 2. API (JSON, solo `127.0.0.1`)
- `GET /api/runs` → lista de runs con estado, actualización, retorno medio reciente, CPU y memoria del árbol de procesos.
- `GET /api/runs/<run>/metrics?since=N` → filas de `metrics.jsonl` desde la actualización `N`.
- `GET /api/runs/<run>/status`, `/architecture`, `/goals`, `/log?tail=200`.
- `GET /api/configs` → archivos de `trainer/configs/*.yaml` (sin `goals.yaml`).
- `POST /api/runs` `{config, name?, seed?, workers?, updates?, resume?}` → lanza un entrenamiento. `config` debe ser un archivo de `configs/`; cualquier otro campo se rechaza (400).
- `POST /api/runs/<run>/control` `{command: "pause"\|"run"\|"stop"}`; `POST /api/runs/<run>/kill`.

## 3. Metas (`configs/goals.yaml`)
```yaml
goals:
  - {id: first_tribe_attack, title: "Primer ataque a una tribu", metric: ep_ev_attack_tribe_orders, op: ">", value: 0}
```
`metric` es una columna de `metrics.jsonl`; el avance es el máximo histórico del run (para `<`, el mínimo). Una meta está cumplida cuando `op`/`value` se cumplen con ese valor. Las metas son del agente, no del run: sirven para ver en qué etapa va.

## 4. Pantallas
1. **Procesos**: tabla de runs (estado, actualización/total, pasos/s, CPU, memoria) con botones Pausar, Reanudar, Detener, Matar; formulario para lanzar (config, nombre, semilla, workers, actualizaciones).
2. **Run**: gráficos en vivo (retorno por componente, eventos por episodio, entropía, KL, velocidad), tarjetas de workers (tick, longitud del episodio, eventos), registro de salida.
3. **Arquitectura**: ramas → núcleo → cabezas con parámetros reales (`architecture.json`).
4. **Metas**: lista de hitos con barra de avance y estado.

## 5. Seguridad
Escucha solo en `127.0.0.1`; no hay autenticación. Los nombres de run se validan (`[A-Za-z0-9_.-]+`) para no salir de `runs/`. Solo se lanza `python -m openfront_rl.train` con argumentos de la lista cerrada.
