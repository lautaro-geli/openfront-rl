# Spec: panel de control (ADR 0008)

## 1. Archivos por run (`runs/<run>/`)
| Archivo | Lo escribe | Contenido |
|---|---|---|
| `config.yaml`, `metrics.jsonl`, `*.pt` | `train.py` | ya existentes |
| `status.json` | `train.py`, tras cada actualización y mientras está en pausa | `{pid, state: "running"\|"paused"\|"stopping"\|"finished", update, total, episodes_done, episodes_total, steps, time, steps_per_s, updated_at, workers: [{i, paused, tick, ep_len, events: {nombre: n}, stats}]}`; `stats` ver §5b |
| `architecture.json` | `train.py`, al arrancar | `{params, total, modules: [{name, params, shape?}], branches: [...]}` |
| `control.json` | el panel | `{command: "run"\|"pause"\|"stop", paused_workers: [int]}`; ausente = `run` y ningún worker pausado |
| `train.log` | el panel (salida del proceso) | texto |
| `workers/worker-<i>.log` | `train.py` | registro de ese worker: una línea por partida terminada, pausas/reanudaciones y la salida de error de su proceso Node |

`updated_at` es epoch en segundos. Un run está **vivo** si su `pid` existe y su proceso es de Python; si el estado dice `running` pero el proceso no existe, se muestra `caído`.

## 2. API (JSON, solo `127.0.0.1`)
- `GET /api/runs` → lista de runs con estado, actualización, retorno medio reciente, CPU y memoria del árbol de procesos.
- `GET /api/runs/<run>/metrics?since=N` → filas de `metrics.jsonl` desde la actualización `N`.
- `GET /api/runs/<run>/status`, `/architecture`, `/goals`, `/log?tail=200`, `/workers/<i>/log?tail=200`.
- `GET /api/configs` → archivos de `trainer/configs/*.yaml` (sin `goals.yaml`).
- `POST /api/runs` `{config, name?, seed?, workers?, episodes?, updates?, resume?, env?: {map?, bots?, nations?, difficulty?}}` → lanza un entrenamiento. `config` debe ser un archivo de `configs/`; cualquier otro campo se rechaza (400).
- `POST /api/runs/<run>/workers` `{worker: i, paused: bool}` → pausa o reanuda un worker.
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

## 4b. Escenario del entrenamiento
El formulario de lanzamiento muestra, para la configuración elegida, el escenario que va a jugar el agente y deja cambiarlo: **mapa**, **tribus** (`bots`, número), **naciones** (`nations`: `disabled`, un número o `default`) y **nivel** (`difficulty`: Easy, Medium, Hard, Impossible). La API acepta esos cuatro campos en `env` con valores validados y los pasa a `train.py` como `--env clave=valor`; el valor final queda guardado en el `config.yaml` del run. El nivel (`difficulty`) lo leen las **naciones** (ritmo de ataque, alianzas, donaciones: `NationExecution.ts:92-103`, `NationAllianceBehavior.ts:101,185`) y las relaciones tras un ataque (`AttackExecution.ts:199-222`); `TribeExecution` no lo usa, así que sin naciones el nivel casi no influye.

## 4b'. Recompensas editables
El formulario de lanzamiento muestra los presupuestos de la configuración elegida y deja cambiarlos: `territory`, `attack`, `economy`, `defense` (`env.reward*Weight`) y el bonus por tribus (`conquest.budget`, `conquest.until`; `docs/specs/reward.md` §1 y §9). La API acepta `rewards: {territory, attack, economy, defense, milestones, troops, conquest_budget, conquest_until, territory_knee, territory_knee_share}`; cada peso debe estar en [0, 1], `conquest_until` en (0, 1], y `territory_knee` en [0, 0.8) y `territory_knee_share` en [0, 1] (curva por tramos, reward.md §2); **la suma de los siete presupuestos debe ser < 1** (la recompensa terminal, ±1, tiene que seguir pesando más que todo lo que se cobra por el camino). Se pasan a `train.py` como `--env rewardTerritoryWeight=…` y `--conquest-budget` / `--conquest-until`; el valor final queda en el `config.yaml` del run.

## 4c. Run nuevo en una carpeta con datos
Si se lanza sin `resume` y la carpeta del run ya tiene `metrics.jsonl`, `train.py` mueve los archivos anteriores (métricas, estadísticas, checkpoints, registros) a `anterior-<fecha>/` dentro del mismo run, para no mezclar dos entrenamientos en un solo `metrics.jsonl`.

## 5. Duración del run y workers pausados
**Duración.** El run se mide en **partidas** (episodios terminados por los workers de entrenamiento; las de evaluación no cuentan): `episodes` en la configuración, `--episodes N` en la línea de comandos o el campo "Partidas" del panel. El run termina al cierre de la primera actualización en la que `episodes_done >= episodes`, así que puede pasarse de la cifra hasta `workers − 1` partidas. Si `episodes` está definido, `updates` se ignora; si no, `updates` es el límite como antes. `episodes_done` se guarda en el checkpoint, así que al reanudar con un `episodes` mayor el run sigue. El avance del panel es `episodes_done / episodes`. Los ritmos de `lr` y de entropía bajan con ese mismo avance (si no hay `episodes`, con `update / updates`).

**Pausar un worker.** `paused_workers` indica los workers que no juegan. Se atiende al comienzo de cada actualización: el worker no recibe pasos (su partida queda congelada, su proceso Node en espera y sin gastar CPU) y sus muestras de ese tramo se **marcan como no válidas** (`valid[t, i] = false` en el buffer): no entran en las ventajas normalizadas, ni en los minibatches, ni en la escala de recompensa. Al reanudarlo, su partida sigue donde estaba. Si todos los workers están pausados, el run queda en `paused`.

## 5b. Estadísticas por worker
Cada worker acumula, desde que empezó el run (se guardan en `workers.json` y se recuperan al reanudar), sobre las **partidas terminadas** (una partida termina por victoria, eliminación o `maxTicks`; no es un paso):
`stats = {games, wins, eliminated, avg_max_territory, avg_final_territory, avg_length, conquered_by_tribe, conquered_by_nation, tribes_conquered, nations_conquered, built: {city, port, factory, post, sam, silo, warship}, nukes_launched, recent: [...]}`.
- `avg_max_territory`: promedio, entre partidas, del mayor `land_share` alcanzado durante la partida; `avg_final_territory`: promedio del `land_share` al terminar. Ambos en fracción del mapa (0–1).
- `conquered_by_*`: veces que el agente fue eliminado porque una tribu o una nación tomó su último territorio (la eliminación por otras causas cuenta en `eliminated` pero no aquí). `*_conquered`: veces que el agente eliminó a una tribu o una nación.
- `built`: total de estructuras colocadas por el agente en las partidas terminadas del worker (por tipo); `result` es `victoria`, `eliminado`, `tiempo` (se acabaron los ticks) o `ganó otro` (otro jugador llegó al 80 % antes, p. ej. una nación); `recent`: las últimas 10 partidas `{n, built: {...}, nukes_launched, result: "victoria"|"eliminado"|"tiempo", max_territory, final_territory, ticks, tribes_conquered, nations_conquered, conquered_by: "tribu"|"nación"|null}`.
En la pantalla Run, cada worker muestra el resumen y al hacer clic se expande con el detalle, las últimas partidas y su propio registro.

## 6. CPU y prioridad
`cpu` es el uso del árbol de procesos del run **como porcentaje de toda la PC** (0–100, igual que el Administrador de tareas): la suma de `cpu_percent` de cada proceso dividida por el número de CPU lógicas. `train.py` acepta dos ajustes de configuración: `torch_threads` (hilos de PyTorch; por defecto, los núcleos físicos menos 2, mínimo 1, para dejar dos núcleos libres a la PC) y `low_priority` (por defecto `true`: el entrenamiento y sus procesos Node corren con prioridad por debajo de lo normal, para que la PC siga usable).

## 7. Seguridad
Escucha solo en `127.0.0.1`; no hay autenticación. Los nombres de run se validan (`[A-Za-z0-9_.-]+`) para no salir de `runs/`. Solo se lanza `python -m openfront_rl.train` con argumentos de la lista cerrada.
