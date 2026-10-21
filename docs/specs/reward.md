# Spec: recompensa (v1)

> Estado: **aprobada para implementar**. Se calcula en `bridge/src/reward.ts` (tiene acceso al motor) y viaja por el protocolo como **componentes separados** (`docs/specs/protocol.md` §4). La recompensa total es su suma.

## 1. Componentes
Los pesos son campos de `EnvConfig` (`rewardTerritoryWeight`, `rewardIncomeWeight`, `rewardSecurityWeight`; en el entrenamiento, bajo `env:` de `configs/*.yaml`). Con peso 0 el componente no se calcula.

| # | Componente | Fórmula | Peso por defecto en el puente | Peso en `ppo_base.yaml` |
|---|---|---|---|---|
| 0 | `territory` | `w · (frac_t − frac_{t−1})`, con `frac` = tiles propios / tiles de tierra | 1.0 | 1.0 |
| 1 | `income` | `w · (ln r̄_t − ln r̄_{t−1})`, con `r̄` = media móvil (0.8) del oro ganado por tick, mínimo 1 | 0 | 0.2 |
| 2 | `security` | `w · (Φ_t − Φ_{t−1})`, ver §2 | 0 | 0.1 |
| 3 | `terminal` | +1 al ganar, −1 al ser eliminado, 0 en empate o truncado | 1 | 1 |

`income` usa `Player.goldEarned()` (el mismo contador que la observación). Es **shaping basado en potencial** (`Φ = ln r̄`): premia que el ingreso *crezca* (ciudades, puertos, comercio), no tenerlo. Se descartó un flujo `tanh(oro/ref)`: sin ciudades el motor da un ingreso fijo de 100 oro por tick, así que saturaba y solo sumaba una constante por paso (medido). Duplicar el ingreso rinde `w · ln 2`.

Ningún componente se paga durante la fase de spawn ni después de que el agente fue eliminado.

## 2. Seguridad (no morir ni quedar en una posición complicada)
`security` es **shaping basado en potencial**: se premia la *mejora* de `Φ`, no estar seguro en sí. Así no cambia la política óptima (Ng et al., 1999) y no se puede cobrar infinitamente quedándose quieto.

`Φ ∈ [0, 1]` es el promedio de tres términos, calculados del estado público del motor:

| Término | Definición | Significa |
|---|---|---|
| `power` | `T / (T + Tmax_hostil)`; 1 si no hay vecinos hostiles | mis tropas frente al vecino hostil más fuerte |
| `pressure` | `1 − min(1, tropas_de_ataques_entrantes / T)` | cuánto me están atacando |
| `exposure` | `1 − min(1, n_vecinos_hostiles / 6)` | qué tan rodeado estoy (muchos frentes = posición complicada) |

`T` = tropas propias. Un vecino es hostil si no es aliado (`isFriendly`). Cuando el agente muere `Φ` no se evalúa (lo cubre el terminal −1).

Con pesos 0.1 y `Φ ∈ [0,1]`, el componente aporta como máximo ±0.1 por paso; el terminal sigue dominando.

## 3. Registro
Cada componente se registra por separado en las métricas del entrenamiento (suma por episodio, media por paso) para ver cuál guía el aprendizaje y detectar hacking (p. ej., territorio sube pero `security` cae).

## 4. Riesgos
- Si `territory` pesa demasiado el agente expande sin ganar. Por eso `terminal` (±1) debe seguir siendo el término mayor del retorno.
- `income` puede premiar construir ciudades sin defenderse. `security` lo compensa; si aun así ocurre, bajar `income` o sumarle `power` como condición.
- Pasividad: no se penaliza con un término aparte; `noop` prolongado ya pierde territorio relativo. Se revisa con la métrica "fracción de `noop`".

## 5. Victoria con varios jugadores y alianzas
`terminal = +1` solo si `game.getWinner() === me` (el motor decide el ganador al llegar al umbral de territorio). Un aliado ganador **no** cuenta. Pendiente medir con qué frecuencia el motor declara ganador en 6000 ticks.

## 6. Tests
- Con semilla fija las sumas de cada componente son reproducibles (mismo hash y mismas recompensas).
- Con pesos por defecto del puente, la recompensa total coincide con la de la v0 (`territory + terminal`).
- `security`: sin vecinos hostiles Φ = 1; con un ataque entrante grande, Φ baja.
- La suma de los componentes que llegan por el protocolo es igual a la recompensa total.
