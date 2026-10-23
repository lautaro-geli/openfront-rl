# Spec: recompensa (v2.2)

> Estado: **aprobada para implementar**. Sustituye a la v1. Se calcula en `bridge/src/reward.ts` (tiene acceso al motor) y viaja por el protocolo como **componentes separados** (`docs/specs/protocol.md` §4). La recompensa total es su suma.

## 1. Idea: un presupuesto máximo por categoría
Cada categoría paga por **mejorar** un potencial `Φ_k ∈ [0, 1]` y tiene un **presupuesto** `B_k`: lo máximo que puede dar en toda la partida.

```
r_k = B_k · (Φ_k(s_t) − Φ_k(s_{t−1}))
```

Es shaping basado en potencial (Ng et al., 1999): se cobra al mejorar, se devuelve al empeorar, y no se puede cobrar quedándose quieto ni repitiendo. Como cada categoría está acotada por `B_k`, **el orden de importancia queda garantizado** por los presupuestos, no por la frecuencia de los eventos.

| Orden | Categoría | Presupuesto `B_k` (configuración de entrenamiento) | Potencial `Φ_k` |
|---|---|---|---|
| 1 | `terminal` | +1 al ganar, −1 al ser eliminado, 0 si no | — |
| 2 | `territory` | 0.40 | `min(frac / 0.8, 1)` |
| 3 | `attack` | 0.08 | 0.8 · `Φ_tropas` + 0.2 · `Φ_silos` |
| 4 | `economy` | 0.04 | `Φ_ingreso` |
| 5 | `defense` | 0.05 | 0.5 · `Φ_seguridad` + 0.5 · `Φ_puestos` |
| 6 | `milestones` | 0.06 | `(ciudad + puerto + fábrica) / 3` (§2b) |
| 7 | `troops` | 0.05 | por paso, no potencial: tropas en la zona justa (§2c) |
| (trainer) | `conquest` | 0.20, decae a 0 | bonus por tribu conquistada (§9) |

Con los valores de entrenamiento (`configs/v1_ppo_base.yaml`) la suma de los presupuestos es 0.88 < 1: **ganar siempre pesa más** que todo lo que se puede ganar por el camino. Los pesos son campos de `EnvConfig` (`rewardTerritoryWeight`, `rewardAttackWeight`, `rewardEconomyWeight`, `rewardDefenseWeight`, `rewardMilestonesWeight`, `rewardTroopsWeight`). Con peso 0 el componente no se calcula. Valores por defecto del puente: territorio 0.8 (equivale a `1.0 · Δ%` de la v0), el resto 0. Ningún componente se paga durante la fase de spawn ni después de que el agente fue eliminado.

## 2. Potenciales
Todos salen del estado del motor, tal como lo ve cualquier jugador.

**`territory`** — `frac` = casillas propias / casillas de tierra. Llegar al 80 % (`PERCENT_TILES_OWNED_TO_WIN`, `packages/engine-lib/src/configuration/Config.ts:74`) rinde `B_T` y, además, el +1 de ganar. La curva es **por tramos** (campos `territoryKnee` = `k` y `territoryKneeShare` = `s`):

```
Φ_territorio(frac) = s · min(frac / k, 1) + (1 − s) · clip((frac − k) / (0.8 − k), 0, 1)
```

El primer tramo (tierra libre, hasta `k`) paga la fracción `s` del presupuesto y el resto va lineal hasta el 80 %. Configuración de entrenamiento: `k = 0.15`, `s = 0.35` (con `B_T = 0.40`: el primer 15 % del mapa paga 0.14, es decir 0.0093 por cada 1 %, y del 15 % al 80 % paga 0.26, 0.0040 por cada 1 %). `k = 0` (valor por defecto del puente) deja la curva lineal de antes: cada 1 % vale `B_T / 80`.

**`attack`** (poder de ataque) — mejora de la capacidad, no de las tropas actuales: estas bajan al atacar, así que premiarlas castigaría atacar.
- `Φ_tropas = clamp((maxTroops − maxTroopsWithoutCities) / 1 000 000, 0, 1)`: solo cuenta lo que aportan las **ciudades** terminadas (250 000 por nivel, `Config.ts:162`, `Config.ts:330-346`); 4 niveles de ciudad → 1. **No** usa el tope total de tropas: ese crece con `casillas^0.6`, así que premiaría el territorio dos veces (en la v2 inicial medimos que `attack` pagaba 5–10 veces más que `territory` por el mismo avance; ver §6).
- `Φ_silos = min(1, silos terminados / 2)`.

**`economy`** — `r̄` = media móvil (factor 0.95 por paso) del oro ganado por tick, que incluye comercio, trenes y conquista (`Player.goldEarned()`). El oro base es 100 por tick fijo (`EngineConfig.ts:668-677`) y no cuenta como mejora. `Φ_ingreso = clamp(ln(r̄ / 100) / ln 10, 0, 1)`: 100 → 0; 1000 por tick → 1.

**`defense`** —
- `Φ_seguridad` = la de la v1 (promedio de `power`, `pressure`, `exposure`; ver §3).
- `Φ_puestos = min(1, puestos de defensa terminados / 4)`. Cada puesto multiplica ×5 las pérdidas del atacante y ×3 su lentitud dentro de 30 casillas (`EngineConfig.ts:141-147`, `Config.ts:175`).

### 2b. Hitos de construcción (`milestones`)
`Φ_hitos = (hay ciudad + hay puerto + hay fábrica) / 3`, donde "hay" = al menos una unidad **terminada** (no en construcción) y activa del agente. Es un potencial: paga una vez al aparecer cada estructura (`B_H / 3` por hito; construir más no suma) y devuelve lo cobrado si se pierde. Enseña la secuencia economía básica; el puerto solo rinde comercio con naciones (§4), pero sirve como hito.

### 2c. Tropas en la zona justa (`troops`)
Las tropas crecen más rápido cuando están en torno al 42 % del máximo: `(10 + T^0.73 / 4) · (1 − T / máx)` (`packages/engine-lib/src/configuration/Config.ts:370-376`; el máximo de esa expresión está en ≈ 0.42 para cualquier tope). Por encima del ~80 % casi no regeneran y por debajo del ~20 % no hay con qué atacar. Con `ρ = tropas / tropas máximas`:

```
banda(ρ) = 1 si 0.30 ≤ ρ ≤ 0.60; cae linealmente a 0 en ρ = 0.10 y ρ = 0.90
r_troops = B_R · banda(ρ) / N_pasos       (N_pasos = maxTicks / decisionInterval)
```

No es un potencial (atacar baja las tropas y el efecto se anularía): es una tasa por paso, acotada a `B_R` en una partida de duración máxima. Solo cobra tras el spawn y mientras el agente vive.

## 3. Seguridad (no morir ni quedar en una posición complicada)
`Φ_seguridad ∈ [0, 1]` es el promedio de tres términos:

| Término | Definición | Significa |
|---|---|---|
| `power` | `T / (T + Tmax_hostil)`; 1 si no hay vecinos hostiles | mis tropas frente al vecino hostil más fuerte |
| `pressure` | `1 − min(1, tropas_de_ataques_entrantes / T)` | cuánto me están atacando |
| `exposure` | `1 − min(1, n_vecinos_hostiles / 6)` | qué tan rodeado estoy (muchos frentes = posición complicada) |

`T` = tropas propias. Un vecino es hostil si no es aliado (`isFriendly`).

## 4. Qué puede hacer crecer cada categoría (verificado en el motor)
- **Tropas máximas:** ciudades (125 k × 2ⁿ, máx. 1 M; `EngineConfig.ts:378-382`) y territorio. Una ciudad triplica el tope de un jugador recién aparecido.
- **Ingreso:** barcos mercantes de puertos (pagan a los dos dueños, `TradeShipExecution.ts:183-215`) y trenes de fábricas (`EngineConfig.ts:236-259`). Los barcos **solo salen hacia puertos de otros jugadores** (`PortExecution.ts`, `tradingPorts`) y las tribus no construyen nada (`TribeExecution.ts`). Por eso el comercio necesita **naciones**: sin ellas, un puerto no rinde.
- **Medición con naciones (Britannia, 20 tribus Easy, política aleatoria):** con `nations: 1` la nación llega a ~90 000 casillas y termina la partida (≥80 %) en ~3200 ticks sin haber construido ningún puerto; con `nations: 2`, ~194 000 casillas en 2700 ticks. Una nación acaba la partida antes de que el comercio exista, así que **la configuración de entrenamiento sigue en `nations: disabled`** y `economy` solo mide oro de conquista hasta resolver esto (pendiente: ADR de currículo de naciones, p. ej. nación con tamaño limitado o más ticks).
- **Presupuesto de oro por episodio:** el oro base da ~600 000 por cada 6000 ticks (~1,8 M en los 18 000 del entrenamiento), es decir 3–4 mejoras de ciudad como mucho. Las mejoras son pocas y discretas.

## 5. Registro
Cada componente viaja por separado y se registra por episodio (suma) y por paso (media), para detectar si alguno se explota (p. ej., el territorio no sube pero `defense` sí).

## 6. Riesgos
- Solapamiento `attack`/`territory`: resuelto en la v2.1 (`Φ_tropas` mide solo ciudades). Medido en el primer entrenamiento (50 actualizaciones): con el tope total de tropas en escala log, pasar de ~0 a ~1 % del mapa subía `Φ_tropas` ≈ 0.4 y pagaba ≈ 0.05 frente a 0.005 de `territory`.
- `economy` es lenta de mover y depende de las naciones; si en la práctica no sube, el componente no aporta señal y conviene revisar el número de naciones.
- `Φ_seguridad` puede bajar por acciones de otros jugadores, no del agente; con `B_D = 0.05 · 0.5` el efecto es pequeño.
- `terminal` sigue siendo el término mayor del retorno; no subir los demás presupuestos sin comprobarlo.

## 7. Victoria con varios jugadores y alianzas
`terminal = +1` solo si `game.getWinner() === me`. Un aliado ganador no cuenta.

## 8. Tests
- Con los pesos por defecto del puente la recompensa total coincide con la de la v0 (`territory + terminal`).
- Con semilla fija, las sumas de cada componente son reproducibles; activar el shaping no cambia el hash del motor.
- Cada potencial está en [0, 1]; `Φ_seguridad = 1` sin vecinos hostiles.
- El valor acumulado de cada categoría nunca supera su presupuesto, y la suma de los presupuestos activos es menor que 1.
- La suma de los componentes que llegan por el protocolo es igual a la recompensa total.

## 9. Bonus por tribus conquistadas (currículo, lo aplica el trainer)
Al empezar, la mayor ganancia está en conquistar tribus (no tienen economía ni defienden). Para empujar al agente a atacarlas hay un bonus que **decae con el progreso del entrenamiento** hasta desaparecer:

```
bonus por tribu = B_c · max(0, 1 − p / p_fin) / N_tribus       (p = progreso del run en [0, 1])
```

- `B_c` = `conquest.budget` (0.20 por defecto): lo máximo que paga el bonus en una partida si se conquistan todas las tribus. `p_fin` = `conquest.until` (0.5): a partir de ese progreso el bonus es 0. `N_tribus` = `env.bots`.
- Se paga por cada tribu eliminada porque el agente tomó su último territorio (evento `tribes_conquered`, `docs/specs/protocol.md` §4). No se paga por ataques ni por tribus que mató otro jugador.
- Lo calcula `trainer/openfront_rl/train.py` (el puente no conoce el progreso del entrenamiento), se suma a la recompensa del paso y se registra como componente aparte `conquest` (`ep_part_conquest`). La evaluación no lo incluye.
- Presupuestos: 0.88 < 1, así que ganar sigue pesando más que todo lo que se cobra por el camino. Con `conquest.budget: 0` está apagado.
