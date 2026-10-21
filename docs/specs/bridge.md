# Spec: bridge v0 (Fase 1)

> Estado: **borrador implementable**. Alcance de la Fase 1: el lado TypeScript. La conexión con Python (ADR 0002) es de la Fase 2.
> Basado en `docs/research/engine-findings.md` y ADR 0003 (usar `GameRunner` como librería).

## 1. Objetivo
Una clase `OpenFrontEnv` que arma una partida del motor sin navegador, controla **un jugador** (el agente) y devuelve observaciones reducidas, recompensa y fin de episodio. Una política aleatoria debe poder jugar partidas completas, y la misma semilla con las mismas acciones debe dar el mismo resultado.

## 2. API

```ts
interface EnvConfig {
  map: GameMapType;          // por defecto Britannia (rápido); World disponible
  bots: number;              // tribus, 0..400
  nations: number | "default" | "disabled";
  difficulty: Difficulty;
  decisionInterval: number;  // ticks del motor por step(), por defecto 10
  maxTicks: number;          // truncado: ticks tras el fin del spawn, por defecto 6000
  grid: { h: number; w: number }; // por defecto 32 x 64; gridForMap() la calcula según la forma del mapa
  samples: number;           // muestras por eje y celda, por defecto 4 (ADR 0005)
  agents: number;            // jugadores controlados; 1 en OpenFrontEnv
  maxNeighbors: number;      // slots de ataque a jugadores, por defecto 8
  rewardTerritoryWeight: number;  // por defecto 1.0
  rewardIncomeWeight: number;     // por defecto 0 (apagado); ver reward.md
  rewardSecurityWeight: number;   // por defecto 0 (apagado)
}

class OpenFrontEnv {
  static create(cfg: Partial<EnvConfig>): Promise<OpenFrontEnv>;
  reset(seed: string): Promise<Observation>;
  step(action: Action): StepResult;
  stateHash(): string;       // hash del estado del motor (último HashUpdate) + tick
  close(): void;
}
```

`StepResult = { obs, reward, terminated, truncated, info }`.

### Varios agentes (self-play)
`MultiAgentEnv` es el núcleo; `OpenFrontEnv` es un envoltorio con `agents = 1`.

```ts
reset(seed): Promise<Observation[]>            // una observación por agente, cada una relativa a su agente
step(actions: Action[]): MultiStepResult       // { obs[], reward[], terminated[], truncated, done, info[] }
```
- Todas las acciones se calculan sobre el mismo estado y viajan en el primer tick del paso.
- La fase de spawn se termina sola cuando todos los agentes han aparecido.
- `terminated[i]` es verdadero si el agente `i` fue eliminado o la partida acabó con victoria; `done` es verdadero cuando acaba la partida (victoria, todos eliminados o truncado). Un agente eliminado sigue recibiendo observaciones vacías y su acción se ignora.
- Un agente eliminado cobra −1 una sola vez; si la partida acaba con la victoria de otro, el resto cobra 0.

## 3. Ciclo de un episodio
1. `reset(seed)` crea una partida nueva (el `gameID` del motor es la semilla). Devuelve la observación en el tick 0, en **fase de spawn**.
2. En fase de spawn solo es válida la acción `spawn`. Tras ella, el entorno avanza solo hasta que acaba el spawn y devuelve la primera observación de juego. Cualquier otra acción en spawn se trata como `noop` y no avanza el juego más de `decisionInterval` ticks.
3. Después, cada `step(action)` envía el intent (si lo hay) y avanza `decisionInterval` ticks.
4. El episodio acaba (`terminated`) si el agente es eliminado o llega una actualización de victoria. Se trunca (`truncated`) al llegar a `maxTicks` ticks de juego tras el spawn.

## 4. Observación (v2)

La definición completa (canales, tablas, escalares, normalización, fuentes en el motor) está en `docs/specs/observations.md`; los nombres y tamaños están en `bridge/src/features.ts`. Resumen:

- `grid` `[20, h, w]`: los 3 canales de la v1 (0 tierra, 1 propio, 2 otro; índices estables) más relieve, costa, aliados, vecinos, top 3, fallout, estructuras, cobertura SAM y de defensas, objetivos de bombas, barcos y ataques.
- `economy`: tabla `[48, 9]` (ciudades, puertos, fábricas) y 38 escalares (tropas máximas, oro, ingresos, costes y capacidad de pago).
- `military`: tabla de unidades `[96, 30]`, tabla de ataques `[16, 10]` y 23 escalares.
- `comparison`: tabla de jugadores `[16, 46]` (fila 0 = el agente, filas 1-8 = ranuras de ataque, 3 del top, 4 aliados) y 14 escalares, en [-1, 1].
- `globals` 14 valores; los 2 primeros siguen siendo los de la v1.
- `mask`, `tick`, `inSpawnPhase` sin cambios.

Información pública: el motor entrega oro, tropas, estructuras, alianzas y ataques de todos los jugadores a cada cliente (`PlayerUpdate`, `GameUpdates.ts:237-282`), así que verlos no es trampa.

## 5. Acciones (v1)

El catálogo completo, con los 24 tipos, su vector `[kind, row, cell, fraction, type, aux]`, la máscara y los intents del motor, está en [`docs/specs/actions.md`](actions.md). Código: `bridge/src/actions.ts` (traducción) y `bridge/src/mask.ts` (máscara).

Se conservan las acciones de la v0 con su forma original (`noop`, `spawn {cx,cy}`, `expand {fraction}`, `attack {slot,fraction}`) y la máscara antigua (`spawn`, `expand`, `attackSlots`).

Una acción inválida o enmascarada se ejecuta como `noop` y se cuenta en `info.invalidAction = true`. Un tipo de acción desconocido (el protocolo no puede decodificarlo) también.

## 5b. Conexión con Python
Ver [`docs/specs/protocol.md`](protocol.md). Servidor: `bridge/src/server.ts`; cliente: `trainer/openfront_rl/client.py`.

## 6. Recompensa
Según `docs/specs/reward.md`, con estos valores concretos:

```
r = territory + income + security + terminal      (docs/specs/reward.md)
```
`info.rewardParts = { territory, income, security, terminal }` para registrar cada componente; por el protocolo v2 llegan separados.

## 7. Requisitos de aceptación (tests)
1. **Determinismo**: dos entornos con misma config, misma semilla y misma secuencia de acciones dan el mismo `stateHash()` final y la misma suma de recompensas.
2. **Contrato**: `obs.grid.length === 3·h·w`, `obs.globals.length === 6`, todos los valores finitos y en rango.
3. **Máscara**: con la política que solo elige acciones enmascaradas como válidas, `info.invalidAction` nunca es verdadero.
4. **Episodio completo**: la política aleatoria acaba un episodio (`terminated` o `truncated`) sin lanzar errores.
5. **Reset**: `reset()` repetido con la misma semilla devuelve la misma observación inicial.
6. **Velocidad (informativa, no un criterio de paso)**: se imprimen ticks/s y pasos/s.
7. **Multiagente**: con 2 agentes, uno que no actúa conserva su territorio y otro que se expande crece; las observaciones son relativas a cada agente; misma semilla y acciones dan el mismo resultado.
8. **Jugabilidad**: una política heurística (`HeuristicPolicy`) debe crecer mucho más que la aleatoria y no hacer acciones inválidas.

## 8. Fuera de alcance
Comunicación con Python, varias partidas en paralelo, construcción de edificios, diplomacia, ataques anfibios, y `miniGameMap`. Cada uno se añade con su propia spec.

## 9. Preguntas abiertas
- Valores correctos de los pesos de la recompensa (ver `reward.md`; `ppo_base.yaml` trae un punto de partida).
- ~~Si la malla de muestreo 4 × 4 pierde territorios pequeños en World~~ Medido en ADR 0005: no ve territorios de 1 tile; con 32 × 64 ve el 67–84 % de las celdas ocupadas.
- Si `maxTicks = 6000` (10 min de juego) alcanza para que la política aleatoria llegue a ver victorias o eliminaciones.
