# Fase 0: hallazgos sobre el motor de OpenFront

- **Commit del motor analizado**: `42636d8a9f8001f999cb0ce1eabed7359b5e9ddb` (`v0.3.6-4876`)
- **Fecha**: 2026-10-18
- **Rutas**: relativas a `vendor/openfront/`. Formato `archivo:línea`.

Cada afirmación está marcada así:
- **[leído]**: visto en el código (se cita archivo y línea).
- **[medido]**: observado ejecutando el motor en esta sesión.
- **[no verificado]**: inferido o pendiente de comprobar.

## Resumen

| Pregunta | Respuesta corta |
|---|---|
| ¿Separación motor/cliente? | Sí, muy limpia. El motor es una librería TS pura en `packages/engine*`. |
| ¿Headless? | Sí. El repo ya trae un script que corre partidas completas sin navegador. |
| ¿Velocidad? | 190–490 ticks/s por proceso (el juego real va a 10 ticks/s). |
| ¿Estado y acciones? | Estado: objetos `Game`/`Player` + actualizaciones en buffers. Acciones: 25 tipos de intent validados con Zod. |
| ¿Bots? | Tribus (bots simples) y Naciones (IA más compleja), con 4 dificultades. |
| ¿Determinismo? | Sí. Dos partidas con la misma semilla dieron el mismo hash final. |

**Conclusión: el proyecto es viable y el motor es un buen candidato para RL.**

## 1. Separación motor/cliente

- **[leído]** El motor es "simulación determinista, TypeScript puro, sin dependencias salvo zod" y está dividido en paquetes npm: `engine-api`, `engine-lib`, `engine` (`CLAUDE.md` del motor, sección Architecture).
- **[leído]** Un test (`tests/LayerBoundaries.test.ts`) prohíbe que esos paquetes importen cliente, servidor, `Math.random`, `Date.now` o fetch. El mapa se le pasa ya cargado.
- **[leído]** La simulación corre en cada cliente, no en el servidor. El servidor solo agrupa intents por turno y los reenvía (`CLAUDE.md` del motor, "Simulation Flow").
- **Consecuencia**: no hace falta servidor, WebSocket ni navegador. Se usa `GameRunner` directamente.

## 2. Ejecución headless

- **[leído]** `createGameRunner(gameStart, clientID, map, callBack)` en `packages/engine/src/GameRunner.ts:44` arma la partida.
- **[leído]** El bucle es: `addTurn(turn)` (`GameRunner.ts:202`) y luego `executeNextTick()` (`GameRunner.ts:206`). `init()` está en `GameRunner.ts:176`.
- **[leído]** El repo ya incluye una prueba que lo hace sin pantalla: `tests/perf/fullgame/FullGamePerf.ts` ("Runs the real simulation pipeline … headlessly"). Se lanza con `npm run perf:game`.
- **[medido]** Requisito de entorno: el motor pide Node `>=24.15 <25` y npm `>=12.1 <13` (`package.json`, campo `engines`). Con Node 22 `npm ci` falla con `EBADENGINE`. Con Node 24.22 y npm 12.2.1 instaló bien.
- **[no verificado]** Si Node en tu PC cumple ese requisito. Hay que comprobarlo con `node -v`.

## 3. Velocidad

**[leído]** El juego real avanza a 100 ms por tick, o sea 10 ticks/s (`packages/engine-lib/src/configuration/Config.ts:165`).

**[medido]** Un solo proceso, sin perfiles, `FullGamePerf.ts`, en la nube (no en tu PC):

| Configuración | Ticks/s |
|---|---|
| World, 400 bots + naciones, 1800 ticks (2 ejecuciones) | 191 y 201 |
| Britannia, 20 bots, 3000 ticks | 492 |
| World, 0 bots, 3000 ticks | 437 |
| World, 50 bots + 1 jugador controlado por código (PoC) | 294 |

- Aproximadamente **20 a 50 veces más rápido que el tiempo real** por proceso.
- Ningún tick superó el presupuesto de 100 ms (`Over 100ms budget: 0 / 1800`).
- **[leído]** La fase de spawn dura 200 turnos (150 con spawn aleatorio, 100 en Singleplayer): `Config.ts:316`.
- **[leído]** Hay un límite duro de 170 min de juego (`WinCheckExecution.ts`, `HARD_TIME_LIMIT_SECONDS`), unos 102 000 ticks. A ~200 ticks/s serían unos 8,5 min por partida en el peor caso. **[no verificado]**: la velocidad baja cuando la partida se llena de unidades, y no medí partidas tan largas.
- **[no verificado]** Cuántas instancias en paralelo caben por máquina, y si comparten estado global (no busqué variables globales mutables en el motor).
- **Advertencia**: son números de mi máquina en la nube. Hay que repetirlos en tu PC y en la máquina de entrenamiento.

## 4. Estado del juego

- **[leído]** El mapa es una cuadrícula de tiles (`TileRef`). Interfaz `GameMap` en `packages/engine-api/src/game/GameMap.ts:5`: `isLand`, `isOcean`, `ownerID`, `hasOwner`, `hasFallout`, `isBorder`, `neighbors`, etc.
- **[medido]** El mapa `World` mide 2000 × 1000 = 2 millones de tiles. Una observación completa a esa resolución es demasiado grande; habrá que reducirla.
- **[leído]** `createGame` recibe además un `miniGameMap`, y existen los archivos `map4x.bin` y `map16x.bin`. **[no verificado]** que sea una versión reducida usable como observación.
- **[leído]** Cada tick, `GameRunner` entrega por callback (`GameRunner.ts`, `executeNextTick`):
  - `packedTileUpdates`: pares `[tileRef, estado]` en `Uint32Array`.
  - `packedPlayerUpdates`: quintetas `[smallID, tiles, oro, tropas, oroGanado]` en `Float64Array` (`packages/engine-api/src/game/GameUpdates.ts:30-47`).
  - `updates`: eventos tipados (`GameUpdateType`, `GameUpdates.ts:92`), entre ellos `Win`, alianzas, unidades.
- **[leído]** También se puede consultar directamente `game.players()`, `player.troops()`, `gold()`, `numTilesOwned()`, `tiles()`, `borderTiles()`, `outgoingAttacks()` (`packages/engine/src/game/Game.ts`, interfaces de `Player`).
- **[medido]** En el PoC se leyeron tiles, tropas y oro del jugador cada pocos ticks sin problema.

## 5. Acciones (intents)

- **[leído]** Hay **25 tipos de intent** en `IntentSchema` (`packages/engine-api/src/Schemas.ts:552-578`). Los relevantes para jugar:

| Intent | Qué hace | Parámetros |
|---|---|---|
| `spawn` | Elegir dónde aparecer | `tile` |
| `attack` | Atacar a un jugador o a tierra libre | `targetID` (null = tierra libre), `troops` (null = por defecto) |
| `boat` | Ataque anfibio | `dst` (tile), `troops` |
| `build_unit` | Construir (ciudad, puerto, silo…) | `unit`, `tile` |
| `upgrade_structure` | Mejorar estructura | `unit`, `unitId` |
| `allianceRequest` / `allianceReject` / `breakAlliance` / `allianceExtension` | Diplomacia | `recipient` |
| `donate_gold` / `donate_troops` | Donaciones | `recipient`, cantidad |
| `cancel_attack` / `cancel_boat` | Retirada | `attackID` / `unitID` |
| `embargo` / `embargo_all` | Comercio | `targetID`, acción |
| `move_warship`, `delete_unit` | Gestión de unidades | |
| `targetPlayer`, `emoji`, `quick_chat` | Señales | |

- Los intents de administración (`kick_player`, `toggle_pause`, `update_game_config`, `toggle_game_start_timer`) no deben ser parte del espacio de acciones del agente.
- **[leído]** El motor ejecuta un intent solo si existe un jugador con ese `clientID` (`packages/engine/src/execution/ExecutionManager.ts:60-61`); si no, lo ignora con un aviso.
- **[leído]** Un `Turn` es `{ turnNumber, intents: [...] }`, cada intent con su `clientID` (`Schemas.ts:596-601`). Nada impide meter intents de varios jugadores en el mismo turno. **[no verificado]** con más de un jugador controlado por código.
- **[leído]** Hay validación de acciones disponibles: `GameRunner.playerActions()` y `playerBuildables()` devuelven qué se puede hacer en un tile (`GameRunner.ts`). Sirven para construir máscaras de acciones inválidas.
- **[medido]** En el PoC, un jugador humano con `clientID` hizo `spawn` y luego `attack` (`targetID: null`) y conquistó tierra: pasó de 52 a 3439 tiles en ~500 ticks.

## 6. Oponentes integrados

- **[leído]** Dos tipos: **Tribus** (`TribeExecution.ts`, 204 líneas, comportamiento simple con tasas de ataque aleatorias por tribu) y **Naciones** (`NationExecution.ts`, 525 líneas, IA más rica, con módulos de alianzas en `execution/nation/`).
- **[leído]** Dificultad: `Easy`, `Medium`, `Hard`, `Impossible` (`packages/engine-api/src/game/GameTypes.ts:66`).
- **[leído]** Se configuran con `bots` (0–400) y `nations` (número, `"default"` o `"disabled"`) en `GameConfig` (`Schemas.ts:274-280`).
- **[no verificado]** Qué tan buenos son contra un humano. Útiles como primeros oponentes de entrenamiento; la calidad real se mide en la Fase 3.

## 7. Condición de victoria y fin de partida

- **[leído]** `WinCheckExecution` revisa cada 10 ticks (`tick % 10`). En FFA gana quien supere un porcentaje de tiles de tierra (`percentageTilesOwnedToWin`, `Config.ts:293`), o quien sea el mayor al agotarse el temporizador o el límite duro de 170 min.
- **[leído]** El umbral base es `PERCENT_TILES_OWNED_TO_WIN = 80`, es decir, más del 80 % de la tierra no contaminada (`Config.ts:74`). Con "overtime" activado el umbral baja con el tiempo (`Config.ts:293-310`).
- **Consecuencia para la recompensa**: la victoria es alcanzable y definida; la fracción de territorio es una señal natural de progreso.

## 8. Determinismo y reproducibilidad

- **[leído]** El motor exige determinismo bit a bit: PRNG con semilla y `DetMath` en lugar de `Math.*` (`CLAUDE.md` del motor).
- **[medido]** Dos ejecuciones con la misma semilla (`rl-test`, World, 400 bots, 1800 ticks) dieron el mismo hash final: `26347989348602096` en el tick 2000.
- **[leído]** Hay snapshots de la partida completa: `GameRunner.snapshot()` y `createGameRunnerFromSnapshot()` (`GameRunner.ts:166` y `:118`). Una partida tardía de World tarda unos 90 ms en guardarse y 150 ms en restaurarse (`packages/engine/src/snapshot/README.md`, cifra del propio repo, **no medida por mí**).
- **Uso posible**: reiniciar episodios desde situaciones intermedias, o evaluar desde posiciones fijas.

## 9. Pendientes e incertidumbres

1. Repetir las mediciones de velocidad en tu PC y en la máquina que entrenará.
2. Comprobar cuántas partidas en paralelo caben por proceso/máquina y si hay estado global compartido.
3. Medir partidas largas (decenas de miles de ticks).
4. Probar dos o más jugadores controlados por código en la misma partida (self-play).
5. Confirmar cómo se reduce el mapa (`miniGameMap`) y si sirve como observación.
6. En el PoC apareció `cannot build Defense Post` en la salida; no investigué su origen (probablemente una IA de nación).
7. La prueba de velocidad con 400 bots mostró "472 jugadores aparecieron" pero "77 / 77 vivos"; no sé explicar la diferencia (puede ser que `players()` solo cuente a los que tienen tiles).
8. El agente del PoC, con una política ingenua, llegó a 3439 tiles y fue eliminado hacia el tick 2200. Eso solo prueba que el ciclo funciona, no dice nada del aprendizaje.

## 10. Riesgos para el proyecto

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Requisito Node 24 / npm 12 | Fallo de instalación | Fijar versiones en el repo (`.nvmrc` o `engines`) |
| Mapa de 2 M de tiles | Observaciones enormes | Mapas pequeños al inicio y observaciones reducidas |
| Partidas largas | Pocos episodios por hora | Límite de ticks por episodio (truncado) |
| Motor cambia (el repo es muy activo: v0.3.6-4876) | Se rompen specs | Fijar el commit del submódulo y actualizar con ADR |
| Licencia del motor | `LICENSE` es **AGPL v3** (`LICENSING.md` explica el historial MIT → GPL → AGPL, y hay assets con licencia aparte). Si publicas código que enlaza o incorpora el motor, las condiciones de la AGPL pueden aplicarte | Mantener el motor como submódulo sin modificar; revisar `LICENSING.md` y decidir la licencia del proyecto antes de publicar. No es asesoría legal |

## 11. Reproducir las mediciones

```bash
# Con Node 24.x y npm 12.x
cd vendor/openfront
npm ci --ignore-scripts
npx tsx tests/perf/fullgame/FullGamePerf.ts --map world --ticks 1800 --bots 400 \
  --seed rl-test --no-cpu-profile --no-exec-profile --no-gc-profile --no-alloc-profile
```

El script de referencia con un jugador controlado por código está en `docs/research/poc/phase0_poc.ts` (ver el aviso al inicio del archivo).
