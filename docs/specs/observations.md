# Spec: observaciones (v2)

> Estado: **implementada en `bridge/src/observation.ts` y `bridge/src/features.ts`** (fase 1b). Sustituye al borrador anterior.
> Cada dato está marcado: **[leído]** (visto en el código del motor, con `archivo:línea`), **[medido]** (ejecutado) o **[no verificado]**.
> La wiki (`openfront.wiki`) se usó como fuente secundaria para entender las reglas; **los números de esta spec salen del motor**, no de la wiki (coinciden en lo que se contrastó: costes, límite de tropas, radios, 80 % de tierra).

## 1. Principios

1. **Solo información pública.** El motor corre en cada cliente con el estado completo y `PlayerUpdate` lleva oro, tropas, tiles, alianzas, embargos y ataques de **todos** los jugadores (`engine-api/src/game/GameUpdates.ts:237-282`). Un humano ve lo mismo. No se usa nada que el cliente no reciba. *(Esto corrige lo que decía `bridge.md` §4: oro y tropas de los rivales sí son visibles.)*
2. **Relativa al agente.** Todo se calcula desde el punto de vista del agente que observa.
3. **Tamaño fijo.** Cada bloque tiene filas máximas fijas, relleno con ceros y una máscara de filas válidas. Se descartan filas por prioridad determinista.
4. **Normalizada.** Todo valor en `[0, 1]` salvo los marcados `[-1, 1]` (razones con `tanh`). Las magnitudes grandes (oro, tropas) usan `log1p`.
5. **Determinista.** El orden de las filas depende solo del estado del motor (desempate por `id`/`smallID`). La observación no modifica el estado del juego.
6. **Estado propio mínimo.** Solo las **tasas de ingreso** necesitan el paso anterior (se calculan con la diferencia de contadores acumulados). Se guardan en el entorno y se reinician en `reset()`.

## 2. Estructura

| Bloque | Rama que lo lee | Forma | Contenido |
|---|---|---|---|
| `grid` | mapa | `[20, h, w]` | 20 canales (§3) |
| `economy` (tabla + escalares) | economía | `[48, 9]` + 38 escalares | ciudades, puertos, fábricas propios; costes; ingresos; tropas |
| `military` (tabla de unidades) | defensa/ataque | `[96, 30]` | defensas, SAM, silos, barcos, bombas (propios y ajenos) |
| `attacks` (tabla) | defensa/ataque | `[16, 10]` | ataques en curso, salientes y entrantes |
| `military` (escalares) | defensa/ataque | 23 escalares | cobertura SAM, munición, bombas en vuelo, ataques, fallout |
| `players` (tabla) | comparación | `[16, 46]` | yo, 8 vecinos, top 3 y 4 aliados |
| `comparison` (escalares) | comparación | 14 escalares | ranking, brecha con el líder, fuerza de los vecinos |
| `globals` | todas (se concatena al núcleo) | 14 escalares | tiempo, fase, umbral de victoria, estado de traidor |

Cada tabla trae `valid: Uint8Array[filas]`. Los nombres de cada columna y escalar están en `bridge/src/features.ts` (`*_FEATURES`); los tests comprueban que la spec y el código tienen las mismas longitudes.

## 3. Mapa: `grid` `[20, h, w]`

Cada celda cubre un bloque del mapa y se estima con una malla de `S × S` tiles (ADR 0005, `S = 4`). Los canales **0 a 2 conservan su significado** de la v1.

| # | Canal | Definición | Fuente |
|---|---|---|---|
| 0 | `land` | fracción de muestras de tierra | `GameMap.isLand` (`GameMap.ts:17`) |
| 1 | `own` | muestras cuyo dueño es el agente | `GameMap.ownerID` (`GameMap.ts:33`) |
| 2 | `other` | muestras con dueño distinto del agente | ídem |
| 3 | `highland` | muestras de terreno `Highland` | `GameMap.terrainType` (`GameMap.ts:60`), `GameTypes.ts:296-302` |
| 4 | `mountain` | muestras de terreno `Mountain` (la llanura es `land − highland − mountain`) | ídem |
| 5 | `shore` | muestras de costa | `GameMap.isShore` (`GameMap.ts:58`) |
| 6 | `ally` | muestras cuyo dueño es aliado | `Player.isFriendly` (`Game.ts:430`) |
| 7 | `neighbor` | muestras cuyo dueño comparte frontera con el agente | `Player.sharesBorderWith` (`PlayerImpl.ts:573`) |
| 8 | `top3` | muestras de los 3 jugadores con más tiles (sin contar al agente) | `Player.numTilesOwned` |
| 9 | `fallout` | muestras con fallout (radiación de bombas) | `GameMap.hasFallout` (`GameMap.ts:37`) |
| 10 | `own_struct` | estructuras propias en la celda (ciudad, puerto, fábrica, defensa, SAM, silo), `min(n/4, 1)` | `Structures` (`GameTypes.ts:232-239`) |
| 11 | `enemy_struct` | estructuras de otros jugadores, igual | ídem |
| 12 | `own_sam_cov` | 1 si el centro de la celda está dentro del radio de algún SAM propio | `EngineConfig.dynamicSamRange` (`EngineConfig.ts:708-721`) |
| 13 | `own_post_cov` | 1 si está a ≤ 30 tiles de una defensa propia | `Config.defensePostRange` (`Config.ts:175-177`) |
| 14 | `enemy_sam_cov` | como el 12, con SAM de otros jugadores no aliados | ídem |
| 15 | `nuke_target` | bombas en vuelo con destino en la celda, `min(n/2, 1)` | `Unit.targetTile` (`Game.ts:234`) |
| 16 | `friendly_ships` | barcos propios y aliados (guerra, transporte) | `UnitType` (`GameTypes.ts:193-210`) |
| 17 | `hostile_ships` | barcos de otros jugadores | ídem |
| 18 | `attack_out` | tiles en disputa de los ataques propios (`Attack.borderTiles`) | `Game.ts:153` |
| 19 | `attack_in` | tiles en disputa de los ataques que recibo | ídem |

- Los canales 0, 3, 4 y 5 son **estáticos** mientras `waterNukes` esté apagado (lo está: `multiEnv.ts` lo deja por defecto, `EngineConfig.ts:163`). Se calculan una vez por episodio.
- Los canales 10 a 19 se pintan desde las listas de unidades y ataques, no desde el muestreo.
- Los canales 6 a 8 usan una tabla `smallID → clase` para que cada muestra cueste una sola consulta.
- **[no verificado]** El muestreo no ve territorios más pequeños que una celda; el error de masa del jugador mayor ronda el 5 % (ADR 0005).

## 4. Economía

### 4.1 Tabla `economy` `[48, 9]`: estructuras económicas propias
Ciudades, puertos y fábricas **del agente**, en orden de `id` del `Unit`. Hasta 48 filas.

| Col | Nombre | Definición |
|---|---|---|
| 0-2 | `is_city`, `is_port`, `is_factory` | tipo (`UnitType`, `GameTypes.ts:193-210`) |
| 3 | `level` | `Unit.level() / 10`, recortado (`Game.ts:292`) |
| 4-5 | `x`, `y` | posición en `[0, 1]` |
| 6 | `under_construction` | `Unit.isUnderConstruction` (`Game.ts:288`) |
| 7 | `has_station` | `Unit.hasTrainStation` (`Game.ts:222`): conectada a la red de trenes |
| 8 | `dist_centroid` | distancia al centroide propio / diagonal del mapa (centroide = media de las estructuras propias; si no hay, el tile de spawn; si no, el centro del mapa) |

### 4.2 Escalares `economy` (38)
- **Tropas** (4): `max_troops_log` (`Config.maxTroops`, `Config.ts:326-368`); `troop_growth` (`troopIncreaseRate / maxTroops`, `Config.ts:370-402`); `city_troop_share` (parte del máximo que aportan las ciudades: 250 000 por nivel, `Config.ts:162-164`); `troops_fill`.
- **Oro** (1): `gold_log`.
- **Ingresos por tick** (4, `log1p`): `income_total` (`Player.goldEarned`, `Game.ts:380`), `income_trade` (`tradeGold`, `Game.ts:368`), `income_train` (`trainGold`, `Game.ts:370`), `income_piracy` (`piracyGold`, `Game.ts:375`). Son diferencias entre pasos. La renta base por trabajadores es 100 por tick para humanos (`EngineConfig.goldAdditionRate`, `EngineConfig.ts:668-677`).
- **Costes** (20): para cada tipo (`city, port, factory, post, sam, silo, warship, atom, hydrogen, mirv`) el coste siguiente en `log1p` (`unitInfo(t).cost(game, me)`, `EngineConfig.ts:339-399`) y `afford = min(1, oro / coste)`.
- **Contadores** (8): niveles de ciudades, puertos y fábricas (`unitsOwned`, `PlayerImpl.ts:551-571`) / 10; estaciones de tren / 20; barcos mercantes propios / 20 y trenes propios / 20 (`UnitType.TradeShip`, `UnitType.Train`); socios comerciales / 10 (`tradingPartners`, `Game.ts:473`).
- **Embargos** (2): embargos puestos por mí / 10 y sobre mí / 10 (`getEmbargoes`, `hasEmbargoAgainst`, `Game.ts:469-476`).

**[leído]** Los costes suben con las unidades ya poseídas, y puertos y fábricas comparten escalera (`EngineConfig.ts:346-388`). El agente conoce siempre el precio real del siguiente.
**[no verificado]** El tope de 20 trenes y barcos mercantes por jugador es una escala de normalización mía, no una regla del motor.

## 5. Defensa y ataque

### 5.1 Tabla `military` `[96, 30]`: unidades y estructuras militares
Incluye **propias y ajenas**: defensas, SAM, silos, barcos de guerra, transportes y bombas (átomo, hidrógeno, MIRV y cabezas). Prioridad de filas: grupo 0 = unidades propias y bombas que apuntan a territorio propio; grupo 1 = el resto, ordenado por distancia al centroide propio y luego por `id`. Se toman las primeras 96.

| Col | Nombre | Definición |
|---|---|---|
| 0-8 | tipo | `is_post, is_sam, is_silo, is_warship, is_transport, is_atom, is_hydrogen, is_mirv, is_warhead` |
| 9-11 | dueño | `own`, `friendly` (aliado), `other` |
| 12-13 | `x`, `y` | posición `[0, 1]` |
| 14 | `level` | nivel / 10 |
| 15 | `under_construction` | |
| 16 | `health` | `Unit.health() / maxHealth()` (`Game.ts:257-259`) |
| 17 | `ready_slots` | SAM y silo: casillas de munición libres / nivel. `isInCooldown` es `cola == nivel` (`UnitImpl.ts:567-569`), así que libres = `nivel − cola` |
| 18 | `range` | SAM: radio actual / 150 (`maxSamRange`, `Config.ts:425`); defensa: 30/150; bombas: radio exterior / 100 (`nukeMagnitudes`, `Config.ts:404-414`) |
| 19 | `troops` | transportes: tropas / máximo del dueño |
| 20-21 | `target_x`, `target_y` | destino (bombas) |
| 22 | `progress` | `trajectoryIndex / longitud` (`NukeState`, `GameTypes.ts:36-41`) |
| 23 | `targeted_by_sam` | `Unit.targetedBySAM` (`Game.ts:242`) |
| 24 | `targets_me` | el tile de destino es del agente |
| 25 | `dist_centroid` | distancia al centroide propio / diagonal |
| 26-28 | estado de barco | `patrolling`, `retreating`, `docked` (`WarshipState`, `GameTypes.ts:19-29`) |
| 29 | `veterancy` | nivel / 3 (`warshipMaxVeterancy`, `EngineConfig.ts:791`) |

### 5.2 Tabla `attacks` `[16, 10]`
Hasta 8 ataques salientes y 8 entrantes (`Player.outgoingAttacks`, `incomingAttacks`, `Game.ts:491-492`). Orden: salientes primero, luego entrantes; dentro de cada grupo por tropas descendente y `id`.

| Col | Nombre | Definición |
|---|---|---|
| 0-1 | `outgoing`, `incoming` | sentido |
| 2 | `troops_over_max` | tropas del ataque / máximo de tropas del agente |
| 3 | `troops_over_mine` | tropas / tropas actuales del agente, recortado a 2 y dividido entre 2 |
| 4 | `retreating` | `Attack.retreating` (`Game.ts:136`) |
| 5 | `border_size` | `log1p(borderSize) / log1p(1e4)` (`Game.ts:151`) |
| 6 | `vs_nobody` | el objetivo es tierra sin dueño (expansión) |
| 7 | `from_landing` | `sourceTile() != null`. **[leído en parte]** `AttackExecution.ts:139-149` lo usa para arrancar desde un desembarco; la semántica exacta de "barco" no se contrastó |
| 8 | `counterpart_in_table` | la contraparte está en la tabla `players` |
| 9 | `counterpart_row` | fila en `players` / 15 (0 si no está) |

### 5.3 Escalares `military` (23)
Defensas, SAM y silos (niveles / 10); silos y SAM listos (casillas libres / niveles); fracción de estructuras propias dentro del radio de algún SAM propio; barcos de guerra / 5, su salud media y los que están en combate; transportes propios / 3 (`boatMaxNumber`, `EngineConfig.ts:500-505`) y sus tropas / máximo; ataques salientes (número / 8, tropas / máximo) y entrantes (número / 8, tropas / mis tropas, mayor ataque / mis tropas); bombas propias en vuelo / 10; bombas que apuntan a mi territorio por tipo (átomo, hidrógeno, MIRV) / 5; mayor progreso de una bomba entrante; fracción propia con fallout (del muestreo) y fracción del mapa con fallout (`numTilesWithFallout / numLandTiles`, `GameMap.ts:106`).

## 6. Comparación

### 6.1 Tabla `players` `[16, 46]`
Filas **fijas por posición**: 0 = el agente; 1 a 8 = vecinos en el mismo orden que las ranuras de ataque (la fila `1 + i` corresponde a `attack{slot: i}`); 9 a 11 = top 3 por tiles (sin el agente; puede repetir un vecino); 12 a 15 = aliados que no estén ya en la tabla, por tiles descendente.

| Grupo | Columnas |
|---|---|
| Tipo (4) | `is_self, is_human, is_nation, is_bot` (`PlayerType`, `GameTypes.ts:304-308`) |
| Tamaño (6) | `tiles_share`, `tiles_ratio` (`tanh(ln((t+1)/(t_yo+1))/2)`, `[-1,1]`), `troops_fill`, `troops_log`, `troops_ratio` (`[-1,1]`), `max_troops_ratio` (`[-1,1]`) |
| Oro (3) | `gold_log`, `gold_ratio` (`[-1,1]`), `income_rate` |
| Estructuras (7) | niveles de ciudades, puertos, fábricas, defensas, SAM, silos y barcos de guerra / 10 |
| Relación (8) | `is_neighbor`, `can_attack` (`canAttackPlayer`, `PlayerImpl.ts:1924-1933`), `rel_hostile, rel_distrustful, rel_neutral, rel_friendly` (`Relation`, `GameTypes.ts:258-263`; umbrales en `PlayerImpl.ts:947-958`), `is_ally`, `targets_me` |
| Diplomacia (6) | `req_in`, `req_out` (solicitudes de alianza), `alliance_left` (ticks restantes / duración, `Config.allianceDuration`, `Config.ts:285-291`), `can_request` (`canSendAllianceRequest`), `embargo_by_me`, `embargo_on_me` |
| Reputación (5) | `is_traitor` (`PlayerImpl.ts:870-880`), `betrayals / 5`, `is_immune` (`PlayerImpl.ts:1914-1922`), `disconnected`, `targeted_by_me` |
| Ranking (4) | `rank` (1 − posición/N), `is_top1`, `is_top3`, `spawn_dist` (distancia entre spawns / diagonal) |
| Presión (3) | `attacks_on_me` (tropas de sus ataques contra mí / mi máximo), `my_attack_on_them` (ídem al revés), `alive` |

(Los 46 valores se numeran en `PLAYER_ROW_FEATURES`.)

### 6.2 Escalares `comparison` (14)
Posición propia en el ranking; cuota del líder; razón líder/yo (`[-1,1]`); suma de las cuotas del top 3; distancia al umbral de victoria; número de vecinos / 8; cuota de tierra de los vecinos; tropas de los vecinos / mis tropas (`[-1,1]`); vecinos hostiles / 8; razón del vecino más fuerte y del más débil (`[-1,1]`); aliados / 5; cuota de tierra de los aliados; enemigos con los que hay un ataque abierto / 8.

## 7. Escalares generales `globals` (14)
Los **6 primeros conservan su significado de la v1** (`bridge.md` §4): cuota de tierra; tropas / máximo; oro; vivos / iniciales; tiempo / `maxTicks`; fase de spawn. Después: umbral de victoria (`percentageTilesOwnedToWin`, `Config.ts:293-312`) / 100; cuota del líder; posición propia; jugadores vivos (log); humanos vivos; naciones vivas; inmunidad de spawn activa; estado de traidor del agente.

## 8. Qué no se incluye (y por qué)
- **Trenes y barcos mercantes por separado**: son muy numerosos y rotan; entran agregados (§4.2).
- **Emojis, chat, nombres, cosméticos**: no afectan al juego.
- **Modos de equipo y Doomsday Clock**: el entorno usa FFA y el reloj está apagado por defecto (`Config.ts:35-71`). Si se activan hay que ampliar la observación (`inDoomsdayClock`, `isDecaying`, `Game.ts:329-331`).
- **Acciones nuevas** (construir, aliarse, bombas): esta spec es solo de **entradas**; el espacio de acciones sigue en `bridge.md` §5 y ampliarlo es el siguiente paso.

## 9. Coste y aceptación
1. Coste por observación medido en la nube; objetivo: menos de 5 ms con 20 bots y menos de 15 ms con 400 bots. **[no medido todavía]**
2. Tests: forma y rango de cada bloque en todos los pasos; la fila 0 de `players` es el propio agente; las filas 1 a 8 coinciden con las ranuras de ataque; dos agentes ven filas distintas (relativas); misma semilla, mismas observaciones; el hash de una partida de referencia **no cambia** (la observación no modifica el estado).
3. Prueba de utilidad: tras construir una ciudad (en un test que fuerza el estado del motor) la tabla `economy` gana una fila y `afford` cambia.
