# Spec: acciones (v1)

> Estado: **aprobada para implementar**. Sustituye al borrador. Cada acción se asocia con un intent real del motor (`packages/engine-api/src/Schemas.ts:398-550`, creación en `packages/engine/src/execution/ExecutionManager.ts:60-150`).

## 1. Principios
- Una acción es un **vector de 6 enteros**: `[kind, row, cell, fraction, type, aux]`. Cada tipo de acción usa solo algunos campos; el resto va a 0.
- El agente decide cada `decisionInterval` ticks (10). Las acciones de todos los agentes se calculan sobre el mismo estado y viajan en el primer tick del paso.
- **Máscara en dos niveles.** Una máscara barata y conservadora acompaña a cada observación (oro suficiente, hay una fila válida, la celda contiene terreno adecuado). La **resolución exacta** (qué tile concreto, si el motor lo acepta) se hace al traducir la acción, con las mismas funciones que usa el motor (`canBuild`, `canUpgradeUnit`, `canSendAllianceRequest`…). Si la resolución falla, la acción se convierte en `noop` y `info.invalidAction = true`. Nada enmascarado llega al motor.
- Las filas apuntan a las tablas de la observación (`docs/specs/observations.md`), así que el agente elige "la fila 3 de jugadores" y no un id opaco.

## 2. Campos
| Campo | Rango | Significado |
|---|---|---|
| `kind` | 0..23 | Tipo de acción (§3) |
| `row` | 0..143 | Según el tipo: fila de jugadores (0..15), fila de ataques (0..15) o fila de unidades `U` (0..143 = 48 filas de economía + 96 de militar) |
| `cell` | 0..2047 | Celda del grid (`cy * w + cx`) |
| `fraction` | 0..4 | Índice en `[0.1, 0.25, 0.5, 0.75, 1.0]` |
| `type` | 0..6 | Tipo de estructura a construir (`city, port, factory, post, sam, silo, warship`) o de bomba (`atom, hydrogen, mirv`) |
| `aux` | 0..7 | Índice de emoji |

## 3. Catálogo
| # | Acción | Campos | Intent del motor | Válida cuando (máscara) | Resolución exacta |
|---|---|---|---|---|---|
| 0 | `noop` | | ninguno | siempre | |
| 1 | `spawn` | `cell` | `spawn` | fase de spawn; celda con tierra libre | tile de tierra libre más cercano (`cellToSpawnTile`) |
| 2 | `expand` | `fraction` | `attack` con `targetID: null` | fuera de spawn, vivo y con tropas | |
| 3 | `attack` | `row` 1..8, `fraction` | `attack` con `targetID` | la fila es una ranura de vecino y `canAttackPlayer` | |
| 4 | `boat_attack` | `cell`, `fraction` | `boat` | menos de `boatMaxNumber` (3) transportes; la celda tiene tierra ajena y costa | tile de tierra no propia de la celda con `canBuild(TransportShip, tile)` |
| 5 | `cancel_attack` | `row` 0..7 (ataque saliente) | `cancel_attack` | el ataque existe, activo y no se está retirando | |
| 6 | `cancel_boat` | `row` `U` | `cancel_boat` | fila de un transporte propio activo | |
| 7 | `build` | `type` 0..6, `cell` | `build_unit` | tipo habilitado y oro ≥ coste; la celda tiene tierra propia; para `warship`, una celda de agua y un puerto propio terminado | tile propio de la celda con `canBuild(type, tile)` |
| 8 | `upgrade` | `row` `U` | `upgrade_structure` | fila de estructura propia con `canUpgradeUnit` (ciudad, puerto, fábrica, SAM, silo) | |
| 9 | `delete_unit` | `row` `U` | `delete_unit` | unidad propia en tierra propia, no en spawn, y `canDeleteUnit` | |
| 10 | `nuke` | `type` 0..2, `cell` | `build_unit` con `AtomBomb`, `HydrogenBomb` o `MIRV` | oro ≥ coste; un silo listo; la celda tiene tierra no propia | objetivo: estructura enemiga de la celda, si no un tile ajeno, si no tierra; `canBuild` |
| 11 | `move_warship` | `row` `U`, `cell` | `move_warship` | fila de barco de guerra propio; la celda tiene agua | tile de agua de la celda en la misma componente de agua que el barco |
| 12 | `alliance_request` | `row` 1..15 | `allianceRequest` | `canSendAllianceRequest`. Si el otro ya nos la envió, **esto la acepta** (`AllianceRequestExecution.ts:35-49`) | |
| 13 | `alliance_reject` | `row` | `allianceReject` | el otro nos envió una petición pendiente | |
| 14 | `alliance_break` | `row` | `breakAlliance` | aliado actual. **Rompe y marca al agente como traidor** | |
| 15 | `alliance_extend` | `row` | `allianceExtension` | `allianceInfo(other).canExtend` | |
| 16 | `embargo_start` | `row` | `embargo` `start` | jugador vivo sin embargo nuestro | |
| 17 | `embargo_stop` | `row` | `embargo` `stop` | embargo nuestro activo | |
| 18 | `embargo_all_start` | | `embargo_all` `start` | siempre fuera de spawn | |
| 19 | `embargo_all_stop` | | `embargo_all` `stop` | siempre fuera de spawn | |
| 20 | `target_player` | `row` | `targetPlayer` | `canTarget` (no aliado, cooldown) | |
| 21 | `emoji` | `row` 0..15, `aux` | `emoji` (fila 0 = todos) | jugador vivo | índice en una tabla fija de 8 emojis que usan las naciones |
| 22 | `donate_gold` | `row`, `fraction` | `donate_gold` con `fraction * oro` | `canDonateGold` y oro > 0 | |
| 23 | `donate_troops` | `row`, `fraction` | `donate_troops` con `fraction * tropas` | `canDonateTroops` y tropas > 0 | |

Notas:
- Las donaciones están desactivadas por defecto para receptores humanos (`donateGold/donateTroops: false` en la config de la partida); con naciones o bots sí valen. Se incluyen para no tocar el protocolo si se activan.
- Los emojis importan porque las naciones reaccionan (`NationEmojiBehavior.ts:26-40`). La tabla de 8: 👍 🤝 🎯 😡 😈 🕊️ 🥺 💀.
- `alliance_break` es una decisión de alto riesgo; queda enmascarable pero disponible.
- **No incluidas**: `quick_chat` (sin efecto de juego), y los intents del servidor `mark_disconnected`, `kick_player`, `toggle_pause`, `update_game_config`, `toggle_game_start_timer`. No hay intent para "subir la población": las tropas crecen solas (`troopIncreaseRate`) y la reserva se controla con las fracciones de `expand`/`attack`.

## 4. Máscara
Acompaña a cada observación. Se calcula del mismo estado y se cachea hasta el siguiente `step`.
| Bloque | Forma | Contenido |
|---|---|---|
| `kind` | `[24]` | tipos con al menos una opción válida |
| `player` | `[11, 16]` | para los tipos con fila de jugador, en orden `attack, alliance_request, alliance_reject, alliance_break, alliance_extend, embargo_start, embargo_stop, target_player, emoji, donate_gold, donate_troops`, qué filas valen |
| `unit` | `[4, 144]` | para `cancel_boat, upgrade, delete_unit, move_warship`, qué filas `U` valen |
| `attack_row` | `[8]` | ataques salientes que se pueden cancelar |
| `build_type` | `[7]` | estructuras que se pueden pagar y están habilitadas |
| `nuke_type` | `[3]` | bombas que se pueden pagar con un silo listo |
| `cell` | `[6, 2048]` | celdas válidas para `spawn`, `build` (estructuras en tierra propia), `build_ship` (agua, para `build` con `warship`), `nuke`, `boat_attack`, `move_warship` |

Los campos antiguos `spawn`, `expand` y `attackSlots` se conservan.

### 4b. Acciones desactivadas por configuración
`EnvConfig.disabledActions` (lista de nombres de tipo, p. ej. `["embargo_start", "emoji"]`; por defecto vacía) apaga esos tipos en `mask.kind`; una acción desactivada cuenta como inválida. Sirve para quitar durante el entrenamiento lo que no aporta contra tribus: embargos (las tribus no comercian), emoji, donaciones, alianzas y marcar objetivo. `configs/v1_ppo_base.yaml` las desactiva. Tras analizar el run `Day1` también se desactivan `cancel_attack`, `cancel_boat` y `delete_unit`: el agente cancelaba casi cada desembarco que lanzaba (19–33 `cancel_boat` por 20–34 `boat_attack`), lo que le impedía cruzar el mar de Britannia (se quedaba en ~70 % o ~26 % del mapa), y borraba casi cada estructura que construía.

`EnvConfig.maxStructures` (p. ej. `{post: 4}`) limita cuántas estructuras de un tipo (`city, port, factory, post, sam, silo, warship`) puede tener el agente a la vez, contando las que están en construcción: con el tope alcanzado ese tipo se apaga en `mask.build_type` y, si no queda ningún tipo construible, también `mask.kind[build]`. Por defecto no hay topes. El entrenamiento limita los puestos de defensa a 4, lo que paga la recompensa (`docs/specs/reward.md` §2); en `Day1` el agente construía 31–43 por partida.

## 5. Tests (todos contra el motor real)
- Ninguna acción enmascarada produce un intent; una acción cuya resolución falla produce `invalidAction` y no llega al motor.
- Cada tipo de acción, forzando un estado donde es válida, cambia el estado esperado del motor (se construye una ciudad, aparece una petición de alianza, etc.).
- La máscara de `kind` es coherente con las demás.

## 6. Preguntas abiertas
- Si "construir en una celda" es suficiente precisión o hará falta elegir el tile exacto (se mide cuántas construcciones fallan en la resolución).
- Qué recompensa guía la economía y la diplomacia: ver `docs/specs/reward.md`; hoy solo hay territorio y victoria.
