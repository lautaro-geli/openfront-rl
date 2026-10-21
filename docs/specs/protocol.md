# Spec: protocolo Node ⇄ Python (v3)

> Estado: **aprobada para implementar** (ADR 0002). Código: `bridge/src/server.ts` (Node) y `trainer/openfront_rl/client.py` (Python).

## 1. Modelo
- Python lanza **un proceso Node por partida** (`tsx bridge/src/server.ts`) y habla con él por stdin/stdout. Para entrenar en paralelo lanza varios (por defecto núcleos − 1; en una CPU de 6 núcleos, 5).
- Cada partida puede tener 1 o más agentes (`agents` en la config), para self-play.
- stdout solo lleva mensajes del protocolo; los registros del motor van a stderr.

## 2. Mensajes
Todo mensaje es `uint32 little-endian con la longitud del cuerpo` seguido del cuerpo. El primer byte del cuerpo es el tipo.

### Peticiones (Python → Node)
| Tipo | Nombre | Cuerpo tras el tipo | Respuesta |
|---|---|---|---|
| `0x01` | `init` | JSON UTF-8 con `EnvConfig` parcial | `0x81` manifiesto |
| `0x02` | `reset` | JSON `{"seed": "..."}` | `0x82` resultado |
| `0x03` | `step` | `int32[agents × 6]`: una acción por agente (`docs/specs/actions.md` §2) | `0x82` resultado |
| `0x04` | `hash` | vacío | `0x83` texto UTF-8 con el hash del motor |
| `0x05` | `close` | vacío | `0x84` y el proceso termina |

### Respuestas (Node → Python)
| Tipo | Nombre | Cuerpo |
|---|---|---|
| `0x81` | manifiesto | JSON: versión, agentes, tamaños y desplazamientos de cada bloque (§3) |
| `0x82` | resultado | cabecera + una sección por agente (§4) |
| `0x83` | hash | texto |
| `0x84` | cerrado | vacío |
| `0xFF` | error | texto UTF-8 con el mensaje y la pila. El proceso sigue vivo salvo error fatal del motor |

Un `step` después del fin del episodio, o antes de `reset`, devuelve `0xFF`.

## 3. Manifiesto
```
{ "version": 3, "agents": A, "grid": {"h":32,"w":64}, "decisionInterval": 10,
  "obs": { "floats": N, "blocks": [ {"name": "grid", "shape": [20,32,64], "offset": 0}, ... ] },
  "mask": { "bytes": M, "blocks": [ {"name": "kind", "shape": [24], "offset": 0}, ... ] },
  "actionDims": 6 }
```
Los bloques salen de `features.ts` y de la spec de acciones; Python no tiene números mágicos y falla al arrancar si no reconoce un bloque.

## 4. Resultado
Cabecera de 8 bytes: `uint8 done`, `uint8 truncated`, `uint8 agents`, `uint8 reservado`, `int32 tick`. Luego, para cada agente, un bloque de longitud fija:
`float32 reward` (suma de los componentes), `uint8 terminated`, `uint8 invalid`, `uint8 won`, `uint8 eliminated`, `float32[5] reward_parts` (`territory, attack, economy, defense, terminal`; `docs/specs/reward.md`), `float32[8] events` (recuentos del paso, ver abajo), `float32[N]` observación (orden del manifiesto), `uint8[M]` máscara (orden del manifiesto). Todo little-endian, sin relleno entre campos salvo el de alineación al final de cada bloque a 4 bytes. El manifiesto da `result.agentHeaderBytes` (= 8 + 4 · 5 + 4 · 8 = 60), `result.rewardParts` y `result.eventNames` (los nombres).

**Eventos (v3).** Recuentos **del paso** por agente, para registrar qué hace el agente (no entran en la recompensa). Salen de las órdenes válidas del agente y de `conquerPlayer` del motor (`packages/engine/src/game/GameImpl.ts:1556`, envuelto desde el bridge sin tocar `vendor/`):
`attack_orders` (ataques a un jugador), `attack_tribe_orders` (los que apuntan a una tribu, `PlayerType.Bot`), `expand_orders` (expansión a tierra sin dueño), `boat_orders` (desembarcos), `build_orders` (construcciones que no son misiles), `nuke_orders` (misiles: bomba atómica, de hidrógeno o MIRV), `tribes_conquered` y `nations_conquered` (jugadores eliminados porque el agente tomó su último territorio). Una orden es válida si pasó la máscara; que el motor la ejecute después no se garantiza (p. ej. el misil necesita silo y oro).

Cambio de la v2 a la v3: se añadieron `events` (la observación empieza en `agentHeaderBytes`; es 60 en vez de 28).

Cambio de la v1 a la v2: se añadieron `reward_parts` (la observación ya no empieza en el byte 8 del agente sino en `agentHeaderBytes`).

## 5. Garantías
- Determinismo: misma `seed` y mismas acciones dan los mismos bytes (test).
- Un `reset` con la misma semilla en el mismo proceso es equivalente a un proceso nuevo.
- Si el proceso Node muere, el cliente lanza una excepción con el contenido de stderr y no se queda colgado (lectura con tiempo límite configurable).

## 6. Tests
- Ida y vuelta: una partida de 100 pasos por el protocolo da los mismos bytes que la API TypeScript directa.
- Error: una acción mal formada devuelve `0xFF` sin matar el proceso.
- Varios procesos en paralelo dan lo mismo que en serie.
- Medición: coste de serializar y copiar por paso y pasos/s con 1 y 5 procesos.
