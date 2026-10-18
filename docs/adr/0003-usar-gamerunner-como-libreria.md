# ADR 0003: Usar `GameRunner` del motor como librería, sin modificar `vendor/`

- **Estado**: propuesta
- **Fecha**: 2026-10-18

## Contexto
El motor está separado del cliente y del servidor, y sus paquetes no dependen de ellos (`engine-findings.md` §1). `GameRunner` (`packages/engine/src/GameRunner.ts:146`) recibe turnos con `addTurn` y avanza con `executeNextTick`. El repo ya ejecuta partidas completas así en `tests/perf/fullgame/FullGamePerf.ts`, y se comprobó con un jugador controlado por código (`docs/research/poc/phase0_poc.ts`).

## Opciones consideradas
1. **Usar `GameRunner`, `createGame`, `Executor` y `createNationsForGame` desde el bridge**, como hace `FullGamePerf.ts`. No se toca `vendor/`.
2. **Levantar el servidor completo** (`src/server`) y conectar clientes por WebSocket. Más fiel al juego real, pero más lento y con más piezas.
3. **Hacer un fork del motor** con una API de RL. Da libertad, pero rompe la regla de no modificar `vendor/` y complica las actualizaciones.

## Decisión (propuesta)
Opción 1. El bridge arma la partida como `FullGamePerf.ts` (mapa, jugadores `Human` con `clientID`, naciones, tribus) y alimenta `Turn` con los intents del agente.

## Consecuencias
- Dependemos de APIs internas del motor (no son una API pública estable). Hay que fijar el commit del submódulo y revisar cada actualización.
- Se aprovecha la validación de intents del motor; no hay que reimplementar reglas.
- Para actualizar el motor: crear un ADR que registre el cambio y repetir las pruebas de contrato.

## Cómo validarlo
Test de humo en el bridge: partida de N ticks, mismo hash final con la misma semilla en dos ejecuciones (ya observado en la Fase 0 con `FullGamePerf.ts`).
