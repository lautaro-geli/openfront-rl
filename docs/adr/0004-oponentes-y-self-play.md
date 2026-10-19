# ADR 0004: Oponentes de entrenamiento y estrategia de self-play

- **Estado**: propuesta
- **Fecha**: 2026-10-18

## Contexto
El motor trae dos IA propias: Tribus (`TribeExecution.ts`, simples) y Naciones (`NationExecution.ts`, más ricas), con dificultades `Easy`–`Impossible` (`engine-findings.md` §6). Los jugadores controlados por código son jugadores `Human` con `clientID`, y el motor ejecuta los intents de cualquier jugador cuyo `clientID` exista (`ExecutionManager.ts:60-61`).

## Opciones consideradas
1. **Solo oponentes integrados** (tribus y naciones). Sencillo; el agente puede quedarse en explotar sus debilidades.
2. **Self-play puro**: varios jugadores controlados por copias del agente. Más rico; riesgo de inestabilidad al inicio.
3. **Currículo mixto**: empezar contra tribus `Easy`, subir a naciones y después añadir copias pasadas del agente (liga).

## Decisión (propuesta)
Opción 3, por fases: Fase 3 contra tribus/naciones; Fase 4 con liga de checkpoints.

## Consecuencias
- Verificado (2026-10-19): el motor ejecuta los intents de varios jugadores controlados por código en la misma partida. Tests con 2 agentes (`bridge/tests/multi.test.ts`) y ejecuciones con 2 y 4 agentes heurísticos. Un agente sin acciones no cambia su territorio mientras otro se expande, o sea que cada intent llega a su jugador.
- Observado, sin investigar: con 4 agentes heurísticos en Britannia, 2 acabaron eliminados. Pudo ser por spawns cercanos (la heurística elige el mismo mejor sitio para todos); no lo comprobé.
- La evaluación se hace siempre contra un conjunto fijo de oponentes y semillas, para poder comparar versiones.

## Cómo validarlo
Hecho en la Fase 1 para el soporte técnico (ver arriba). Falta la liga de checkpoints (Fase 4). Fase 3: curva de victorias contra tribus `Easy`.
