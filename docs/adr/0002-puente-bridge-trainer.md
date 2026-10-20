# ADR 0002: Mecanismo de comunicación bridge ⇄ trainer

- **Estado**: aceptada (2026-10-20); la medición se hace con la implementación (`docs/specs/protocol.md` §6)
- **Fecha**: 2026-10-18

## Contexto
El motor es TypeScript puro y corre sin navegador (`docs/research/engine-findings.md` §1-2). El entrenamiento irá en Python. El motor simula a ~200–490 ticks/s por proceso (§3), así que el canal entre ambos no debe volverse el cuello de botella.

Datos relevantes:
- El agente decide cada N ticks, no en cada tick. Con N = 10 (1 decisión por segundo de juego), un proceso a 200 ticks/s produce ~20 decisiones/s.
- Por decisión viajan una observación (capas reducidas + vector) y una acción pequeña.
- Cada tick el motor ya entrega `packedTileUpdates` y `packedPlayerUpdates` en buffers tipados (`GameUpdates.ts:30-47`).

## Opciones consideradas
1. **Proceso Node por entorno (o por lote) hablando con Python por stdio/sockets locales**, con mensajes binarios. Simple, desacoplado, aísla fallos. Coste: serializar observaciones.
2. **Memoria compartida** para las capas espaciales. Más rápido; más complejo y frágil entre sistemas operativos.
3. **Entrenar del lado TS** (política en ONNX u otra librería JS). Evita IPC, pero el ecosistema de RL en JS es pobre.

## Decisión
Opción 1, con estas precisiones acordadas:
- **Un proceso Node por partida**, lanzado por Python y comunicado por stdin/stdout con mensajes binarios con longitud. Python gestiona un entorno vectorizado con N procesos (por defecto núcleos − 1).
- La observación se construye en TypeScript y viaja como **un buffer contiguo de `float32`** descrito por un manifiesto; Python lo corta con `numpy` sin copiar.
- La acción es un **vector de 6 enteros**; el catálogo completo (24 tipos) está en `docs/specs/actions.md`.
- La API de Python es multiagente (`reset()`/`step(acciones)` con listas) y encima un envoltorio Gymnasium de un agente.
- Se pasa a la Opción 2 solo si la medición muestra que el IPC pasa del 20 % del tiempo por paso.

## Consecuencias
- Fácil de depurar y de paralelizar con varios procesos Node.
- La reducción del mapa vive en el bridge, no en Python.
- Hay que fijar un contrato de mensajes versionado (spec pendiente).

## Cómo validarlo
En la Fase 1: bucle `reset`/`step` con política aleatoria, midiendo pasos/s de extremo a extremo con 1, 4 y 8 entornos. Criterio: el IPC debe costar menos del 20 % del tiempo total por paso. **No medido todavía.**
