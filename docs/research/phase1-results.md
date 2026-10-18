# Fase 1: resultados del bridge (lado TypeScript)

- **Fecha**: 2026-10-18
- **Entorno de medición**: contenedor en la nube, Node 24.22.0, npm 12.2.1. **No es tu PC**: repetir allí.
- **Motor**: commit `42636d8a9f8001f999cb0ce1eabed7359b5e9ddb`.

## Qué se probó
`bridge/tests/env.test.ts` (8 tests, todos pasan):

| Test | Resultado |
|---|---|
| Misma semilla y mismas acciones → mismo hash, pasos, recompensa y observación final | pasa |
| Semillas distintas → hash distinto | pasa |
| Contrato de observación (tamaños, finitos, rango [0, 1]) en todos los pasos | pasa |
| La política que respeta la máscara no produce acciones inválidas (3 semillas) | pasa |
| Episodio completo sin errores | pasa |
| `reset` con la misma semilla → misma observación inicial | pasa |
| Acción inválida → se trata como noop y se informa | pasa |
| `step` antes de `reset` lanza error | pasa |

`npm run typecheck` pasa sin errores.

## Velocidad (política aleatoria, un episodio, un proceso)

| Mapa | Bots | Ticks/s | Pasos/s (1 paso = 10 ticks) |
|---|---|---|---|
| Britannia | 20 | 461–501 | 41–45 |
| World | 400 | 171 | 14,9 |

Incluye el coste de construir la observación en cada paso. Comparación con la Fase 0 (sin observaciones): ~490 ticks/s en Britannia y ~190–200 en World, así que la observación cuesta poco. **No medí el coste por separado**, es una comparación entre dos ejecuciones distintas.

## Comportamiento de la política aleatoria
En los 5 episodios ejecutados (4 en Britannia, 1 en World) el agente acabó **eliminado** (recompensa −1,0), entre 1480 y 2070 ticks. Ninguno se truncó ni ganó. Una traza de un episodio mostró que el agente gasta sus tropas con ataques frecuentes (tropas ≈ 0 durante casi todo el episodio), llega a ~0,6 % del mapa y luego lo eliminan. Es coherente con una política sin estrategia; **no verifiqué** con una política escrita a mano que el entorno permita jugar bien.

Consecuencia para la recompensa: con estas políticas casi toda la señal sale del término de territorio y del −1 de eliminación. La victoria (+1) aún no se ha visto en ninguna partida.

## Problemas encontrados
1. **Transpilación de clases del motor.** Sin incluir los paquetes del motor en `bridge/tsconfig.json`, `tsx` transpila con semántica de campos distinta y el motor falla ("Cannot read properties of undefined (reading 'unitInfo')"). Documentado en `bridge/README.md`.
2. **Spawn en el mar.** La primera versión de la política aleatoria elegía celdas de océano y la acción de spawn era inválida. Ahora elige celdas con ≥ 75 % de tierra y sin dueño. La máscara aún no informa de qué celdas son válidas para spawn (mejora pendiente).

## Pendiente
- Medir el coste de la observación por separado y con varios entornos por proceso.
- Probar `World` con partidas largas (aquí solo 3000 ticks).
- Dos o más jugadores controlados por código (self-play).
- Validar que una política razonable escrita a mano puede ganar o crecer (criterio de "el entorno es jugable").
- Fase 2: comunicación con Python y wrapper Gymnasium.
