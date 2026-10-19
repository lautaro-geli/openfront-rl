# ADR 0005: Cómo reducir el mapa para las observaciones

- **Estado**: propuesta (con mediciones de un solo estado de partida; ver "Límites")
- **Fecha**: 2026-10-19

## Contexto
- `World` mide 2000 × 1000 = 2 millones de tiles (`resources/maps/world/manifest.json`). `Britannia` mide 1600 × 2088 (vertical, no apaisado).
- El agente necesita una observación pequeña y de tamaño fijo. Una red convolucional razonable trabaja con unas pocas miles de celdas.
- El motor trae mapas reducidos (`map4x`: mitad de ancho y alto; `map16x`: un cuarto). El motor los pasa como `miniGameMap` y los usa en `PathFinder.ts`. **No encontré en `GameImpl.ts` ninguna escritura de propiedad sobre ese mapa**, así que no sirve para saber quién posee cada tile. No lo probé ejecutando.
- La propiedad de cada tile solo está al día en el mapa principal (`ownerID(ref)`, `GameMap.ts`).

## Opciones consideradas
1. **Muestreo por celdas** (implementado): cada celda del grid cubre un bloque del mapa y se estima con una malla de S × S tiles. Coste fijo y predecible, sin depender del tamaño del mapa.
2. **`miniGameMap`**: descartada, por lo dicho arriba.
3. **Contadores exactos incrementales** usando `packedTileUpdates` de cada tick: daría fracciones exactas sin muestreo y coste proporcional a los cambios. **No lo probé.** Posible optimización futura si el muestreo se queda corto.

## Datos (opción 1)
Medidos en World, 400 bots, tick 1502, 121 jugadores vivos, comparando contra el conteo exacto de todos los tiles. "Recall" = de las celdas donde el jugador tiene al menos 1 tile, cuántas ve el muestreo; las que falla son franjas de borde con pocos tiles. "Masa" = total estimado de tiles del jugador / total real.

| Rejilla | S | Coste de la observación completa | Mayor (18 472 tiles): recall / masa | Mediano (2 042 tiles): recall / masa |
|---|---|---|---|---|
| 32 × 64 | 2 | 0,15 ms | 70 % / 104 % | 44 % / 95 % |
| 32 × 64 | 4 | 0,63 ms | 84 % / 100 % | 67 % / 101 % |
| 64 × 128 | 2 | 0,55 ms | 79 % / 100 % | 65 % / 102 % |
| 64 × 128 | 4 | 1,53 ms | 90 % / 99 % | 85 % / 98 % |
| 128 × 256 | 2 | 1,87 ms | 91 % / 100 % | 81 % / 98 % |
| 128 × 256 | 4 | 4,48 ms | 97 % / 100 % | 96 % / 100 % |

- El error medio por celda (MAE) fue ≤ 1,5·10⁻³ en todas las configuraciones.
- Un jugador con 1 tile no es visible con ninguna configuración con S ≤ 4 (esperable).
- A S = 8 el coste sube mucho (6–66 ms por canal en las pruebas de error) sin mejora clara frente a S = 4.
- Las mediciones de coste son de una máquina concreta; en otra serán distintas.

## Decisión (propuesta)
- Mantener el muestreo por celdas. Valores por defecto: rejilla 32 × 64 con S = 4 (coste bajo; la masa estimada del jugador mayor quedó entre el 100 % y el 105 % de la real en dos mediciones, ticks 602 y 1502).
- Subir a 64 × 128 con S = 4 cuando importen los territorios pequeños o los frentes: sigue costando menos de 2 ms.
- Elegir la rejilla según la forma del mapa con `gridForMap(ancho, alto)` para que las celdas sean casi cuadradas: World → 32 × 64, Britannia → 52 × 40. Con una rejilla fija, un mapa vertical queda deformado.
- No usar `miniGameMap` en observaciones.

## Consecuencias
- Un agente entrenado en una forma de rejilla no se puede usar tal cual en un mapa de otra forma. Entrenar con varios mapas exige un tamaño de rejilla único (con relleno) o redes por mapa. **Pendiente de decidir.**
- El muestreo no ve territorios muy pequeños; su pérdida es pequeña en la masa total, pero afecta a la detección de enemigos diminutos.

## Límites de estas mediciones
Un solo estado de partida (una semilla, un tick) y tres jugadores de referencia. No se probó en otros mapas ni en etapas distintas de la partida. Hay que repetirlo antes de fijar el valor definitivo.

## Cómo validarlo
Entrenar con las dos resoluciones (32 × 64 y 64 × 128) y comparar la curva de aprendizaje (Fase 3).
