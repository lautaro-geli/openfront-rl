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
- ~~Dos o más jugadores controlados por código (self-play)~~ Hecho, ver adenda.
- ~~Validar que una política razonable escrita a mano puede crecer (el entorno es jugable)~~ Hecho, ver adenda (aún sin victorias).
- Fase 2: comunicación con Python y wrapper Gymnasium.

---

# Adenda 2026-10-19: varios agentes, reducción del mapa y política heurística

Medido en un contenedor nuevo, **más lento que el de la primera sesión** (el motor solo, con `FullGamePerf.ts`, dio 84 ticks/s frente a 191–201 antes, con el **mismo hash final** `26347989348602096`). Las velocidades de esta adenda y las de arriba **no son comparables entre sí**.

## 1. Varios agentes en la misma partida
`MultiAgentEnv` (con `OpenFrontEnv` como envoltorio de un agente). Tests en `bridge/tests/multi.test.ts` (4, pasan):
- Dos agentes aparecen en la partida. Con el agente 0 sin acciones y el agente 1 expandiéndose, el territorio del 0 no cambia y el del 1 crece: cada intent llega a su jugador.
- Cada observación es relativa a su agente: lo que uno ve como "propio" es lo que el otro ve como "otro".
- Misma semilla y misma secuencia de acciones con 2 agentes y 10 bots dan el mismo resultado.
- `OpenFrontEnv` rechaza `agents > 1`.

Ejecuciones sueltas con política heurística en cada agente (3000 ticks):

| Mapa | Bots | Agentes | Resultado | Ticks/s |
|---|---|---|---|---|
| Britannia | 20 | 2 | ambos sobreviven (83 726 y 116 230 tiles) | 194 |
| Britannia | 20 | 4 | 2 eliminados; sobreviven 41 384 y 63 422 tiles | 199 |
| World | 400 | 2 | 1 eliminado; el otro 40 210 tiles | 84 |

No investigué por qué se eliminó a esos agentes. La heurística elige el mismo "mejor sitio" para todos, así que los spawns pueden quedar muy cerca.

## 2. Reducción del mapa
Ver `docs/adr/0005-reduccion-de-observaciones.md`. Resumen: el muestreo por celdas con S = 4 estima la masa del territorio de los jugadores grandes con error de hasta el 5 %; 32 × 64 cuesta 0,63 ms por observación y 64 × 128 cuesta 1,53 ms. Los territorios de 1 tile no se ven. El `miniGameMap` del motor no sirve para propiedad.

Hallazgo de rendimiento: la **máscara de acciones** costaba 3–5 ms por agente (recorre todos los jugadores vivos buscando vecinos), más que construir la observación. Ahora se calcula una vez por paso y se reutiliza (`MaskInfo`). El resultado del juego no cambió: el hash de una partida de referencia (`w-0`, World, 400 bots, política aleatoria) es idéntico antes y después, `30948628230160724@1480`.

Otro hallazgo: Britannia es vertical (1600 × 2088) y World apaisado (2000 × 1000). `gridForMap` elige una rejilla con celdas casi cuadradas.

## 3. Política escrita a mano (`HeuristicPolicy`)
Reglas, usando solo la observación: spawn en la celda de tierra libre con más tierra alrededor y sin jugadores; expandirse mientras la frontera toque tierra libre; si no hay, atacar al vecino válido más pequeño; no gastar tropas por debajo del 40 % del máximo; enviar el 25 % de las tropas.

Ajuste de 6 variantes de parámetros en 3 semillas (`d0`–`d2`, 4000 ticks). Evaluación en **10 semillas distintas** (`e0`–`e9`) con los parámetros por defecto, sin volver a ajustar. Britannia, 20 bots, 6000 ticks de juego tras el spawn:

| Política | Territorio máximo medio | Territorio final medio | Sobreviven | Victorias |
|---|---|---|---|---|
| Aleatoria | 0,44 % | 0,00 % | 0 / 10 | 0 |
| Heurística | 10,40 % | 6,80 % | 6 / 10 | 0 |

Por semilla, la heurística acabó con: 0, 17,2, 17,4, 8,6, 4,0, 4,3, 16,5, 0, 0 y 0 % (los ceros son eliminaciones).

Conclusión: **el entorno es jugable**, porque una política simple lo supera con claridad y la aleatoria nunca sobrevive. Pero la heurística no es buena: la eliminan 4 de cada 10 veces y nunca gana. Es una línea base, no un listón alto.

Límites:
- Solo Britannia con 20 bots. En **World con 400 bots**, una ejecución (semilla `w-0`) la eliminó antes del tick 2300, igual que a la aleatoria, así que la heurística no generaliza a mapas muy poblados. Es un único episodio.
- Los parámetros se ajustaron con pocas semillas; la evaluación en `e0`–`e9` es independiente de ese ajuste, pero sigue siendo una muestra pequeña.
- Ninguna política ganó nunca (>80 % de la tierra). La señal de victoria sigue sin observarse.

## Pendiente
- Medir en tu PC (las cifras cambian mucho de una máquina a otra).
- Decidir qué hacer con mapas de distinta forma al entrenar (ADR 0005, "Consecuencias").
- Investigar las eliminaciones de la heurística y el spawn de varios agentes.
- Fase 2: comunicación con Python.
