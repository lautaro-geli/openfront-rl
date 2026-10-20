# bridge

Puente TypeScript entre el motor de OpenFront (`vendor/openfront`) y el entrenador en Python.

**Estado: Fase 1 hecha y Fase 2 en marcha.** `MultiAgentEnv`/`OpenFrontEnv` con `reset`/`step`, observación v2 de 4 ramas, 24 tipos de acción con máscara, políticas aleatoria y heurística, y un servidor que habla con Python (`src/server.ts`, ADR 0002).

Especificación: [`docs/specs/bridge.md`](../docs/specs/bridge.md).

## Requisitos
- Node `>=24.15 <25` y npm `>=12.1 <13` (hay un `.nvmrc` en la raíz del repo con `24.22.0`).
- El submódulo del motor descargado (`git submodule update --init --recursive`) y sus dependencias instaladas:

```bash
cd vendor/openfront && npm ci --ignore-scripts && cd ../../bridge
npm install
```

## Comandos
```bash
npm run typecheck                 # tsc --noEmit
npm test                          # vitest (33 tests, ~2-3 min)
npm run serve                     # servidor del protocolo (lo lanza Python; ver docs/specs/protocol.md)
npm run random -- --bots 20 --seed s --episodes 4            # política aleatoria, Britannia
npm run random -- --policy heuristic --bots 20 --episodes 4  # política escrita a mano
npm run random -- --map world --bots 400 --max-ticks 3000    # mapa grande
```

## Estructura
| Archivo | Qué hace |
|---|---|
| `src/multiEnv.ts` | `MultiAgentEnv`: arma la partida con `GameRunner`, uno o más agentes, `reset`, `step`, recompensa, fin de episodio |
| `src/env.ts` | `OpenFrontEnv`: envoltorio de un solo agente |
| `src/features.ts` | Nombres y tamaños de cada bloque de la observación y de las acciones |
| `src/observation.ts` | Observación v2: grid de 20 canales, tablas y escalares por rama |
| `src/actions.ts` | Traducción acción → intent del motor (24 tipos) |
| `src/mask.ts` | Máscara de acciones ampliada |
| `src/protocol.ts` | Formato binario del manifiesto y del resultado |
| `src/server.ts` | Servidor stdin/stdout para Python |
| `src/randomPolicy.ts` | Política aleatoria que respeta la máscara, con semilla |
| `src/heuristicPolicy.ts` | Política escrita a mano (línea base) |
| `src/mapLoader.ts` | Carga de mapas desde `vendor/openfront/resources/maps` |
| `src/cli/runRandom.ts` | Juega episodios completos y muestra velocidad |
| `tests/env.test.ts` | Determinismo, contrato de observación, máscara, episodio completo, reducción del mapa |
| `tests/multi.test.ts` | Varios agentes en la misma partida |
| `tests/policies.test.ts` | La heurística supera a la aleatoria |
| `tests/observation.test.ts` | Formas, rangos y contenido de la observación v2 |
| `tests/actions.test.ts` | Cada tipo de acción contra el motor real |
| `tests/protocol.test.ts` | El protocolo da los mismos bytes que la API directa |

## Trampa conocida: la configuración de TypeScript
El motor necesita que sus clases se transpilen con `useDefineForClassFields: false` (usa propiedades de parámetro en inicializadores de campo). `tsx` solo aplica las opciones de `tsconfig.json` a los archivos que ese `tsconfig` incluye, así que `bridge/tsconfig.json` incluye `../vendor/openfront/packages/*/src/**/*`. **Si se quita esa línea, el motor falla en la primera partida con "Cannot read properties of undefined (reading 'unitInfo')".**

## Límites actuales
Sin construcción, diplomacia ni ataques anfibios. Ver §8 de la spec. Varios agentes soportados con `MultiAgentEnv`; el spawn simultáneo de varios agentes no se ha estudiado a fondo.
