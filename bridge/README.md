# bridge

Puente TypeScript entre el motor de OpenFront (`vendor/openfront`) y el entrenador en Python.

**Estado: Fase 1 (lado TypeScript).** Hay un entorno `OpenFrontEnv` con `reset`/`step`, una observación reducida, 4 acciones y una política aleatoria. La conexión con Python (ADR 0002) es de la Fase 2.

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
npm test                          # vitest (16 tests, ~1-2 min)
npm run random -- --bots 20 --seed s --episodes 4            # política aleatoria, Britannia
npm run random -- --policy heuristic --bots 20 --episodes 4  # política escrita a mano
npm run random -- --map world --bots 400 --max-ticks 3000    # mapa grande
```

## Estructura
| Archivo | Qué hace |
|---|---|
| `src/multiEnv.ts` | `MultiAgentEnv`: arma la partida con `GameRunner`, uno o más agentes, `reset`, `step`, recompensa, fin de episodio |
| `src/env.ts` | `OpenFrontEnv`: envoltorio de un solo agente |
| `src/observation.ts` | Grid `[3, h, w]` por muestreo de tiles + vector global de 6 valores |
| `src/actions.ts` | Máscara de acciones válidas y traducción acción → intent del motor |
| `src/randomPolicy.ts` | Política aleatoria que respeta la máscara, con semilla |
| `src/heuristicPolicy.ts` | Política escrita a mano (línea base) |
| `src/mapLoader.ts` | Carga de mapas desde `vendor/openfront/resources/maps` |
| `src/cli/runRandom.ts` | Juega episodios completos y muestra velocidad |
| `tests/env.test.ts` | Determinismo, contrato de observación, máscara, episodio completo, reducción del mapa |
| `tests/multi.test.ts` | Varios agentes en la misma partida |
| `tests/policies.test.ts` | La heurística supera a la aleatoria |

## Trampa conocida: la configuración de TypeScript
El motor necesita que sus clases se transpilen con `useDefineForClassFields: false` (usa propiedades de parámetro en inicializadores de campo). `tsx` solo aplica las opciones de `tsconfig.json` a los archivos que ese `tsconfig` incluye, así que `bridge/tsconfig.json` incluye `../vendor/openfront/packages/*/src/**/*`. **Si se quita esa línea, el motor falla en la primera partida con "Cannot read properties of undefined (reading 'unitInfo')".**

## Límites actuales
Sin construcción, diplomacia ni ataques anfibios. Ver §8 de la spec. Varios agentes soportados con `MultiAgentEnv`; el spawn simultáneo de varios agentes no se ha estudiado a fondo.
