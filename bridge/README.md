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
npm test                          # vitest (8 tests, ~45 s)
npm run random -- --bots 20 --seed s --episodes 4            # política aleatoria, Britannia
npm run random -- --map world --bots 400 --max-ticks 3000    # mapa grande
```

## Estructura
| Archivo | Qué hace |
|---|---|
| `src/env.ts` | `OpenFrontEnv`: arma la partida con `GameRunner`, `reset`, `step`, recompensa, fin de episodio |
| `src/observation.ts` | Grid `[3, h, w]` por muestreo de tiles + vector global de 6 valores |
| `src/actions.ts` | Máscara de acciones válidas y traducción acción → intent del motor |
| `src/randomPolicy.ts` | Política aleatoria que respeta la máscara, con semilla |
| `src/mapLoader.ts` | Carga de mapas desde `vendor/openfront/resources/maps` |
| `src/cli/runRandom.ts` | Juega episodios completos y muestra velocidad |
| `tests/env.test.ts` | Determinismo, contrato de observación, máscara, episodio completo |

## Trampa conocida: la configuración de TypeScript
El motor necesita que sus clases se transpilen con `useDefineForClassFields: false` (usa propiedades de parámetro en inicializadores de campo). `tsx` solo aplica las opciones de `tsconfig.json` a los archivos que ese `tsconfig` incluye, así que `bridge/tsconfig.json` incluye `../vendor/openfront/packages/*/src/**/*`. **Si se quita esa línea, el motor falla en la primera partida con "Cannot read properties of undefined (reading 'unitInfo')".**

## Límites actuales
Un solo jugador controlado, sin construcción, diplomacia ni ataques anfibios. Ver §8 de la spec.
