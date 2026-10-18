# openfront-rl

Agente de aprendizaje por refuerzo (RL) que aprende a jugar **OpenFront** (juego RTS de conquista territorial en navegador, repo `openfrontio/OpenFrontIO`).

## Objetivo
Entrenar una política (PyTorch, PPO + self-play) que juegue partidas completas de OpenFront contra bots y contra copias pasadas de sí misma.

## Stack
- **Motor del juego**: OpenFront (TypeScript), como submódulo git en `vendor/openfront`. **SOLO LECTURA.**
- **Bridge** (`bridge/`): TypeScript. Corre el motor en modo headless y expone estado y acciones.
- **Trainer** (`trainer/`): Python 3.11+, PyTorch, Gymnasium. Entorno Gym, PPO y self-play.
- **Comunicación bridge ⇄ trainer**: pendiente de decidir (ver `docs/adr/`).

## Reglas (obligatorias)
1. **Nunca modificar `vendor/`.** Si el motor necesita cambios, se envuelve desde `bridge/` o se documenta en un ADR.
2. **Specs antes que código.** Cada componente tiene su spec en `docs/specs/`. Si el código y la spec difieren, se actualiza primero la spec.
3. **Decisiones de arquitectura = ADR** en `docs/adr/` (usar `0000-template.md`). No decidir en silencio.
4. **Tests antes de dar algo por terminado.** Nada se marca como hecho sin tests que pasen.
5. **Fases pequeñas.** Trabajar una fase de `docs/architecture.md` a la vez. Usar modo plan antes de implementar.
6. **No inventar la API del motor.** Leer el código de `vendor/openfront` y citar archivo y línea en las specs y ADRs.
7. **Reproducibilidad.** Toda ejecución de entrenamiento fija semillas y guarda su configuración junto al checkpoint.
8. **Uso responsable.** Entrenar solo en local/headless. No conectar bots a partidas públicas de OpenFront.

## Comandos
> Se completarán cuando existan (actualizar este bloque en cada fase).
- Typecheck bridge: `cd bridge && npm run typecheck`
- Tests bridge: `cd bridge && npm test` (8 tests, ~45 s)
- Política aleatoria: `cd bridge && npm run random -- --bots 20 --seed s --episodes 4`
- Tests trainer: `cd trainer && pytest`
- Entrenar: `cd trainer && python -m openfront_rl.train --config configs/ppo_base.yaml`

## Convenciones
- Python: tipado con type hints, `ruff` + `black`, `pytest`.
- TypeScript: `strict: true`, `eslint` + `prettier`.
- Commits pequeños y descriptivos (Conventional Commits).
- Documentación y comentarios en español; nombres de código en inglés.

## Mapa de documentación
- `docs/architecture.md`: visión general, diagramas y fases.
- `docs/adr/`: decisiones de arquitectura.
- `docs/specs/`: observaciones, acciones, recompensa.
- `docs/phase0-investigation.md`: prompt de la primera sesión (investigación del motor).

## Entorno
- El motor exige **Node >=24.15 <25 y npm >=12.1 <13** (`vendor/openfront/package.json`). Con Node 22 `npm ci` falla.
- Instalar el motor con `npm ci --ignore-scripts` dentro de `vendor/openfront` (no `npm install`). `node_modules` queda ignorado por git.

## Estado actual
**Fase 0 completada** (`docs/research/engine-findings.md`). **Fase 1, lado TypeScript hecho** (`bridge/`, `docs/specs/bridge.md`, `docs/research/phase1-results.md`). Siguiente: Fase 2, comunicación con Python y wrapper Gymnasium. ADRs 0002–0004 siguen en "propuesta".

## Trampa conocida
`bridge/tsconfig.json` DEBE incluir `../vendor/openfront/packages/*/src/**/*`; si no, `tsx` transpila mal las clases del motor y falla en la primera partida (ver `bridge/README.md`).
