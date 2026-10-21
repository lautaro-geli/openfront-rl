# openfront-rl

Agente de aprendizaje por refuerzo (RL) que aprende a jugar **OpenFront** (juego RTS de conquista territorial en navegador, repo `openfrontio/OpenFrontIO`).

## Objetivo
Entrenar una política (PyTorch, PPO + self-play) que juegue partidas completas de OpenFront contra bots y contra copias pasadas de sí misma.

## Stack
- **Motor del juego**: OpenFront (TypeScript), como submódulo git en `vendor/openfront`. **SOLO LECTURA.**
- **Bridge** (`bridge/`): TypeScript. Corre el motor en modo headless y expone estado y acciones.
- **Trainer** (`trainer/`): Python 3.11+, PyTorch, Gymnasium. Entorno Gym, PPO y self-play.
- **Comunicación bridge ⇄ trainer**: un proceso Node por partida por stdin/stdout binario (ADR 0002, `docs/specs/protocol.md`).

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
- Tests bridge: `cd bridge && npm test` (36 tests, ~2-3 min)
- Política aleatoria o heurística: `cd bridge && npm run random -- --policy heuristic --bots 20 --seed s --episodes 4`
- Tests trainer: `cd trainer && pytest` (tests contra el motor real, incluida la red y PPO; necesita Node 24, o `OPENFRONT_NODE`)
- Medir velocidad: `cd trainer && PYTHONPATH=. python scripts/bench.py --workers 1 5 --map World --bots 400`
- Entrenar: `cd trainer && python -m openfront_rl.train --config configs/ppo_base.yaml` (crea `runs/<nombre>-<semilla>/` con config, `metrics.jsonl` y checkpoints; `--resume runs/.../latest.pt`)

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
**Fases 0, 1 y 2 hechas; Fase 3 escrita y en pruebas** (falta entrenar en el PC del usuario). `bridge/`: observación v2, 24 acciones con máscara, recompensa por componentes (territorio, ingreso, seguridad, terminal; `docs/specs/reward.md`) y protocolo v2. `trainer/openfront_rl/`: cliente, entorno en paralelo, red por ramas con cabezas autorregresivas (`model.py`), PPO (`ppo.py`) y `train.py` (ADR 0007, aceptada). Siguiente: entrenar contra tribus `Easy` en Britannia, comparar con la heurística y pasar a World. ADRs 0003–0005 siguen en "propuesta".

## Trampa conocida
`bridge/tsconfig.json` DEBE incluir `../vendor/openfront/packages/*/src/**/*`; si no, `tsx` transpila mal las clases del motor y falla en la primera partida (ver `bridge/README.md`).
