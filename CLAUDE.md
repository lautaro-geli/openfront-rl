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
- Tests bridge: `cd bridge && npm test` (47 tests, ~5 min)
- Política aleatoria o heurística: `cd bridge && npm run random -- --policy heuristic --bots 20 --seed s --episodes 4`
- Tests trainer: `cd trainer && pytest` (tests contra el motor real, incluida la red y PPO; necesita Node 24, o `OPENFRONT_NODE`)
- Medir velocidad: `cd trainer && PYTHONPATH=. python scripts/bench.py --workers 1 5 --map World --bots 400`
- Panel de control: `cd trainer && python -m openfront_rl.dashboard` → http://127.0.0.1:8765 (lanza, pausa, reanuda, detiene y mata entrenamientos; métricas, workers, arquitectura y metas; ADR 0008, `docs/specs/dashboard.md`)
- Entrenar: `cd trainer && python -m openfront_rl.train --config configs/ppo_base.yaml` (crea `runs/<nombre>-<semilla>/` con config, `metrics.jsonl` y checkpoints; `--resume runs/.../latest.pt`; `--episodes 50` fija la duración en partidas, `--workers N`, `--env bots=30 --env nations=disabled --env difficulty=Easy --env map=World` cambia el escenario; un run nuevo en una carpeta con datos archiva lo anterior en `anterior-<fecha>/`)
- Exportar un modelo (solo pesos, ADR 0009): `cd trainer && python -m openfront_rl.export --ckpt runs/<run>/latest.pt --out ../models/<nombre> --results "…" --note "…"`; cargar con `openfront_rl.export.load_policy`. Modelos guardados en `models/` (uno por escenario del currículo: `tribus-day2`, `nacion1-day3-u175`).

## Convenciones
- Python: tipado con type hints, `ruff` + `black`, `pytest`.
- TypeScript: `strict: true`, `eslint` + `prettier`.
- Commits pequeños y descriptivos (Conventional Commits).
- Documentación y comentarios en español; nombres de código en inglés.

## Mapa de documentación
- `docs/architecture.md`: visión general, diagramas y fases.
- `docs/adr/`: decisiones de arquitectura (0009: modelos en git).
- `docs/specs/`: observaciones, acciones, recompensa, protocolo, panel (`dashboard.md`).
- `docs/phase0-investigation.md`: prompt de la primera sesión (investigación del motor).

## Entorno
- El motor exige **Node >=24.15 <25 y npm >=12.1 <13** (`vendor/openfront/package.json`). Con Node 22 `npm ci` falla.
- Instalar el motor con `npm ci --ignore-scripts` dentro de `vendor/openfront` (no `npm install`). `node_modules` queda ignorado por git.

## Estado actual
**Fases 0, 1 y 2 hechas; Fase 3 escrita y en pruebas** (falta entrenar en el PC del usuario). `bridge/`: observación v2, 24 acciones con máscara, recompensa v2.2 por categorías con presupuesto (terminal > territorio por tramos > ataque > economía > defensa > hitos de construcción > tropas en la zona justa; `docs/specs/reward.md`), registro de eventos (ataques, tribus conquistadas, misiles, construcciones realmente colocadas por tipo) y protocolo v3. `trainer/openfront_rl/`: cliente, entorno en paralelo, red por ramas con cabezas autorregresivas (`model.py`), PPO (`ppo.py`) `train.py` (ADR 0007, aceptada) y el panel web local `dashboard.py` (ADR 0008). El entrenamiento base desactiva las acciones diplomáticas (`disabledActions`) y paga un bonus decreciente por tribu conquistada (`conquest`, reward.md §9). Siguiente: entrenar contra tribus `Easy` en Britannia, comparar con la heurística y pasar a World. ADRs 0003–0005 siguen en "propuesta".

## Trampa conocida
`bridge/tsconfig.json` DEBE incluir `../vendor/openfront/packages/*/src/**/*`; si no, `tsx` transpila mal las clases del motor y falla en la primera partida (ver `bridge/README.md`).
