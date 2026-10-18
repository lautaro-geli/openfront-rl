# openfront-rl

Agente de aprendizaje por refuerzo para jugar [OpenFront](https://github.com/openfrontio/OpenFrontIO).

> Estado: **Fase 0** (investigación del motor). Aún no hay código.

## Puesta en marcha

```bash
git init
git submodule add https://github.com/openfrontio/OpenFrontIO vendor/openfront
git submodule update --init --recursive
```

Si el repositorio de OpenFront está en otra URL, usa esa en el comando `submodule add`.

Después abre Claude Code en la raíz del proyecto y sigue `docs/phase0-investigation.md`.

## Estructura

| Ruta | Contenido |
|---|---|
| `CLAUDE.md` | Reglas y contexto para Claude Code |
| `docs/architecture.md` | Arquitectura, diagramas y fases |
| `docs/adr/` | Decisiones de arquitectura |
| `docs/specs/` | Observaciones, acciones, recompensa |
| `.claude/` | Permisos y subagentes |
| `vendor/openfront/` | Motor del juego (solo lectura) |
| `bridge/` | Puente TypeScript con el motor |
| `trainer/` | Entorno Gym y entrenamiento en Python |
