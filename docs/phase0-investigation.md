# Fase 0: investigación del motor (prompt para Claude Code)

Pega esto en tu primera sesión de Claude Code, **en modo plan** (Shift+Tab hasta "plan mode"), con el submódulo `vendor/openfront` ya clonado.

---

Lee `CLAUDE.md` y `docs/architecture.md`. Luego investiga `vendor/openfront` SIN modificar nada y responde con evidencia (archivo y línea):

1. **Separación motor/cliente**: ¿qué parte del código es la simulación pura y qué parte depende del navegador (renderizado, DOM, WebSocket)? ¿Hay un servidor autoritativo que ejecute la lógica?
2. **Headless**: ¿se puede ejecutar una partida completa en Node sin navegador? ¿Qué habría que envolver desde `bridge/`?
3. **Velocidad**: ¿hay un tick fijo o ritmo atado al reloj real? ¿Se puede simular más rápido que tiempo real? Propón cómo medir ticks/segundo.
4. **Estado**: ¿qué estructura expone el estado del juego (territorio, tropas, jugadores, edificios) y cómo se lee de forma eficiente?
5. **Acciones**: lista todos los intents/acciones que un jugador puede enviar, con sus parámetros y condiciones de validez.
6. **Bots/IA existentes**: ¿qué bots trae el juego? ¿Sirven como oponentes de entrenamiento?
7. **Determinismo**: ¿se pueden fijar semillas para reproducir partidas?

Entrega:
- Un informe en `docs/research/engine-findings.md`.
- Un borrador de actualización de `docs/architecture.md`.
- Propuestas de ADR (0003 en adelante) con opciones y recomendación.
- Una lista de riesgos y de cosas que NO pudiste confirmar.

No escribas código todavía.
