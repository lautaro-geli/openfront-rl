# ADR 0001: Registrar las decisiones de arquitectura

- **Estado**: aceptada
- **Fecha**: 2026-10-18

## Contexto
El proyecto lo desarrolla en gran parte una IA (Claude Code) a partir de especificaciones. Sin un registro de decisiones, el razonamiento se pierde entre sesiones y se repiten debates.

## Decisión
Cada decisión de arquitectura relevante se documenta como ADR numerado en `docs/adr/`, usando `0000-template.md`.

## Consecuencias
- Claude Code lee los ADRs como contexto en cada sesión.
- Cambiar una decisión implica crear un ADR nuevo que reemplace al anterior, no borrarlo.
