# 0006. Observación v2: entradas por rama

- Estado: propuesta
- Fecha: 2026-10-19

## Contexto
La red tendrá ramas (mapa, economía, defensa/ataque, comparación) que alimentan un núcleo común (ver la sección "Anatomía del agente"). La v1 solo tenía 3 canales y 6 escalares. Se entrena en un solo mapa (World, rejilla 32×64, S=4).

## Decisión
La observación pasa a: grid de 20 canales, tabla de economía, tablas de unidades y ataques, tabla de jugadores y escalares por rama (`docs/specs/observations.md`). Solo información pública (el motor ya la envía a todos los clientes). Tamaño fijo con máscaras `valid`. Único estado entre pasos: el rastreador de tasas de ingreso. Los canales 0-2 y `globals[0..1]` conservan su índice para no romper las políticas heurística y aleatoria.

## Consecuencias
- Coste medido: 6 ms (Britannia, 20 bots), 7 ms (World, 20 bots), 15 ms (World, 400 bots) por observación de un agente.
- La observación pesa más; la comunicación con Python (Fase 2) deberá enviarla en bloques contiguos.
- Normalizaciones como /20 para trenes y barcos comerciales son escalas mías, no reglas del motor.
- Pendiente: acciones de construir, aliarse y misiles; hasta entonces las ramas de economía y defensa solo observan.
