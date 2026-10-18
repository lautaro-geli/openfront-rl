# Spec: observaciones

> Estado: **borrador**. Los tamaños y campos se fijan tras la Fase 0.

## Formato propuesto
Observación = `dict` con dos partes:

1. **Capas espaciales** (tensor `C × H × W`), un canal por concepto:
   - terreno (tierra, agua, montaña)
   - propietario (propio, aliado, enemigo, nadie)
   - densidad de tropas
   - edificios / estructuras
   - frontera propia (celdas adyacentes a otros jugadores)
2. **Vector global**:
   - oro, tropas totales, número de territorios
   - número de jugadores vivos, proporción del mapa propia
   - tick actual / tiempo restante

## Requisitos
- Valores normalizados (rango documentado por campo).
- Solo información que un jugador humano podría ver (sin trampas).
- Dimensiones fijas por mapa; definir cómo se maneja el tamaño variable de mapas.
- Test de contrato: la observación coincide con `observation_space` en cada paso.

## Preguntas abiertas
- ¿Se recorta o reescala el mapa a una resolución fija?
- ¿Cómo se representan los demás jugadores individuales (ids vs. relación)?
