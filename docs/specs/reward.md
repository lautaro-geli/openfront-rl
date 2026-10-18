# Spec: recompensa

> Estado: **borrador**.

## Diseño inicial
- **Terminal**: +1 por ganar, −1 por perder o ser eliminado, 0 en empate o truncado.
- **Shaping ligero**: pequeña recompensa proporcional al cambio en la fracción de territorio propio por paso.

```
r_t = w_territorio * (frac_t − frac_{t−1}) + r_terminal
```

## Riesgos conocidos
- Si el shaping pesa demasiado, el agente expande sin ganar (reward hacking).
- Si solo hay recompensa terminal, la señal es muy escasa al principio.

## Requisitos
- Pesos configurables por archivo de configuración.
- Registrar cada componente de la recompensa por separado en las métricas.
- Test: sumas de recompensa reproducibles con una semilla fija.

## Preguntas abiertas
- ¿Se penaliza la pasividad (no-op prolongado)?
- ¿Cómo se define la victoria cuando hay varios jugadores y alianzas?
