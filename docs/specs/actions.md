# Spec: acciones

> Estado: **borrador**. El catálogo real de acciones se obtiene del motor en la Fase 0.

## Espacio jerárquico propuesto
1. **Tipo de acción** (p. ej. atacar, expandirse, construir, lanzar barco, lanzar bomba, diplomacia, no-op).
2. **Objetivo** (jugador o celda, según el tipo).
3. **Proporción de tropas** (valores discretos, p. ej. 10/25/50/75/100 %).

## Requisitos
- **Máscara de acciones inválidas** en cada paso (sin oro, sin objetivo válido, etc.).
- Toda acción debe mapearse a un intent real del motor; citar archivo y línea.
- Frecuencia de decisión (cada N ticks) configurable y documentada.
- Test: ninguna acción enmascarada llega al motor.

## Preguntas abiertas
- ¿Qué acciones se incluyen en la primera versión? (propuesta: atacar y expandirse)
- ¿Cómo se manejan acciones con efecto retardado?
