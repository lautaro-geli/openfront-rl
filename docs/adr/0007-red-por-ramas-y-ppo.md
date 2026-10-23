# ADR 0007: Red por ramas, cabezas autorregresivas y PPO con máscaras

- **Estado**: aceptada
- **Fecha**: 2026-10-21

## Contexto
La observación v2 (ADR 0006) tiene cuatro familias de datos: grid `[20,32,64]`, economía, militar y comparación de jugadores. La acción es un vector `[kind, row, cell, fraction, type, aux]` con 24 tipos (`docs/specs/actions.md`) y una máscara por bloque (`kind`, `player`, `unit`, `attack_row`, `build_type`, `nuke_type`, `cell`). Un producto cartesiano `24 × 2048 × ...` es inviable como una sola salida. La Fase 3 pide PPO contra bots simples (`docs/architecture.md`). El entrenamiento largo será en un i5-9400 (6 núcleos, probablemente sin GPU), así que el modelo debe ser pequeño.

## Opciones consideradas
1. **Una sola cabeza plana** sobre el producto de acciones: no escala (2048 celdas × 24 tipos) y ignora la estructura de la máscara.
2. **Cabezas independientes** (cada campo se muestrea sin condicionarse a `kind`): simple, pero `cell` y `row` no saben para qué acción se piden, y la máscara correcta depende de `kind`.
3. **Cabezas autorregresivas** (elegir `kind`, luego los parámetros que ese `kind` usa, condicionados al `kind` elegido): la máscara de cada campo se selecciona según el `kind`; solo se paga log-prob de los campos usados.

## Decisión
**Opción 3**, con esta estructura:

**Red** (`trainer/openfront_rl/model.py`):
- Rama *mapa*: CNN pequeña sobre el grid (3 conv 3×3 con stride, GELU, 32→64→64 canales) → aplanado → lineal a 256. También se produce un mapa de características `[C, 8, 16]` para los *logits de celda*.
- Rama *economía*: MLP por fila de la tabla (9→32), *masked mean + max pooling* sobre filas válidas, concatenado con los 38 escalares → 128.
- Rama *defensa/ataque*: igual con unidades `[96,30]` y ataques `[16,10]` + 23 escalares → 128.
- Rama *comparación*: MLP por fila (46→64), concatena las filas ordenadas (16×64) y pasa por MLP; + 14 escalares + 14 globales → 128.
- *Núcleo*: concatenación de ramas → MLP de 2 capas (512) con LayerNorm → vector de estado `h`.
- Las ramas son módulos independientes (se pueden apagar o sustituir).

**Política** (cada campo con su máscara, valores enmascarados a −1e9 antes del softmax):
1. `kind ~ softmax(W_k h)` con la máscara `kind`.
2. Se calcula un embedding `e_k = Emb(kind)` y `g = MLP([h, e_k])`.
3. Campos condicionados a `kind` (solo los que ese tipo usa, ver tabla en `docs/specs/actions.md`):
   - `row` de jugador: logits por fila = `⟨q(g), K(fila_jugador)⟩` (atención sobre las filas de la tabla de comparación), máscara `player[PLAYER_KINDS.index(kind)]`; para `attack` la fila es la ranura de vecino.
   - `row` de unidad: igual sobre la tabla de unidades (economía + militar, 144 filas), máscara `unit[...]`.
   - `row` de ataque saliente: logits sobre las 8 filas con `attack_row`.
   - `cell`: logits por celda = producto escalar entre `g` y las características del mapa de esa celda (más un sesgo por kind), máscara `cell[CELL_KINDS.index(...)]`.
   - `fraction`: 5 logits.
   - `type`: 7 (build) o 3 (nuke), con `build_type` / `nuke_type`.
   - `aux`: 8 emojis.
4. `log π(a|s) = Σ log π(campo)` solo de los campos usados por `kind`; entropía igual. Los campos no usados quedan en 0 en el vector enviado.
5. *Value head*: MLP de `h` → escalar.

**PPO** (`trainer/openfront_rl/ppo.py`): GAE(γ=0.997, λ=0.95), clip 0.2, minibatches, 4 épocas, normalización de ventajas, clip de gradiente 0.5, entropía con coeficiente decreciente, Adam. Las máscaras de cada paso se guardan en el buffer y se reaplican al recalcular log-probs (no hay acciones fuera de máscara). Normalización de recompensa por desviación del retorno acumulado. `truncated` hace *bootstrap* con el valor del último estado; `terminated` no.

**Almacenamiento**: la observación son ~182 KB por paso por agente. Un rollout de 5 partidas × 128 pasos = 640 pasos ≈ 117 MB en float32; se aceptan. Se guarda el grid en float16 en el buffer.

**Configuración**: `configs/v1_ppo_base.yaml`; cada ejecución guarda la config, las semillas y la versión del manifiesto junto con el checkpoint (regla 7).

## Consecuencias
- Simple de enmascarar y de depurar: cada campo tiene su propia distribución.
- Los gradientes de campos no usados no existen (log-prob 0), así que `kind` raro (p. ej. `nuke`) recibe pocas señales; se compensa con la entropía y el currículo.
- Si PPO plano se estanca, esta cabeza de `kind` es el punto natural para sustituir por una jerarquía (opciones).
- Velocidad: el CNN y los MLP caben en CPU; el cuello de botella seguirá siendo el motor (40 pasos/s por proceso).

## Cómo validarlo
- Test: ninguna acción muestreada viola la máscara (10 000 muestras aleatorias, estados reales).
- Test: `log_prob` recalculado al actualizar es igual al del muestreo (mismo estado, mismos pesos).
- Test: un paso de PPO baja la pérdida sobre un lote fijo.
- Humo: 10 000 pasos de entrenamiento sin NaN y con el porcentaje de acciones inválidas bajo.
- Criterio de la fase: la curva de recompensa sube contra tribus `Easy`, y supera a la heurística en territorio final.
