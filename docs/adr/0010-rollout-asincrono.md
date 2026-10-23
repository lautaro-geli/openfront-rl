# ADR 0010: Rollout asíncrono (workers sin esperarse entre sí)

- **Estado**: implementada como opción experimental (`rollout_mode: async`; por defecto `sync`); la decisión de adoptarla depende de lo medido en el run de prueba `ppo_prueba` (Day6)
- **Fecha**: 2026-10-22

## Contexto
En la PC del usuario (i5-9400, 5 workers + RTX 3050) una actualización PPO tarda ~7 s: ~5,9 s de rollout y ~1,5 s de actualización (en CPU eran 7,5 s). El rollout es el 80 % del tiempo. `VecGame.step` (`trainer/openfront_rl/vec_env.py`) envía las acciones a los N procesos Node y **espera a todos** antes del siguiente paso: cada paso dura lo que tarda el worker más lento. En una captura del Administrador de tareas el motor más cargado estaba al 15 % de la CPU total (casi un núcleo de 6) y la media de los 5 era ~9 % (~55 % de un núcleo): los workers pasan ~45 % del tiempo esperando. La CPU total no está saturada, así que más núcleos no lo arreglan. El coste del motor depende de la fase de la partida (los workers estaban en los ticks 12.902, 12.902, 7.992, 3.322 y 12.872).

## Medición previa (ya implementada)
Cada rollout registra en `metrics.jsonl` y en el log: `t_net` (red), `t_env` (motor, incluye reinicios) y `worker_util` = Σ(latencia media de los workers) / Σ(latencia del más lento) por paso. `1/worker_util` es el máximo de aceleración que daría que nadie esperase. Cada worker se lee en su propio hilo para medir su latencia real.

## Opciones consideradas
1. **No hacer nada**: simple; el rollout sigue limitado por el más lento.
2. **Más núcleos / otra CPU**: sube las muestras por segundo, no acorta el paso, y la CPU no está saturada.
3. **Cada worker avanza por su cuenta con T pasos fijos**: quita el sincronismo paso a paso, pero un worker en una fase pesada sigue tardando más en sus T pasos (el desequilibrio de fase es persistente), así que gana poco.
4. **Recolectar hasta N·T muestras totales con trayectorias de largo variable por worker** (decisión propuesta): los workers rápidos aportan más pasos y ninguno espera. Se mantiene PPO en política: la política no cambia durante la recolección; solo cambia cómo se reparten las muestras.
5. **Solapar la actualización con el rollout (política desfasada)**: añade datos fuera de política y requiere V-trace o similar. Con la actualización en GPU (~1,5 s de ~7) el beneficio es poco.

## Decisión propuesta
Opción 4:
- La inferencia de la red atiende a los workers que están listos (lote de 1..N), no a los N a la vez.
- El buffer pasa de `[T, N]` a una lista de segmentos por worker. GAE se calcula por segmento con bootstrap del valor del último estado; luego se aplanan todas las muestras para el minibatch.
- Tope por worker (p. ej. 2·T pasos) para limitar el sesgo de la siguiente sección.
- `ppo_update` y la red no cambian.

## Riesgos
- **Sesgo hacia fases baratas**: los pasos baratos (inicio de partida, motor más rápido) quedan sobrerrepresentados frente a los caros (final de partida). Mitigación: el tope por worker y medir la distribución de ticks de las muestras antes y después.
- **Reproducibilidad**: el orden en que llegan las respuestas depende de la CPU; las semillas dejan de dar exactamente las mismas muestras. Se documenta; la regla 7 (guardar semillas y configuración) se mantiene.
- **Complejidad** en `Runner.collect`, estadísticas por worker y pausa de workers (`active`).

## Cómo validarlo
1. Medir `worker_util` en un run real (esta ADR sigue en propuesta hasta tener el dato). Si `worker_util` > 85 %, no se implementa.
2. Tests: con la misma política y semillas, una colección asíncrona y una síncrona producen el mismo conjunto de transiciones por worker (el orden entre workers puede cambiar); GAE por segmento coincide con el de un buffer rectangular cuando todos los largos son T.
3. Comparar pasos/s y la curva de aprendizaje en un run corto (50 partidas) contra el síncrono.

## Implementación (experimental)
- Config: `rollout_mode: sync|async` y `rollout_cap` (tope por worker = `rollout_cap × rollout_steps`, 2 por defecto). Con `async`, `Runner.collect_async` (`trainer/openfront_rl/train.py`) usa un hilo por worker para enviar el paso, recibir el resultado y reiniciar la partida si terminó; el hilo principal atiende en lote a los workers que ya respondieron. Se reúnen `rollout_steps × workers activos` muestras en total.
- `RolloutBuffer.lengths[i]` = pasos de la columna i; `compute_gae(..., use_lengths=True)` hace GAE por columna con bootstrap del valor del último estado de cada worker (test: equivale a buffers independientes).
- Métricas por rollout: `samples`, `t_net` (red), `t_env` (en async: tiempo del hilo principal esperando a los workers) y `worker_util` (tiempo ocupado de los workers / tiempo de pared × workers; en sync, Σ media / Σ máximo por paso).
- Medición en el sandbox (2 workers, 2 núcleos, red en CPU): sin ganancia (la red atiende lotes de 1 y compite por CPU con los motores). No es representativa de la PC del usuario (GPU, 5–6 núcleos); la decisión se toma con el run real.
- El run `ppo_prueba` cambia varias cosas a la vez (3 workers, async, evaluación de 20, `lr_floor` 0,3): sirve para ver si el conjunto funciona, no para atribuir efectos.
