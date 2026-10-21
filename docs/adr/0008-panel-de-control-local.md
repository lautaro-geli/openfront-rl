# ADR 0008: Panel de control web local para entrenamientos

- **Estado**: aceptada
- **Fecha**: 2026-10-21

## Contexto
Los entrenamientos son largos (horas) y corren en el PC del usuario (Windows, 4 núcleos). Hoy se lanzan a mano desde una consola, no se pueden pausar sin matarlos, y las métricas solo se ven como líneas de texto o leyendo `metrics.jsonl`. El usuario pidió un frontend para ver los procesos en ejecución, lanzarlos, pararlos y cerrarlos, ver los workers, la arquitectura del agente y las metas a cumplir.

## Opciones consideradas
1. **Web local** (servidor Python con la biblioteca estándar + página estática, solo en `127.0.0.1`): sin dependencias de frontend, se abre en el navegador, y el servidor puede lanzar y vigilar procesos. Contra: el usuario tiene que arrancar el servidor.
2. **Panel de solo lectura** (página que lee `runs/`): más simple, pero no lanza ni controla nada.
3. **App de escritorio (Electron)**: ventana propia; mucho empaquetado y no aporta sobre la opción 1.
4. **TensorBoard/Grafana**: gráficos buenos, pero no controlan procesos ni muestran metas o arquitectura propias.

## Decisión
**Opción 1**, con estas reglas:
- **Servidor**: `python -m openfront_rl.dashboard` (`trainer/openfront_rl/dashboard.py`), `http.server` de la biblioteca estándar, escucha **solo en 127.0.0.1**. Una API JSON y una página estática (`trainer/openfront_rl/web/index.html`, JavaScript sin librerías ni CDN, funciona sin internet).
- **Control cooperativo**: el servidor no manda señales al proceso de entrenamiento; escribe `runs/<run>/control.json` (`{"command": "run" | "pause" | "stop"}`) y `train.py` lo lee tras cada actualización. `pause` guarda un checkpoint y espera con los workers inactivos; `stop` guarda un checkpoint y termina limpio. Así funciona igual en Windows y Linux (no hay `SIGSTOP` en Windows) y nunca deja un checkpoint a medias. Además hay un botón **Matar** que termina el árbol de procesos (con `psutil`) para los casos en que el entrenamiento no responde.
- **Estado**: `train.py` escribe `runs/<run>/status.json` tras cada actualización (pid, estado, actualización, pasos, velocidad, y por worker: tick, longitud del episodio y eventos del episodio en curso) y `architecture.json` al arrancar (módulos, formas y parámetros reales del modelo).
- **Lanzar**: la API solo ejecuta `python -m openfront_rl.train` con una configuración de `trainer/configs/` y un conjunto cerrado de argumentos (`name`, `seed`, `workers`, `updates`). No ejecuta comandos arbitrarios.
- **Metas**: `trainer/configs/goals.yaml` define hitos con una métrica de `metrics.jsonl` y un umbral; el servidor los evalúa contra el máximo histórico del run.

## Consecuencias
- Pausar/detener tarda hasta lo que dure una actualización (~10 s con 2 workers).
- Se agrega `psutil` como dependencia (CPU, memoria, árbol de procesos).
- Sin autenticación: seguro mientras escuche en `127.0.0.1`; no exponerlo a la red.
- `train.py` gana dos puntos de integración pequeños (control y estado). El formato de `metrics.jsonl` no cambia.

## Cómo validarlo
- Test: el servidor lanza un entrenamiento corto, ve su estado, lo pausa, lo reanuda y lo detiene, y quedan los checkpoints.
- Test: el servidor rechaza configuraciones fuera de `configs/` y argumentos desconocidos.
- Test: las metas se evalúan bien contra un `metrics.jsonl` de ejemplo.

## Enmienda 2026-10-21: duración en partidas y workers pausables
- **Duración en partidas** (`episodes`), no en actualizaciones: el usuario piensa en "50 partidas", no en bloques de 128 pasos. El run termina al cierre de la actualización en la que se alcanza; las curvas de `lr` y entropía siguen el avance en partidas. Alternativa descartada: cortar la actualización a la mitad al llegar a la partida N (rompe el tamaño fijo del lote de PPO).
- **Pausar un worker** congela su partida y marca sus muestras como no válidas en el buffer, en vez de rellenar con datos falsos o reducir el lote (la forma `[T, N]` se mantiene y no hay que reconstruir nada). La pausa de un worker se aplica entre actualizaciones, como el resto del control cooperativo. Alternativa descartada: cerrar el proceso Node del worker y recrearlo (pierde la partida en curso y tarda varios segundos).
