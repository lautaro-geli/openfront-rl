# ADR 0009: Guardar los mejores modelos en git (solo pesos)

- **Estado**: aceptada
- **Fecha**: 2026-10-22

## Contexto
Los checkpoints de entrenamiento viven en `trainer/runs/<run>/` y están fuera de git (`runs/` y `*.pt` en `.gitignore`). Cada uno pesa ~40 MB porque incluye el estado del optimizador, y el único sitio donde existen es la PC del usuario. Si se borra la carpeta, o se quiere reproducir una partida en otra máquina (p. ej. el análisis que hace Claude en la nube), se pierde el mejor agente. Los pesos de la red (3,37 M parámetros) pesan ~13,5 MB.

## Opciones consideradas
1. **No versionar modelos**: repositorio liviano, pero el mejor agente depende de una carpeta local.
2. **Git LFS**: soporta archivos grandes, pero añade dependencia y cuotas de GitHub para algo que son pocos archivos.
3. **Pesos sueltos en `models/` dentro de git**: simple y reproducible. Cada modelo ~13,5 MB (por debajo del aviso de 50 MB de GitHub); el historial crece con cada modelo nuevo.

## Decisión
La opción 3, con reglas para que no crezca sin control:
- Solo se versionan **pesos** (sin optimizador) con `python -m openfront_rl.export`, que escribe `models/<nombre>.pt` y un `models/<nombre>.json` con el origen (run, checkpoint, actualización, partidas), la versión del protocolo, la configuración de entrenamiento y los resultados medidos y una nota.
- Se guardan pocos modelos: uno por escenario del currículo, el mejor medido. Al reemplazar uno se borra el anterior (queda en el historial).
- Un modelo se carga con `openfront_rl.export.load_policy(ruta, manifest)`, que comprueba la versión del protocolo.
- Los checkpoints completos siguen fuera de git: sirven para reanudar entrenamientos, no para archivar.

## Consecuencias
- Cualquier sesión puede reproducir partidas del mejor agente sin pedir archivos a la PC del usuario.
- El repositorio crece ~13,5 MB por modelo nuevo; si hiciera falta guardar muchos, se reconsidera Git LFS.
- Un modelo es válido solo con la versión del protocolo/observación con que se entrenó (`protocol_version` en el sidecar); si cambia la observación o las acciones, los modelos viejos dejan de servir (el `load_policy` avisa).

## Cómo validarlo
Test: exportar un checkpoint, cargarlo en una política nueva y comprobar que produce la misma acción y valor que la original (`trainer/tests/test_export.py`).
