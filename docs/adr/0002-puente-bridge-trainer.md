# ADR 0002: Mecanismo de comunicación bridge ⇄ trainer

- **Estado**: propuesta (pendiente de la Fase 0)
- **Fecha**: 2026-10-18

## Contexto
El motor está en TypeScript y el entrenamiento irá en Python. La velocidad de este canal puede ser el cuello de botella del entrenamiento.

## Opciones consideradas
1. **Sockets / IPC local** (p. ej. ZeroMQ o WebSocket) entre un proceso Node y Python. Simple y desacoplado; la serialización cuesta si el estado es grande.
2. **Memoria compartida** para el estado en rejilla. Muy rápido; más complejo de implementar y depurar.
3. **Entrenar del lado TS** exportando/importando política vía ONNX. Evita IPC en el bucle, pero el ecosistema de RL en TS es pobre.

## Decisión
Pendiente. Criterio: elegir la opción más simple que alcance la velocidad objetivo medida en la Fase 0.

## Cómo validarlo
Medir pasos/segundo con una política aleatoria usando la opción 1. Si no basta, pasar a la 2.
