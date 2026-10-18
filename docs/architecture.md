# Arquitectura de openfront-rl

> Estado: **borrador inicial (Fase 0)**. Las partes marcadas con ⚠️ dependen de la investigación del motor y se confirmarán con ADRs.

## 1. Visión general

```mermaid
flowchart LR
  subgraph Node["Node.js (TypeScript)"]
    E["Motor OpenFront headless<br/>(vendor/openfront)"] --> B["Bridge<br/>estado ⇄ acciones"]
  end
  B <-->|"⚠️ IPC / sockets"| G["Python: Gym env<br/>+ action masking"]
  G --> T["Trainer PPO<br/>+ self-play"]
  T --> C[("Checkpoints")]
  C --> L["Liga de oponentes"]
  L --> G
  T --> M["Métricas / evaluación"]
```

## 2. Bucle de interacción

```mermaid
sequenceDiagram
  participant T as Trainer (Python)
  participant G as Gym Env
  participant B as Bridge (TS)
  participant E as Motor OpenFront
  T->>G: step(acción)
  G->>B: acción (máscara validada)
  B->>E: intent(s) del jugador
  E-->>B: tick(s) simulados
  B-->>G: observación + eventos
  G-->>T: obs, recompensa, terminated, truncated, info
```

## 3. Componentes

| Componente | Lenguaje | Responsabilidad | Spec |
|---|---|---|---|
| Motor | TS | Simulación (no se toca) | n/a |
| Bridge | TS | Headless, serializar estado, aplicar acciones | ⚠️ pendiente |
| Gym env | Python | Observaciones, acciones, máscaras, recompensa | `specs/observations.md`, `specs/actions.md`, `specs/reward.md` |
| Trainer | Python | PPO, rollouts paralelos, checkpoints | ⚠️ pendiente |
| Liga | Python | Gestión de oponentes para self-play | ⚠️ pendiente |

## 4. Puntos de decisión abiertos (ADRs)

1. **Headless y velocidad**: ¿el motor corre sin cliente y cuántos ticks/segundo alcanza?
2. **Puente TS ⇄ Python**: sockets, memoria compartida, o política exportada a ONNX.
3. **Representación de observaciones**: capas tipo imagen + vector global.
4. **Espacio de acciones**: jerárquico con máscaras.
5. **Recompensa**: victoria + término ligero de territorio.
6. **Algoritmo**: PPO y diseño de self-play.

## 5. Fases

```mermaid
flowchart TD
  F0["Fase 0<br/>Investigar motor"] --> F1["Fase 1<br/>Entorno headless<br/>con bots aleatorios"]
  F1 --> F2["Fase 2<br/>Wrapper Gym + tests"]
  F2 --> F3["Fase 3<br/>PPO vs bots simples"]
  F3 --> F4["Fase 4<br/>Self-play y liga"]
  F4 --> F5["Fase 5<br/>Evaluación<br/>(+ capa Claude opcional)"]
```

| Fase | Entregable | Criterio de salida |
|---|---|---|
| 0 | Informe de investigación + ADRs 0002-0003 | Se sabe si es viable headless y a qué velocidad |
| 1 | Partida completa headless con bots aleatorios | Reproducible con semilla |
| 2 | `OpenFrontEnv` (Gymnasium) | Tests de contrato pasan |
| 3 | Agente PPO | Curva de recompensa sube frente a bots simples |
| 4 | Self-play y liga | Gana de forma consistente a checkpoints anteriores |
| 5 | Evaluación y reportes | Métricas reproducibles |

## 6. Riesgos

- **Velocidad de simulación insuficiente** → paralelizar instancias o simplificar el mapa.
- **Motor acoplado al cliente** → extraer lo mínimo desde `bridge/` sin modificar `vendor/`.
- **Recompensa explotable** → mantenerla simple y vigilar comportamientos degenerados.
- **Espacio de acciones demasiado grande** → jerarquía y máscaras.
