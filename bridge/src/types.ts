import type { Difficulty, GameMapType } from "@openfront/engine-api/game/GameTypes";

export const FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0] as const;
export type Fraction = (typeof FRACTIONS)[number];

export { NUM_CHANNELS, NUM_GLOBALS } from "./features";

export interface EnvConfig {
  map: GameMapType;
  bots: number;
  nations: number | "default" | "disabled";
  difficulty: Difficulty;
  decisionInterval: number;
  maxTicks: number;
  grid: { h: number; w: number };
  /** Muestras por eje y por celda al reducir el mapa (S x S tiles por celda). */
  samples: number;
  maxNeighbors: number;
  rewardTerritoryWeight: number;
  /** Número de jugadores controlados por código en la partida (MultiAgentEnv). */
  agents: number;
}

export type Action =
  | { kind: "noop" }
  | { kind: "spawn"; cx: number; cy: number }
  | { kind: "expand"; fraction: Fraction }
  | { kind: "attack"; slot: number; fraction: Fraction };

export interface ActionMask {
  spawn: boolean;
  expand: boolean;
  attackSlots: boolean[];
}

/** Tabla de filas de tamaño fijo, aplanada por filas. `valid[i]` = 1 si la fila `i` tiene datos. */
export interface Table {
  data: Float32Array;
  valid: Uint8Array;
  rows: number;
  cols: number;
}

export interface Observation {
  /** [NUM_CHANNELS, h, w] aplanado. Canales 0 tierra, 1 propio, 2 otro jugador; el resto en features.ts (GRID_CHANNELS). */
  grid: Float32Array;
  /** Escalares generales (GLOBAL_FEATURES). Los 6 primeros son los de la v1. */
  globals: Float32Array;
  mask: ActionMask;
  tick: number;
  inSpawnPhase: boolean;
  /** Entrada de la rama de economía. Ver docs/specs/observations.md §4. */
  economy: { table: Table; scalars: Float32Array };
  /** Entrada de la rama de defensa y ataque. Ver docs/specs/observations.md §5. */
  military: { units: Table; attacks: Table; scalars: Float32Array };
  /** Entrada de la rama de comparación. Ver docs/specs/observations.md §6. La fila 0 es el agente; las filas 1 a 8 son las ranuras de ataque. */
  comparison: { players: Table; scalars: Float32Array };
}

export interface StepInfo {
  invalidAction: boolean;
  rewardParts: { territory: number; terminal: number };
  won: boolean;
  eliminated: boolean;
  tick: number;
}

export interface StepResult {
  obs: Observation;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: StepInfo;
}

export interface MultiStepResult {
  /** Una observación por agente, relativa a ese agente. */
  obs: Observation[];
  reward: number[];
  /** Por agente: eliminado, o la partida terminó con victoria. */
  terminated: boolean[];
  /** La partida alcanzó maxTicks. Es de toda la partida. */
  truncated: boolean;
  /** La partida terminó: victoria, todos eliminados, o truncada. */
  done: boolean;
  info: StepInfo[];
}
