import type { Difficulty, GameMapType } from "@openfront/engine-api/game/GameTypes";

export const FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0] as const;
export type Fraction = (typeof FRACTIONS)[number];

export const NUM_CHANNELS = 3;
export const NUM_GLOBALS = 6;

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

export interface Observation {
  /** [NUM_CHANNELS, h, w] aplanado: 0 tierra, 1 propio, 2 otro jugador. */
  grid: Float32Array;
  /** Ver docs/specs/bridge.md §4. */
  globals: Float32Array;
  mask: ActionMask;
  tick: number;
  inSpawnPhase: boolean;
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
