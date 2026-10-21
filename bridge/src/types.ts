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
  /** Pesos de los demás componentes (docs/specs/reward.md). 0 = apagado. */
  rewardAttackWeight: number;
  rewardEconomyWeight: number;
  rewardDefenseWeight: number;
  /** Número de jugadores controlados por código en la partida (MultiAgentEnv). */
  agents: number;
}

import type { ActionKind } from "./features";
import type { RewardParts } from "./reward";

/** Acciones de la v0, con su forma original (las usan las políticas aleatoria y heurística). */
export type LegacyAction =
  | { kind: "noop" }
  | { kind: "spawn"; cx: number; cy: number }
  | { kind: "expand"; fraction: Fraction }
  | { kind: "attack"; slot: number; fraction: Fraction };

/** El resto del catálogo. Los campos que no usa un tipo se ignoran. Ver docs/specs/actions.md §2. */
export interface ExtraAction {
  kind: Exclude<ActionKind, LegacyAction["kind"]>;
  row?: number;
  cell?: number;
  fraction?: Fraction;
  type?: number;
  aux?: number;
}

export type Action = LegacyAction | ExtraAction;

/** Máscara de acciones (docs/specs/actions.md §4). Los tres primeros campos son los de la v0. */
export interface ActionMask {
  spawn: boolean;
  expand: boolean;
  attackSlots: boolean[];
  /** [NUM_KINDS] */
  kind: Uint8Array;
  /** [PLAYER_KINDS.length, PLAYER_ROWS] */
  player: Uint8Array;
  /** [UNIT_KINDS.length, UNIT_ROWS] */
  unit: Uint8Array;
  /** [MAX_ATTACKS_PER_SIDE] */
  attackRow: Uint8Array;
  /** [BUILD_TYPES.length] */
  buildType: Uint8Array;
  /** [NUKE_KINDS.length] */
  nukeType: Uint8Array;
  /** [CELL_KINDS.length, h * w] */
  cell: Uint8Array;
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
  rewardParts: RewardParts;
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
