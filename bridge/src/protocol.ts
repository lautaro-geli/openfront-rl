// Formato binario de la observación y del resultado (docs/specs/protocol.md). Funciones puras: sin estado ni entrada/salida.
import {
  ACTION_DIMS,
  ACTION_KINDS,
  BUILD_TYPES,
  CELL_KINDS,
  ECON_ROW_FEATURES,
  ECON_SCALARS,
  GLOBAL_FEATURES,
  GRID_CHANNELS,
  MAX_ATTACK_ROWS,
  MAX_ECON_ROWS,
  MAX_ATTACKS_PER_SIDE,
  MAX_MIL_ROWS,
  MIL_ROW_FEATURES,
  MIL_SCALARS,
  NUKE_KINDS,
  PLAYER_KINDS,
  PLAYER_ROW_FEATURES,
  PLAYER_ROWS,
  ATTACK_ROW_FEATURES,
  CMP_SCALARS,
  UNIT_KINDS,
  UNIT_ROWS,
} from "./features";
import type { EnvConfig, MultiStepResult, Observation } from "./types";

export const PROTOCOL_VERSION = 1;

export const REQ = { init: 0x01, reset: 0x02, step: 0x03, hash: 0x04, close: 0x05 } as const;
export const RES = { manifest: 0x81, result: 0x82, hash: 0x83, closed: 0x84, error: 0xff } as const;

interface Block<T> {
  name: string;
  shape: number[];
  get(x: T): ArrayLike<number>;
}

const f = (a: ArrayLike<number>): ArrayLike<number> => a;
const bool = (b: boolean): ArrayLike<number> => [b ? 1 : 0];

export function obsBlocks(cfg: Pick<EnvConfig, "grid">): Block<Observation>[] {
  const { h, w } = cfg.grid;
  return [
    { name: "grid", shape: [GRID_CHANNELS.length, h, w], get: (o) => f(o.grid) },
    { name: "globals", shape: [GLOBAL_FEATURES.length], get: (o) => f(o.globals) },
    { name: "econ_table", shape: [MAX_ECON_ROWS, ECON_ROW_FEATURES.length], get: (o) => f(o.economy.table.data) },
    { name: "econ_valid", shape: [MAX_ECON_ROWS], get: (o) => f(o.economy.table.valid) },
    { name: "econ_scalars", shape: [ECON_SCALARS.length], get: (o) => f(o.economy.scalars) },
    { name: "mil_units", shape: [MAX_MIL_ROWS, MIL_ROW_FEATURES.length], get: (o) => f(o.military.units.data) },
    { name: "mil_units_valid", shape: [MAX_MIL_ROWS], get: (o) => f(o.military.units.valid) },
    { name: "mil_attacks", shape: [MAX_ATTACK_ROWS, ATTACK_ROW_FEATURES.length], get: (o) => f(o.military.attacks.data) },
    { name: "mil_attacks_valid", shape: [MAX_ATTACK_ROWS], get: (o) => f(o.military.attacks.valid) },
    { name: "mil_scalars", shape: [MIL_SCALARS.length], get: (o) => f(o.military.scalars) },
    { name: "players", shape: [PLAYER_ROWS, PLAYER_ROW_FEATURES.length], get: (o) => f(o.comparison.players.data) },
    { name: "players_valid", shape: [PLAYER_ROWS], get: (o) => f(o.comparison.players.valid) },
    { name: "cmp_scalars", shape: [CMP_SCALARS.length], get: (o) => f(o.comparison.scalars) },
  ];
}

export function maskBlocks(cfg: Pick<EnvConfig, "grid" | "maxNeighbors">): Block<Observation>[] {
  const cells = cfg.grid.h * cfg.grid.w;
  return [
    { name: "spawn", shape: [1], get: (o) => bool(o.mask.spawn) },
    { name: "expand", shape: [1], get: (o) => bool(o.mask.expand) },
    { name: "attack_slots", shape: [cfg.maxNeighbors], get: (o) => o.mask.attackSlots.map((b) => (b ? 1 : 0)) },
    { name: "kind", shape: [ACTION_KINDS.length], get: (o) => o.mask.kind },
    { name: "player", shape: [PLAYER_KINDS.length, PLAYER_ROWS], get: (o) => o.mask.player },
    { name: "unit", shape: [UNIT_KINDS.length, UNIT_ROWS], get: (o) => o.mask.unit },
    { name: "attack_row", shape: [MAX_ATTACKS_PER_SIDE], get: (o) => o.mask.attackRow },
    { name: "build_type", shape: [BUILD_TYPES.length], get: (o) => o.mask.buildType },
    { name: "nuke_type", shape: [NUKE_KINDS.length], get: (o) => o.mask.nukeType },
    { name: "cell", shape: [CELL_KINDS.length, cells], get: (o) => o.mask.cell },
  ];
}

const size = (shape: number[]): number => shape.reduce((a, b) => a * b, 1);
const align4 = (n: number): number => (n + 3) & ~3;

export interface Layout {
  obsFloats: number;
  maskBytes: number;
  /** Bytes de un agente en el resultado: reward + 4 banderas + observación + máscara (alineado a 4). */
  agentBytes: number;
}

export function layout(cfg: Pick<EnvConfig, "grid" | "maxNeighbors">): Layout {
  const obsFloats = obsBlocks(cfg).reduce((s, b) => s + size(b.shape), 0);
  const maskBytes = maskBlocks(cfg).reduce((s, b) => s + size(b.shape), 0);
  return { obsFloats, maskBytes, agentBytes: align4(8 + obsFloats * 4 + maskBytes) };
}

export function buildManifest(cfg: EnvConfig): object {
  const place = (blocks: Block<Observation>[]) => {
    let offset = 0;
    return blocks.map((b) => {
      const out = { name: b.name, shape: b.shape, offset };
      offset += size(b.shape);
      return out;
    });
  };
  const l = layout(cfg);
  return {
    version: PROTOCOL_VERSION,
    agents: cfg.agents,
    grid: cfg.grid,
    decisionInterval: cfg.decisionInterval,
    maxNeighbors: cfg.maxNeighbors,
    actionDims: ACTION_DIMS,
    actionKinds: ACTION_KINDS,
    obs: { dtype: "float32", floats: l.obsFloats, blocks: place(obsBlocks(cfg)) },
    mask: { dtype: "uint8", bytes: l.maskBytes, blocks: place(maskBlocks(cfg)) },
    result: { headerBytes: 8, agentBytes: l.agentBytes },
  };
}

/** Escribe un resultado en el formato de la spec §4. */
export function encodeResult(cfg: EnvConfig, r: MultiStepResult, tick: number): Uint8Array {
  const l = layout(cfg);
  const A = r.obs.length;
  const ab = new ArrayBuffer(8 + A * l.agentBytes);
  const dv = new DataView(ab);
  dv.setUint8(0, r.done ? 1 : 0);
  dv.setUint8(1, r.truncated ? 1 : 0);
  dv.setUint8(2, A);
  dv.setInt32(4, tick, true);
  const oBlocks = obsBlocks(cfg);
  const mBlocks = maskBlocks(cfg);
  for (let a = 0; a < A; a++) {
    const base = 8 + a * l.agentBytes;
    dv.setFloat32(base, r.reward[a], true);
    dv.setUint8(base + 4, r.terminated[a] ? 1 : 0);
    dv.setUint8(base + 5, r.info[a].invalidAction ? 1 : 0);
    dv.setUint8(base + 6, r.info[a].won ? 1 : 0);
    dv.setUint8(base + 7, r.info[a].eliminated ? 1 : 0);
    const floats = new Float32Array(ab, base + 8, l.obsFloats);
    let o = 0;
    for (const b of oBlocks) {
      floats.set(b.get(r.obs[a]) as ArrayLike<number>, o);
      o += size(b.shape);
    }
    const bytes = new Uint8Array(ab, base + 8 + l.obsFloats * 4, l.maskBytes);
    let m = 0;
    for (const b of mBlocks) {
      bytes.set(b.get(r.obs[a]) as ArrayLike<number>, m);
      m += size(b.shape);
    }
  }
  return new Uint8Array(ab);
}

/** Resultado de un `reset`: sin recompensa ni banderas. */
export function resetResult(obs: Observation[]): MultiStepResult {
  const A = obs.length;
  return {
    obs,
    reward: Array(A).fill(0),
    terminated: Array(A).fill(false),
    truncated: false,
    done: false,
    info: Array.from({ length: A }, () => ({
      invalidAction: false,
      rewardParts: { territory: 0, terminal: 0 },
      won: false,
      eliminated: false,
      tick: obs[0]?.tick ?? 0,
    })),
  };
}
