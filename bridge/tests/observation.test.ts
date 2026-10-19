import { describe, expect, it } from "vitest";
import {
  ATTACK_ROW_FEATURES,
  ECON_ROW_FEATURES,
  ECON_SCALARS,
  GLOBAL_FEATURES,
  MAX_ATTACK_ROWS,
  MAX_ECON_ROWS,
  MAX_MIL_ROWS,
  MIL_ROW_FEATURES,
  MIL_SCALARS,
  CMP_SCALARS,
  PLAYER_ROW_FEATURES,
  PLAYER_ROWS,
  NUM_CHANNELS,
} from "../src/features";
import { HeuristicPolicy } from "../src/heuristicPolicy";
import { MultiAgentEnv } from "../src/multiEnv";
import type { Observation, Table } from "../src/types";

function bad(a: ArrayLike<number>, lo: number, hi: number): number {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (!(a[i] >= lo && a[i] <= hi)) n++;
  return n;
}
function invalidRowsNonZero(t: Table): number {
  let n = 0;
  for (let r = 0; r < t.rows; r++) {
    if (t.valid[r] === 1) continue;
    for (let c = 0; c < t.cols; c++) if (t.data[r * t.cols + c] !== 0) n++;
  }
  return n;
}

async function run(agents: number, steps: number, seed = "obs-1") {
  const env = await MultiAgentEnv.create({ bots: 10, maxTicks: 3000, agents });
  let obs = await env.reset(seed);
  const pols = Array.from({ length: agents }, (_, i) => new HeuristicPolicy(`${seed}-${i}`, env.cfg.grid));
  const all: Observation[][] = [obs];
  for (let i = 0; i < steps; i++) {
    const r = env.step(obs.map((o, j) => pols[j].act(o)));
    obs = r.obs;
    all.push(obs);
    if (r.done) break;
  }
  return { env, all };
}

describe("observación v2", () => {
  it("tiene las formas y los rangos de la spec", async () => {
    const { env, all } = await run(1, 120);
    for (const [o] of all) {
      expect(o.grid.length).toBe(NUM_CHANNELS * 32 * 64);
      expect(o.globals.length).toBe(GLOBAL_FEATURES.length);
      expect(o.economy.table.rows).toBe(MAX_ECON_ROWS);
      expect(o.economy.table.cols).toBe(ECON_ROW_FEATURES.length);
      expect(o.economy.scalars.length).toBe(ECON_SCALARS.length);
      expect(o.military.units.rows).toBe(MAX_MIL_ROWS);
      expect(o.military.units.cols).toBe(MIL_ROW_FEATURES.length);
      expect(o.military.attacks.rows).toBe(MAX_ATTACK_ROWS);
      expect(o.military.attacks.cols).toBe(ATTACK_ROW_FEATURES.length);
      expect(o.military.scalars.length).toBe(MIL_SCALARS.length);
      expect(o.comparison.players.rows).toBe(PLAYER_ROWS);
      expect(o.comparison.players.cols).toBe(PLAYER_ROW_FEATURES.length);
      expect(o.comparison.scalars.length).toBe(CMP_SCALARS.length);

      expect(bad(o.grid, 0, 1)).toBe(0);
      expect(bad(o.globals, 0, 1)).toBe(0);
      expect(bad(o.economy.table.data, 0, 1)).toBe(0);
      expect(bad(o.economy.scalars, 0, 1)).toBe(0);
      expect(bad(o.military.units.data, 0, 1)).toBe(0);
      expect(bad(o.military.attacks.data, 0, 1)).toBe(0);
      expect(bad(o.military.scalars, 0, 1)).toBe(0);
      expect(bad(o.comparison.players.data, -1, 1)).toBe(0);
      expect(bad(o.comparison.scalars, -1, 1)).toBe(0);

      expect(invalidRowsNonZero(o.economy.table)).toBe(0);
      expect(invalidRowsNonZero(o.military.units)).toBe(0);
      expect(invalidRowsNonZero(o.military.attacks)).toBe(0);
      expect(invalidRowsNonZero(o.comparison.players)).toBe(0);
    }
    env.close();
  });

  it("la fila 0 de jugadores es el agente y las filas 1..8 son las ranuras de ataque", async () => {
    const { env, all } = await run(1, 120);
    const f = (name: (typeof PLAYER_ROW_FEATURES)[number]) => PLAYER_ROW_FEATURES.indexOf(name);
    let checkedSlots = 0;
    for (const [o] of all) {
      const t = o.comparison.players;
      if (o.inSpawnPhase) continue;
      expect(t.valid[0]).toBe(1);
      expect(t.data[f("is_self")]).toBe(1);
      for (let i = 0; i < 8; i++) {
        const r = 1 + i;
        if (t.valid[r] === 1) {
          expect(t.data[r * t.cols + f("is_self")]).toBe(0);
          expect(t.data[r * t.cols + f("is_neighbor")]).toBe(1);
          checkedSlots++;
        }
      }
    }
    expect(checkedSlots).toBeGreaterThan(0);
    env.close();
  });

  it("dos agentes ven vistas relativas distintas de la misma partida", async () => {
    const { env, all } = await run(2, 60);
    const last = all[all.length - 1];
    const own = (o: Observation) => o.grid.slice(2048, 2 * 2048);
    expect(Array.from(own(last[0]))).not.toEqual(Array.from(own(last[1])));
    // El canal "propio" de un agente no se solapa con el del otro.
    const a = own(last[0]);
    const b = own(last[1]);
    let overlap = 0;
    for (let i = 0; i < a.length; i++) if (a[i] > 0 && b[i] > 0) overlap += Math.min(a[i], b[i]);
    expect(overlap).toBeLessThan(1);
    env.close();
  });

  it("es determinista con la misma semilla", async () => {
    const a = await run(1, 60, "det-obs");
    const b = await run(1, 60, "det-obs");
    const la = a.all[a.all.length - 1][0];
    const lb = b.all[b.all.length - 1][0];
    expect(Array.from(la.economy.scalars)).toEqual(Array.from(lb.economy.scalars));
    expect(Array.from(la.comparison.players.data)).toEqual(Array.from(lb.comparison.players.data));
    expect(Array.from(la.military.units.data)).toEqual(Array.from(lb.military.units.data));
    a.env.close();
    b.env.close();
  });

  it("la economía refleja el estado: una ciudad construida aparece en la tabla y baja la capacidad de pagar", async () => {
    const { env, all } = await run(1, 40);
    const before = all[all.length - 1][0];
    const e = (name: (typeof ECON_SCALARS)[number]) => ECON_SCALARS.indexOf(name);
    expect(before.economy.table.valid[0]).toBe(0);
    const game = env.unsafeGame();
    const me = game.playerByClientID("rl-agent-0");
    if (me === null) throw new Error("sin jugador");
    // Se da oro y se construye una ciudad con una intención real del motor.
    const tile = [...me.tiles()][0];
    const { Executor } = await import("@openfront/engine/execution/ExecutionManager");
    void Executor;
    (me as unknown as { addGold(n: bigint): void }).addGold(5_000_000n);
    const { ConstructionExecution } = await import("@openfront/engine/execution/ConstructionExecution");
    const { UnitType } = await import("@openfront/engine-api/game/GameTypes");
    game.addExecution(new ConstructionExecution(me, UnitType.City, tile));
    const ep = env as unknown as { observeAll(): Observation[] };
    for (let i = 0; i < 3; i++) game.executeNextTick();
    const after = ep.observeAll()[0];
    expect(after.economy.table.valid[0]).toBe(1);
    expect(after.economy.table.data[ECON_ROW_FEATURES.indexOf("is_city")]).toBe(1);
    expect(after.economy.scalars[e("afford_city")]).toBeGreaterThanOrEqual(0);
    expect(after.economy.scalars[e("city_levels")]).toBeGreaterThan(0);
    expect(after.economy.scalars[e("gold_log")]).toBeGreaterThan(before.economy.scalars[e("gold_log")]);
    env.close();
  });
});
