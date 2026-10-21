import { describe, expect, it } from "vitest";
import { MultiAgentEnv } from "../src/multiEnv";
import { CELL_KINDS } from "../src/features";
import { HeuristicPolicy } from "../src/heuristicPolicy";
import { REWARD_PARTS, securityPotential, territoryPotential, troopBand } from "../src/reward";
import { RandomPolicy } from "../src/randomPolicy";
import type { EnvConfig } from "../src/types";

const BASE: Partial<EnvConfig> = { bots: 10, maxTicks: 1200 };

async function play(cfg: Partial<EnvConfig>, seed: string, steps = 80) {
  const env = await MultiAgentEnv.create(cfg);
  const policy = new RandomPolicy(seed, env.cfg.grid);
  let obs = await env.reset(seed);
  const totals = { territory: 0, attack: 0, economy: 0, defense: 0, milestones: 0, troops: 0, terminal: 0 };
  let reward = 0;
  let maxSecAbs = 0;
  for (let i = 0; i < steps; i++) {
    const r = env.step(obs.map((o) => policy.act(o)));
    obs = r.obs;
    reward += r.reward[0];
    for (const k of REWARD_PARTS) totals[k] += r.info[0].rewardParts[k];
    maxSecAbs = Math.max(maxSecAbs, Math.abs(r.info[0].rewardParts.defense));
    expect(Math.abs(r.reward[0] - REWARD_PARTS.reduce((s, k) => s + r.info[0].rewardParts[k], 0))).toBeLessThan(1e-6);
    if (r.done) break;
  }
  const hash = env.stateHash();
  return { totals, reward, maxSecAbs, hash, env };
}

describe("recompensa por componentes", () => {
  it("con los pesos por defecto solo hay territorio y terminal (igual que la v0)", async () => {
    const a = await play(BASE, "rw-1");
    expect(a.totals.attack).toBe(0);
    expect(a.totals.economy).toBe(0);
    expect(a.totals.defense).toBe(0);
    expect(a.totals.milestones).toBe(0);
    expect(a.totals.troops).toBe(0);
    expect(Math.abs(a.reward - (a.totals.territory + a.totals.terminal))).toBeLessThan(1e-6);
  }, 120000);

  it("con todas las categorías activas las partes son reproducibles y respetan su presupuesto", async () => {
    const cfg = { ...BASE, rewardTerritoryWeight: 0.4, rewardAttackWeight: 0.12, rewardEconomyWeight: 0.1, rewardDefenseWeight: 0.05, rewardMilestonesWeight: 0.06, rewardTroopsWeight: 0.05, territoryKnee: 0.15, territoryKneeShare: 0.35 };
    const a = await play(cfg, "rw-2");
    const b = await play(cfg, "rw-2");
    expect(a.hash).toBe(b.hash);
    for (const k of REWARD_PARTS) expect(a.totals[k]).toBe(b.totals[k]);
    // Telescópicas: el acumulado de cada categoría está en [−peso, +peso].
    expect(Math.abs(a.totals.territory)).toBeLessThanOrEqual(0.4 + 1e-6);
    expect(Math.abs(a.totals.attack)).toBeLessThanOrEqual(0.12 + 1e-6);
    expect(Math.abs(a.totals.economy)).toBeLessThanOrEqual(0.1 + 1e-6);
    expect(Math.abs(a.totals.defense)).toBeLessThanOrEqual(0.05 + 1e-6);
    expect(Math.abs(a.totals.milestones)).toBeLessThanOrEqual(0.06 + 1e-6);
    expect(a.totals.troops).toBeGreaterThanOrEqual(0);
    expect(a.totals.troops).toBeLessThanOrEqual(0.05 + 1e-6);
    expect(a.maxSecAbs).toBeLessThanOrEqual(0.05 + 1e-9);
    // La partida no cambia por activar el shaping (la recompensa no toca el motor).
    const c = await play(BASE, "rw-2");
    expect(c.hash).toBe(a.hash);
  }, 240000);

  it("la seguridad vale 1 sin vecinos hostiles y baja cuando hay un enemigo más fuerte", async () => {
    const alone = await play({ bots: 0, nations: "disabled", maxTicks: 600 }, "rw-3", 20);
    const g = alone.env.unsafeGame();
    const me = g.playerByClientID("rl-agent-0");
    expect(me).not.toBeNull();
    expect(securityPotential(g, me!)).toBe(1);

    const env = await MultiAgentEnv.create({ bots: 30, maxTicks: 3000 });
    const policy = new RandomPolicy("rw-4", env.cfg.grid);
    let obs = await env.reset("rw-4");
    let seenLow = false;
    for (let i = 0; i < 150 && !seenLow; i++) {
      const r = env.step(obs.map((o) => policy.act(o)));
      obs = r.obs;
      const gm = env.unsafeGame();
      const p = gm.playerByClientID("rl-agent-0");
      if (p !== null && p.isAlive()) {
        const phi = securityPotential(gm, p);
        expect(phi).toBeGreaterThanOrEqual(0);
        expect(phi).toBeLessThanOrEqual(1);
        if (phi < 0.9) seenLow = true;
      }
      if (r.done) break;
    }
    expect(seenLow).toBe(true);
  }, 240000);

  it("la curva de territorio por tramos paga el 35 % en el primer 15 % y llega a 1 en el 80 %", () => {
    const f = (x: number) => territoryPotential(x, 0.15, 0.35);
    expect(f(0)).toBe(0);
    expect(f(0.15)).toBeCloseTo(0.35, 12);
    expect(f(0.8)).toBeCloseTo(1, 12);
    expect(f(0.95)).toBeCloseTo(1, 12);
    expect(f(0.075)).toBeCloseTo(0.175, 12);
    expect(f(0.475)).toBeCloseTo(0.35 + 0.65 * 0.5, 12); // mitad del segundo tramo
    // monótona y con más pendiente en el primer tramo
    expect(f(0.01) / 0.01).toBeGreaterThan((f(0.41) - f(0.4)) / 0.01);
    // knee 0 = lineal (valor por defecto del puente)
    expect(territoryPotential(0.4, 0, 0.5)).toBeCloseTo(0.5, 12);
  });

  it("la banda de tropas vale 1 entre 30 % y 60 % y cae a 0 en 10 % y 90 %", () => {
    for (const r of [0.3, 0.42, 0.6]) expect(troopBand(r)).toBe(1);
    expect(troopBand(0.1)).toBe(0);
    expect(troopBand(0.9)).toBe(0);
    expect(troopBand(0.2)).toBeCloseTo(0.5, 12);
    expect(troopBand(0.75)).toBeCloseTo(0.5, 12);
    expect(troopBand(0)).toBe(0);
    expect(troopBand(1)).toBe(0);
  });

  it("los hitos pagan B/3 al terminar la primera ciudad (y solo una vez)", async () => {
    const env = await MultiAgentEnv.create({ bots: 5, nations: "disabled", maxTicks: 6000, rewardMilestonesWeight: 0.06 });
    const pol = new HeuristicPolicy("rw-m", env.cfg.grid);
    let obs = await env.reset("rw-m");
    const cells = env.cfg.grid.h * env.cfg.grid.w;
    const city = 0;
    let total = 0;
    let ordered = 0;
    for (let i = 0; i < 200; i++) {
      const o = obs[0];
      let act = pol.act(o);
      const c = o.mask.cell.slice(CELL_KINDS.indexOf("build") * cells, (CELL_KINDS.indexOf("build") + 1) * cells).findIndex((x) => x === 1);
      if (i >= 40 && ordered < 3 && c >= 0 && o.mask.buildType[city] === 1) {
        act = { kind: "build", row: 0, cell: c, fraction: 0.5, type: city, aux: 0 } as never;
        ordered++;
      }
      const r = env.step([act]);
      obs = r.obs;
      total += r.info[0].rewardParts.milestones;
      if (r.done) break;
    }
    expect(ordered).toBeGreaterThan(0);
    expect(total).toBeCloseTo(0.02, 9);
  }, 120000);
});
