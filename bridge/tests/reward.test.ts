import { describe, expect, it } from "vitest";
import { MultiAgentEnv } from "../src/multiEnv";
import { REWARD_PARTS, securityPotential } from "../src/reward";
import { RandomPolicy } from "../src/randomPolicy";
import type { EnvConfig } from "../src/types";

const BASE: Partial<EnvConfig> = { bots: 10, maxTicks: 1200 };

async function play(cfg: Partial<EnvConfig>, seed: string, steps = 80) {
  const env = await MultiAgentEnv.create(cfg);
  const policy = new RandomPolicy(seed, env.cfg.grid);
  let obs = await env.reset(seed);
  const totals = { territory: 0, attack: 0, economy: 0, defense: 0, terminal: 0 };
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
    expect(Math.abs(a.reward - (a.totals.territory + a.totals.terminal))).toBeLessThan(1e-6);
  }, 120000);

  it("con todas las categorías activas las partes son reproducibles y respetan su presupuesto", async () => {
    const cfg = { ...BASE, rewardTerritoryWeight: 0.4, rewardAttackWeight: 0.15, rewardEconomyWeight: 0.1, rewardDefenseWeight: 0.05 };
    const a = await play(cfg, "rw-2");
    const b = await play(cfg, "rw-2");
    expect(a.hash).toBe(b.hash);
    for (const k of REWARD_PARTS) expect(a.totals[k]).toBe(b.totals[k]);
    // Telescópicas: el acumulado de cada categoría está en [−peso, +peso].
    expect(Math.abs(a.totals.territory)).toBeLessThanOrEqual(0.4 + 1e-6);
    expect(Math.abs(a.totals.attack)).toBeLessThanOrEqual(0.15 + 1e-6);
    expect(Math.abs(a.totals.economy)).toBeLessThanOrEqual(0.1 + 1e-6);
    expect(Math.abs(a.totals.defense)).toBeLessThanOrEqual(0.05 + 1e-6);
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
});
