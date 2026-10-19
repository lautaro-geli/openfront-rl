import { describe, expect, it } from "vitest";
import { MultiAgentEnv } from "../src/multiEnv";
import { RandomPolicy } from "../src/randomPolicy";
import type { Action, EnvConfig, Observation } from "../src/types";

const EMPTY: Partial<EnvConfig> = { agents: 2, bots: 0, nations: "disabled", maxTicks: 1200 };

/** Spawn en celdas de tierra distintas: elige las celdas con más tierra, separadas entre sí. */
function pickSpawns(obs: Observation, grid: { h: number; w: number }, n: number): Action[] {
  const cells = grid.h * grid.w;
  const land: number[] = [];
  for (let c = 0; c < cells; c++) if (obs.grid[c] >= 0.95) land.push(c);
  const chosen: number[] = [];
  const step = Math.floor(land.length / (n + 1));
  for (let i = 1; i <= n; i++) chosen.push(land[i * step]);
  return chosen.map((c) => ({ kind: "spawn", cx: c % grid.w, cy: Math.floor(c / grid.w) }));
}

describe("MultiAgentEnv", () => {
  it("dos agentes aparecen en la partida y cada uno actúa solo sobre su propio jugador", async () => {
    const env = await MultiAgentEnv.create(EMPTY);
    const obs = await env.reset("multi-1");
    expect(obs.length).toBe(2);
    expect(obs.every((o) => o.mask.spawn)).toBe(true);

    let r = env.step(pickSpawns(obs[0], env.cfg.grid, 2));
    expect(r.done).toBe(false);
    const [t0, t1] = env.agentTiles();
    expect(t0).toBeGreaterThan(0);
    expect(t1).toBeGreaterThan(0);

    // El agente 0 no hace nada; el agente 1 se expande en cada paso.
    for (let i = 0; i < 30 && !r.done; i++) {
      r = env.step([{ kind: "noop" }, { kind: "expand", fraction: 0.5 }]);
      expect(r.info[0].invalidAction).toBe(false);
    }
    const [a0, a1] = env.agentTiles();
    expect(a0).toBe(t0); // sin acciones, su territorio no cambia
    expect(a1).toBeGreaterThan(t1); // el agente 1 sí creció
    expect(r.reward[1]).toBeDefined();
    env.close();
  });

  it("las observaciones son relativas a cada agente", async () => {
    const env = await MultiAgentEnv.create(EMPTY);
    const obs = await env.reset("multi-2");
    env.step(pickSpawns(obs[0], env.cfg.grid, 2));
    let last = obs;
    for (let i = 0; i < 5; i++) {
      const r = env.step([{ kind: "expand", fraction: 0.5 }, { kind: "expand", fraction: 0.5 }]);
      last = r.obs;
    }
    const cells = env.cfg.grid.h * env.cfg.grid.w;
    const sum = (o: Observation, ch: number) => {
      let s = 0;
      for (let c = 0; c < cells; c++) s += o.grid[ch * cells + c];
      return s;
    };
    // Lo que el agente 0 ve como "propio" es lo que el agente 1 ve como "otro", y al revés.
    expect(sum(last[0], 1)).toBeGreaterThan(0);
    expect(sum(last[1], 1)).toBeGreaterThan(0);
    expect(sum(last[0], 1)).toBeCloseTo(sum(last[1], 2), 4);
    expect(sum(last[1], 1)).toBeCloseTo(sum(last[0], 2), 4);
    env.close();
  });

  it("es determinista con dos agentes y bots", async () => {
    const play = async () => {
      const env = await MultiAgentEnv.create({ agents: 2, bots: 10, maxTicks: 800 });
      const pols = [0, 1].map((i) => new RandomPolicy(`m-${i}`, env.cfg.grid));
      let obs = await env.reset("multi-det");
      let rewards = [0, 0];
      for (let i = 0; i < 400; i++) {
        const r = env.step(obs.map((o, k) => pols[k].act(o)));
        obs = r.obs;
        rewards = rewards.map((x, k) => x + r.reward[k]);
        if (r.done) break;
      }
      const out = { hash: env.stateHash(), rewards, tiles: env.agentTiles() };
      env.close();
      return out;
    };
    const a = await play();
    const b = await play();
    expect(a).toEqual(b);
  });

  it("OpenFrontEnv rechaza más de un agente", async () => {
    const { OpenFrontEnv } = await import("../src/env");
    await expect(OpenFrontEnv.create({ agents: 2 })).rejects.toThrow();
  });
});
