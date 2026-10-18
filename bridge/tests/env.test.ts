import { describe, expect, it } from "vitest";
import { OpenFrontEnv } from "../src/env";
import { RandomPolicy } from "../src/randomPolicy";
import { type EnvConfig, NUM_CHANNELS, NUM_GLOBALS, type Observation } from "../src/types";

const SMALL: Partial<EnvConfig> = { bots: 10, maxTicks: 1500 };

async function playEpisode(seed: string, cfg: Partial<EnvConfig> = SMALL) {
  const env = await OpenFrontEnv.create(cfg);
  const policy = new RandomPolicy(seed, env.cfg.grid);
  let obs = await env.reset(seed);
  const first = obs;
  let steps = 0;
  let reward = 0;
  let invalid = 0;
  let ended = "none";
  const obsChecks: Observation[] = [obs];
  for (let i = 0; i < 2000; i++) {
    const r = env.step(policy.act(obs));
    obs = r.obs;
    obsChecks.push(obs);
    steps++;
    reward += r.reward;
    if (r.info.invalidAction) invalid++;
    if (r.terminated || r.truncated) {
      ended = r.terminated ? "terminated" : "truncated";
      break;
    }
  }
  const hash = env.stateHash();
  env.close();
  return { first, steps, reward, invalid, ended, hash, obsChecks };
}

describe("OpenFrontEnv", () => {
  it("es determinista: misma semilla y mismas acciones dan el mismo resultado", async () => {
    const a = await playEpisode("det-1");
    const b = await playEpisode("det-1");
    expect(a.hash).toBe(b.hash);
    expect(a.steps).toBe(b.steps);
    expect(a.reward).toBe(b.reward);
    expect(Array.from(a.obsChecks[a.obsChecks.length - 1].grid)).toEqual(
      Array.from(b.obsChecks[b.obsChecks.length - 1].grid),
    );
  });

  it("semillas distintas dan partidas distintas", async () => {
    const a = await playEpisode("det-1");
    const c = await playEpisode("det-2");
    expect(a.hash).not.toBe(c.hash);
  });

  it("cumple el contrato de observación", async () => {
    const { obsChecks } = await playEpisode("contract-1");
    const cfg = { h: 32, w: 64 };
    for (const o of obsChecks) {
      expect(o.grid.length).toBe(NUM_CHANNELS * cfg.h * cfg.w);
      expect(o.globals.length).toBe(NUM_GLOBALS);
      for (const v of o.grid) {
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
      for (const v of o.globals) {
        expect(Number.isFinite(v)).toBe(true);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      }
      expect(o.mask.attackSlots.length).toBe(8);
    }
  });

  it("la política que respeta la máscara no produce acciones inválidas", async () => {
    for (const seed of ["mask-1", "mask-2", "mask-3"]) {
      const r = await playEpisode(seed);
      expect(r.invalid).toBe(0);
    }
  });

  it("acaba un episodio completo sin errores", async () => {
    const r = await playEpisode("full-1");
    expect(["terminated", "truncated"]).toContain(r.ended);
    expect(Number.isFinite(r.reward)).toBe(true);
  });

  it("reset con la misma semilla devuelve la misma observación inicial", async () => {
    const env = await OpenFrontEnv.create(SMALL);
    const o1 = await env.reset("reset-1");
    const o2 = await env.reset("reset-1");
    expect(Array.from(o1.grid)).toEqual(Array.from(o2.grid));
    expect(Array.from(o1.globals)).toEqual(Array.from(o2.globals));
    expect(o1.inSpawnPhase).toBe(true);
    env.close();
  });

  it("trata una acción inválida como noop y lo informa", async () => {
    const env = await OpenFrontEnv.create(SMALL);
    await env.reset("invalid-1");
    // En fase de spawn, expand no es válida.
    const r = env.step({ kind: "expand", fraction: 0.5 });
    expect(r.info.invalidAction).toBe(true);
    expect(r.terminated).toBe(false);
    env.close();
  });

  it("step antes de reset y después del fin lanza error", async () => {
    const env = await OpenFrontEnv.create(SMALL);
    expect(() => env.step({ kind: "noop" })).toThrow();
    env.close();
  });
});
