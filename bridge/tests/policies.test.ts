import { describe, expect, it } from "vitest";
import { OpenFrontEnv } from "../src/env";
import { HeuristicPolicy } from "../src/heuristicPolicy";
import { RandomPolicy } from "../src/randomPolicy";
import type { Action, EnvConfig, Observation } from "../src/types";

const CFG: Partial<EnvConfig> = { bots: 20, maxTicks: 3000 };
const SEEDS = ["e0", "e1", "e2"];

async function play(seed: string, make: (seed: string, grid: EnvConfig["grid"]) => { act(o: Observation): Action }) {
  const env = await OpenFrontEnv.create(CFG);
  const policy = make(seed, env.cfg.grid);
  let obs = await env.reset(seed);
  let peak = 0;
  let invalid = 0;
  let survived = true;
  for (let i = 0; i < 2000; i++) {
    const r = env.step(policy.act(obs));
    obs = r.obs;
    peak = Math.max(peak, obs.globals[0]);
    if (r.info.invalidAction) invalid++;
    if (r.info.eliminated) survived = false;
    if (r.terminated || r.truncated) break;
  }
  env.close();
  return { peak, invalid, survived };
}

describe("política heurística como línea base", () => {
  it("el entorno es jugable: la heurística crece mucho más que la aleatoria y no hace acciones inválidas", async () => {
    const heur = [];
    const rand = [];
    for (const s of SEEDS) {
      heur.push(await play(s, (seed, g) => new HeuristicPolicy(seed, g)));
      rand.push(await play(s, (seed, g) => new RandomPolicy(seed, g)));
    }
    const mean = (xs: { peak: number }[]) => xs.reduce((a, x) => a + x.peak, 0) / xs.length;
    expect(heur.every((x) => x.invalid === 0)).toBe(true);
    expect(mean(heur)).toBeGreaterThan(3 * mean(rand));
    expect(mean(heur)).toBeGreaterThan(0.02); // más del 2 % del mapa de media
  });

  it("es determinista", async () => {
    const a = await play("det", (seed, g) => new HeuristicPolicy(seed, g));
    const b = await play("det", (seed, g) => new HeuristicPolicy(seed, g));
    expect(a).toEqual(b);
  });
});
