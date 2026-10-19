import { HeuristicPolicy } from "../heuristicPolicy";
import { MultiAgentEnv } from "../multiEnv";
const map = process.argv[2] ?? "Britannia";
const bots = Number(process.argv[3] ?? 20);
const env = await MultiAgentEnv.create({ bots, maxTicks: 3000, map: map as never });
let obs = await env.reset("prof");
const pol = new HeuristicPolicy("prof", env.cfg.grid);
for (let i = 0; i < 100; i++) {
  const r = env.step([pol.act(obs[0])]);
  obs = r.obs;
  if (r.done) break;
}
const o = env as any;
const s2 = performance.now();
for (let i = 0; i < 20; i++) o.observeAll();
console.log(map, bots, "ms/observeAll", ((performance.now() - s2) / 20).toFixed(1));
const s3 = performance.now();
let k = 0;
for (let i = 0; i < 20; i++) { const r = env.step([{ kind: "noop" }]); k++; if (r.done) break; }
console.log("ms/step (10 ticks + obs)", ((performance.now() - s3) / k).toFixed(1));
