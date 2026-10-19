// Juega episodios completos con una política y muestra velocidad y resultado.
// Uso: npm run random -- [--policy random|heuristic] [--map britannia] [--bots 20] [--seed s1] [--episodes 1] [--max-ticks 6000]
import { GameMapType } from "@openfront/engine-api/game/GameTypes";
import { OpenFrontEnv } from "../env";
import { HeuristicPolicy } from "../heuristicPolicy";
import { RandomPolicy } from "../randomPolicy";
import type { EnvConfig } from "../types";

function parse(argv: string[]) {
  const o = { policy: "random", map: "britannia", bots: 20, seed: "random-1", episodes: 1, maxTicks: 6000 };
  for (let i = 0; i < argv.length; i++) {
    const v = argv[i + 1];
    switch (argv[i]) {
      case "--policy": o.policy = v; i++; break;
      case "--map": o.map = v; i++; break;
      case "--bots": o.bots = parseInt(v, 10); i++; break;
      case "--seed": o.seed = v; i++; break;
      case "--episodes": o.episodes = parseInt(v, 10); i++; break;
      case "--max-ticks": o.maxTicks = parseInt(v, 10); i++; break;
      default: throw new Error(`argumento desconocido: ${argv[i]}`);
    }
  }
  return o;
}

async function main() {
  const o = parse(process.argv.slice(2));
  const key = Object.keys(GameMapType).find((k) => k.toLowerCase() === o.map.toLowerCase());
  if (!key) throw new Error(`mapa desconocido "${o.map}"`);
  if (o.policy !== "random" && o.policy !== "heuristic") throw new Error(`política desconocida "${o.policy}"`);
  const cfg: Partial<EnvConfig> = {
    map: GameMapType[key as keyof typeof GameMapType],
    bots: o.bots,
    maxTicks: o.maxTicks,
  };
  const env = await OpenFrontEnv.create(cfg);
  for (let ep = 0; ep < o.episodes; ep++) {
    const seed = `${o.seed}-${ep}`;
    const policy = o.policy === "heuristic" ? new HeuristicPolicy(seed, env.cfg.grid) : new RandomPolicy(seed, env.cfg.grid);
    let obs = await env.reset(seed);
    let steps = 0;
    let total = 0;
    let invalid = 0;
    const t0 = performance.now();
    const tick0 = obs.tick;
    for (;;) {
      const r = env.step(policy.act(obs));
      obs = r.obs;
      steps++;
      total += r.reward;
      if (r.info.invalidAction) invalid++;
      if (r.terminated || r.truncated) {
        const secs = (performance.now() - t0) / 1000;
        const ticks = r.info.tick - tick0;
        console.log(
          `[${o.policy}] episodio ${ep} seed=${seed}: ${r.terminated ? (r.info.won ? "VICTORIA" : r.info.eliminated ? "eliminado" : "terminado") : "truncado"} ` +
            `| pasos=${steps} ticks=${ticks} recompensa=${total.toFixed(4)} mapa=${(obs.globals[0] * 100).toFixed(1)}% inválidas=${invalid} ` +
            `| ${(ticks / secs).toFixed(0)} ticks/s, ${(steps / secs).toFixed(1)} pasos/s | hash=${env.stateHash()}`,
        );
        break;
      }
    }
  }
  env.close();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
