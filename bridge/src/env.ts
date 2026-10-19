import { DEFAULT_CONFIG, DEFAULT_MAPS_DIR, MultiAgentEnv } from "./multiEnv";
import type { Action, EnvConfig, Observation, StepResult } from "./types";

export { DEFAULT_CONFIG };

/** Entorno de un solo agente: envoltorio de MultiAgentEnv con agents = 1. */
export class OpenFrontEnv {
  readonly cfg: EnvConfig;
  private constructor(private inner: MultiAgentEnv) {
    this.cfg = inner.cfg;
  }

  static async create(cfg: Partial<EnvConfig> = {}, mapsDir: string = DEFAULT_MAPS_DIR): Promise<OpenFrontEnv> {
    if (cfg.agents !== undefined && cfg.agents !== 1) {
      throw new Error("OpenFrontEnv controla un solo agente; usa MultiAgentEnv para varios");
    }
    return new OpenFrontEnv(await MultiAgentEnv.create({ ...cfg, agents: 1 }, mapsDir));
  }

  async reset(seed: string): Promise<Observation> {
    return (await this.inner.reset(seed))[0];
  }

  step(action: Action): StepResult {
    const r = this.inner.step([action]);
    return {
      obs: r.obs[0],
      reward: r.reward[0],
      terminated: r.terminated[0],
      truncated: r.truncated,
      info: r.info[0],
    };
  }

  stateHash(): string {
    return this.inner.stateHash();
  }

  close(): void {
    this.inner.close();
  }
}
