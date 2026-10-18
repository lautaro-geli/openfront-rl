import type { Game, Player } from "@openfront/engine/game/Game";
import {
  type ActionMask,
  type EnvConfig,
  NUM_CHANNELS,
  NUM_GLOBALS,
  type Observation,
} from "./types";

const SAMPLES = 4; // malla de SAMPLES x SAMPLES tiles por celda
const GOLD_LOG_NORM = Math.log1p(1e7);

/** Tiles que se muestrean para cada celda del grid. Se calcula una vez por mapa. */
export function buildSamplePlan(game: Game, h: number, w: number): Int32Array {
  const map = game.map();
  const W = map.width();
  const H = map.height();
  const plan = new Int32Array(h * w * SAMPLES * SAMPLES);
  let k = 0;
  for (let cy = 0; cy < h; cy++) {
    for (let cx = 0; cx < w; cx++) {
      for (let sy = 0; sy < SAMPLES; sy++) {
        for (let sx = 0; sx < SAMPLES; sx++) {
          const x = Math.min(W - 1, Math.floor(((cx + (sx + 0.5) / SAMPLES) * W) / w));
          const y = Math.min(H - 1, Math.floor(((cy + (sy + 0.5) / SAMPLES) * H) / h));
          plan[k++] = map.ref(x, y);
        }
      }
    }
  }
  return plan;
}

export interface ObsContext {
  cfg: EnvConfig;
  plan: Int32Array;
  initialPlayers: number;
  ticksSinceSpawnEnd: number;
}

export function buildObservation(
  game: Game,
  me: Player,
  ctx: ObsContext,
  mask: ActionMask,
): Observation {
  const { h, w } = ctx.cfg.grid;
  const map = game.map();
  const grid = new Float32Array(NUM_CHANNELS * h * w);
  const cells = h * w;
  const per = SAMPLES * SAMPLES;
  const meSmall = me.smallID();
  let k = 0;
  for (let c = 0; c < cells; c++) {
    let land = 0;
    let own = 0;
    let other = 0;
    for (let s = 0; s < per; s++) {
      const ref = ctx.plan[k++];
      if (map.isLand(ref)) land++;
      if (map.hasOwner(ref)) {
        if (map.ownerID(ref) === meSmall) own++;
        else other++;
      }
    }
    grid[c] = land / per;
    grid[cells + c] = own / per;
    grid[2 * cells + c] = other / per;
  }

  const inSpawn = game.inSpawnPhase();
  const globals = new Float32Array(NUM_GLOBALS);
  const landTiles = Math.max(1, game.numLandTiles());
  globals[0] = Math.min(1, me.numTilesOwned() / landTiles);
  const maxTroops = Math.max(1, game.config().maxTroops(me));
  globals[1] = Math.min(1, Math.max(0, me.troops() / maxTroops));
  globals[2] = Math.min(1, Math.log1p(Number(me.gold())) / GOLD_LOG_NORM);
  globals[3] = inSpawn
    ? 1
    : Math.min(1, game.players().length / Math.max(1, ctx.initialPlayers));
  globals[4] = Math.min(1, ctx.ticksSinceSpawnEnd / ctx.cfg.maxTicks);
  globals[5] = inSpawn ? 1 : 0;

  return { grid, globals, mask, tick: game.ticks(), inSpawnPhase: inSpawn };
}
