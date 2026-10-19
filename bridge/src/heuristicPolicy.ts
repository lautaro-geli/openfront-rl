import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { type Action, type Fraction, type Observation } from "./types";

export interface HeuristicOptions {
  /** Fracción mínima de tropas (sobre el máximo) para lanzar cualquier ataque. */
  troopFloor: number;
  /** Fracción de tropas que se envía en cada ataque. */
  fraction: Fraction;
  /** Cada cuántos pasos se permite atacar (evita gastar todas las tropas en seguida). */
  cooldownSteps: number;
}

export const DEFAULT_HEURISTIC: HeuristicOptions = { troopFloor: 0.4, fraction: 0.25, cooldownSteps: 1 };

/**
 * Política escrita a mano, sin aprendizaje. Solo lee la observación (como lo haría un agente de RL):
 *  - Spawn: la celda de tierra libre con más tierra a su alrededor y sin otros jugadores cerca.
 *  - Expandir sobre tierra libre mientras la frontera propia toque tierra sin dueño.
 *  - Si no hay tierra libre accesible, atacar al vecino válido más pequeño (el último slot: van ordenados por tiles).
 *  - Nunca gasta tropas por debajo de `troopFloor`.
 * Sirve para comprobar que el entorno es jugable y como línea base que un agente aprendido debe superar.
 */
export class HeuristicPolicy {
  private rng: PseudoRandom;
  private sinceAttack = Number.POSITIVE_INFINITY;

  constructor(
    seed: string,
    private grid: { h: number; w: number },
    private opts: HeuristicOptions = DEFAULT_HEURISTIC,
  ) {
    this.rng = new PseudoRandom(simpleHash(seed));
  }

  act(obs: Observation): Action {
    if (obs.mask.spawn) return this.spawn(obs);
    this.sinceAttack++;
    const troops = obs.globals[1];
    if (troops < this.opts.troopFloor || this.sinceAttack < this.opts.cooldownSteps) return { kind: "noop" };

    const fraction = this.opts.fraction;
    if (obs.mask.expand && this.touchesFreeLand(obs)) {
      this.sinceAttack = 0;
      return { kind: "expand", fraction };
    }
    for (let slot = obs.mask.attackSlots.length - 1; slot >= 0; slot--) {
      if (obs.mask.attackSlots[slot]) {
        this.sinceAttack = 0;
        return { kind: "attack", slot, fraction };
      }
    }
    if (obs.mask.expand) {
      this.sinceAttack = 0;
      return { kind: "expand", fraction };
    }
    return { kind: "noop" };
  }

  private spawn(obs: Observation): Action {
    const { h, w } = this.grid;
    const cells = h * w;
    const R = 2;
    let best = -1;
    let bestScore = -1;
    const ties: number[] = [];
    for (let cy = R; cy < h - R; cy++) {
      for (let cx = R; cx < w - R; cx++) {
        const c = cy * w + cx;
        if (obs.grid[c] < 0.95 || obs.grid[cells + c] + obs.grid[2 * cells + c] > 0) continue;
        let land = 0;
        let taken = 0;
        for (let dy = -R; dy <= R; dy++) {
          for (let dx = -R; dx <= R; dx++) {
            const k = (cy + dy) * w + (cx + dx);
            land += obs.grid[k];
            taken += obs.grid[cells + k] + obs.grid[2 * cells + k];
          }
        }
        const score = land - 4 * taken;
        if (score > bestScore) {
          bestScore = score;
          best = c;
          ties.length = 0;
          ties.push(c);
        } else if (score === bestScore) {
          ties.push(c);
        }
      }
    }
    if (ties.length > 1) best = ties[this.rng.nextInt(0, ties.length)];
    if (best < 0) best = this.rng.nextInt(0, cells);
    return { kind: "spawn", cx: best % w, cy: Math.floor(best / w) };
  }

  /** ¿Hay una celda propia vecina (4 direcciones) de una celda con tierra sin dueño? */
  private touchesFreeLand(obs: Observation): boolean {
    const { h, w } = this.grid;
    const cells = h * w;
    const free = (c: number) => obs.grid[c] - obs.grid[cells + c] - obs.grid[2 * cells + c] > 0.1;
    for (let cy = 0; cy < h; cy++) {
      for (let cx = 0; cx < w; cx++) {
        const c = cy * w + cx;
        if (obs.grid[cells + c] <= 0) continue;
        if (free(c)) return true; // la propia celda ya tiene tierra libre en su borde
        if (cx > 0 && free(c - 1)) return true;
        if (cx < w - 1 && free(c + 1)) return true;
        if (cy > 0 && free(c - w)) return true;
        if (cy < h - 1 && free(c + w)) return true;
      }
    }
    return false;
  }
}
