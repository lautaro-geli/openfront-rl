import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { type Action, FRACTIONS, type Observation } from "./types";

/**
 * Política aleatoria que solo elige acciones válidas según la máscara.
 * Usa el PRNG del motor con semilla, para que las partidas sean reproducibles.
 */
export class RandomPolicy {
  private rng: PseudoRandom;
  constructor(seed: string, private grid: { h: number; w: number }) {
    this.rng = new PseudoRandom(simpleHash(seed));
  }

  act(obs: Observation): Action {
    const m = obs.mask;
    if (m.spawn) {
      // Solo celdas mayormente de tierra y sin dueño, para no elegir océano.
      const { h, w } = this.grid;
      const cells = h * w;
      const candidates: number[] = [];
      for (let c = 0; c < cells; c++) {
        if (obs.grid[c] >= 0.75 && obs.grid[cells + c] + obs.grid[2 * cells + c] === 0) candidates.push(c);
      }
      const c = candidates.length > 0 ? candidates[this.rng.nextInt(0, candidates.length)] : this.rng.nextInt(0, cells);
      return { kind: "spawn", cx: c % w, cy: Math.floor(c / w) };
    }
    const options: Action[] = [{ kind: "noop" }];
    const fraction = FRACTIONS[this.rng.nextInt(0, FRACTIONS.length)];
    if (m.expand) options.push({ kind: "expand", fraction });
    m.attackSlots.forEach((ok, slot) => {
      if (ok) options.push({ kind: "attack", slot, fraction });
    });
    return options[this.rng.nextInt(0, options.length)];
  }
}
