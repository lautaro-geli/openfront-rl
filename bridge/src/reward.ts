// Recompensa por componentes (docs/specs/reward.md). La total es la suma de las partes.
import type { Game, Player } from "@openfront/engine/game/Game";
import { allNeighbors } from "./actions";
import type { EnvConfig } from "./types";

export const REWARD_PARTS = ["territory", "income", "security", "terminal"] as const;
export type RewardParts = Record<(typeof REWARD_PARTS)[number], number>;

export const zeroParts = (): RewardParts => ({ territory: 0, income: 0, security: 0, terminal: 0 });

/** Estado por agente entre pasos. */
export interface RewardState {
  prevFrac: number;
  prevEarned: number | null;
  /** Media móvil del oro ganado por tick (suaviza los ingresos a golpes del comercio). */
  rateEma: number | null;
  prevLogRate: number | null;
  prevPhi: number | null;
}

const INCOME_EMA = 0.8;

export const newRewardState = (): RewardState => ({ prevFrac: 0, prevEarned: null, rateEma: null, prevLogRate: null, prevPhi: null });

/**
 * Potencial de seguridad Φ ∈ [0,1] (spec §2): fuerza frente al vecino hostil más fuerte, presión de ataques entrantes
 * y cuántos frentes hostiles tengo.
 */
export function securityPotential(game: Game, me: Player): number {
  const troops = Math.max(0, me.troops());
  const hostile = allNeighbors(game, me).filter((p) => !me.isFriendly(p));
  const strongest = hostile.reduce((m, p) => Math.max(m, p.troops()), 0);
  const power = hostile.length === 0 ? 1 : troops / Math.max(1, troops + strongest);
  const incoming = me
    .incomingAttacks()
    .filter((a) => a.isActive())
    .reduce((s, a) => s + a.troops(), 0);
  const pressure = 1 - Math.min(1, incoming / Math.max(1, troops));
  const exposure = 1 - Math.min(1, hostile.length / 6);
  return (power + pressure + exposure) / 3;
}

/**
 * Calcula los componentes de un agente tras un paso y actualiza su estado. `afterSpawn` es false mientras dura la
 * fase de spawn; un agente ya eliminado no cobra nada.
 */
export function computeRewardParts(
  game: Game,
  me: Player,
  st: RewardState,
  cfg: EnvConfig,
  f: { afterSpawn: boolean; wasEliminated: boolean; eliminatedNow: boolean; won: boolean; landTiles: number },
): RewardParts {
  const parts = zeroParts();
  const alive = f.afterSpawn && !f.wasEliminated;
  const frac = me.numTilesOwned() / f.landTiles;
  if (alive) parts.territory = cfg.rewardTerritoryWeight * (frac - st.prevFrac);
  st.prevFrac = f.afterSpawn ? frac : 0;

  if (cfg.rewardIncomeWeight !== 0 || cfg.rewardSecurityWeight !== 0) {
    const earned = Number(me.goldEarned());
    if (alive && st.prevEarned !== null && cfg.rewardIncomeWeight !== 0) {
      const rate = Math.max(0, earned - st.prevEarned) / cfg.decisionInterval;
      st.rateEma = st.rateEma === null ? rate : INCOME_EMA * st.rateEma + (1 - INCOME_EMA) * rate;
      const logRate = Math.log(Math.max(1, st.rateEma));
      if (st.prevLogRate !== null) parts.income = cfg.rewardIncomeWeight * (logRate - st.prevLogRate);
      st.prevLogRate = logRate;
    }
    st.prevEarned = f.afterSpawn ? earned : null;
    if (!alive) {
      st.rateEma = null;
      st.prevLogRate = null;
    }
    if (cfg.rewardSecurityWeight !== 0) {
      if (alive && !f.eliminatedNow) {
        const phi = securityPotential(game, me);
        if (st.prevPhi !== null) parts.security = cfg.rewardSecurityWeight * (phi - st.prevPhi);
        st.prevPhi = phi;
      } else {
        st.prevPhi = null;
      }
    }
  }

  parts.terminal = f.wasEliminated ? 0 : f.won ? 1 : f.eliminatedNow ? -1 : 0;
  return parts;
}

export const sumParts = (p: RewardParts): number => p.territory + p.income + p.security + p.terminal;
