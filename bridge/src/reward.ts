// Recompensa por componentes v2 (docs/specs/reward.md). La total es la suma de las partes.
import type { Game, Player } from "@openfront/engine/game/Game";
import { UnitType } from "@openfront/engine-api/game/GameTypes";
import { allNeighbors } from "./actions";
import type { EnvConfig } from "./types";

export const REWARD_PARTS = ["territory", "attack", "economy", "defense", "terminal"] as const;
export type RewardParts = Record<(typeof REWARD_PARTS)[number], number>;

export const zeroParts = (): RewardParts => ({ territory: 0, attack: 0, economy: 0, defense: 0, terminal: 0 });

/** Estado por agente entre pasos: el potencial anterior de cada categoría. */
export interface RewardState {
  prevTerritory: number;
  prevAttack: number | null;
  prevEconomy: number | null;
  prevDefense: number | null;
  prevEarned: number | null;
  /** Media móvil del oro ganado por tick (suaviza los ingresos a golpes del comercio). */
  rateEma: number | null;
}

const INCOME_EMA = 0.95;
const WIN_FRAC = 0.8;
const CITY_TROOPS_FULL = 1_000_000;
const BASE_GOLD_RATE = 100;

export const newRewardState = (): RewardState => ({
  prevTerritory: 0,
  prevAttack: null,
  prevEconomy: null,
  prevDefense: null,
  prevEarned: null,
  rateEma: null,
});

const clip01 = (x: number): number => Math.max(0, Math.min(1, x));
const logScale = (x: number, base: number, factor: number): number => clip01(Math.log(Math.max(x, 1) / base) / Math.log(factor));

const finished = (me: Player, t: UnitType): number => me.units(t).filter((u) => u.isActive() && !u.isUnderConstruction()).length;

/**
 * Potencial de ataque: tropas máximas que aportan las ciudades terminadas y silos terminados (spec §2). No usa el tope
 * total de tropas porque crece con el territorio y premiaría dos veces lo mismo.
 */
export function attackPotential(game: Game, me: Player): number {
  const cfg = game.config();
  const troops = clip01((cfg.maxTroops(me) - cfg.maxTroopsWithoutCities(me)) / CITY_TROOPS_FULL);
  const silos = Math.min(1, finished(me, UnitType.MissileSilo) / 2);
  return 0.8 * troops + 0.2 * silos;
}

/** Potencial de defensa: seguridad y puestos de defensa terminados (spec §2). */
export function defensePotential(game: Game, me: Player): number {
  return 0.5 * securityPotential(game, me) + 0.5 * Math.min(1, finished(me, UnitType.DefensePost) / 4);
}

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
 * fase de spawn; un agente ya eliminado no cobra nada. Cada categoría paga `peso · Δ potencial` (spec §1).
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
  const live = alive && !f.eliminatedNow;

  const terr = clip01(me.numTilesOwned() / f.landTiles / WIN_FRAC);
  if (alive) parts.territory = cfg.rewardTerritoryWeight * (terr - st.prevTerritory);
  st.prevTerritory = f.afterSpawn ? terr : 0;

  if (cfg.rewardAttackWeight !== 0) {
    if (live) {
      const phi = attackPotential(game, me);
      if (st.prevAttack !== null) parts.attack = cfg.rewardAttackWeight * (phi - st.prevAttack);
      st.prevAttack = phi;
    } else st.prevAttack = null;
  }

  if (cfg.rewardEconomyWeight !== 0) {
    const earned = Number(me.goldEarned());
    if (live && st.prevEarned !== null) {
      const rate = Math.max(0, earned - st.prevEarned) / cfg.decisionInterval;
      st.rateEma = st.rateEma === null ? rate : INCOME_EMA * st.rateEma + (1 - INCOME_EMA) * rate;
      const phi = logScale(st.rateEma, BASE_GOLD_RATE, 10);
      if (st.prevEconomy !== null) parts.economy = cfg.rewardEconomyWeight * (phi - st.prevEconomy);
      st.prevEconomy = phi;
    }
    st.prevEarned = live ? earned : null;
    if (!live) {
      st.rateEma = null;
      st.prevEconomy = null;
    }
  }

  if (cfg.rewardDefenseWeight !== 0) {
    if (live) {
      const phi = defensePotential(game, me);
      if (st.prevDefense !== null) parts.defense = cfg.rewardDefenseWeight * (phi - st.prevDefense);
      st.prevDefense = phi;
    } else st.prevDefense = null;
  }

  parts.terminal = f.wasEliminated ? 0 : f.won ? 1 : f.eliminatedNow ? -1 : 0;
  return parts;
}

export const sumParts = (p: RewardParts): number => REWARD_PARTS.reduce((s, k) => s + p[k], 0);
