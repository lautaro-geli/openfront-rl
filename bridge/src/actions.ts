import type { Game, Player } from "@openfront/engine/game/Game";
import type { StampedIntent } from "@openfront/engine-api/Schemas";
import { type Action, type ActionMask, FRACTIONS } from "./types";

/** Jugadores vivos que comparten frontera con el agente, ordenados por tiles desc y smallID asc. */
export function neighborSlots(game: Game, me: Player, maxNeighbors: number): Player[] {
  if (!me.isAlive()) return [];
  return game
    .players()
    .filter((p) => p !== me && p.isAlive() && me.sharesBorderWith(p))
    .sort((a, b) => b.numTilesOwned() - a.numTilesOwned() || a.smallID() - b.smallID())
    .slice(0, maxNeighbors);
}

export function computeMask(game: Game, me: Player, maxNeighbors: number): ActionMask {
  const inSpawn = game.inSpawnPhase();
  const canAct = !inSpawn && me.isAlive() && me.troops() > 0;
  const slots = neighborSlots(game, me, maxNeighbors);
  const attackSlots: boolean[] = [];
  for (let i = 0; i < maxNeighbors; i++) {
    const p = slots[i];
    attackSlots.push(canAct && p !== undefined && me.canAttackPlayer(p));
  }
  return { spawn: inSpawn, expand: canAct, attackSlots };
}

/** Tile de tierra libre más cercano al centro de la celda (búsqueda en anillos, determinista). */
export function cellToSpawnTile(game: Game, cx: number, cy: number, h: number, w: number): number | null {
  const map = game.map();
  const W = map.width();
  const H = map.height();
  const x0 = Math.floor(((cx + 0.5) * W) / w);
  const y0 = Math.floor(((cy + 0.5) * H) / h);
  const MAX_R = 64;
  for (let r = 0; r <= MAX_R; r++) {
    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
        const x = x0 + dx;
        const y = y0 + dy;
        if (!map.isValidCoord(x, y)) continue;
        const ref = map.ref(x, y);
        if (map.isLand(ref) && !map.hasOwner(ref)) return ref;
      }
    }
  }
  return null;
}

export interface BuiltIntents {
  intents: StampedIntent[];
  invalid: boolean;
}

export function actionToIntents(
  action: Action,
  game: Game,
  me: Player,
  clientID: string,
  mask: ActionMask,
  grid: { h: number; w: number },
  maxNeighbors: number,
): BuiltIntents {
  const none: BuiltIntents = { intents: [], invalid: false };
  switch (action.kind) {
    case "noop":
      return none;
    case "spawn": {
      if (!mask.spawn) return { intents: [], invalid: true };
      if (action.cx < 0 || action.cx >= grid.w || action.cy < 0 || action.cy >= grid.h) {
        return { intents: [], invalid: true };
      }
      const tile = cellToSpawnTile(game, action.cx, action.cy, grid.h, grid.w);
      if (tile === null) return { intents: [], invalid: true };
      return { intents: [{ type: "spawn", tile, clientID }], invalid: false };
    }
    case "expand": {
      if (!mask.expand || !FRACTIONS.includes(action.fraction)) return { intents: [], invalid: true };
      return {
        intents: [{ type: "attack", targetID: null, troops: me.troops() * action.fraction, clientID }],
        invalid: false,
      };
    }
    case "attack": {
      if (
        !Number.isInteger(action.slot) ||
        action.slot < 0 ||
        action.slot >= maxNeighbors ||
        !mask.attackSlots[action.slot] ||
        !FRACTIONS.includes(action.fraction)
      ) {
        return { intents: [], invalid: true };
      }
      const target = neighborSlots(game, me, maxNeighbors)[action.slot];
      if (target === undefined) return { intents: [], invalid: true };
      return {
        intents: [{ type: "attack", targetID: target.id(), troops: me.troops() * action.fraction, clientID }],
        invalid: false,
      };
    }
  }
}
