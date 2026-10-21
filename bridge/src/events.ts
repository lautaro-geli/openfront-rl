// Recuentos de eventos por paso (docs/specs/protocol.md §4). No entran en la recompensa: sirven para registrar la conducta.
import { PlayerType } from "@openfront/engine-api/game/GameTypes";
import type { StampedIntent } from "@openfront/engine-api/Schemas";
import type { Game, Player } from "@openfront/engine/game/Game";
import { NUKE_UNIT } from "./mask";

export const EVENT_NAMES = [
  "attack_orders",
  "attack_tribe_orders",
  "expand_orders",
  "boat_orders",
  "build_orders",
  "nuke_orders",
  "tribes_conquered",
  "nations_conquered",
] as const;
export type Events = Record<(typeof EVENT_NAMES)[number], number>;

export const zeroEvents = (): Events => ({
  attack_orders: 0,
  attack_tribe_orders: 0,
  expand_orders: 0,
  boat_orders: 0,
  build_orders: 0,
  nuke_orders: 0,
  tribes_conquered: 0,
  nations_conquered: 0,
});

const NUKE_UNITS = new Set<unknown>(Object.values(NUKE_UNIT));

/** Cuenta las órdenes válidas de un agente a partir de las intenciones ya construidas. */
export function countIntentEvents(game: Game, intents: StampedIntent[], ev: Events): void {
  for (const it of intents) {
    switch (it.type) {
      case "attack": {
        if (it.targetID === null) {
          ev.expand_orders++;
          break;
        }
        ev.attack_orders++;
        if (game.player(it.targetID).type() === PlayerType.Bot) ev.attack_tribe_orders++;
        break;
      }
      case "boat":
        ev.boat_orders++;
        break;
      case "build_unit":
        if (NUKE_UNITS.has(it.unit)) ev.nuke_orders++;
        else ev.build_orders++;
        break;
    }
  }
}

/** Suma una conquista (jugador eliminado por `conqueror`) al agente que la hizo. */
export function countConquest(ev: Events, conquered: Player): void {
  if (conquered.type() === PlayerType.Bot) ev.tribes_conquered++;
  else if (conquered.type() === PlayerType.Nation) ev.nations_conquered++;
}
