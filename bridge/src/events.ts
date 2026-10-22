// Recuentos de eventos por paso (docs/specs/protocol.md §4). No entran en la recompensa: sirven para registrar la conducta.
import { PlayerType } from "@openfront/engine-api/game/GameTypes";
import type { StampedIntent } from "@openfront/engine-api/Schemas";
import type { Game, Player } from "@openfront/engine/game/Game";
import { BUILD_UNIT, NUKE_UNIT } from "./mask";

export const EVENT_NAMES = [
  "attack_orders",
  "attack_tribe_orders",
  "expand_orders",
  "boat_orders",
  "build_orders",
  "nuke_orders",
  "tribes_conquered",
  "nations_conquered",
  "conquered_by_tribe",
  "conquered_by_nation",
  "built_city",
  "built_port",
  "built_factory",
  "built_post",
  "built_sam",
  "built_silo",
  "built_warship",
  "nukes_launched",
  "nuke_at_tribe",
  "nuke_at_nation",
  "nuke_at_nobody",
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
  conquered_by_tribe: 0,
  conquered_by_nation: 0,
  built_city: 0,
  built_port: 0,
  built_factory: 0,
  built_post: 0,
  built_sam: 0,
  built_silo: 0,
  built_warship: 0,
  nukes_launched: 0,
  nuke_at_tribe: 0,
  nuke_at_nation: 0,
  nuke_at_nobody: 0,
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
        if (NUKE_UNITS.has(it.unit)) {
          ev.nuke_orders++;
          const owner = game.owner(it.tile);
          if (!owner.isPlayer()) ev.nuke_at_nobody++;
          else if (owner.type() === PlayerType.Bot) ev.nuke_at_tribe++;
          else if (owner.type() === PlayerType.Nation) ev.nuke_at_nation++;
        } else ev.build_orders++;
        break;
    }
  }
}

/** Suma una conquista (jugador eliminado por `conqueror`) al agente que la hizo. */
export function countConquest(ev: Events, conquered: Player): void {
  if (conquered.type() === PlayerType.Bot) ev.tribes_conquered++;
  else if (conquered.type() === PlayerType.Nation) ev.nations_conquered++;
}

/** El agente fue eliminado por `conqueror`: se anota si fue una tribu o una nación. */
export function countConqueredBy(ev: Events, conqueror: Player): void {
  if (conqueror.type() === PlayerType.Bot) ev.conquered_by_tribe++;
  else if (conqueror.type() === PlayerType.Nation) ev.conquered_by_nation++;
}

const BUILT_EVENT = Object.fromEntries(Object.entries(BUILD_UNIT).map(([name, unit]) => [String(unit), `built_${name}`])) as Record<string, keyof Events>;

/** Suma una unidad que el agente acaba de colocar (las capturadas no pasan por `buildUnit`). */
export function countBuilt(ev: Events, unit: unknown): void {
  const key = BUILT_EVENT[String(unit)];
  if (key !== undefined) ev[key]++;
  else if (NUKE_UNITS.has(unit)) ev.nukes_launched++;
}
