// Máscara ampliada de acciones (docs/specs/actions.md §4). Es conservadora: la resolución exacta se hace al traducir la acción.
import { UnitType } from "@openfront/engine-api/game/GameTypes";
import type { Game, Player, Unit } from "@openfront/engine/game/Game";
import {
  BUILD_TYPES,
  CELL_KINDS,
  CH,
  KIND,
  MAX_ATTACKS_PER_SIDE,
  MAX_ECON_ROWS,
  NUKE_KINDS,
  NUM_KINDS,
  PLAYER_KINDS,
  PLAYER_ROWS,
  UNIT_KINDS,
  UNIT_ROWS,
} from "./features";
import type { ActionMask } from "./types";

export const BUILD_UNIT: Record<(typeof BUILD_TYPES)[number], UnitType> = {
  city: UnitType.City,
  port: UnitType.Port,
  factory: UnitType.Factory,
  post: UnitType.DefensePost,
  sam: UnitType.SAMLauncher,
  silo: UnitType.MissileSilo,
  warship: UnitType.Warship,
};
export const NUKE_UNIT: Record<(typeof NUKE_KINDS)[number], UnitType> = {
  atom: UnitType.AtomBomb,
  hydrogen: UnitType.HydrogenBomb,
  mirv: UnitType.MIRV,
};

/** Las filas de las tablas de la observación, para poder traducir una fila elegida por el agente a un objeto del motor. */
export interface ObsRows {
  /** Tabla de jugadores: fila 0 = el agente. */
  players: (Player | null)[];
  /** Filas de unidad U: 0..47 economía, 48..143 militar. `null` si la fila está vacía. */
  units: (Unit | null)[];
  /** Ataques salientes, en el orden de la tabla de ataques. */
  outgoing: { id(): string; retreating(): boolean; isActive(): boolean }[];
}

export function emptyMask(maxNeighbors: number, cells: number): ActionMask {
  return {
    spawn: false,
    expand: false,
    attackSlots: Array(maxNeighbors).fill(false),
    kind: new Uint8Array(NUM_KINDS),
    player: new Uint8Array(PLAYER_KINDS.length * PLAYER_ROWS),
    unit: new Uint8Array(UNIT_KINDS.length * UNIT_ROWS),
    attackRow: new Uint8Array(MAX_ATTACKS_PER_SIDE),
    buildType: new Uint8Array(BUILD_TYPES.length),
    nukeType: new Uint8Array(NUKE_KINDS.length),
    cell: new Uint8Array(CELL_KINDS.length * cells),
  };
}

export function unitRowsFrom(econ: Unit[], mil: Unit[]): (Unit | null)[] {
  const out: (Unit | null)[] = Array(UNIT_ROWS).fill(null);
  econ.forEach((u, i) => {
    out[i] = u;
  });
  mil.forEach((u, i) => {
    out[MAX_ECON_ROWS + i] = u;
  });
  return out;
}

const any = (a: Uint8Array): boolean => a.some((v) => v !== 0);

/**
 * Rellena los bloques ampliados de `mask`. `grid` es el grid de la observación (canales land, own, other, shore).
 * Los campos de la v0 (spawn, expand, attackSlots) ya vienen puestos por `computeMaskInfo`.
 */
export function extendMask(game: Game, me: Player, mask: ActionMask, rows: ObsRows, grid: Float32Array, cells: number): void {
  const inSpawn = game.inSpawnPhase();
  const canAct = mask.expand;
  const config = game.config();
  const gold = Number(me.gold());
  mask.kind.fill(0);
  mask.kind[KIND.noop] = 1;

  // Jugadores.
  const P = (kind: (typeof PLAYER_KINDS)[number], row: number, ok: boolean): void => {
    if (ok) mask.player[PLAYER_KINDS.indexOf(kind) * PLAYER_ROWS + row] = 1;
  };
  const incoming = new Set<Player>(me.incomingAllianceRequests().map((r) => r.requestor()));
  const goldPos = gold > 0;
  const troopsPos = me.troops() > 0;
  for (let r = 0; r < PLAYER_ROWS; r++) {
    const p = rows.players[r] ?? null;
    if (r === 0) {
      P("emoji", 0, canAct); // fila 0 = todos los jugadores
      continue;
    }
    if (p === null || !p.isAlive() || !canAct) continue;
    if (r <= mask.attackSlots.length) P("attack", r, mask.attackSlots[r - 1]);
    P("alliance_request", r, me.canSendAllianceRequest(p));
    P("alliance_reject", r, incoming.has(p));
    P("alliance_break", r, me.isAlliedWith(p));
    P("alliance_extend", r, me.allianceInfo(p)?.canExtend === true);
    const embargoed = me.hasEmbargoAgainst(p);
    P("embargo_start", r, !embargoed);
    P("embargo_stop", r, embargoed);
    P("target_player", r, me.canTarget(p));
    P("emoji", r, true);
    P("donate_gold", r, goldPos && me.canDonateGold(p));
    P("donate_troops", r, troopsPos && me.canDonateTroops(p));
  }

  // Unidades.
  const U = (kind: (typeof UNIT_KINDS)[number], row: number, ok: boolean): void => {
    if (ok) mask.unit[UNIT_KINDS.indexOf(kind) * UNIT_ROWS + row] = 1;
  };
  if (canAct) {
    const canDelete = me.canDeleteUnit();
    rows.units.forEach((u, i) => {
      if (u === null || !u.isActive() || u.owner() !== me) return;
      const t = u.type();
      U("upgrade", i, me.canUpgradeUnit(u));
      U(
        "delete_unit",
        i,
        canDelete &&
          !u.isMarkedForDeletion() &&
          game.isLand(u.tile()) &&
          game.owner(u.tile()) === me,
      );
      U("cancel_boat", i, t === UnitType.TransportShip && !u.transportShipState().isRetreating);
      U("move_warship", i, t === UnitType.Warship);
    });
  }

  // Ataques salientes que se pueden cancelar.
  rows.outgoing.slice(0, MAX_ATTACKS_PER_SIDE).forEach((a, i) => {
    if (canAct && a.isActive() && !a.retreating()) mask.attackRow[i] = 1;
  });

  // Tipos de estructura y de bomba.
  const hasPort = me.units(UnitType.Port).some((u) => u.isActive() && !u.isUnderConstruction());
  BUILD_TYPES.forEach((name, i) => {
    const t = BUILD_UNIT[name];
    if (!canAct || config.isUnitDisabled(t)) return;
    if (name === "warship" && !hasPort) return;
    if (gold >= Number(game.unitInfo(t).cost(game, me))) mask.buildType[i] = 1;
  });
  const siloReady = me
    .units(UnitType.MissileSilo)
    .some((s) => s.isActive() && !s.isInCooldown() && !s.isUnderConstruction());
  NUKE_KINDS.forEach((name, i) => {
    const t = NUKE_UNIT[name];
    if (!canAct || !siloReady || game.isSpawnImmunityActive() || config.isUnitDisabled(t)) return;
    if (gold >= Number(game.unitInfo(t).cost(game, me))) mask.nukeType[i] = 1;
  });

  // Celdas.
  const shipIdx = BUILD_TYPES.indexOf("warship");
  const canBuildLand = mask.buildType.some((v, i) => v === 1 && i !== shipIdx);
  const canBuildShip = mask.buildType[shipIdx] === 1;
  const canNuke = any(mask.nukeType);
  const boatsFree = me.unitCount(UnitType.TransportShip) < config.boatMaxNumber();
  const hasWarship = me.units(UnitType.Warship).some((u) => u.isActive());
  const cellRow = (kind: (typeof CELL_KINDS)[number]): number => CELL_KINDS.indexOf(kind) * cells;
  const sp = cellRow("spawn");
  const bd = cellRow("build");
  const bs = cellRow("build_ship");
  const nk = cellRow("nuke");
  const bt = cellRow("boat_attack");
  const mw = cellRow("move_warship");
  const EPS = 0.01;
  for (let c = 0; c < cells; c++) {
    const land = grid[CH.land * cells + c];
    const own = grid[CH.own * cells + c];
    const other = grid[CH.other * cells + c];
    const shore = grid[CH.shore * cells + c];
    const nonOwnLand = land - own > EPS;
    if (inSpawn && !me.hasSpawned() && land - own - other > EPS) mask.cell[sp + c] = 1;
    if (canBuildLand && own > 0) mask.cell[bd + c] = 1;
    if (canBuildShip && land < 1) mask.cell[bs + c] = 1;
    if (canNuke && nonOwnLand) mask.cell[nk + c] = 1;
    if (canAct && boatsFree && shore > 0 && nonOwnLand) mask.cell[bt + c] = 1;
    if (canAct && hasWarship && land < 1) mask.cell[mw + c] = 1;
  }

  // Tipos de acción.
  const k = mask.kind;
  const has = (kind: (typeof PLAYER_KINDS)[number]): boolean => {
    const i = PLAYER_KINDS.indexOf(kind);
    return any(mask.player.subarray(i * PLAYER_ROWS, (i + 1) * PLAYER_ROWS));
  };
  const hasU = (kind: (typeof UNIT_KINDS)[number]): boolean => {
    const i = UNIT_KINDS.indexOf(kind);
    return any(mask.unit.subarray(i * UNIT_ROWS, (i + 1) * UNIT_ROWS));
  };
  const hasC = (kind: (typeof CELL_KINDS)[number]): boolean => any(mask.cell.subarray(cellRow(kind), cellRow(kind) + cells));
  k[KIND.spawn] = mask.spawn && hasC("spawn") ? 1 : 0;
  k[KIND.expand] = canAct ? 1 : 0;
  k[KIND.attack] = mask.attackSlots.some(Boolean) ? 1 : 0;
  k[KIND.boat_attack] = hasC("boat_attack") ? 1 : 0;
  k[KIND.cancel_attack] = any(mask.attackRow) ? 1 : 0;
  k[KIND.cancel_boat] = hasU("cancel_boat") ? 1 : 0;
  k[KIND.build] = (canBuildLand && hasC("build")) || (canBuildShip && hasC("build_ship")) ? 1 : 0;
  k[KIND.upgrade] = hasU("upgrade") ? 1 : 0;
  k[KIND.delete_unit] = hasU("delete_unit") ? 1 : 0;
  k[KIND.nuke] = canNuke && hasC("nuke") ? 1 : 0;
  k[KIND.move_warship] = hasU("move_warship") && hasC("move_warship") ? 1 : 0;
  k[KIND.alliance_request] = has("alliance_request") ? 1 : 0;
  k[KIND.alliance_reject] = has("alliance_reject") ? 1 : 0;
  k[KIND.alliance_break] = has("alliance_break") ? 1 : 0;
  k[KIND.alliance_extend] = has("alliance_extend") ? 1 : 0;
  k[KIND.embargo_start] = has("embargo_start") ? 1 : 0;
  k[KIND.embargo_stop] = has("embargo_stop") ? 1 : 0;
  k[KIND.embargo_all_start] = canAct ? 1 : 0;
  k[KIND.embargo_all_stop] = canAct ? 1 : 0;
  k[KIND.target_player] = has("target_player") ? 1 : 0;
  k[KIND.emoji] = has("emoji") ? 1 : 0;
  k[KIND.donate_gold] = has("donate_gold") ? 1 : 0;
  k[KIND.donate_troops] = has("donate_troops") ? 1 : 0;
}
