import { AllPlayers, UnitType } from "@openfront/engine-api/game/GameTypes";
import type { StampedIntent } from "@openfront/engine-api/Schemas";
import { flattenedEmojiTable } from "@openfront/engine-api/Schemas";
import type { Game, Player } from "@openfront/engine/game/Game";
import {
  ACTION_KINDS,
  BUILD_TYPES,
  CELL_KINDS,
  EMOJI_CHARS,
  KIND,
  NUKE_KINDS,
  NUM_EMOJIS,
  PLAYER_KINDS,
  PLAYER_ROWS,
  UNIT_KINDS,
  UNIT_ROWS,
} from "./features";
import { BUILD_UNIT, emptyMask, NUKE_UNIT, type ObsRows } from "./mask";
import { type Action, type ActionMask, FRACTIONS, type Fraction } from "./types";

/** Todos los jugadores vivos que comparten frontera con el agente, ordenados por tiles desc y smallID asc. */
export function allNeighbors(game: Game, me: Player): Player[] {
  if (!me.isAlive()) return [];
  return game
    .players()
    .filter((p) => p !== me && p.isAlive() && me.sharesBorderWith(p))
    .sort((a, b) => b.numTilesOwned() - a.numTilesOwned() || a.smallID() - b.smallID());
}

/** Los primeros `maxNeighbors` vecinos: las ranuras de ataque. */
export function neighborSlots(game: Game, me: Player, maxNeighbors: number): Player[] {
  return allNeighbors(game, me).slice(0, maxNeighbors);
}

/** Máscara de acciones junto con los vecinos de cada slot (para no recalcularlos al traducir la acción). */
export interface MaskInfo {
  mask: ActionMask;
  slots: Player[];
  /** Todos los vecinos (las ranuras son sus primeros `maxNeighbors`). La observación los usa. */
  neighbors: Player[];
  /** Filas de las tablas de la observación. Las rellena `buildObservation`. */
  rows?: ObsRows;
}

export function computeMaskInfo(game: Game, me: Player, maxNeighbors: number, cells: number): MaskInfo {
  const inSpawn = game.inSpawnPhase();
  const canAct = !inSpawn && me.isAlive() && me.troops() > 0;
  const neighbors = allNeighbors(game, me);
  const slots = neighbors.slice(0, maxNeighbors);
  const mask = emptyMask(maxNeighbors, cells);
  mask.spawn = inSpawn;
  mask.expand = canAct;
  for (let i = 0; i < maxNeighbors; i++) {
    const p = slots[i];
    mask.attackSlots[i] = canAct && p !== undefined && me.canAttackPlayer(p);
  }
  return { mask, slots, neighbors };
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

const NONE: BuiltIntents = { intents: [], invalid: false };
const INVALID: BuiltIntents = { intents: [], invalid: true };
const ok = (intent: unknown): BuiltIntents => ({ intents: [intent as StampedIntent], invalid: false });

/** Convierte el vector `[kind, row, cell, fraction, type, aux]` del protocolo en una acción. */
export function vecToAction(v: ArrayLike<number>, grid: { h: number; w: number }): Action | null {
  const kind = ACTION_KINDS[v[0]];
  if (kind === undefined) return null; // tipo desconocido: el entorno lo cuenta como acción inválida
  const fraction = FRACTIONS[v[3]];
  switch (kind) {
    case "noop":
      return { kind };
    case "spawn":
      return { kind, cx: v[2] % grid.w, cy: Math.floor(v[2] / grid.w) };
    case "expand":
      return { kind, fraction: (fraction ?? NaN) as Fraction };
    case "attack":
      return { kind, slot: v[1] - 1, fraction: (fraction ?? NaN) as Fraction };
    default:
      return { kind, row: v[1], cell: v[2], fraction, type: v[4], aux: v[5] };
  }
}

/** Tiles candidatos de una celda: una malla de n × n puntos, de los más cercanos al centro a los más lejanos. */
export function cellTiles(game: Game, cell: number, grid: { h: number; w: number }, n = 6): number[] {
  const map = game.map();
  const W = map.width();
  const H = map.height();
  const cx = cell % grid.w;
  const cy = Math.floor(cell / grid.w);
  const x0 = Math.floor((cx * W) / grid.w);
  const x1 = Math.max(x0 + 1, Math.floor(((cx + 1) * W) / grid.w));
  const y0 = Math.floor((cy * H) / grid.h);
  const y1 = Math.max(y0 + 1, Math.floor(((cy + 1) * H) / grid.h));
  const mx = (x0 + x1) / 2;
  const my = (y0 + y1) / 2;
  const pts: { ref: number; d: number }[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const x = Math.min(W - 1, Math.floor(x0 + ((i + 0.5) * (x1 - x0)) / n));
      const y = Math.min(H - 1, Math.floor(y0 + ((j + 0.5) * (y1 - y0)) / n));
      pts.push({ ref: map.ref(x, y), d: (x - mx) ** 2 + (y - my) ** 2 });
    }
  }
  pts.sort((a, b) => a.d - b.d || a.ref - b.ref);
  return pts.map((p) => p.ref);
}

const STRUCT_FOR_NUKE = [UnitType.City, UnitType.Factory, UnitType.Port, UnitType.MissileSilo, UnitType.SAMLauncher, UnitType.DefensePost];

export function actionToIntents(
  action: Action,
  game: Game,
  me: Player,
  clientID: string,
  info: MaskInfo,
  grid: { h: number; w: number },
  maxNeighbors: number,
): BuiltIntents {
  const { mask, slots } = info;
  const rows = info.rows;
  const cells = grid.h * grid.w;
  const map = game.map();
  const frac = (f: number | undefined): number | null => (f !== undefined && FRACTIONS.includes(f as Fraction) ? f : null);
  const cellOk = (kind: (typeof CELL_KINDS)[number], c: number | undefined): c is number =>
    c !== undefined && Number.isInteger(c) && c >= 0 && c < cells && mask.cell[CELL_KINDS.indexOf(kind) * cells + c] === 1;
  const playerRow = (kind: (typeof PLAYER_KINDS)[number], r: number | undefined): Player | null | undefined => {
    // undefined = fila inválida; null = fila 0 (todos los jugadores) válida solo para emoji.
    if (r === undefined || !Number.isInteger(r) || r < 0 || r >= PLAYER_ROWS) return undefined;
    if (mask.player[PLAYER_KINDS.indexOf(kind) * PLAYER_ROWS + r] !== 1) return undefined;
    return r === 0 ? null : (rows?.players[r] ?? undefined);
  };
  const unitRow = (kind: (typeof UNIT_KINDS)[number], r: number | undefined) => {
    if (r === undefined || !Number.isInteger(r) || r < 0 || r >= UNIT_ROWS) return undefined;
    if (mask.unit[UNIT_KINDS.indexOf(kind) * UNIT_ROWS + r] !== 1) return undefined;
    return rows?.units[r] ?? undefined;
  };
  if (action.kind !== "noop" && mask.kind[KIND[action.kind]] !== 1) return INVALID;

  switch (action.kind) {
    case "noop":
      return NONE;
    case "spawn": {
      if (!mask.spawn) return INVALID;
      if (action.cx < 0 || action.cx >= grid.w || action.cy < 0 || action.cy >= grid.h) return INVALID;
      const tile = cellToSpawnTile(game, action.cx, action.cy, grid.h, grid.w);
      if (tile === null) return INVALID;
      return ok({ type: "spawn", tile, clientID });
    }
    case "expand": {
      if (!mask.expand || !FRACTIONS.includes(action.fraction)) return INVALID;
      return ok({ type: "attack", targetID: null, troops: me.troops() * action.fraction, clientID });
    }
    case "attack": {
      if (
        !Number.isInteger(action.slot) ||
        action.slot < 0 ||
        action.slot >= maxNeighbors ||
        !mask.attackSlots[action.slot] ||
        !FRACTIONS.includes(action.fraction)
      ) {
        return INVALID;
      }
      const target = slots[action.slot];
      if (target === undefined) return INVALID;
      return ok({ type: "attack", targetID: target.id(), troops: me.troops() * action.fraction, clientID });
    }
    case "boat_attack": {
      const f = frac(action.fraction);
      if (f === null || !cellOk("boat_attack", action.cell)) return INVALID;
      const mine = me.smallID();
      const cands = cellTiles(game, action.cell, grid).filter((t) => map.isLand(t) && !(map.hasOwner(t) && map.ownerID(t) === mine));
      // Se prefieren los tiles de otros jugadores: es donde un desembarco tiene sentido.
      cands.sort((a, b) => Number(map.hasOwner(b)) - Number(map.hasOwner(a)));
      for (const t of cands) {
        if (me.canBuild(UnitType.TransportShip, t) !== false) {
          return ok({ type: "boat", troops: me.troops() * f, dst: t, clientID });
        }
      }
      return INVALID;
    }
    case "cancel_attack": {
      const a = action.row !== undefined && Number.isInteger(action.row) ? rows?.outgoing[action.row] : undefined;
      if (a === undefined || action.row === undefined || mask.attackRow[action.row] !== 1) return INVALID;
      return ok({ type: "cancel_attack", attackID: a.id(), clientID });
    }
    case "cancel_boat": {
      const u = unitRow("cancel_boat", action.row);
      return u ? ok({ type: "cancel_boat", unitID: u.id(), clientID }) : INVALID;
    }
    case "build": {
      const ti = action.type;
      if (ti === undefined || !Number.isInteger(ti) || ti < 0 || ti >= BUILD_TYPES.length || mask.buildType[ti] !== 1) return INVALID;
      const unit = BUILD_UNIT[BUILD_TYPES[ti]];
      if (!cellOk(unit === UnitType.Warship ? "build_ship" : "build", action.cell)) return INVALID;
      const mine = me.smallID();
      let cands = cellTiles(game, action.cell, grid);
      if (unit === UnitType.Warship) cands = cands.filter((t) => map.isWater(t));
      else {
        cands = cands.filter((t) => map.isLand(t) && map.hasOwner(t) && map.ownerID(t) === mine);
        if (unit === UnitType.Port) cands = cands.filter((t) => map.isShore(t));
      }
      for (const t of cands) {
        if (me.canBuild(unit, t) !== false) return ok({ type: "build_unit", unit, tile: t, clientID });
      }
      return INVALID;
    }
    case "upgrade": {
      const u = unitRow("upgrade", action.row);
      return u ? ok({ type: "upgrade_structure", unit: u.type(), unitId: u.id(), clientID }) : INVALID;
    }
    case "delete_unit": {
      const u = unitRow("delete_unit", action.row);
      return u ? ok({ type: "delete_unit", unitId: u.id(), clientID }) : INVALID;
    }
    case "nuke": {
      const ni = action.type;
      if (ni === undefined || !Number.isInteger(ni) || ni < 0 || ni >= NUKE_KINDS.length || mask.nukeType[ni] !== 1) return INVALID;
      if (!cellOk("nuke", action.cell)) return INVALID;
      const unit = NUKE_UNIT[NUKE_KINDS[ni]];
      const mine = me.smallID();
      const cands: number[] = [];
      // Primero, estructuras ajenas dentro de la celda (el objetivo con más valor).
      const cx = action.cell % grid.w;
      const cy = Math.floor(action.cell / grid.w);
      const W = map.width();
      const H = map.height();
      const x0 = Math.floor((cx * W) / grid.w);
      const x1 = Math.floor(((cx + 1) * W) / grid.w);
      const y0 = Math.floor((cy * H) / grid.h);
      const y1 = Math.floor(((cy + 1) * H) / grid.h);
      const structs = game
        .units(STRUCT_FOR_NUKE)
        .filter((s) => s.isActive() && s.owner() !== me && !me.isFriendly(s.owner()))
        .filter((s) => {
          const x = map.x(s.tile());
          const y = map.y(s.tile());
          return x >= x0 && x < x1 && y >= y0 && y < y1;
        })
        .sort((a, b) => a.id() - b.id());
      for (const s of structs) cands.push(s.tile());
      const land = cellTiles(game, action.cell, grid).filter((t) => map.isLand(t) && !(map.hasOwner(t) && map.ownerID(t) === mine));
      land.sort((a, b) => Number(map.hasOwner(b)) - Number(map.hasOwner(a)));
      cands.push(...land);
      for (const t of cands) {
        if (unit === UnitType.MIRV && !map.hasOwner(t)) continue;
        if (me.canBuild(unit, t) !== false) return ok({ type: "build_unit", unit, tile: t, clientID });
      }
      return INVALID;
    }
    case "move_warship": {
      const u = unitRow("move_warship", action.row);
      if (!u || !cellOk("move_warship", action.cell)) return INVALID;
      for (const t of cellTiles(game, action.cell, grid)) {
        if (!map.isWater(t)) continue;
        const comp = game.getWaterComponent(t);
        if (comp !== null && game.hasWaterComponent(u.tile(), comp)) {
          return ok({ type: "move_warship", unitIds: [u.id()], tile: t, clientID });
        }
      }
      return INVALID;
    }
    case "alliance_request": {
      const p = playerRow("alliance_request", action.row);
      return p ? ok({ type: "allianceRequest", recipient: p.id(), clientID }) : INVALID;
    }
    case "alliance_reject": {
      const p = playerRow("alliance_reject", action.row);
      return p ? ok({ type: "allianceReject", requestor: p.id(), clientID }) : INVALID;
    }
    case "alliance_break": {
      const p = playerRow("alliance_break", action.row);
      return p ? ok({ type: "breakAlliance", recipient: p.id(), clientID }) : INVALID;
    }
    case "alliance_extend": {
      const p = playerRow("alliance_extend", action.row);
      return p ? ok({ type: "allianceExtension", recipient: p.id(), clientID }) : INVALID;
    }
    case "embargo_start":
    case "embargo_stop": {
      const p = playerRow(action.kind, action.row);
      return p ? ok({ type: "embargo", targetID: p.id(), action: action.kind === "embargo_start" ? "start" : "stop", clientID }) : INVALID;
    }
    case "embargo_all_start":
      return ok({ type: "embargo_all", action: "start", clientID });
    case "embargo_all_stop":
      return ok({ type: "embargo_all", action: "stop", clientID });
    case "target_player": {
      const p = playerRow("target_player", action.row);
      return p ? ok({ type: "targetPlayer", target: p.id(), clientID }) : INVALID;
    }
    case "emoji": {
      const p = playerRow("emoji", action.row);
      const e = action.aux;
      if (p === undefined || e === undefined || !Number.isInteger(e) || e < 0 || e >= NUM_EMOJIS) return INVALID;
      const emoji = flattenedEmojiTable.indexOf(EMOJI_CHARS[e] as (typeof flattenedEmojiTable)[number]);
      if (emoji < 0) return INVALID;
      return ok({ type: "emoji", recipient: p === null ? AllPlayers : p.id(), emoji, clientID });
    }
    case "donate_gold": {
      const p = playerRow("donate_gold", action.row);
      const f = frac(action.fraction);
      if (!p || f === null) return INVALID;
      return ok({ type: "donate_gold", recipient: p.id(), gold: Number(me.gold()) * f, clientID });
    }
    case "donate_troops": {
      const p = playerRow("donate_troops", action.row);
      const f = frac(action.fraction);
      if (!p || f === null) return INVALID;
      return ok({ type: "donate_troops", recipient: p.id(), troops: me.troops() * f, clientID });
    }
  }
}

