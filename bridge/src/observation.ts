// Construye la observación de un agente. La spec es docs/specs/observations.md.
// Regla: aquí solo se LEE el estado del motor. Nada de lo que se hace en este archivo cambia la partida.
import { PlayerType, Relation, UnitType } from "@openfront/engine-api/game/GameTypes";
import type { Game, Player, Unit } from "@openfront/engine/game/Game";
import type { MaskInfo } from "./actions";
import {
  ALLY_ROWS,
  ATTACK_ROW_FEATURES,
  BUILDABLE_COST_TYPES,
  CH,
  CMP_SCALARS,
  ECON_ROW_FEATURES,
  ECON_SCALARS,
  GLOBAL_FEATURES,
  MAX_ATTACK_ROWS,
  MAX_ATTACKS_PER_SIDE,
  MAX_ECON_ROWS,
  MAX_MIL_ROWS,
  MAX_NEIGHBOR_ROWS,
  MIL_ROW_FEATURES,
  MIL_SCALARS,
  NUM_CHANNELS,
  PLAYER_ROW_FEATURES,
  PLAYER_ROWS,
  STATIC_CHANNELS,
  TOP_ROWS,
} from "./features";
import type { EnvConfig, Observation, Table } from "./types";

// --- Normalización ----------------------------------------------------------------
const clip01 = (v: number): number => (Number.isFinite(v) ? (v < 0 ? 0 : v > 1 ? 1 : v) : 0);
/** log1p(v) / log1p(ref), en [0, 1]. */
const logn = (v: number, ref: number): number => clip01(Math.log1p(Math.max(0, v)) / Math.log1p(ref));
/** Razón a/b comprimida a [-1, 1]: 0 si son iguales, positivo si a > b. */
const ratio = (a: number, b: number): number => {
  const r = Math.tanh(Math.log((Math.max(0, a) + 1) / (Math.max(0, b) + 1)) / 2);
  return Number.isFinite(r) ? r : 0;
};

const STRUCT_TYPES = [
  UnitType.City,
  UnitType.DefensePost,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
  UnitType.Port,
  UnitType.Factory,
] as const;
const ECON_TYPES = [UnitType.City, UnitType.Port, UnitType.Factory] as const;
const SHIP_TYPES = [UnitType.Warship, UnitType.TransportShip] as const;
const NUKE_TYPES = [UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRV, UnitType.MIRVWarhead] as const;
/** Todo lo que se pide al motor de una sola vez cada paso. */
const OBS_UNIT_TYPES = [...STRUCT_TYPES, ...SHIP_TYPES, ...NUKE_TYPES] as const;
const MIL_TYPES = [
  UnitType.DefensePost,
  UnitType.SAMLauncher,
  UnitType.MissileSilo,
  UnitType.Warship,
  UnitType.TransportShip,
  UnitType.AtomBomb,
  UnitType.HydrogenBomb,
  UnitType.MIRV,
  UnitType.MIRVWarhead,
] as const;
const COST_UNIT_TYPE: Record<(typeof BUILDABLE_COST_TYPES)[number], UnitType> = {
  city: UnitType.City,
  port: UnitType.Port,
  factory: UnitType.Factory,
  post: UnitType.DefensePost,
  sam: UnitType.SAMLauncher,
  silo: UnitType.MissileSilo,
  warship: UnitType.Warship,
  atom: UnitType.AtomBomb,
  hydrogen: UnitType.HydrogenBomb,
  mirv: UnitType.MIRV,
};

const MAX_SAM_RANGE = 150; // Config.maxSamRange (Config.ts:425)
const DEFENSE_POST_RANGE = 30; // Config.defensePostRange (Config.ts:175)
const NUKE_RADIUS_REF = 100; // radio exterior de la bomba de hidrógeno (Config.ts:411)

// Bits de la clase de dueño de un tile.
const B_OWN = 1;
const B_ALLY = 2;
const B_NEIGHBOR = 4;
const B_TOP = 8;
const B_OTHER = 16;
const MAX_OWNER_ID = 4096; // ownerID usa 12 bits (GameMap.ts:100)

const index = <T extends readonly string[]>(names: T) =>
  Object.fromEntries(names.map((n, i) => [n, i])) as Record<T[number], number>;
const ECON = index(ECON_ROW_FEATURES);
const MIL = index(MIL_ROW_FEATURES);
const ATK = index(ATTACK_ROW_FEATURES);
const PLR = index(PLAYER_ROW_FEATURES);
const ES = index(ECON_SCALARS);
const MS = index(MIL_SCALARS);
const CS = index(CMP_SCALARS);
const GS = index(GLOBAL_FEATURES);

// --- Plan de muestreo --------------------------------------------------------------
export interface SamplePlan {
  /** Tiles que se muestrean para cada celda (S × S por celda). */
  refs: Int32Array;
  /** Grid [NUM_CHANNELS, h, w] con solo los canales estáticos rellenos (tierra, relieve, costa). */
  statics: Float32Array;
}

/** Tiles que se muestrean para cada celda del grid y canales estáticos. Se calcula una vez por mapa. */
export function buildSamplePlan(game: Game, h: number, w: number, samples: number): SamplePlan {
  const map = game.map();
  const W = map.width();
  const H = map.height();
  const cells = h * w;
  const refs = new Int32Array(cells * samples * samples);
  const statics = new Float32Array(NUM_CHANNELS * cells);
  const per = samples * samples;
  let k = 0;
  for (let cy = 0; cy < h; cy++) {
    for (let cx = 0; cx < w; cx++) {
      const c = cy * w + cx;
      let land = 0;
      let high = 0;
      let mount = 0;
      let shore = 0;
      for (let sy = 0; sy < samples; sy++) {
        for (let sx = 0; sx < samples; sx++) {
          const x = Math.min(W - 1, Math.floor(((cx + (sx + 0.5) / samples) * W) / w));
          const y = Math.min(H - 1, Math.floor(((cy + (sy + 0.5) / samples) * H) / h));
          const ref = map.ref(x, y);
          refs[k++] = ref;
          if (map.isLand(ref)) {
            land++;
            // TerrainType: Plains 0, Highland 1, Mountain 2 (GameTypes.ts:296-302)
            const t = map.terrainType(ref);
            if (t === 1) high++;
            else if (t === 2) mount++;
          }
          if (map.isShore(ref)) shore++;
        }
      }
      statics[CH.land * cells + c] = land / per;
      statics[CH.highland * cells + c] = high / per;
      statics[CH.mountain * cells + c] = mount / per;
      statics[CH.shore * cells + c] = shore / per;
    }
  }
  // Defensa: solo los canales de STATIC_CHANNELS pueden estar rellenos.
  for (let ch = 0; ch < NUM_CHANNELS; ch++) {
    const isStatic = STATIC_CHANNELS.some((n) => CH[n] === ch);
    if (!isStatic) for (let c = 0; c < cells; c++) statics[ch * cells + c] = 0;
  }
  return { refs, statics };
}

// --- Tasas de ingreso (único estado entre pasos) --------------------------------------
type Counter = "earned" | "trade" | "train" | "piracy";
interface Counters {
  earned: number;
  trade: number;
  train: number;
  piracy: number;
}

/** Calcula oro por tick a partir de los contadores acumulados del motor (`goldEarned`, `tradeGold`...). */
export class RateTracker {
  private prev = new Map<number, Counters>();
  private cur = new Map<number, Counters>();
  private dt = 0;
  private lastTick = -1;

  update(game: Game, players: readonly Player[]): void {
    const tick = game.ticks();
    this.dt = this.lastTick < 0 ? 0 : tick - this.lastTick;
    this.lastTick = tick;
    this.prev = this.cur;
    this.cur = new Map();
    for (const p of players) {
      this.cur.set(p.smallID(), {
        earned: Number(p.goldEarned()),
        trade: Number(p.tradeGold()),
        train: Number(p.trainGold()),
        piracy: Number(p.piracyGold()),
      });
    }
  }

  /** Oro por tick ganado entre las dos últimas observaciones. 0 si no hay dos puntos. */
  rate(smallID: number, key: Counter): number {
    const c = this.cur.get(smallID);
    const p = this.prev.get(smallID);
    if (c === undefined || p === undefined || this.dt <= 0) return 0;
    return Math.max(0, (c[key] - p[key]) / this.dt);
  }
}

// --- Datos que se calculan una vez por paso y comparten todos los agentes ------------------
export interface Frame {
  tick: number;
  /** Jugadores vivos por tiles desc y smallID asc. */
  alive: Player[];
  /** smallID → posición (0 = el mayor). */
  rank: Map<number, number>;
  unitsByType: Map<UnitType, Unit[]>;
  landTiles: number;
  rates: RateTracker;
}

export function buildFrame(game: Game, rates: RateTracker): Frame {
  const alive = game
    .players()
    .filter((p) => p.isAlive())
    .sort((a, b) => b.numTilesOwned() - a.numTilesOwned() || a.smallID() - b.smallID());
  rates.update(game, alive);
  const rank = new Map<number, number>();
  alive.forEach((p, i) => rank.set(p.smallID(), i));
  const unitsByType = new Map<UnitType, Unit[]>();
  for (const u of game.units(OBS_UNIT_TYPES)) {
    if (!u.isActive()) continue;
    const list = unitsByType.get(u.type());
    if (list === undefined) unitsByType.set(u.type(), [u]);
    else list.push(u);
  }
  // Los jugadores se recorren en el orden del motor; se ordena por id para que el resultado no dependa de él.
  for (const list of unitsByType.values()) list.sort((a, b) => a.id() - b.id());
  return { tick: game.ticks(), alive, rank, unitsByType, landTiles: Math.max(1, game.numLandTiles()), rates };
}

export interface ObsContext {
  cfg: EnvConfig;
  plan: SamplePlan;
  initialPlayers: number;
  ticksSinceSpawnEnd: number;
  frame: Frame;
}

function makeTable(rows: number, cols: number): Table {
  return { data: new Float32Array(rows * cols), valid: new Uint8Array(rows), rows, cols };
}

// --- Observación ---------------------------------------------------------------------
export function buildObservation(game: Game, me: Player, ctx: ObsContext, info: MaskInfo): Observation {
  const { h, w } = ctx.cfg.grid;
  const cells = h * w;
  const map = game.map();
  const W = map.width();
  const H = map.height();
  const diag = Math.hypot(W, H);
  const config = game.config();
  const frame = ctx.frame;
  const tick = frame.tick;
  const meSmall = me.smallID();
  const myTiles = me.numTilesOwned();
  const myTroops = me.troops();
  const myMax = Math.max(1, config.maxTroops(me));
  const gold = Number(me.gold());
  const landTiles = frame.landTiles;
  const inSpawn = game.inSpawnPhase();
  const neighbors = info.neighbors;
  const neighborSet = new Set<number>(neighbors.map((p) => p.smallID()));
  const top = frame.alive.filter((p) => p !== me).slice(0, TOP_ROWS);
  const topSet = new Set<number>(top.map((p) => p.smallID()));

  // 1. Clase de dueño de cada jugador (una consulta por muestra).
  const classOf = new Uint8Array(MAX_OWNER_ID);
  for (const p of frame.alive) {
    const id = p.smallID();
    if (id >= MAX_OWNER_ID) continue;
    if (p === me) {
      classOf[id] = B_OWN;
      continue;
    }
    classOf[id] =
      B_OTHER | (me.isFriendly(p) ? B_ALLY : 0) | (neighborSet.has(id) ? B_NEIGHBOR : 0) | (topSet.has(id) ? B_TOP : 0);
  }
  const cls = (u: Unit): number => classOf[u.owner().smallID()] ?? 0;

  // 2. Grid: canales estáticos + muestreo de propiedad y fallout.
  const grid = new Float32Array(NUM_CHANNELS * cells);
  grid.set(ctx.plan.statics);
  const per = ctx.cfg.samples * ctx.cfg.samples;
  const refs = ctx.plan.refs;
  let k = 0;
  let ownSamples = 0;
  let ownFallout = 0;
  for (let c = 0; c < cells; c++) {
    let own = 0;
    let other = 0;
    let ally = 0;
    let nb = 0;
    let tp = 0;
    let fall = 0;
    for (let s = 0; s < per; s++) {
      const ref = refs[k++];
      const owned = map.hasOwner(ref);
      const bits = owned ? classOf[map.ownerID(ref)] : 0;
      if (bits !== 0) {
        if (bits & B_OWN) own++;
        else {
          other++;
          if (bits & B_ALLY) ally++;
          if (bits & B_NEIGHBOR) nb++;
          if (bits & B_TOP) tp++;
        }
      }
      if (map.hasFallout(ref)) {
        fall++;
        if (bits & B_OWN) ownFallout++;
      }
    }
    ownSamples += own;
    grid[CH.own * cells + c] = own / per;
    grid[CH.other * cells + c] = other / per;
    grid[CH.ally * cells + c] = ally / per;
    grid[CH.neighbor * cells + c] = nb / per;
    grid[CH.top3 * cells + c] = tp / per;
    grid[CH.fallout * cells + c] = fall / per;
  }

  const cellOf = (x: number, y: number): number =>
    Math.min(h - 1, Math.max(0, Math.floor((y * h) / H))) * w + Math.min(w - 1, Math.max(0, Math.floor((x * w) / W)));
  const bump = (ch: number, x: number, y: number, amount: number): void => {
    const i = ch * cells + cellOf(x, y);
    grid[i] = Math.min(1, grid[i] + amount);
  };
  const paintCircle = (ch: number, x: number, y: number, r: number): void => {
    const cw = W / w;
    const chh = H / h;
    const x0 = Math.max(0, Math.floor((x - r) / cw));
    const x1 = Math.min(w - 1, Math.floor((x + r) / cw));
    const y0 = Math.max(0, Math.floor((y - r) / chh));
    const y1 = Math.min(h - 1, Math.floor((y + r) / chh));
    const r2 = r * r;
    for (let cy = y0; cy <= y1; cy++) {
      const dy = (cy + 0.5) * chh - y;
      for (let cx = x0; cx <= x1; cx++) {
        const dx = (cx + 0.5) * cw - x;
        if (dx * dx + dy * dy <= r2) grid[ch * cells + cy * w + cx] = 1;
      }
    }
  };

  // 3. Centroide propio: media de las estructuras propias; si no hay, el spawn; si no, el centro del mapa.
  const myStructs = me.units(STRUCT_TYPES).filter((u) => u.isActive());
  let cx0 = W / 2;
  let cy0 = H / 2;
  if (myStructs.length > 0) {
    cx0 = 0;
    cy0 = 0;
    for (const u of myStructs) {
      cx0 += map.x(u.tile());
      cy0 += map.y(u.tile());
    }
    cx0 /= myStructs.length;
    cy0 /= myStructs.length;
  } else {
    const sp = me.spawnTile();
    if (sp !== undefined) {
      cx0 = map.x(sp);
      cy0 = map.y(sp);
    }
  }
  const distCentroid = (x: number, y: number): number => clip01(Math.hypot(x - cx0, y - cy0) / diag);

  // 4. Estructuras, barcos y bombas en el grid.
  for (const t of STRUCT_TYPES) {
    for (const u of frame.unitsByType.get(t) ?? []) {
      const own = (cls(u) & B_OWN) !== 0;
      const x = map.x(u.tile());
      const y = map.y(u.tile());
      bump(own ? CH.own_struct : CH.enemy_struct, x, y, 0.25);
      if (u.isUnderConstruction()) continue;
      if (t === UnitType.SAMLauncher) {
        const friendly = (cls(u) & (B_OWN | B_ALLY)) !== 0;
        if (own) paintCircle(CH.own_sam_cov, x, y, config.dynamicSamRange(u, tick));
        else if (!friendly) paintCircle(CH.enemy_sam_cov, x, y, config.dynamicSamRange(u, tick));
      } else if (t === UnitType.DefensePost && own) {
        paintCircle(CH.own_post_cov, x, y, DEFENSE_POST_RANGE);
      }
    }
  }
  for (const t of SHIP_TYPES) {
    for (const u of frame.unitsByType.get(t) ?? []) {
      const friendly = (cls(u) & (B_OWN | B_ALLY)) !== 0;
      bump(friendly ? CH.friendly_ships : CH.hostile_ships, map.x(u.tile()), map.y(u.tile()), 0.5);
    }
  }
  for (const t of NUKE_TYPES) {
    for (const u of frame.unitsByType.get(t) ?? []) {
      const target = u.targetTile();
      if (target !== undefined) bump(CH.nuke_target, map.x(target), map.y(target), 0.5);
    }
  }

  // 5. Ataques en curso: tiles en disputa en el grid y tabla de ataques.
  const outgoing = me.outgoingAttacks().filter((a) => a.isActive());
  const incoming = me.incomingAttacks().filter((a) => a.isActive());
  for (const a of outgoing) for (const t of a.borderTiles()) bump(CH.attack_out, map.x(t), map.y(t), 0.125);
  for (const a of incoming) for (const t of a.borderTiles()) bump(CH.attack_in, map.x(t), map.y(t), 0.125);

  // --- Economía ---------------------------------------------------------------------------
  const econ = makeTable(MAX_ECON_ROWS, ECON_ROW_FEATURES.length);
  const econUnits = me
    .units(ECON_TYPES)
    .filter((u) => u.isActive())
    .sort((a, b) => a.id() - b.id())
    .slice(0, MAX_ECON_ROWS);
  econUnits.forEach((u, r) => {
    const o = r * econ.cols;
    const x = map.x(u.tile());
    const y = map.y(u.tile());
    econ.valid[r] = 1;
    econ.data[o + ECON.is_city] = u.type() === UnitType.City ? 1 : 0;
    econ.data[o + ECON.is_port] = u.type() === UnitType.Port ? 1 : 0;
    econ.data[o + ECON.is_factory] = u.type() === UnitType.Factory ? 1 : 0;
    econ.data[o + ECON.level] = clip01(u.level() / 10);
    econ.data[o + ECON.x] = clip01(x / W);
    econ.data[o + ECON.y] = clip01(y / H);
    econ.data[o + ECON.under_construction] = u.isUnderConstruction() ? 1 : 0;
    econ.data[o + ECON.has_station] = u.hasTrainStation() ? 1 : 0;
    econ.data[o + ECON.dist_centroid] = distCentroid(x, y);
  });

  const es = new Float32Array(ECON_SCALARS.length);
  const cityLevels = me.unitsOwned(UnitType.City);
  es[ES.max_troops_log] = logn(myMax, 5e6);
  es[ES.troop_growth] = clip01((config.troopIncreaseRate(me) / myMax) * 10);
  es[ES.city_troop_share] = clip01(1 - config.maxTroopsWithoutCities(me) / myMax);
  es[ES.troops_fill] = clip01(myTroops / myMax);
  es[ES.gold_log] = logn(gold, 1e7);
  es[ES.income_total] = logn(frame.rates.rate(meSmall, "earned"), 1e5);
  es[ES.income_trade] = logn(frame.rates.rate(meSmall, "trade"), 1e5);
  es[ES.income_train] = logn(frame.rates.rate(meSmall, "train"), 1e5);
  es[ES.income_piracy] = logn(frame.rates.rate(meSmall, "piracy"), 1e5);
  for (const name of BUILDABLE_COST_TYPES) {
    const cost = Number(game.unitInfo(COST_UNIT_TYPE[name]).cost(game, me));
    es[ES[`cost_${name}` as keyof typeof ES]] = logn(cost, 3e7);
    es[ES[`afford_${name}` as keyof typeof ES]] = cost <= 0 ? 1 : clip01(gold / cost);
  }
  es[ES.city_levels] = clip01(cityLevels / 10);
  es[ES.port_levels] = clip01(me.unitsOwned(UnitType.Port) / 10);
  es[ES.factory_levels] = clip01(me.unitsOwned(UnitType.Factory) / 10);
  es[ES.stations] = clip01(myStructs.filter((u) => u.hasTrainStation()).length / 20);
  es[ES.trade_ships] = clip01(me.units(UnitType.TradeShip).length / 20);
  es[ES.trains] = clip01(me.units(UnitType.Train).length / 20);
  es[ES.trade_partners] = clip01(me.tradingPartners().length / 10);
  es[ES.embargoes_by_me] = clip01(me.getEmbargoes().length / 10);
  es[ES.embargoes_on_me] = clip01(frame.alive.filter((p) => p !== me && p.hasEmbargoAgainst(me)).length / 10);

  // --- Defensa y ataque -----------------------------------------------------------------------
  // Candidatos: grupo 0 = unidades propias y bombas que apuntan a mi territorio; grupo 1 = el resto por cercanía.
  interface Cand {
    u: Unit;
    group: number;
    d2: number;
    targetsMe: boolean;
  }
  const cands: Cand[] = [];
  for (const t of MIL_TYPES) {
    for (const u of frame.unitsByType.get(t) ?? []) {
      const bits = cls(u);
      const target = u.targetTile();
      const targetsMe =
        target !== undefined && (t === UnitType.AtomBomb || t === UnitType.HydrogenBomb || t === UnitType.MIRV || t === UnitType.MIRVWarhead)
          ? map.hasOwner(target) && map.ownerID(target) === meSmall
          : false;
      const x = map.x(u.tile());
      const y = map.y(u.tile());
      cands.push({
        u,
        group: bits & B_OWN || targetsMe ? 0 : 1,
        d2: (x - cx0) * (x - cx0) + (y - cy0) * (y - cy0),
        targetsMe,
      });
    }
  }
  cands.sort((a, b) => a.group - b.group || a.d2 - b.d2 || a.u.id() - b.u.id());
  const mil = makeTable(MAX_MIL_ROWS, MIL_ROW_FEATURES.length);
  const typeCol: Partial<Record<UnitType, number>> = {
    [UnitType.DefensePost]: MIL.is_post,
    [UnitType.SAMLauncher]: MIL.is_sam,
    [UnitType.MissileSilo]: MIL.is_silo,
    [UnitType.Warship]: MIL.is_warship,
    [UnitType.TransportShip]: MIL.is_transport,
    [UnitType.AtomBomb]: MIL.is_atom,
    [UnitType.HydrogenBomb]: MIL.is_hydrogen,
    [UnitType.MIRV]: MIL.is_mirv,
    [UnitType.MIRVWarhead]: MIL.is_warhead,
  };
  cands.slice(0, MAX_MIL_ROWS).forEach(({ u, targetsMe }, r) => {
    const o = r * mil.cols;
    const t = u.type();
    const bits = cls(u);
    const x = map.x(u.tile());
    const y = map.y(u.tile());
    const d = mil.data;
    mil.valid[r] = 1;
    d[o + (typeCol[t] as number)] = 1;
    d[o + MIL.own] = bits & B_OWN ? 1 : 0;
    d[o + MIL.friendly] = bits & B_ALLY ? 1 : 0;
    d[o + MIL.other] = bits & B_OWN || bits & B_ALLY ? 0 : 1;
    d[o + MIL.x] = clip01(x / W);
    d[o + MIL.y] = clip01(y / H);
    d[o + MIL.level] = clip01(u.level() / 10);
    d[o + MIL.under_construction] = u.isUnderConstruction() ? 1 : 0;
    if (u.hasHealth()) d[o + MIL.health] = clip01(u.health() / Math.max(1, u.maxHealth()));
    if (t === UnitType.SAMLauncher || t === UnitType.MissileSilo) {
      const level = Math.max(1, u.level());
      d[o + MIL.ready_slots] = clip01((level - u.missileTimerQueue().length) / level);
    }
    if (t === UnitType.SAMLauncher) d[o + MIL.range] = clip01(config.dynamicSamRange(u, tick) / MAX_SAM_RANGE);
    else if (t === UnitType.DefensePost) d[o + MIL.range] = clip01(DEFENSE_POST_RANGE / MAX_SAM_RANGE);
    else if (t === UnitType.AtomBomb || t === UnitType.HydrogenBomb || t === UnitType.MIRVWarhead) {
      d[o + MIL.range] = clip01(config.nukeMagnitudes(t).outer / NUKE_RADIUS_REF);
    }
    if (t === UnitType.TransportShip) d[o + MIL.troops] = clip01(u.troops() / Math.max(1, config.maxTroops(u.owner())));
    const target = u.targetTile();
    if (target !== undefined && t !== UnitType.Warship && t !== UnitType.TransportShip) {
      d[o + MIL.target_x] = clip01(map.x(target) / W);
      d[o + MIL.target_y] = clip01(map.y(target) / H);
    }
    if (t === UnitType.AtomBomb || t === UnitType.HydrogenBomb || t === UnitType.MIRVWarhead) {
      d[o + MIL.progress] = nukeProgress(u);
      d[o + MIL.targeted_by_sam] = u.targetedBySAM() ? 1 : 0;
    }
    d[o + MIL.targets_me] = targetsMe ? 1 : 0;
    d[o + MIL.dist_centroid] = distCentroid(x, y);
    if (t === UnitType.Warship) {
      const st = u.warshipState().state;
      d[o + MIL.ws_patrolling] = st === "patrolling" ? 1 : 0;
      d[o + MIL.ws_retreating] = st === "retreating" ? 1 : 0;
      d[o + MIL.ws_docked] = st === "docked" ? 1 : 0;
      d[o + MIL.veterancy] = clip01(u.veterancy() / config.warshipMaxVeterancy());
    }
  });

  // --- Tabla de jugadores (comparación) --------------------------------------------------------
  const rows: (Player | null)[] = Array(PLAYER_ROWS).fill(null);
  rows[0] = me;
  const slots = info.slots.slice(0, MAX_NEIGHBOR_ROWS);
  slots.forEach((p, i) => {
    rows[1 + i] = p;
  });
  top.forEach((p, i) => {
    rows[1 + MAX_NEIGHBOR_ROWS + i] = p;
  });
  const inTable = new Set<number>(rows.filter((p): p is Player => p !== null).map((p) => p.smallID()));
  const myAllies = me
    .allies()
    .filter((p) => p.isAlive() && !inTable.has(p.smallID()))
    .sort((a, b) => b.numTilesOwned() - a.numTilesOwned() || a.smallID() - b.smallID())
    .slice(0, ALLY_ROWS);
  myAllies.forEach((p, i) => {
    rows[1 + MAX_NEIGHBOR_ROWS + TOP_ROWS + i] = p;
  });
  const rowOf = new Map<number, number>();
  rows.forEach((p, r) => {
    if (p !== null && !rowOf.has(p.smallID())) rowOf.set(p.smallID(), r);
  });

  const outTroops = new Map<number, number>();
  const inTroops = new Map<number, number>();
  for (const a of outgoing) {
    const t = a.target();
    if (t.isPlayer()) outTroops.set(t.smallID(), (outTroops.get(t.smallID()) ?? 0) + a.troops());
  }
  for (const a of incoming) {
    const s = a.attacker().smallID();
    inTroops.set(s, (inTroops.get(s) ?? 0) + a.troops());
  }
  const reqIn = new Set<number>(me.incomingAllianceRequests().map((r) => r.requestor().smallID()));
  const reqOut = new Set<number>(me.outgoingAllianceRequests().map((r) => r.recipient().smallID()));
  const myTargets = new Set<number>(me.targets().map((p) => p.smallID()));
  const winThreshold = config.percentageTilesOwnedToWin(game.elapsedGameSeconds()) / 100;
  const allianceDuration = Math.max(1, config.allianceDuration());
  const alivePlayers = Math.max(1, frame.alive.length);

  const players = makeTable(PLAYER_ROWS, PLAYER_ROW_FEATURES.length);
  rows.forEach((p, r) => {
    if (p === null) return;
    const o = r * players.cols;
    const d = players.data;
    const isSelf = p === me;
    const pid = p.smallID();
    const tiles = p.numTilesOwned();
    const pMax = Math.max(1, config.maxTroops(p));
    players.valid[r] = 1;
    d[o + PLR.is_self] = isSelf ? 1 : 0;
    d[o + PLR.is_human] = p.type() === PlayerType.Human ? 1 : 0;
    d[o + PLR.is_nation] = p.type() === PlayerType.Nation ? 1 : 0;
    d[o + PLR.is_bot] = p.type() === PlayerType.Bot ? 1 : 0;
    d[o + PLR.tiles_share] = clip01(tiles / landTiles);
    d[o + PLR.tiles_ratio] = isSelf ? 0 : ratio(tiles, myTiles);
    d[o + PLR.troops_fill] = clip01(p.troops() / pMax);
    d[o + PLR.troops_log] = logn(p.troops(), 5e6);
    d[o + PLR.troops_ratio] = isSelf ? 0 : ratio(p.troops(), myTroops);
    d[o + PLR.max_troops_ratio] = isSelf ? 0 : ratio(pMax, myMax);
    const pGold = Number(p.gold());
    d[o + PLR.gold_log] = logn(pGold, 1e7);
    d[o + PLR.gold_ratio] = isSelf ? 0 : ratio(pGold, gold);
    d[o + PLR.income_rate] = logn(frame.rates.rate(pid, "earned"), 1e5);
    d[o + PLR.cities] = clip01(p.unitsOwned(UnitType.City) / 10);
    d[o + PLR.ports] = clip01(p.unitsOwned(UnitType.Port) / 10);
    d[o + PLR.factories] = clip01(p.unitsOwned(UnitType.Factory) / 10);
    d[o + PLR.posts] = clip01(p.unitsOwned(UnitType.DefensePost) / 10);
    d[o + PLR.sams] = clip01(p.unitsOwned(UnitType.SAMLauncher) / 10);
    d[o + PLR.silos] = clip01(p.unitsOwned(UnitType.MissileSilo) / 10);
    d[o + PLR.warships] = clip01(p.unitsOwned(UnitType.Warship) / 10);
    d[o + PLR.is_traitor] = p.isTraitor() ? 1 : 0;
    d[o + PLR.betrayals] = clip01(p.betrayals() / 5);
    d[o + PLR.is_immune] = p.isImmune() ? 1 : 0;
    d[o + PLR.disconnected] = p.isDisconnected() ? 1 : 0;
    const rk = frame.rank.get(pid);
    if (rk !== undefined) {
      d[o + PLR.rank] = clip01(1 - rk / Math.max(1, alivePlayers - 1));
      d[o + PLR.is_top1] = rk === 0 ? 1 : 0;
      d[o + PLR.is_top3] = rk < TOP_ROWS ? 1 : 0;
    }
    d[o + PLR.alive] = p.isAlive() ? 1 : 0;
    if (isSelf) return;
    d[o + PLR.is_neighbor] = neighborSet.has(pid) ? 1 : 0;
    d[o + PLR.can_attack] = me.canAttackPlayer(p) ? 1 : 0;
    const rel = me.relation(p);
    d[o + PLR.rel_hostile] = rel === Relation.Hostile ? 1 : 0;
    d[o + PLR.rel_distrustful] = rel === Relation.Distrustful ? 1 : 0;
    d[o + PLR.rel_neutral] = rel === Relation.Neutral ? 1 : 0;
    d[o + PLR.rel_friendly] = rel === Relation.Friendly ? 1 : 0;
    const alliance = me.allianceWith(p);
    d[o + PLR.is_ally] = alliance !== null ? 1 : 0;
    d[o + PLR.alliance_left] = alliance !== null ? clip01((alliance.expiresAt() - tick) / allianceDuration) : 0;
    d[o + PLR.targets_me] = p.targets().some((t) => t === me) ? 1 : 0;
    d[o + PLR.req_in] = reqIn.has(pid) ? 1 : 0;
    d[o + PLR.req_out] = reqOut.has(pid) ? 1 : 0;
    d[o + PLR.can_request] = me.canSendAllianceRequest(p) ? 1 : 0;
    d[o + PLR.embargo_by_me] = me.hasEmbargoAgainst(p) ? 1 : 0;
    d[o + PLR.embargo_on_me] = p.hasEmbargoAgainst(me) ? 1 : 0;
    d[o + PLR.targeted_by_me] = myTargets.has(pid) ? 1 : 0;
    const a = me.spawnTile();
    const b = p.spawnTile();
    if (a !== undefined && b !== undefined) {
      d[o + PLR.spawn_dist] = clip01(Math.hypot(map.x(a) - map.x(b), map.y(a) - map.y(b)) / diag);
    }
    d[o + PLR.attacks_on_me] = clip01((inTroops.get(pid) ?? 0) / myMax);
    d[o + PLR.my_attack_on_them] = clip01((outTroops.get(pid) ?? 0) / myMax);
  });

  // --- Tabla de ataques ------------------------------------------------------------------------
  const attacks = makeTable(MAX_ATTACK_ROWS, ATTACK_ROW_FEATURES.length);
  const byTroops = <T extends { troops(): number; id(): string }>(list: T[]): T[] =>
    list.slice().sort((x, y) => y.troops() - x.troops() || (x.id() < y.id() ? -1 : 1));
  let ar = 0;
  const addAttack = (a: (typeof outgoing)[number], isOut: boolean): void => {
    const o = ar * attacks.cols;
    const d = attacks.data;
    attacks.valid[ar] = 1;
    d[o + ATK.outgoing] = isOut ? 1 : 0;
    d[o + ATK.incoming] = isOut ? 0 : 1;
    d[o + ATK.troops_over_max] = clip01(a.troops() / myMax);
    d[o + ATK.troops_over_mine] = clip01(a.troops() / Math.max(1, myTroops) / 2);
    d[o + ATK.retreating] = a.retreating() ? 1 : 0;
    d[o + ATK.border_size] = logn(a.borderSize(), 1e4);
    const other = isOut ? a.target() : a.attacker();
    d[o + ATK.vs_nobody] = !other.isPlayer() ? 1 : 0;
    d[o + ATK.from_landing] = a.sourceTile() !== null ? 1 : 0;
    const row = other.isPlayer() ? rowOf.get(other.smallID()) : undefined;
    d[o + ATK.counterpart_in_table] = row !== undefined ? 1 : 0;
    d[o + ATK.counterpart_row] = row !== undefined ? clip01(row / (PLAYER_ROWS - 1)) : 0;
    ar++;
  };
  byTroops(outgoing).slice(0, MAX_ATTACKS_PER_SIDE).forEach((a) => addAttack(a, true));
  byTroops(incoming).slice(0, MAX_ATTACKS_PER_SIDE).forEach((a) => addAttack(a, false));

  // --- Escalares de defensa y ataque -----------------------------------------------------------
  const ms = new Float32Array(MIL_SCALARS.length);
  const mySams = myStructs.filter((u) => u.type() === UnitType.SAMLauncher && !u.isUnderConstruction());
  const mySilos = myStructs.filter((u) => u.type() === UnitType.MissileSilo && !u.isUnderConstruction());
  const slotsOf = (list: Unit[]): { free: number; total: number } =>
    list.reduce(
      (acc, u) => ({ free: acc.free + (u.level() - u.missileTimerQueue().length), total: acc.total + u.level() }),
      { free: 0, total: 0 },
    );
  const samSlots = slotsOf(mySams);
  const siloSlots = slotsOf(mySilos);
  ms[MS.posts] = clip01(me.unitsOwned(UnitType.DefensePost) / 10);
  ms[MS.sams] = clip01(me.unitsOwned(UnitType.SAMLauncher) / 10);
  ms[MS.silos] = clip01(me.unitsOwned(UnitType.MissileSilo) / 10);
  ms[MS.silos_ready] = siloSlots.total > 0 ? clip01(siloSlots.free / siloSlots.total) : 0;
  ms[MS.sams_ready] = samSlots.total > 0 ? clip01(samSlots.free / samSlots.total) : 0;
  const protectable = myStructs.filter((u) => u.type() !== UnitType.DefensePost);
  if (protectable.length > 0) {
    const covered = protectable.filter((s) =>
      mySams.some((m) => {
        const r = config.dynamicSamRange(m, tick);
        const dx = map.x(s.tile()) - map.x(m.tile());
        const dy = map.y(s.tile()) - map.y(m.tile());
        return dx * dx + dy * dy <= r * r;
      }),
    ).length;
    ms[MS.struct_sam_covered] = covered / protectable.length;
  }
  const myShips = me.units(UnitType.Warship).filter((u) => u.isActive());
  ms[MS.warships] = clip01(myShips.length / 5);
  if (myShips.length > 0) {
    ms[MS.warship_health] = clip01(myShips.reduce((s, u) => s + u.health() / Math.max(1, u.maxHealth()), 0) / myShips.length);
    ms[MS.warships_in_combat] = clip01(myShips.filter((u) => u.warshipState().isInCombat === true).length / myShips.length);
  }
  const myBoats = me.units(UnitType.TransportShip).filter((u) => u.isActive());
  ms[MS.transports] = clip01(myBoats.length / Math.max(1, config.boatMaxNumber()));
  ms[MS.transport_troops] = clip01(myBoats.reduce((s, u) => s + u.troops(), 0) / myMax);
  const sum = (list: { troops(): number }[]): number => list.reduce((s, a) => s + a.troops(), 0);
  ms[MS.attacks_out_n] = clip01(outgoing.length / MAX_ATTACKS_PER_SIDE);
  ms[MS.attacks_out_troops] = clip01(sum(outgoing) / myMax);
  ms[MS.attacks_in_n] = clip01(incoming.length / MAX_ATTACKS_PER_SIDE);
  ms[MS.attacks_in_troops] = clip01(sum(incoming) / Math.max(1, myTroops) / 2);
  ms[MS.attack_in_max] = clip01(incoming.reduce((m, a) => Math.max(m, a.troops()), 0) / Math.max(1, myTroops) / 2);
  let ownFlying = 0;
  let atomIn = 0;
  let hydroIn = 0;
  let mirvIn = 0;
  let progressIn = 0;
  for (const t of NUKE_TYPES) {
    for (const u of frame.unitsByType.get(t) ?? []) {
      if (cls(u) & B_OWN) ownFlying++;
      const target = u.targetTile();
      if (target === undefined || !map.hasOwner(target) || map.ownerID(target) !== meSmall) continue;
      if (cls(u) & B_OWN) continue;
      if (t === UnitType.AtomBomb) atomIn++;
      else if (t === UnitType.HydrogenBomb) hydroIn++;
      else mirvIn++;
      progressIn = Math.max(progressIn, t === UnitType.MIRV ? 0 : nukeProgress(u));
    }
  }
  ms[MS.nukes_own_flying] = clip01(ownFlying / 10);
  ms[MS.atom_inbound] = clip01(atomIn / 5);
  ms[MS.hydrogen_inbound] = clip01(hydroIn / 5);
  ms[MS.mirv_inbound] = clip01(mirvIn / 5);
  ms[MS.nuke_inbound_progress] = progressIn;
  ms[MS.fallout_own] = ownSamples > 0 ? clip01(ownFallout / ownSamples) : 0;
  ms[MS.fallout_map] = clip01(game.numTilesWithFallout() / landTiles);

  // --- Escalares de comparación ------------------------------------------------------------------
  const cs = new Float32Array(CMP_SCALARS.length);
  const myRank = frame.rank.get(meSmall);
  const leader = frame.alive[0];
  const leaderTiles = leader?.numTilesOwned() ?? 0;
  const myShare = clip01(myTiles / landTiles);
  cs[CS.my_rank] = myRank === undefined ? 0 : clip01(1 - myRank / Math.max(1, alivePlayers - 1));
  cs[CS.leader_share] = clip01(leaderTiles / landTiles);
  cs[CS.leader_ratio] = ratio(leaderTiles, myTiles);
  cs[CS.top3_share] = clip01(frame.alive.slice(0, TOP_ROWS).reduce((s, p) => s + p.numTilesOwned(), 0) / landTiles);
  cs[CS.win_gap] = clip01(winThreshold - myShare);
  cs[CS.neighbors_n] = clip01(neighbors.length / MAX_NEIGHBOR_ROWS);
  cs[CS.neighbors_share] = clip01(neighbors.reduce((s, p) => s + p.numTilesOwned(), 0) / landTiles);
  if (neighbors.length > 0) {
    const troops = neighbors.map((p) => p.troops());
    cs[CS.neighbors_troops_ratio] = ratio(troops.reduce((s, v) => s + v, 0), myTroops);
    cs[CS.strongest_ratio] = ratio(Math.max(...troops), myTroops);
    cs[CS.weakest_ratio] = ratio(Math.min(...troops), myTroops);
    cs[CS.hostile_neighbors] = clip01(neighbors.filter((p) => me.relation(p) <= Relation.Distrustful).length / MAX_NEIGHBOR_ROWS);
  }
  cs[CS.allies_n] = clip01(me.allies().length / 5);
  cs[CS.allies_share] = clip01(me.allies().reduce((s, p) => s + p.numTilesOwned(), 0) / landTiles);
  const openEnemies = new Set<number>([...outTroops.keys(), ...inTroops.keys()]);
  cs[CS.open_attack_enemies] = clip01(openEnemies.size / MAX_NEIGHBOR_ROWS);

  // --- Generales (los 6 primeros son los de la v1) -------------------------------------------------
  const gs = new Float32Array(GLOBAL_FEATURES.length);
  gs[GS.land_share] = myShare;
  gs[GS.troops_fill] = clip01(myTroops / myMax);
  gs[GS.gold_log] = logn(gold, 1e7);
  gs[GS.alive_ratio] = inSpawn ? 1 : clip01(game.players().length / Math.max(1, ctx.initialPlayers));
  gs[GS.time_progress] = clip01(ctx.ticksSinceSpawnEnd / ctx.cfg.maxTicks);
  gs[GS.in_spawn] = inSpawn ? 1 : 0;
  gs[GS.win_threshold] = clip01(winThreshold);
  gs[GS.leader_share] = clip01(leaderTiles / landTiles);
  gs[GS.my_rank] = cs[CS.my_rank];
  gs[GS.alive_log] = logn(alivePlayers, 1000);
  gs[GS.humans_alive] = clip01(frame.alive.filter((p) => p.type() === PlayerType.Human).length / 10);
  gs[GS.nations_alive] = clip01(frame.alive.filter((p) => p.type() === PlayerType.Nation).length / 50);
  gs[GS.spawn_immunity] = me.isImmune() ? 1 : 0;
  gs[GS.is_traitor] = me.isTraitor() ? 1 : 0;

  return {
    grid,
    globals: gs,
    mask: info.mask,
    tick,
    inSpawnPhase: inSpawn,
    economy: { table: econ, scalars: es },
    military: { units: mil, attacks, scalars: ms },
    comparison: { players, scalars: cs },
  };
}

/** Avance de una bomba por su trayectoria, en [0, 1]. Devuelve 0 si el motor aún no la ha calculado. */
function nukeProgress(u: Unit): number {
  const s = u.nukeState();
  const n = s.trajectory.length;
  return n > 0 ? clip01(s.trajectoryIndex / n) : 0;
}

/**
 * Elige una rejilla (h, w) con unas `targetCells` celdas y la proporción del mapa,
 * para que las celdas sean casi cuadradas. Ej.: World 2000x1000 -> 32x64; Britannia 1600x2088 -> 52x40.
 */
export function gridForMap(width: number, height: number, targetCells = 2048): { h: number; w: number } {
  const aspect = width / height;
  const h = Math.max(1, Math.round(Math.sqrt(targetCells / aspect)));
  const w = Math.max(1, Math.round(h * aspect));
  return { h, w };
}
