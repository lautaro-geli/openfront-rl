import { UnitType } from "@openfront/engine-api/game/GameTypes";
import type { Game, Player } from "@openfront/engine/game/Game";
import { describe, expect, it } from "vitest";
import { actionToIntents, vecToAction } from "../src/actions";
import {
  ACTION_KINDS,
  BUILD_TYPES,
  CELL_KINDS,
  KIND,
  NUKE_KINDS,
  PLAYER_KINDS,
  PLAYER_ROWS,
  UNIT_KINDS,
  UNIT_ROWS,
} from "../src/features";
import { MultiAgentEnv } from "../src/multiEnv";
import type { Action, EnvConfig, MultiStepResult, Observation } from "../src/types";

const CFG: Partial<EnvConfig> = { agents: 2, bots: 0, nations: "disabled", maxTicks: 6000 };
const NOOP: Action = { kind: "noop" };

interface Ctx {
  env: MultiAgentEnv;
  game: Game;
  me: Player;
  other: Player;
  obs: Observation[];
  cells: number;
  last: MultiStepResult | null;
}

const give = (p: Player, gold: bigint) => (p as unknown as { addGold(n: bigint): void }).addGold(gold);

function firstValid(a: Uint8Array, offset: number, n: number): number {
  for (let i = 0; i < n; i++) if (a[offset + i] === 1) return i;
  return -1;
}

/** Dos agentes ya en juego: aparecen lejos uno del otro y se expanden unos pasos. */
async function setup(seed: string, growth = 25, coastal = false): Promise<Ctx> {
  const env = await MultiAgentEnv.create(CFG);
  let obs = await env.reset(seed);
  const cells = env.cfg.grid.h * env.cfg.grid.w;
  const land: number[] = [];
  const SHORE = 5; // canal "shore" del grid
  for (let c = 0; c < cells; c++) {
    const ok = coastal ? obs[0].grid[c] >= 0.6 && obs[0].grid[SHORE * cells + c] >= 0.1 : obs[0].grid[c] >= 0.95;
    if (ok) land.push(c);
  }
  const step = Math.floor(land.length / 3);
  const spawns = [land[step], land[2 * step]].map((c) => ({ kind: "spawn", cx: c % env.cfg.grid.w, cy: Math.floor(c / env.cfg.grid.w) }) as Action);
  let r = env.step(spawns);
  for (let i = 0; i < growth; i++) r = env.step([{ kind: "expand", fraction: 0.5 }, { kind: "expand", fraction: 0.5 }]);
  obs = r.obs;
  const game = env.unsafeGame();
  const me = game.playerByClientID("rl-agent-0") as Player;
  const other = game.playerByClientID("rl-agent-1") as Player;
  return { env, game, me, other, obs, cells, last: r };
}

function stepBoth(c: Ctx, a0: Action, a1: Action = NOOP): MultiStepResult {
  const r = c.env.step([a0, a1]);
  c.obs = r.obs;
  c.last = r;
  return r;
}
/** ¿La resolución exacta acepta esta acción con el estado actual? No avanza el juego. */
function resolves(c: Ctx, a: Action): boolean {
  const info = (c.env as unknown as { ep: { agents: { maskInfo: never }[] } }).ep.agents[0].maskInfo;
  const out = actionToIntents(a, c.game, c.me, "rl-agent-0", info, c.env.cfg.grid, 8);
  return !out.invalid && out.intents.length > 0;
}
const idle = (c: Ctx, n: number) => {
  for (let i = 0; i < n; i++) stepBoth(c, NOOP);
};

function playerRow(obs: Observation, kind: (typeof PLAYER_KINDS)[number]): number {
  return firstValid(obs.mask.player, PLAYER_KINDS.indexOf(kind) * PLAYER_ROWS, PLAYER_ROWS);
}
function unitRows(obs: Observation, kind: (typeof UNIT_KINDS)[number]): number[] {
  const out: number[] = [];
  const base = UNIT_KINDS.indexOf(kind) * UNIT_ROWS;
  for (let i = 0; i < UNIT_ROWS; i++) if (obs.mask.unit[base + i] === 1) out.push(i);
  return out;
}
function cellsOf(obs: Observation, kind: (typeof CELL_KINDS)[number], cells: number): number[] {
  const out: number[] = [];
  const base = CELL_KINDS.indexOf(kind) * cells;
  for (let i = 0; i < cells; i++) if (obs.mask.cell[base + i] === 1) out.push(i);
  return out;
}

/** Construye una estructura probando celdas válidas hasta que la resolución exacta acepta una. */
function buildStructure(c: Ctx, name: (typeof BUILD_TYPES)[number]): boolean {
  const type = BUILD_TYPES.indexOf(name);
  const cand = cellsOf(c.obs[0], name === "warship" ? "build_ship" : "build", c.cells);
  for (const cell of cand.slice(0, 60)) {
    if (c.obs[0].mask.buildType[type] !== 1) give(c.me, 20_000_000n);
    const before = c.me.unitsOwned(
      { city: UnitType.City, port: UnitType.Port, factory: UnitType.Factory, post: UnitType.DefensePost, sam: UnitType.SAMLauncher, silo: UnitType.MissileSilo, warship: UnitType.Warship }[name],
    );
    const r = stepBoth(c, { kind: "build", type, cell });
    const after = c.me.unitsOwned(
      { city: UnitType.City, port: UnitType.Port, factory: UnitType.Factory, post: UnitType.DefensePost, sam: UnitType.SAMLauncher, silo: UnitType.MissileSilo, warship: UnitType.Warship }[name],
    );
    if (!r.info[0].invalidAction && after > before) return true;
  }
  return false;
}

describe("acciones: el catálogo completo", () => {
  it("el catálogo tiene 24 tipos y vecToAction decodifica cada uno", async () => {
    expect(ACTION_KINDS.length).toBe(24);
    const grid = { h: 32, w: 64 };
    for (let k = 0; k < 24; k++) {
      const a = vecToAction([k, 1, 70, 2, 1, 3], grid);
      expect(a?.kind).toBe(ACTION_KINDS[k]);
    }
    expect(vecToAction([99, 0, 0, 0, 0, 0], grid)).toBeNull();
  });

  it("una acción enmascarada no llega al motor; un tipo desconocido cuenta como inválido", async () => {
    const c = await setup("mask-guard", 5);
    // Sin silo, lanzar una bomba está enmascarado.
    expect(c.obs[0].mask.kind[KIND.nuke]).toBe(0);
    const r = stepBoth(c, { kind: "nuke", type: 0, cell: 100 });
    expect(r.info[0].invalidAction).toBe(true);
    expect(c.me.units(UnitType.AtomBomb).length).toBe(0);
    const bad = c.env.step([null, NOOP]);
    expect(bad.info[0].invalidAction).toBe(true);
    c.env.close();
  });

  it("construir y mejorar: aparece la ciudad, sube de nivel y se puede borrar", async () => {
    const c = await setup("build-1");
    give(c.me, 50_000_000n);
    idle(c, 1);
    expect(c.obs[0].mask.kind[KIND.build]).toBe(1);
    expect(buildStructure(c, "city")).toBe(true);
    idle(c, 4); // la ciudad tarda 20 ticks en terminar
    const rows = unitRows(c.obs[0], "upgrade");
    expect(rows.length).toBeGreaterThan(0);
    const city = c.me.units(UnitType.City)[0];
    const level = city.level();
    give(c.me, 50_000_000n);
    idle(c, 1);
    const row = unitRows(c.obs[0], "upgrade")[0];
    const r = stepBoth(c, { kind: "upgrade", row });
    expect(r.info[0].invalidAction).toBe(false);
    idle(c, 1);
    expect(c.me.units(UnitType.City).reduce((s, u) => s + u.level(), 0)).toBeGreaterThan(level);

    // Borrar: la unidad queda marcada para borrado.
    const del = unitRows(c.obs[0], "delete_unit");
    if (del.length > 0) {
      const target = c.obs[0].mask.unit.length > 0 ? del[0] : 0;
      const rd = stepBoth(c, { kind: "delete_unit", row: target });
      expect(rd.info[0].invalidAction).toBe(false);
    }
    c.env.close();
  });

  it("alianzas: pedir, rechazar, aceptar, romper (y quedar como traidor)", async () => {
    const c = await setup("alliance-1");
    idle(c, 1);
    const row = playerRow(c.obs[0], "alliance_request");
    expect(row).toBeGreaterThan(0);
    // El agente 0 pide; el agente 1 ve la petición y puede rechazarla.
    stepBoth(c, { kind: "alliance_request", row });
    expect(c.me.outgoingAllianceRequests().length).toBe(1);
    const rej = playerRow(c.obs[1], "alliance_reject");
    expect(rej).toBeGreaterThan(0);
    stepBoth(c, NOOP, { kind: "alliance_reject", row: rej });
    idle(c, 1);
    expect(c.me.isAlliedWith(c.other)).toBe(false);

    // Se pide de nuevo (puede haber enfriamiento): se espera hasta que la máscara lo permita.
    for (let i = 0; i < 80 && playerRow(c.obs[0], "alliance_request") < 0; i++) idle(c, 1);
    const row2 = playerRow(c.obs[0], "alliance_request");
    if (row2 > 0) {
      stepBoth(c, { kind: "alliance_request", row: row2 });
      // Pedir a quien ya nos pidió equivale a aceptar.
      const acc = playerRow(c.obs[1], "alliance_request");
      expect(acc).toBeGreaterThan(0);
      stepBoth(c, NOOP, { kind: "alliance_request", row: acc });
      expect(c.me.isAlliedWith(c.other)).toBe(true);

      // Con alianza ya no se puede atacar, pero sí romperla.
      idle(c, 1);
      const brk = playerRow(c.obs[0], "alliance_break");
      expect(brk).toBeGreaterThan(0);
      const rb = stepBoth(c, { kind: "alliance_break", row: brk });
      expect(rb.info[0].invalidAction).toBe(false);
      expect(c.me.isAlliedWith(c.other)).toBe(false);
      expect(c.me.isTraitor()).toBe(true);
    }
    c.env.close();
  });

  it("embargos, marcar objetivo y emojis", async () => {
    const c = await setup("diplomacy-1");
    idle(c, 1);
    const row = playerRow(c.obs[0], "embargo_start");
    expect(row).toBeGreaterThan(0);
    stepBoth(c, { kind: "embargo_start", row });
    expect(c.me.hasEmbargoAgainst(c.other)).toBe(true);
    idle(c, 1);
    const stop = playerRow(c.obs[0], "embargo_stop");
    expect(stop).toBeGreaterThan(0);
    stepBoth(c, { kind: "embargo_stop", row: stop });
    expect(c.me.hasEmbargoAgainst(c.other)).toBe(false);

    stepBoth(c, { kind: "embargo_all_start" });
    stepBoth(c, { kind: "embargo_all_stop" });

    const t = playerRow(c.obs[0], "target_player");
    expect(t).toBeGreaterThan(0);
    const rt = stepBoth(c, { kind: "target_player", row: t });
    expect(rt.info[0].invalidAction).toBe(false);
    expect(c.me.targets().includes(c.other)).toBe(true);

    for (const aux of [0, 7]) {
      const e = playerRow(c.obs[0], "emoji");
      expect(e).toBeGreaterThanOrEqual(0);
      const re = stepBoth(c, { kind: "emoji", row: e, aux });
      expect(re.info[0].invalidAction).toBe(false);
    }
    // Las donaciones entre humanos están desactivadas: la máscara las prohíbe.
    expect(c.obs[0].mask.kind[KIND.donate_gold]).toBe(0);
    expect(c.obs[0].mask.kind[KIND.donate_troops]).toBe(0);
    c.env.close();
  });

  it("cancelar un ataque en curso", async () => {
    const c = await setup("cancel-1", 5);
    stepBoth(c, { kind: "expand", fraction: 1 });
    expect(c.obs[0].mask.attackRow[0]).toBe(1);
    expect(c.obs[0].mask.kind[KIND.cancel_attack]).toBe(1);
    const r = stepBoth(c, { kind: "cancel_attack", row: 0 });
    expect(r.info[0].invalidAction).toBe(false);
    expect(c.me.outgoingAttacks().some((a) => a.isActive() && !a.retreating())).toBe(false);
    c.env.close();
  });

  it("silo y bomba: se construye el silo, se lanza una bomba y el silo entra en espera", async () => {
    const c = await setup("nuke-1");
    give(c.me, 100_000_000n);
    idle(c, 1);
    expect(buildStructure(c, "silo")).toBe(true);
    idle(c, 12); // el silo tarda 100 ticks
    give(c.me, 100_000_000n);
    idle(c, 1);
    expect(c.obs[0].mask.nukeType[NUKE_KINDS.indexOf("atom")]).toBe(1);
    const cand = cellsOf(c.obs[0], "nuke", c.cells);
    expect(cand.length).toBeGreaterThan(0);
    let launched = false;
    for (const cell of cand.slice(0, 80)) {
      const r = stepBoth(c, { kind: "nuke", type: 0, cell });
      if (!r.info[0].invalidAction) {
        launched = true;
        break;
      }
      give(c.me, 100_000_000n);
    }
    expect(launched).toBe(true);
    expect(c.game.units([UnitType.AtomBomb]).length + c.me.units(UnitType.MissileSilo).filter((s) => s.isInCooldown()).length).toBeGreaterThan(0);
    c.env.close();
  });

  it("puerto, barco de guerra y mover el barco", async () => {
    const c = await setup("navy-1", 30, true);
    give(c.me, 100_000_000n);
    idle(c, 1);
    expect(buildStructure(c, "port")).toBe(true);
    idle(c, 8); // el puerto tarda 50 ticks
    give(c.me, 100_000_000n);
    idle(c, 1);
    expect(c.obs[0].mask.buildType[BUILD_TYPES.indexOf("warship")]).toBe(1);
    expect(buildStructure(c, "warship")).toBe(true);
    idle(c, 2);
    const ships = unitRows(c.obs[0], "move_warship");
    expect(ships.length).toBeGreaterThan(0);
    const water = cellsOf(c.obs[0], "move_warship", c.cells);
    const cell = water.find((w) => resolves(c, { kind: "move_warship", row: ships[0], cell: w }));
    expect(cell).toBeDefined();
    const r = stepBoth(c, { kind: "move_warship", row: ships[0], cell: cell as number });
    expect(r.info[0].invalidAction).toBe(false);
    c.env.close();
  });

  it("desembarco: se envía un barco con tropas a una celda alcanzable", async () => {
    const c = await setup("boat-1", 30, true);
    idle(c, 1);
    const cand = cellsOf(c.obs[0], "boat_attack", c.cells);
    expect(cand.length).toBeGreaterThan(0);
    const cell = cand.find((x) => resolves(c, { kind: "boat_attack", cell: x, fraction: 0.25 }));
    expect(cell).toBeDefined();
    const r = stepBoth(c, { kind: "boat_attack", cell: cell as number, fraction: 0.25 });
    expect(r.info[0].invalidAction).toBe(false);
    expect(c.me.units(UnitType.TransportShip).length).toBeGreaterThan(0);
    expect(c.me.units(UnitType.TransportShip)[0].troops()).toBeGreaterThan(0);
    c.env.close();
  });

  it("actionToIntents rechaza filas y celdas fuera de rango", async () => {
    const c = await setup("range-1", 5);
    const info = (c.env as unknown as { ep: { agents: { maskInfo: never }[] } }).ep.agents[0].maskInfo;
    const g = c.env.cfg.grid;
    const bads: Action[] = [
      { kind: "build", type: 99, cell: 0 },
      { kind: "build", type: 0, cell: -1 },
      { kind: "upgrade", row: 9999 },
      { kind: "alliance_request", row: -3 },
      { kind: "emoji", row: 0, aux: 99 },
      { kind: "donate_gold", row: 1, fraction: 0.5 },
    ];
    for (const a of bads) {
      const out = actionToIntents(a, c.game, c.me, "rl-agent-0", info, g, 8);
      expect(out.intents.length).toBe(0);
      expect(out.invalid).toBe(true);
    }
    c.env.close();
  });
});
