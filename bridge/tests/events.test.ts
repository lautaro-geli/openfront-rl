import { Difficulty, PlayerType, UnitType } from "@openfront/engine-api/game/GameTypes";
import { describe, expect, it } from "vitest";
import { countConqueredBy, countConquest, countIntentEvents, EVENT_NAMES, zeroEvents } from "../src/events";
import { CELL_KINDS } from "../src/features";
import { HeuristicPolicy } from "../src/heuristicPolicy";
import { MultiAgentEnv } from "../src/multiEnv";

describe("eventos del paso", () => {
  it("cuenta misiles, construcciones, ataques y expansiones a partir de las intenciones", async () => {
    const env = await MultiAgentEnv.create({ bots: 10, maxTicks: 600 });
    const pol = new HeuristicPolicy("ev-1", env.cfg.grid);
    let obs = await env.reset("ev-1");
    for (let i = 0; i < 5; i++) obs = env.step(obs.map((o) => pol.act(o))).obs; // las tribus aparecen al terminar el spawn
    const g = env.unsafeGame();
    const tribe = g.allPlayers().find((p) => p.type() === PlayerType.Bot);
    expect(tribe).toBeDefined();
    const ev = zeroEvents();
    const c = "rl-agent-0";
    countIntentEvents(
      g,
      [
        { type: "build_unit", unit: UnitType.AtomBomb, tile: 0, clientID: c },
        { type: "build_unit", unit: UnitType.MIRV, tile: 0, clientID: c },
        { type: "build_unit", unit: UnitType.City, tile: 0, clientID: c },
        { type: "attack", targetID: null, troops: 10, clientID: c },
        { type: "attack", targetID: tribe!.id(), troops: 10, clientID: c },
        { type: "boat", troops: 10, dst: 0, clientID: c },
      ] as never,
      ev,
    );
    expect(ev).toMatchObject({ nuke_orders: 2, build_orders: 1, expand_orders: 1, attack_orders: 1, attack_tribe_orders: 1, boat_orders: 1 });
  }, 60000);

  it("la heurística ordena ataques y los eventos son enteros coherentes", async () => {
    const env = await MultiAgentEnv.create({ bots: 20, maxTicks: 3000 });
    const pol = new HeuristicPolicy("ev-2", env.cfg.grid);
    let obs = await env.reset("ev-2");
    const tot = zeroEvents();
    for (let i = 0; i < 250; i++) {
      const r = env.step(obs.map((o) => pol.act(o)));
      obs = r.obs;
      for (const k of EVENT_NAMES) {
        const v = r.info[0].events[k];
        expect(Number.isInteger(v) && v >= 0).toBe(true);
        tot[k] += v;
      }
      if (r.done) break;
    }
    expect(tot.attack_orders + tot.expand_orders).toBeGreaterThan(0);
    expect(tot.attack_tribe_orders).toBeLessThanOrEqual(tot.attack_orders);
  }, 240000);

  it("cuenta las construcciones realmente colocadas por el agente (no las órdenes ni las capturadas)", async () => {
    const env = await MultiAgentEnv.create({ bots: 20, nations: "disabled", difficulty: Difficulty.Easy, maxTicks: 6000 });
    const pol = new HeuristicPolicy("ev-b", env.cfg.grid);
    let obs = await env.reset("ev-b");
    const tot = zeroEvents();
    const cells = env.cfg.grid.h * env.cfg.grid.w;
    for (let i = 0; i < 120; i++) {
      const o = obs[0];
      let act = pol.act(o);
      // Tras la expansión inicial, el agente pide la primera estructura permitida por la máscara (la que pueda pagar).
      const cell = i >= 40 && i % 10 === 0 ? o.mask.cell.slice(CELL_KINDS.indexOf("build") * cells, (CELL_KINDS.indexOf("build") + 1) * cells).findIndex((x) => x === 1) : -1;
      const type = [0, 1, 2, 3, 4, 5].find((t) => o.mask.buildType[t] === 1);
      if (cell >= 0 && type !== undefined) act = { kind: "build", row: 0, cell, fraction: 0.5, type, aux: 0 } as never;
      const r = env.step([act]);
      obs = r.obs;
      for (const k of EVENT_NAMES) tot[k] += r.info[0].events[k];
      if (r.done) break;
    }
    const me = env.unsafeGame().allPlayers().find((p) => p.clientID() === "rl-agent-0")!;
    const built = tot.built_city + tot.built_port + tot.built_factory + tot.built_post + tot.built_sam + tot.built_silo;
    const owned = [UnitType.City, UnitType.Port, UnitType.Factory, UnitType.DefensePost, UnitType.SAMLauncher, UnitType.MissileSilo].reduce((n, u) => n + me.units(u).length, 0);
    expect(built).toBeGreaterThan(0);
    expect(built).toBeLessThanOrEqual(tot.build_orders);
    expect(built).toBeGreaterThanOrEqual(owned);
    expect(tot.nukes_launched).toBe(0);
  }, 300000);

  it("cuenta las tribus conquistadas por el agente cuando ataca a sus vecinos", async () => {
    const env = await MultiAgentEnv.create({ bots: 20, nations: "disabled", difficulty: Difficulty.Easy, maxTicks: 18000 });
    const heu = new HeuristicPolicy("k-b", env.cfg.grid);
    let obs = await env.reset("k-b");
    const tot = zeroEvents();
    for (let i = 0; i < 1800; i++) {
      const o = obs[0];
      const base = heu.act(o);
      const slot = o.mask.attackSlots.findIndex((x) => x);
      // La heurística nunca ataca a tribus: aquí se cambia cada expansión por un ataque si hay un vecino atacable.
      const act = base.kind === "expand" && slot >= 0 ? ({ kind: "attack", slot, fraction: 0.75 } as const) : base;
      const r = env.step([act]);
      obs = r.obs;
      for (const k of EVENT_NAMES) tot[k] += r.info[0].events[k];
      if (r.done) break;
    }
    const dead = env.unsafeGame().allPlayers().filter((p) => p.type() === PlayerType.Bot && !p.isAlive()).length;
    expect(tot.tribes_conquered).toBeGreaterThan(0);
    expect(tot.tribes_conquered).toBeLessThanOrEqual(dead);
    expect(tot.attack_tribe_orders).toBeGreaterThan(0);
  }, 400000);

  it("distingue tribus y naciones al contar conquistas hechas y recibidas", async () => {
    const env = await MultiAgentEnv.create({ bots: 5, nations: 3, difficulty: Difficulty.Easy, maxTicks: 600 });
    const pol = new HeuristicPolicy("ev-3", env.cfg.grid);
    let obs = await env.reset("ev-3");
    for (let i = 0; i < 5; i++) obs = env.step(obs.map((o) => pol.act(o))).obs;
    const all = env.unsafeGame().allPlayers();
    const tribe = all.find((p) => p.type() === PlayerType.Bot)!;
    const nation = all.find((p) => p.type() === PlayerType.Nation)!;
    const ev = zeroEvents();
    countConqueredBy(ev, tribe);
    countConqueredBy(ev, nation);
    countConqueredBy(ev, nation);
    countConquest(ev, tribe);
    countConquest(ev, nation);
    expect(ev).toMatchObject({ conquered_by_tribe: 1, conquered_by_nation: 2, tribes_conquered: 1, nations_conquered: 1 });
  }, 60000);
});
