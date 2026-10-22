import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  PlayerInfo,
  PlayerType,
} from "@openfront/engine-api/game/GameTypes";
import { GameUpdateType, type HashUpdate } from "@openfront/engine-api/game/GameUpdates";
import type { GameConfig, GameStartInfo, StampedIntent } from "@openfront/engine-api/Schemas";
import { loadTerrainMap } from "@openfront/engine-lib/game/TerrainMapLoader";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { EngineConfig } from "@openfront/engine/configuration/EngineConfig";
import { Executor } from "@openfront/engine/execution/ExecutionManager";
import type { Game, Player } from "@openfront/engine/game/Game";
import type { UnitType } from "@openfront/engine-api/game/GameTypes";
import { createGame } from "@openfront/engine/game/GameImpl";
import { createNationsForGame } from "@openfront/engine/game/NationCreation";
import { GameRunner } from "@openfront/engine/GameRunner";
import { loadMapFiles } from "@openfront/shared/GameMapLoader";
import { actionToIntents, computeMaskInfo, type MaskInfo } from "./actions";
import { emptyMask } from "./mask";
import { ACTION_KINDS, BUILD_TYPES, KIND, type ActionKind } from "./features";
import { BUILD_UNIT } from "./mask";
import { countBuilt, countConqueredBy, countConquest, countIntentEvents, zeroEvents, type Events } from "./events";
import { computeRewardParts, newRewardState, sumParts, type RewardState } from "./reward";
import { FsMapLoader } from "./mapLoader";
import {
  buildFrame,
  buildObservation,
  buildSamplePlan,
  RateTracker,
  type ObsContext,
  type SamplePlan,
} from "./observation";
import type { Action, EnvConfig, MultiStepResult, Observation, StepInfo } from "./types";

const MAX_SPAWN_TICKS = 1000;
const HERE = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_MAPS_DIR = path.resolve(HERE, "../../vendor/openfront/resources/maps");

export const DEFAULT_CONFIG: EnvConfig = {
  map: GameMapType.Britannia,
  bots: 20,
  nations: "default",
  difficulty: Difficulty.Medium,
  disabledActions: [],
  maxStructures: {},
  decisionInterval: 10,
  maxTicks: 6000,
  grid: { h: 32, w: 64 },
  samples: 4,
  maxNeighbors: 8,
  rewardTerritoryWeight: 0.8,
  rewardAttackWeight: 0,
  rewardEconomyWeight: 0,
  rewardDefenseWeight: 0,
  rewardMilestonesWeight: 0,
  rewardTroopsWeight: 0,
  territoryKnee: 0,
  territoryKneeShare: 0.5,
  agents: 1,
};

export function clientIdFor(index: number): string {
  return `rl-agent-${index}`;
}

interface Agent {
  clientID: string;
  me: Player;
  rs: RewardState;
  eliminated: boolean;
  /** Máscara calculada al devolver la última observación; el estado no cambia hasta el siguiente step. */
  maskInfo: MaskInfo | null;
}

interface Episode {
  game: Game;
  runner: GameRunner;
  agents: Agent[];
  plan: SamplePlan;
  rates: RateTracker;
  turn: number;
  lastHash: HashUpdate | undefined;
  sawWin: boolean;
  fatal: string | undefined;
  initialPlayers: number;
  spawnEndTick: number | null;
  done: boolean;
  /** Conquistas ocurridas durante el paso en curso: [conquistador, conquistado]. */
  conquests: [Player, Player][];
  /** Jugadores ya contados como conquistados: el motor llama a `conquerPlayer` varias veces por el mismo jugador. */
  conquered: Set<Player>;
  /** Unidades colocadas por cada agente durante el paso en curso (índice de agente → tipos). */
  built: UnitType[][];
}

/**
 * Entorno con uno o más jugadores controlados por código en la misma partida.
 * Cada agente ve el mundo desde su propia perspectiva (canal "propio" = sus tiles).
 */
export class MultiAgentEnv {
  readonly cfg: EnvConfig;
  private mapsDir: string;
  private ep: Episode | null = null;

  private constructor(cfg: EnvConfig, mapsDir: string) {
    this.cfg = cfg;
    this.mapsDir = mapsDir;
  }

  static async create(cfg: Partial<EnvConfig> = {}, mapsDir: string = DEFAULT_MAPS_DIR): Promise<MultiAgentEnv> {
    const full: EnvConfig = { ...DEFAULT_CONFIG, ...cfg, grid: { ...DEFAULT_CONFIG.grid, ...cfg.grid } };
    const unknown = full.disabledActions.filter((k) => !(ACTION_KINDS as readonly string[]).includes(k));
    const badCap = Object.keys(full.maxStructures).filter((k) => !(BUILD_TYPES as readonly string[]).includes(k));
    if (badCap.length > 0) throw new Error(`maxStructures desconocidas: ${badCap.join(", ")}`);
    if (unknown.length > 0) throw new Error(`disabledActions desconocidas: ${unknown.join(", ")}`);
    if (full.decisionInterval < 1) throw new Error("decisionInterval debe ser >= 1");
    if (full.maxTicks < 1) throw new Error("maxTicks debe ser >= 1");
    if (!Number.isInteger(full.samples) || full.samples < 1) throw new Error("samples debe ser un entero >= 1");
    if (!Number.isInteger(full.agents) || full.agents < 1) throw new Error("agents debe ser un entero >= 1");
    return new MultiAgentEnv(full, mapsDir);
  }

  get numAgents(): number {
    return this.cfg.agents;
  }

  async reset(seed: string): Promise<Observation[]> {
    const cfg = this.cfg;
    const gameConfig: GameConfig = {
      gameMap: cfg.map,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Public,
      difficulty: cfg.difficulty,
      nations: cfg.nations,
      donateGold: false,
      donateTroops: false,
      bots: cfg.bots,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };
    const start: GameStartInfo = {
      gameID: seed,
      lobbyCreatedAt: 0,
      config: gameConfig,
      players: Array.from({ length: cfg.agents }, (_, i) => ({
        clientID: clientIdFor(i),
        username: `RLAgent${i}`,
        clanTag: null,
      })),
    };

    // console.debug del motor es muy ruidoso; se silencia como hace su propio harness de rendimiento.
    console.debug = () => {};

    const engineConfig = new EngineConfig(gameConfig, false);
    const terrain = await loadTerrainMap(
      await loadMapFiles(new FsMapLoader(this.mapsDir), gameConfig.gameMap, gameConfig.gameMapSize),
    );
    const random = new PseudoRandom(simpleHash(start.gameID));
    const humans = start.players.map(
      (p) => new PlayerInfo(p.username, PlayerType.Human, p.clientID, random.nextID(), false, null, [], null),
    );
    const nations = createNationsForGame(start, terrain.nations, terrain.additionalNations, humans.length, random);
    const game = createGame(
      humans,
      nations,
      terrain.gameMap,
      terrain.miniGameMap,
      engineConfig,
      terrain.teamGameSpawnAreas,
    );

    const ep = {
      game,
      turn: 0,
      lastHash: undefined,
      sawWin: false,
      fatal: undefined,
      initialPlayers: cfg.agents,
      spawnEndTick: null,
      done: false,
      conquests: [],
      conquered: new Set(),
      built: [],
    } as unknown as Episode;
    ep.runner = new GameRunner(game, new Executor(game, start.gameID, clientIdFor(0)), (gu) => {
      if ("errMsg" in gu) {
        ep.fatal = `${gu.errMsg}\n${gu.stack ?? ""}`;
        return;
      }
      const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[];
      if (hashes.length > 0) ep.lastHash = hashes[hashes.length - 1];
      if ((gu.updates[GameUpdateType.Win] ?? []).length > 0) ep.sawWin = true;
    });
    ep.runner.init();
    // El motor avisa de cada conquista a través de `conquerPlayer`; se envuelve la instancia (no el código del motor).
    const conquerPlayer = game.conquerPlayer.bind(game);
    game.conquerPlayer = (conqueror: Player, conquered: Player) => {
      if (!ep.conquered.has(conquered)) {
        ep.conquered.add(conquered);
        ep.conquests.push([conqueror, conquered]);
      }
      conquerPlayer(conqueror, conquered);
    };
    ep.agents = start.players.map((p) => {
      const me = game.playerByClientID(p.clientID);
      if (me === null) throw new Error(`el jugador ${p.clientID} no existe en la partida`);
      // Se cuentan las unidades que el agente coloca de verdad (envolviendo su instancia, no el motor).
      const index = ep.built.length;
      ep.built.push([]);
      const buildUnit = me.buildUnit.bind(me) as (...a: unknown[]) => unknown;
      (me as unknown as { buildUnit: (...a: unknown[]) => unknown }).buildUnit = (type: unknown, ...rest: unknown[]) => {
        ep.built[index].push(type as UnitType);
        return buildUnit(type, ...rest);
      };
      return { clientID: p.clientID, me, rs: newRewardState(), eliminated: false, maskInfo: null };
    });
    ep.plan = buildSamplePlan(game, cfg.grid.h, cfg.grid.w, cfg.samples);
    ep.rates = new RateTracker();
    this.ep = ep;
    return this.observeAll();
  }

  step(actions: (Action | null)[]): MultiStepResult {
    const ep = this.requireEpisode();
    if (ep.done) throw new Error("el episodio terminó; llama a reset()");
    const { game, agents } = ep;
    const cfg = this.cfg;

    // Todas las acciones se calculan sobre el mismo estado y viajan en el primer tick del paso.
    const intents: StampedIntent[] = [];
    const invalid: boolean[] = [];
    const events: Events[] = agents.map(() => zeroEvents());
    ep.built.forEach((b) => (b.length = 0));
    agents.forEach((a, i) => {
      if (a.eliminated) {
        invalid.push(false);
        return;
      }
      const given = actions[i];
      if (given === null) {
        invalid.push(true); // el protocolo no pudo decodificar la acción
        return;
      }
      const action: Action = given ?? { kind: "noop" };
      const info = a.maskInfo ?? computeMaskInfo(game, a.me, cfg.maxNeighbors, cfg.grid.h * cfg.grid.w);
      const built = actionToIntents(action, game, a.me, a.clientID, info, cfg.grid, cfg.maxNeighbors);
      intents.push(...built.intents);
      if (!built.invalid) countIntentEvents(game, built.intents, events[i]);
      invalid.push(built.invalid);
    });

    const wasSpawn = game.inSpawnPhase();
    this.advance(cfg.decisionInterval, intents);
    if (wasSpawn && agents.every((a) => a.me.hasSpawned())) {
      // Cuando todos ya eligieron sitio, el resto de la fase de spawn se avanza automáticamente.
      let guard = 0;
      while (game.inSpawnPhase()) {
        if (++guard > MAX_SPAWN_TICKS) throw new Error("la fase de spawn no termina");
        this.advance(1, []);
      }
    }
    if (ep.spawnEndTick === null && !game.inSpawnPhase()) {
      ep.spawnEndTick = game.ticks();
      ep.initialPlayers = Math.max(agents.length, game.players().length);
    }
    const afterSpawn = ep.spawnEndTick !== null;

    const landTiles = Math.max(1, game.numLandTiles());
    const reward: number[] = [];
    const info: StepInfo[] = [];
    const terminatedFlags: boolean[] = [];
    const winner = ep.sawWin ? game.getWinner() : null;
    agents.forEach((a, i) => {
      const wasEliminated = a.eliminated;
      const eliminatedNow = afterSpawn && !a.me.isAlive();
      const won = ep.sawWin && winner === a.me;
      for (const t of ep.built[i]) countBuilt(events[i], t);
      for (const [by, who] of ep.conquests) {
        if (by === a.me) countConquest(events[i], who);
        else if (who === a.me) countConqueredBy(events[i], by);
      }
      const parts = computeRewardParts(game, a.me, a.rs, cfg, { afterSpawn, wasEliminated, eliminatedNow, won, landTiles });
      a.eliminated = a.eliminated || eliminatedNow;
      reward.push(sumParts(parts));
      terminatedFlags.push(a.eliminated || ep.sawWin);
      info.push({
        invalidAction: invalid[i],
        rewardParts: parts,
        events: events[i],
        won,
        eliminated: a.eliminated,
        tick: game.ticks(),
      });
    });

    ep.conquests = []; // ya contadas en este paso
    const allEliminated = agents.every((a) => a.eliminated);
    const ticksSince = afterSpawn ? game.ticks() - (ep.spawnEndTick as number) : 0;
    const terminatedGame = ep.sawWin || allEliminated;
    const truncated = !terminatedGame && afterSpawn && ticksSince >= cfg.maxTicks;
    ep.done = terminatedGame || truncated;

    return { obs: this.observeAll(), reward, terminated: terminatedFlags, truncated, done: ep.done, info };
  }

  /** Hash del motor (último HashUpdate) junto con el tick. Sirve para comprobar determinismo. */
  stateHash(): string {
    const ep = this.requireEpisode();
    return `${ep.lastHash?.hash ?? "none"}@${ep.lastHash?.tick ?? -1}/t${ep.game.ticks()}`;
  }

  /** Tiles de cada agente (para comprobar que de verdad actúan). */
  agentTiles(): number[] {
    return this.requireEpisode().agents.map((a) => a.me.numTilesOwned());
  }

  /** Acceso directo al motor. Solo para diagnóstico y experimentos; no lo uses en el bucle de entrenamiento. */
  unsafeGame(): Game {
    return this.requireEpisode().game;
  }

  close(): void {
    this.ep = null;
  }

  private observeAll(): Observation[] {
    const ep = this.requireEpisode();
    const ctx: ObsContext = {
      cfg: this.cfg,
      plan: ep.plan,
      initialPlayers: ep.initialPlayers,
      ticksSinceSpawnEnd: ep.spawnEndTick === null ? 0 : ep.game.ticks() - ep.spawnEndTick,
      // Una sola vez por paso; lo comparten todos los agentes.
      frame: buildFrame(ep.game, ep.rates),
    };
    return ep.agents.map((a) => {
      a.maskInfo = a.eliminated
        ? {
            mask: emptyMask(this.cfg.maxNeighbors, this.cfg.grid.h * this.cfg.grid.w),
            slots: [],
            neighbors: [],
          }
        : computeMaskInfo(ep.game, a.me, this.cfg.maxNeighbors, this.cfg.grid.h * this.cfg.grid.w);
      const obs = buildObservation(ep.game, a.me, ctx, a.maskInfo);
      for (const k of this.cfg.disabledActions) obs.mask.kind[KIND[k as ActionKind]] = 0;
      for (const [name, max] of Object.entries(this.cfg.maxStructures)) {
        const unit = BUILD_UNIT[name as (typeof BUILD_TYPES)[number]];
        if (a.me.units(unit).filter((u) => u.isActive()).length >= max) obs.mask.buildType[BUILD_TYPES.indexOf(name as (typeof BUILD_TYPES)[number])] = 0;
      }
      if (!obs.mask.buildType.some((v) => v === 1)) obs.mask.kind[KIND.build] = 0;
      return obs;
    });
  }

  private advance(ticks: number, intents: StampedIntent[]): void {
    const ep = this.requireEpisode();
    for (let i = 0; i < ticks; i++) {
      ep.runner.addTurn({ turnNumber: ep.turn++, intents: i === 0 ? intents : [] });
      const ok = ep.runner.executeNextTick();
      if (!ok || ep.fatal !== undefined) {
        throw new Error(`error del motor en el tick ${ep.game.ticks()}:\n${ep.fatal ?? "executeNextTick devolvió false"}`);
      }
      if (ep.sawWin) break;
    }
  }

  private requireEpisode(): Episode {
    if (this.ep === null) throw new Error("llama a reset() primero");
    return this.ep;
  }
}
