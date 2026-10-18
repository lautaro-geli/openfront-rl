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
import { createGame } from "@openfront/engine/game/GameImpl";
import { createNationsForGame } from "@openfront/engine/game/NationCreation";
import { GameRunner } from "@openfront/engine/GameRunner";
import { loadMapFiles } from "@openfront/shared/GameMapLoader";
import { actionToIntents, computeMask } from "./actions";
import { FsMapLoader } from "./mapLoader";
import { buildObservation, buildSamplePlan, type ObsContext } from "./observation";
import type { Action, EnvConfig, Observation, StepResult } from "./types";

export const AGENT_CLIENT_ID = "rl-agent";
const MAX_SPAWN_TICKS = 1000;
const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MAPS_DIR = path.resolve(HERE, "../../vendor/openfront/resources/maps");

export const DEFAULT_CONFIG: EnvConfig = {
  map: GameMapType.Britannia,
  bots: 20,
  nations: "default",
  difficulty: Difficulty.Medium,
  decisionInterval: 10,
  maxTicks: 6000,
  grid: { h: 32, w: 64 },
  maxNeighbors: 8,
  rewardTerritoryWeight: 1.0,
};

interface Episode {
  game: Game;
  runner: GameRunner;
  me: Player;
  plan: Int32Array;
  turn: number;
  lastHash: HashUpdate | undefined;
  sawWin: boolean;
  fatal: string | undefined;
  initialPlayers: number;
  spawnEndTick: number | null;
  prevFrac: number;
  done: boolean;
}

export class OpenFrontEnv {
  readonly cfg: EnvConfig;
  private mapsDir: string;
  private ep: Episode | null = null;

  private constructor(cfg: EnvConfig, mapsDir: string) {
    this.cfg = cfg;
    this.mapsDir = mapsDir;
  }

  static async create(cfg: Partial<EnvConfig> = {}, mapsDir: string = DEFAULT_MAPS_DIR): Promise<OpenFrontEnv> {
    const full: EnvConfig = { ...DEFAULT_CONFIG, ...cfg, grid: { ...DEFAULT_CONFIG.grid, ...cfg.grid } };
    if (full.decisionInterval < 1) throw new Error("decisionInterval debe ser >= 1");
    if (full.maxTicks < 1) throw new Error("maxTicks debe ser >= 1");
    return new OpenFrontEnv(full, mapsDir);
  }

  async reset(seed: string): Promise<Observation> {
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
      players: [{ clientID: AGENT_CLIENT_ID, username: "RLAgent", clanTag: null }],
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
      initialPlayers: 1,
      spawnEndTick: null,
      prevFrac: 0,
      done: false,
    } as unknown as Episode;
    ep.runner = new GameRunner(game, new Executor(game, start.gameID, AGENT_CLIENT_ID), (gu) => {
      if ("errMsg" in gu) {
        ep.fatal = `${gu.errMsg}\n${gu.stack ?? ""}`;
        return;
      }
      const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[];
      if (hashes.length > 0) ep.lastHash = hashes[hashes.length - 1];
      if ((gu.updates[GameUpdateType.Win] ?? []).length > 0) ep.sawWin = true;
    });
    ep.runner.init();
    const me = game.playerByClientID(AGENT_CLIENT_ID);
    if (me === null) throw new Error("el jugador agente no existe en la partida");
    ep.me = me;
    ep.plan = buildSamplePlan(game, cfg.grid.h, cfg.grid.w);
    this.ep = ep;
    return this.observe();
  }

  step(action: Action): StepResult {
    const ep = this.requireEpisode();
    if (ep.done) throw new Error("el episodio terminó; llama a reset()");
    const { game, me } = ep;
    const cfg = this.cfg;

    const mask = computeMask(game, me, cfg.maxNeighbors);
    const built = actionToIntents(action, game, me, AGENT_CLIENT_ID, mask, cfg.grid, cfg.maxNeighbors);

    const wasSpawn = game.inSpawnPhase();
    this.advance(cfg.decisionInterval, built.intents);
    if (wasSpawn && action.kind === "spawn" && !built.invalid) {
      // Tras elegir sitio, el resto de la fase de spawn se avanza automáticamente.
      let guard = 0;
      while (game.inSpawnPhase()) {
        if (++guard > MAX_SPAWN_TICKS) throw new Error("la fase de spawn no termina");
        this.advance(1, []);
      }
    }
    if (ep.spawnEndTick === null && !game.inSpawnPhase()) {
      ep.spawnEndTick = game.ticks();
      ep.initialPlayers = Math.max(1, game.players().length);
    }

    const frac = me.numTilesOwned() / Math.max(1, game.numLandTiles());
    const territory = ep.spawnEndTick === null ? 0 : cfg.rewardTerritoryWeight * (frac - ep.prevFrac);
    ep.prevFrac = ep.spawnEndTick === null ? 0 : frac;

    const afterSpawn = ep.spawnEndTick !== null;
    const eliminated = afterSpawn && !me.isAlive();
    const won = ep.sawWin && game.getWinner() === me;
    const terminated = eliminated || ep.sawWin;
    const ticksSince = afterSpawn ? game.ticks() - (ep.spawnEndTick as number) : 0;
    const truncated = !terminated && afterSpawn && ticksSince >= cfg.maxTicks;
    const terminal = won ? 1 : eliminated ? -1 : 0;
    ep.done = terminated || truncated;

    return {
      obs: this.observe(),
      reward: territory + terminal,
      terminated,
      truncated,
      info: {
        invalidAction: built.invalid,
        rewardParts: { territory, terminal },
        won,
        eliminated,
        tick: game.ticks(),
      },
    };
  }

  /** Hash del motor (último HashUpdate) junto con el tick. Sirve para comprobar determinismo. */
  stateHash(): string {
    const ep = this.requireEpisode();
    return `${ep.lastHash?.hash ?? "none"}@${ep.lastHash?.tick ?? -1}/t${ep.game.ticks()}`;
  }

  close(): void {
    this.ep = null;
  }

  private observe(): Observation {
    const ep = this.requireEpisode();
    const ctx: ObsContext = {
      cfg: this.cfg,
      plan: ep.plan,
      initialPlayers: ep.initialPlayers,
      ticksSinceSpawnEnd: ep.spawnEndTick === null ? 0 : ep.game.ticks() - ep.spawnEndTick,
    };
    const mask = computeMask(ep.game, ep.me, this.cfg.maxNeighbors);
    return buildObservation(ep.game, ep.me, ctx, mask);
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
