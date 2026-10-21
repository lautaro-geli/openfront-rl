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
import { actionToIntents, computeMaskInfo, type MaskInfo } from "./actions";
import { emptyMask } from "./mask";
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
  decisionInterval: 10,
  maxTicks: 6000,
  grid: { h: 32, w: 64 },
  samples: 4,
  maxNeighbors: 8,
  rewardTerritoryWeight: 1.0,
  rewardIncomeWeight: 0,
  rewardSecurityWeight: 0,
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
    ep.agents = start.players.map((p) => {
      const me = game.playerByClientID(p.clientID);
      if (me === null) throw new Error(`el jugador ${p.clientID} no existe en la partida`);
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
      const parts = computeRewardParts(game, a.me, a.rs, cfg, { afterSpawn, wasEliminated, eliminatedNow, won, landTiles });
      a.eliminated = a.eliminated || eliminatedNow;
      reward.push(sumParts(parts));
      terminatedFlags.push(a.eliminated || ep.sawWin);
      info.push({
        invalidAction: invalid[i],
        rewardParts: parts,
        won,
        eliminated: a.eliminated,
        tick: game.ticks(),
      });
    });

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
      return buildObservation(ep.game, a.me, ctx, a.maskInfo);
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
