// REFERENCIA, no es código de producción. Se ejecutó en la sesión de la Fase 0 con otras rutas
// (script fuera del repo, node_modules enlazado a vendor/openfront/node_modules, cwd = vendor/openfront,
// Node 24.22 + npm 12.2). Las rutas de este archivo se ajustaron después para que sean relativas y
// NO se volvió a ejecutar con ellas. El bridge real (Fase 1) resolverá imports y rutas bien.
// PoC Fase 0: un jugador "humano" controlado por código en el motor headless.
import { Difficulty, GameMapSize, GameMapType, GameMode, GameType, PlayerInfo, PlayerType } from "@openfront/engine-api/game/GameTypes";
import { GameConfig, GameStartInfo } from "@openfront/engine-api/Schemas";
import { loadTerrainMap } from "@openfront/engine-lib/game/TerrainMapLoader";
import { PseudoRandom } from "@openfront/engine-lib/PseudoRandom";
import { simpleHash } from "@openfront/engine-lib/Util";
import { EngineConfig } from "@openfront/engine/configuration/EngineConfig";
import { Executor } from "@openfront/engine/execution/ExecutionManager";
import { createGame } from "@openfront/engine/game/GameImpl";
import { createNationsForGame } from "@openfront/engine/game/NationCreation";
import { GameRunner } from "@openfront/engine/GameRunner";
import { loadMapFiles } from "@openfront/shared/GameMapLoader";
import { NodeGameMapLoader } from "../../../vendor/openfront/tests/perf/fullgame/NodeGameMapLoader";

const ME = "rl-agent";
async function main() {
  console.debug = () => {};
  const cfg: GameConfig = { gameMap: GameMapType.World, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
    gameType: GameType.Public, difficulty: Difficulty.Medium, nations: "default", donateGold: false, donateTroops: false,
    bots: 50, infiniteGold: false, infiniteTroops: false, instantBuild: false, randomSpawn: false };
  const start: GameStartInfo = { gameID: "poc-1", lobbyCreatedAt: 0, config: cfg,
    players: [{ clientID: ME, username: "RLAgent", clanTag: null }] } as any;
  const config = new EngineConfig(cfg, false);
  const loader = new NodeGameMapLoader("vendor/openfront/resources/maps");
  const terrain = await loadTerrainMap(await loadMapFiles(loader, cfg.gameMap, cfg.gameMapSize));
  const random = new PseudoRandom(simpleHash(start.gameID));
  const humans = start.players.map((p) => new PlayerInfo(p.username, PlayerType.Human, p.clientID, random.nextID(), false, null, [], null));
  const nations = createNationsForGame(start, terrain.nations, terrain.additionalNations, humans.length, random);
  const game = createGame(humans, nations, terrain.gameMap, terrain.miniGameMap, config, terrain.teamGameSpawnAreas);
  let err: string | undefined;
  const runner = new GameRunner(game, new Executor(game, start.gameID, ME), (gu) => { if ("errMsg" in gu) err = gu.errMsg; });
  runner.init();

  const map = game.map();
  // Elegimos un tile de tierra libre cerca del centro.
  let spawn = -1;
  for (let r = 0; r < 400 && spawn < 0; r += 5)
    for (let dx = -r; dx <= r && spawn < 0; dx += 5) {
      const x = Math.floor(map.width() / 2) + dx, y = Math.floor(map.height() / 2) + r;
      if (map.isValidCoord(x, y) && map.isLand(map.ref(x, y)) && !map.hasOwner(map.ref(x, y))) spawn = map.ref(x, y);
    }
  console.log("spawn tile", spawn, "mapa", map.width(), "x", map.height());

  let turn = 0;
  const me = () => game.playerByClientID(ME)!;
  const step = (intents: any[]) => { runner.addTurn({ turnNumber: turn++, intents }); return runner.executeNextTick(); };
  step([{ type: "spawn", tile: spawn, clientID: ME }]);
  while (game.inSpawnPhase()) step([]);
  console.log("fin spawn en tick", game.ticks(), "| me: vivo?", me().isAlive(), "tiles", me().numTilesOwned(), "tropas", Math.round(me().troops()));
  const t0 = performance.now();
  for (let i = 0; i < 3000 && !err; i++) {
    const intents: any[] = [];
    if (i % 20 === 0 && me().isAlive()) intents.push({ type: "attack", targetID: null, troops: me().troops() * 0.5, clientID: ME });
    step(intents);
    if (i % 500 === 0) console.log(`tick ${game.ticks()} tiles=${me().numTilesOwned()} tropas=${Math.round(me().troops())} oro=${me().gold()} vivos=${game.players().length}`);
  }
  console.log("error:", err ?? "ninguno", "| ticks/s:", Math.round(3000 / ((performance.now() - t0) / 1000)));
}
main();
