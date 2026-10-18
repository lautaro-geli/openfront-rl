import fs from "node:fs";
import path from "node:path";
import { GameMapType } from "@openfront/engine-api/game/GameTypes";
import type { MapManifest } from "@openfront/engine-api/game/MapFiles";
import type { GameMapLoader, MapData } from "@openfront/shared/GameMapLoader";

/**
 * Carga los mapas de producción desde resources/maps/ del motor usando el
 * sistema de archivos. Misma lógica que tests/perf/fullgame/NodeGameMapLoader.ts
 * del motor (que es de tests y no se importa desde aquí).
 */
export class FsMapLoader implements GameMapLoader {
  constructor(private mapsDir: string) {}

  getMapData(map: GameMapType): MapData {
    const key = Object.keys(GameMapType).find(
      (k) => GameMapType[k as keyof typeof GameMapType] === map,
    );
    if (key === undefined) {
      throw new Error(`mapa desconocido: ${map}`);
    }
    const dir = path.join(this.mapsDir, key.toLowerCase());
    const readBin = (name: string) => async () =>
      new Uint8Array(fs.readFileSync(path.join(dir, name)));
    return {
      mapBin: readBin("map.bin"),
      map4xBin: readBin("map4x.bin"),
      map16xBin: readBin("map16x.bin"),
      manifest: async () =>
        JSON.parse(
          fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
        ) as MapManifest,
      webpPath: path.join(dir, "thumbnail.webp"),
      layerImage: async () => {
        throw new Error("layerImage no está soportado en FsMapLoader");
      },
    };
  }
}
