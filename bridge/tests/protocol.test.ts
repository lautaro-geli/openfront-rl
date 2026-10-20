import { type ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { ACTION_DIMS } from "../src/features";
import { MultiAgentEnv } from "../src/multiEnv";
import { buildManifest, encodeResult, REQ, RES, resetResult } from "../src/protocol";
import { vecToAction } from "../src/actions";
import type { EnvConfig } from "../src/types";

const SERVER = path.resolve(__dirname, "../src/server.ts");

/** Cliente mínimo del protocolo, solo para los tests. El cliente real es el de Python. */
class Client {
  private proc: ChildProcessWithoutNullStreams;
  private buf = Buffer.alloc(0);
  private waiters: ((m: { type: number; body: Buffer }) => void)[] = [];
  private inbox: { type: number; body: Buffer }[] = [];
  stderr = "";

  constructor() {
    this.proc = spawn(process.execPath, ["--import", "tsx", SERVER], { cwd: path.resolve(__dirname, "..") });
    this.proc.stderr.on("data", (d: Buffer) => (this.stderr += d.toString()));
    this.proc.stdout.on("data", (d: Buffer) => {
      this.buf = Buffer.concat([this.buf, d]);
      while (this.buf.length >= 4) {
        const len = this.buf.readUInt32LE(0);
        if (this.buf.length < 4 + len) break;
        const msg = { type: this.buf.readUInt8(4), body: Buffer.from(this.buf.subarray(5, 4 + len)) };
        this.buf = this.buf.subarray(4 + len);
        const w = this.waiters.shift();
        if (w) w(msg);
        else this.inbox.push(msg);
      }
    });
  }

  request(type: number, body: Uint8Array = new Uint8Array(0)): Promise<{ type: number; body: Buffer }> {
    const header = Buffer.alloc(5);
    header.writeUInt32LE(body.length + 1, 0);
    header.writeUInt8(type, 4);
    this.proc.stdin.write(Buffer.concat([header, Buffer.from(body)]));
    const ready = this.inbox.shift();
    if (ready) return Promise.resolve(ready);
    return new Promise((resolve) => this.waiters.push(resolve));
  }

  json(type: number, obj: unknown) {
    return this.request(type, Buffer.from(JSON.stringify(obj)));
  }

  step(actions: number[][]) {
    return this.request(REQ.step, Buffer.from(new Int32Array(actions.flat()).buffer));
  }

  kill(): void {
    this.proc.kill();
  }
}

const CFG: Partial<EnvConfig> = { agents: 2, bots: 5, maxTicks: 1200 };

describe("protocolo", () => {
  it("una partida por el protocolo da los mismos bytes que la API directa", async () => {
    const c = new Client();
    try {
      const m = await c.json(REQ.init, CFG);
      expect(m.type).toBe(RES.manifest);
      const manifest = JSON.parse(m.body.toString("utf8"));
      expect(manifest.version).toBe(1);
      expect(manifest.agents).toBe(2);
      expect(manifest.actionDims).toBe(ACTION_DIMS);

      const env = await MultiAgentEnv.create(CFG);
      const direct = buildManifest(env.cfg);
      expect(JSON.stringify(manifest)).toBe(JSON.stringify(direct));

      const r0 = await c.json(REQ.reset, { seed: "proto-1" });
      expect(r0.type).toBe(RES.result);
      const obs0 = await env.reset("proto-1");
      expect(Buffer.compare(r0.body, Buffer.from(encodeResult(env.cfg, resetResult(obs0), obs0[0].tick)))).toBe(0);

      // Spawn en celdas con tierra y luego expandir.
      const cells = env.cfg.grid.h * env.cfg.grid.w;
      const land: number[] = [];
      for (let k = 0; k < cells; k++) if (obs0[0].grid[k] >= 0.95) land.push(k);
      const s = Math.floor(land.length / 3);
      const plan: number[][][] = [[[1, 0, land[s], 0, 0, 0], [1, 0, land[2 * s], 0, 0, 0]]];
      for (let i = 0; i < 12; i++) plan.push([[2, 0, 0, 2, 0, 0], [i % 2 === 0 ? 2 : 0, 0, 0, 2, 0, 0]]);

      for (const vecs of plan) {
        const rs = await c.step(vecs);
        expect(rs.type).toBe(RES.result);
        const rd = env.step(vecs.map((v) => vecToAction(v, env.cfg.grid)));
        expect(Buffer.compare(rs.body, Buffer.from(encodeResult(env.cfg, rd, rd.info[0].tick)))).toBe(0);
      }
      const h = await c.request(REQ.hash);
      expect(h.body.toString("utf8")).toBe(env.stateHash());
    } finally {
      c.kill();
    }
  }, 120000);

  it("los errores no matan el proceso", async () => {
    const c = new Client();
    try {
      const early = await c.step([[0, 0, 0, 0, 0, 0]]);
      expect(early.type).toBe(RES.error);
      await c.json(REQ.init, { agents: 1, bots: 3, maxTicks: 600 });
      await c.json(REQ.reset, { seed: "proto-err" });
      const wrong = await c.step([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]); // 2 vectores para 1 agente
      expect(wrong.type).toBe(RES.error);
      expect(wrong.body.toString("utf8")).toContain("step espera");
      const unknown = await c.request(0x7e);
      expect(unknown.type).toBe(RES.error);
      // Sigue vivo: un tipo de acción desconocido cuenta como inválido y el resultado llega.
      const bad = await c.step([[99, 0, 0, 0, 0, 0]]);
      expect(bad.type).toBe(RES.result);
      expect(bad.body.readUInt8(8 + 5)).toBe(1); // bandera invalidAction del agente 0
      const closed = await c.request(REQ.close);
      expect(closed.type).toBe(RES.closed);
    } finally {
      c.kill();
    }
  }, 120000);
});
