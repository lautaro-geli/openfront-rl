// Servidor del protocolo Node ⇄ Python (docs/specs/protocol.md). Una partida por proceso.
// stdout solo lleva mensajes del protocolo: todo lo que el motor imprima se desvía a stderr.
console.log = console.info = console.debug = console.warn = (...args: unknown[]) => console.error(...args);

const { MultiAgentEnv } = await import("./multiEnv");
const { vecToAction } = await import("./actions");
const { ACTION_DIMS } = await import("./features");
const { REQ, RES, buildManifest, encodeResult, resetResult } = await import("./protocol");
import type { MultiAgentEnv as Env } from "./multiEnv";

let env: Env | null = null;
// Tiempos acumulados por tipo de trabajo (se imprimen a stderr al cerrar si OPENFRONT_TIMINGS=1).
const t = { steps: 0, engine: 0, encode: 0, bytes: 0 };
const out = process.stdout;

function send(type: number, body: Uint8Array = new Uint8Array(0)): void {
  const header = Buffer.alloc(5);
  header.writeUInt32LE(body.length + 1, 0);
  header.writeUInt8(type, 4);
  out.write(header);
  if (body.length > 0) out.write(body);
}
const sendText = (type: number, text: string): void => send(type, Buffer.from(text, "utf8"));
const sendError = (e: unknown): void => sendText(RES.error, e instanceof Error ? `${e.message}\n${e.stack ?? ""}` : String(e));

async function handle(type: number, body: Buffer): Promise<void> {
  switch (type) {
    case REQ.init: {
      const cfg = body.length > 0 ? JSON.parse(body.toString("utf8")) : {};
      env = await MultiAgentEnv.create(cfg);
      sendText(RES.manifest, JSON.stringify(buildManifest(env.cfg)));
      return;
    }
    case REQ.reset: {
      if (env === null) throw new Error("envía init antes de reset");
      const { seed } = JSON.parse(body.toString("utf8")) as { seed: string };
      const obs = await env.reset(String(seed));
      send(RES.result, encodeResult(env.cfg, resetResult(obs), obs[0]?.tick ?? 0));
      return;
    }
    case REQ.step: {
      if (env === null) throw new Error("envía init y reset antes de step");
      const A = env.numAgents;
      if (body.length !== A * ACTION_DIMS * 4) {
        throw new Error(`step espera ${A * ACTION_DIMS * 4} bytes (${A} agentes × ${ACTION_DIMS} enteros) y llegaron ${body.length}`);
      }
      const v = new Int32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.length));
      const actions = Array.from({ length: A }, (_, i) => vecToAction(v.subarray(i * ACTION_DIMS, (i + 1) * ACTION_DIMS), env!.cfg.grid));
      const t0 = performance.now();
      const r = env.step(actions);
      const t1 = performance.now();
      const bytes = encodeResult(env.cfg, r, r.info[0]?.tick ?? 0);
      const t2 = performance.now();
      send(RES.result, bytes);
      t.steps++;
      t.engine += t1 - t0;
      t.encode += t2 - t1;
      t.bytes = bytes.length;
      return;
    }
    case REQ.hash:
      if (env === null) throw new Error("no hay partida");
      sendText(RES.hash, env.stateHash());
      return;
    case REQ.close:
      if (process.env.OPENFRONT_TIMINGS === "1" && t.steps > 0) {
        console.error(
          `timings: steps=${t.steps} engine_obs_ms=${(t.engine / t.steps).toFixed(2)} encode_ms=${(t.encode / t.steps).toFixed(3)} bytes=${t.bytes}`,
        );
      }
      env?.close();
      send(RES.closed);
      setImmediate(() => process.exit(0));
      return;
    default:
      throw new Error(`tipo de petición desconocido: 0x${type.toString(16)}`);
  }
}

// Las peticiones se procesan de una en una, en orden.
let pending = Buffer.alloc(0);
const queue: { type: number; body: Buffer }[] = [];
let running = false;

async function pump(): Promise<void> {
  if (running) return;
  running = true;
  while (queue.length > 0) {
    const { type, body } = queue.shift()!;
    try {
      await handle(type, body);
    } catch (e) {
      sendError(e);
    }
  }
  running = false;
}

process.stdin.on("data", (chunk: Buffer) => {
  pending = Buffer.concat([pending, chunk]);
  while (pending.length >= 4) {
    const len = pending.readUInt32LE(0);
    if (pending.length < 4 + len) break;
    const msg = pending.subarray(4, 4 + len);
    queue.push({ type: msg.readUInt8(0), body: Buffer.from(msg.subarray(1)) });
    pending = pending.subarray(4 + len);
  }
  void pump();
});
process.stdin.on("end", () => process.exit(0));
