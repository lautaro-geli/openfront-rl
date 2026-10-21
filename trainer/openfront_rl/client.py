"""Cliente del protocolo Node ⇄ Python (docs/specs/protocol.md)."""
from __future__ import annotations

import json
import os
import queue
import shutil
import struct
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import Manifest

REQ_INIT, REQ_RESET, REQ_STEP, REQ_HASH, REQ_CLOSE = 0x01, 0x02, 0x03, 0x04, 0x05
RES_MANIFEST, RES_RESULT, RES_HASH, RES_CLOSED, RES_ERROR = 0x81, 0x82, 0x83, 0x84, 0xFF

DEFAULT_BRIDGE = Path(__file__).resolve().parents[2] / "bridge"


class ProtocolError(RuntimeError):
    """El servidor devolvió un error, murió o no respondió a tiempo."""


@dataclass
class Result:
    """Resultado de `reset` o `step` para todos los agentes de la partida. Los arrays tienen el agente en el eje 0."""

    done: bool
    truncated: bool
    tick: int
    reward: np.ndarray  # [A] float32
    reward_parts: np.ndarray  # [A, P] float32; columnas en m.reward_parts
    terminated: np.ndarray  # [A] bool
    invalid: np.ndarray  # [A] bool
    won: np.ndarray  # [A] bool
    eliminated: np.ndarray  # [A] bool
    obs: dict[str, np.ndarray]  # nombre -> [A, *shape] float32
    mask: dict[str, np.ndarray]  # nombre -> [A, *shape] bool


def decode_result(m: Manifest, body: bytes) -> Result:
    a = body[2]
    if len(body) != m.header_bytes + a * m.agent_bytes:
        raise ProtocolError(f"resultado de {len(body)} bytes; se esperaban {m.header_bytes + a * m.agent_bytes}")
    done, truncated = bool(body[0]), bool(body[1])
    (tick,) = struct.unpack_from("<i", body, 4)
    base = m.header_bytes
    stride = m.agent_bytes
    reward = np.ndarray((a,), np.float32, body, base, (stride,)).copy()
    flags = np.ndarray((a, 4), np.uint8, body, base + 4, (stride, 1))
    parts = np.ndarray((a, len(m.reward_parts)), np.float32, body, base + 8, (stride, 4)).copy()
    h = m.agent_header_bytes
    floats = np.ndarray((a, m.obs_floats), np.float32, body, base + h, (stride, 4))
    masks = np.ndarray((a, m.mask_bytes), np.uint8, body, base + h + 4 * m.obs_floats, (stride, 1))
    obs = {n: floats[:, b.offset : b.offset + b.size].reshape((a, *b.shape)) for n, b in m.obs.items()}
    mask = {n: masks[:, b.offset : b.offset + b.size].reshape((a, *b.shape)).astype(bool) for n, b in m.mask.items()}
    return Result(
        done=done, truncated=truncated, tick=tick, reward=reward, reward_parts=parts,
        terminated=flags[:, 0].astype(bool), invalid=flags[:, 1].astype(bool),
        won=flags[:, 2].astype(bool), eliminated=flags[:, 3].astype(bool), obs=obs, mask=mask,
    )


def find_node() -> str:
    node = os.environ.get("OPENFRONT_NODE") or shutil.which("node")
    if node is None:
        raise ProtocolError("no encuentro Node. Instala Node 24 o define OPENFRONT_NODE con su ruta")
    return node


class NodeGame:
    """Una partida de OpenFront en su propio proceso Node. Se puede usar con `with`."""

    def __init__(self, config: dict[str, Any] | None = None, bridge_dir: str | Path | None = None,
                 node: str | None = None, timeout: float = 120.0):
        self.timeout = timeout
        self.bridge = Path(bridge_dir) if bridge_dir else DEFAULT_BRIDGE
        tsx = self.bridge / "node_modules" / "tsx" / "dist" / "cli.mjs"
        if not tsx.exists():
            raise ProtocolError(f"falta {tsx}: ejecuta `npm ci` dentro de bridge/")
        self._stderr: deque[str] = deque(maxlen=200)
        self._frames: queue.Queue[tuple[int, bytes] | None] = queue.Queue()
        self._proc = subprocess.Popen(
            # `node --import tsx` (sin el envoltorio `tsx`, que lanza otro proceso hijo y deja huérfanos al cerrar).
            [node or find_node(), "--import", "tsx", "src/server.ts"], cwd=self.bridge,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        self.manifest: Manifest | None = None
        t, body = self._request(REQ_INIT, json.dumps(config or {}).encode())
        self._expect(t, body, RES_MANIFEST)
        self.manifest = Manifest.parse(json.loads(body))

    # -- E/S ---------------------------------------------------------------------------------
    def _read_stdout(self) -> None:
        out = self._proc.stdout
        assert out is not None
        try:
            while True:
                head = self._read_exact(out, 4)
                if head is None:
                    break
                (n,) = struct.unpack("<I", head)
                msg = self._read_exact(out, n)
                if msg is None:
                    break
                self._frames.put((msg[0], bytes(msg[1:])))
        finally:
            self._frames.put(None)

    @staticmethod
    def _read_exact(f: Any, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            chunk = f.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return bytes(buf)

    def _read_stderr(self) -> None:
        err = self._proc.stderr
        assert err is not None
        for line in iter(err.readline, b""):
            self._stderr.append(line.decode("utf8", "replace").rstrip())

    def _send(self, kind: int, body: bytes = b"") -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(struct.pack("<IB", len(body) + 1, kind) + body)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ProtocolError(f"el proceso Node no acepta datos: {e}\n{self.stderr_tail()}") from e

    def _recv(self) -> tuple[int, bytes]:
        try:
            frame = self._frames.get(timeout=self.timeout)
        except queue.Empty as e:
            raise ProtocolError(f"sin respuesta en {self.timeout} s\n{self.stderr_tail()}") from e
        if frame is None:
            raise ProtocolError(f"el proceso Node terminó (código {self._proc.poll()})\n{self.stderr_tail()}")
        return frame

    def _request(self, kind: int, body: bytes = b"") -> tuple[int, bytes]:
        self._send(kind, body)
        return self._recv()

    def _expect(self, t: int, body: bytes, want: int) -> None:
        if t == RES_ERROR:
            raise ProtocolError(body.decode("utf8", "replace"))
        if t != want:
            raise ProtocolError(f"respuesta inesperada 0x{t:02x}; se esperaba 0x{want:02x}")

    def stderr_tail(self) -> str:
        return "\n".join(list(self._stderr)[-20:])

    # -- API ---------------------------------------------------------------------------------
    def send_reset(self, seed: str) -> None:
        self._send(REQ_RESET, json.dumps({"seed": seed}).encode())

    def send_step(self, actions: np.ndarray) -> None:
        m = self._require_manifest()
        a = np.ascontiguousarray(actions, dtype="<i4")
        if a.shape != (m.agents, m.action_dims):
            raise ValueError(f"acciones con forma {a.shape}; se esperaba {(m.agents, m.action_dims)}")
        self._send(REQ_STEP, a.tobytes())

    def recv_result(self) -> Result:
        t, body = self._recv()
        self._expect(t, body, RES_RESULT)
        return decode_result(self._require_manifest(), body)

    def reset(self, seed: str) -> Result:
        self.send_reset(seed)
        return self.recv_result()

    def step(self, actions: np.ndarray) -> Result:
        self.send_step(actions)
        return self.recv_result()

    def state_hash(self) -> str:
        t, body = self._request(REQ_HASH)
        self._expect(t, body, RES_HASH)
        return body.decode()

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._send(REQ_CLOSE)
                self._proc.wait(timeout=5)
            except (ProtocolError, subprocess.TimeoutExpired):
                self._proc.kill()
        for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass

    def _require_manifest(self) -> Manifest:
        if self.manifest is None:
            raise ProtocolError("la partida no está inicializada")
        return self.manifest

    def __enter__(self) -> "NodeGame":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
