"""Panel de control local (docs/adr/0008-panel-de-control-local.md, docs/specs/dashboard.md).

    python -m openfront_rl.dashboard            # http://127.0.0.1:8765

Escucha solo en 127.0.0.1. Lanza `python -m openfront_rl.train` con argumentos de una lista cerrada y controla los runs
por archivos (`control.json`, `status.json`), sin mandar señales al proceso de entrenamiento.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

from . import runctl

try:  # psutil es opcional para que el servidor arranque sin él; sin él no hay CPU ni "Matar".
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

TRAINER_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent / "web"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
LAUNCH_KEYS = {"config", "name", "seed", "workers", "updates", "resume"}
STALE_S = 180.0  # un run "running" sin actualizar tanto tiempo se muestra como caído si su proceso no existe


class ApiError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


class Panel:
    """Lógica del panel, separada del HTTP para poder probarla."""

    def __init__(self, runs_dir: Path, configs_dir: Path):
        self.runs_dir, self.configs_dir = Path(runs_dir), Path(configs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[int, Any] = {}  # Process de psutil por pid: cpu_percent necesita la llamada anterior para medir
        self._children: dict[int, subprocess.Popen] = {}  # procesos lanzados por este panel (para recogerlos al terminar)

    # --- consultas ---------------------------------------------------------------------------------------------
    def run_dir(self, name: str) -> Path:
        if not NAME_RE.match(name):
            raise ApiError(400, "nombre de run inválido")
        d = self.runs_dir / name
        if not d.is_dir():
            raise ApiError(404, f"el run {name} no existe")
        return d

    def _proc(self, status: dict[str, Any] | None):
        if psutil is None or not status or "pid" not in status:
            return None
        child = self._children.get(int(status["pid"]))
        if child is not None and child.poll() is not None:  # terminó: se recoge para que no quede zombi
            self._children.pop(int(status["pid"]), None)
            return None
        try:
            pid = int(status["pid"])
            p = self._cache.get(pid) or self._cache.setdefault(pid, psutil.Process(pid))
            if p.status() == psutil.STATUS_ZOMBIE:
                return None
            return p if p.is_running() and "python" in p.name().lower() else None
        except (psutil.Error, ValueError):
            return None

    def _usage(self, p) -> dict[str, float]:
        try:
            procs = [p, *p.children(recursive=True)]
            procs = [self._cache.setdefault(q.pid, q) for q in procs]
            return {"cpu": round(sum(q.cpu_percent(interval=None) for q in procs), 1),
                    "mem_mb": round(sum(q.memory_info().rss for q in procs) / 2**20, 0), "procs": len(procs)}
        except psutil.Error:
            return {"cpu": 0.0, "mem_mb": 0.0, "procs": 0}

    def state_of(self, status: dict[str, Any] | None, alive: bool) -> str:
        if status is None:
            return "sin iniciar"
        st = status.get("state", "?")
        if st in ("running", "paused", "stopping") and not alive and psutil is not None:
            return "caído"
        if st in ("running", "paused") and psutil is None and time.time() - status.get("updated_at", 0) > STALE_S:
            return "caído"
        return st

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for d in sorted(self.runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            status = runctl.read_json(d / "status.json")
            p = self._proc(status)
            last = self._last_metric(d)
            out.append({
                "name": d.name, "state": self.state_of(status, p is not None),
                "update": (status or {}).get("update", last.get("update", 0)), "total": (status or {}).get("total"),
                "steps": (status or {}).get("steps", last.get("steps", 0)), "steps_per_s": (status or {}).get("steps_per_s"),
                "ep_return": last.get("ep_return"), "ent": last.get("ent"),
                **(self._usage(p) if p is not None else {}),
            })
        return out

    def _last_metric(self, d: Path) -> dict[str, Any]:
        try:
            with open(d / "metrics.jsonl", "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 8192))
                lines = [x for x in f.read().decode("utf8", "ignore").splitlines() if x.strip()]
            for line in reversed(lines):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
        except OSError:
            pass
        return {}

    def metrics(self, name: str, since: int = 0) -> list[dict[str, Any]]:
        rows = []
        try:
            with open(self.run_dir(name) / "metrics.jsonl", encoding="utf8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("update", 0) > since:
                        rows.append(r)
        except OSError:
            pass
        return rows

    def status(self, name: str) -> dict[str, Any]:
        d = self.run_dir(name)
        st = runctl.read_json(d / "status.json")
        p = self._proc(st)
        return {"status": st, "state": self.state_of(st, p is not None), "usage": self._usage(p) if p is not None else None,
                "control": runctl.read_command(d), "config": yaml.safe_load((d / "config.yaml").read_text(encoding="utf8")) if (d / "config.yaml").exists() else None}

    def architecture(self, name: str) -> dict[str, Any]:
        return runctl.read_json(self.run_dir(name) / "architecture.json") or {}

    def log(self, name: str, tail: int = 200) -> str:
        try:
            return "\n".join((self.run_dir(name) / "train.log").read_text(encoding="utf8", errors="ignore").splitlines()[-tail:])
        except OSError:
            return ""

    def configs(self) -> list[str]:
        return sorted(p.name for p in self.configs_dir.glob("*.yaml") if p.name != "goals.yaml")

    def goals(self, name: str) -> list[dict[str, Any]]:
        spec = yaml.safe_load((self.configs_dir / "goals.yaml").read_text(encoding="utf8")) or {}
        rows = self.metrics(name)
        out = []
        for g in spec.get("goals", []):
            vals = [r[g["metric"]] for r in rows if isinstance(r.get(g["metric"]), (int, float))]
            lower = g["op"] in ("<", "<=")
            best = (min(vals) if lower else max(vals)) if vals else None
            ok = best is not None and {">": best > g["value"], ">=": best >= g["value"], "<": best < g["value"], "<=": best <= g["value"]}[g["op"]]
            out.append({**g, "best": best, "done": bool(ok)})
        return out

    # --- acciones ----------------------------------------------------------------------------------------------
    def launch(self, body: dict[str, Any]) -> dict[str, Any]:
        extra = set(body) - LAUNCH_KEYS
        if extra:
            raise ApiError(400, f"campos no permitidos: {sorted(extra)}")
        cfg_name = str(body.get("config", ""))
        if cfg_name not in self.configs():
            raise ApiError(400, "config debe ser un archivo de configs/")
        cfg = yaml.safe_load((self.configs_dir / cfg_name).read_text(encoding="utf8")) or {}
        args = [sys.executable, "-m", "openfront_rl.train", "--config", str(self.configs_dir / cfg_name)]
        seed = body.get("seed", cfg.get("seed", 0))
        for key, flag in (("seed", "--seed"), ("workers", "--workers"), ("updates", "--updates")):
            if key in body and body[key] is not None:
                if not isinstance(body[key], int) or isinstance(body[key], bool) or body[key] < (0 if key == "seed" else 1):
                    raise ApiError(400, f"{key} debe ser un entero válido")
                args += [flag, str(body[key])]
        name = str(body.get("name") or f"{cfg.get('name', 'run')}-{seed}")
        if not NAME_RE.match(name):
            raise ApiError(400, "nombre de run inválido")
        d = self.runs_dir / name
        if body.get("resume"):
            if not (d / "latest.pt").exists():
                raise ApiError(400, "no hay latest.pt para reanudar")
            args += ["--resume", str(d / "latest.pt")]
        st = runctl.read_json(d / "status.json")
        if self._proc(st) is not None:
            raise ApiError(409, f"el run {name} ya está en ejecución")
        d.mkdir(parents=True, exist_ok=True)
        runctl.write_json(d / "control.json", {"command": "run"})
        args += ["--run-dir", str(d)]
        log = open(d / "train.log", "ab")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(args, cwd=TRAINER_DIR, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags)
        self._children[proc.pid] = proc
        cur = runctl.read_json(d / "status.json") or {}
        if cur.get("pid") != proc.pid:  # aún no escribió el suyo: se deja uno provisional con su pid
            runctl.write_json(d / "status.json", {"pid": proc.pid, "state": "running", "updated_at": time.time(), "update": cur.get("update", 0),
                                                  "total": None, "steps": cur.get("steps", 0), "time": 0.0, "steps_per_s": 0.0, "workers": []})
        return {"name": name, "pid": proc.pid}

    def control(self, name: str, command: str) -> dict[str, str]:
        if command not in ("run", "pause", "stop"):
            raise ApiError(400, "command debe ser run, pause o stop")
        runctl.write_json(self.run_dir(name) / "control.json", {"command": command})
        return {"command": command}

    def kill(self, name: str) -> dict[str, Any]:
        d = self.run_dir(name)
        p = self._proc(runctl.read_json(d / "status.json"))
        if p is None:
            raise ApiError(409, "el run no está en ejecución")
        procs = [p, *p.children(recursive=True)]
        for q in procs:
            try:
                q.kill()
            except psutil.Error:
                pass
        st = runctl.read_json(d / "status.json") or {}
        runctl.write_json(d / "status.json", {**st, "state": "killed", "updated_at": time.time()})
        return {"killed": len(procs)}


def make_handler(panel: Panel):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # silencioso
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode("utf8"), "application/json; charset=utf-8")

        def _route(self, method: str) -> None:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            parts = [x for x in u.path.split("/") if x]
            try:
                if method == "GET" and not parts:
                    return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                if parts[:1] != ["api"]:
                    raise ApiError(404, "no existe")
                body: dict[str, Any] = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ApiError(400, "el cuerpo debe ser un objeto JSON")
                a = parts[1:]
                if method == "GET" and a == ["runs"]:
                    return self._json(panel.list_runs())
                if method == "GET" and a == ["configs"]:
                    return self._json(panel.configs())
                if method == "POST" and a == ["runs"]:
                    return self._json(panel.launch(body), 201)
                if len(a) >= 3 and a[0] == "runs":
                    name, what = a[1], a[2]
                    if method == "GET" and what == "metrics":
                        return self._json(panel.metrics(name, int(q.get("since", 0))))
                    if method == "GET" and what == "status":
                        return self._json(panel.status(name))
                    if method == "GET" and what == "architecture":
                        return self._json(panel.architecture(name))
                    if method == "GET" and what == "goals":
                        panel.run_dir(name)
                        return self._json(panel.goals(name))
                    if method == "GET" and what == "log":
                        return self._json({"log": panel.log(name, int(q.get("tail", 200)))})
                    if method == "POST" and what == "control":
                        return self._json(panel.control(name, str(body.get("command"))))
                    if method == "POST" and what == "kill":
                        return self._json(panel.kill(name))
                raise ApiError(404, "no existe")
            except ApiError as e:
                self._json({"error": e.msg}, e.status)
            except (ValueError, KeyError) as e:
                self._json({"error": f"petición inválida: {e}"}, 400)

        def do_GET(self) -> None:
            self._route("GET")

        def do_POST(self) -> None:
            self._route("POST")

    return Handler


def make_server(panel: Panel, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(panel))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs", default=str(TRAINER_DIR / "runs"))
    a = ap.parse_args(argv)
    srv = make_server(Panel(Path(a.runs), TRAINER_DIR / "configs"), a.port)
    print(f"Panel en http://127.0.0.1:{a.port}  (Ctrl+C para cerrar; los entrenamientos siguen corriendo)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
