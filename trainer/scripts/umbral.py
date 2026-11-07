"""Sale con código 0 si la última línea JSON de un archivo de `evalckpt` tiene `eval_won` ≥ umbral, y 1 si no (para .bat).

    python scripts/umbral.py ARCHIVO UMBRAL
"""
from __future__ import annotations

import json
import sys


def won(path: str) -> float:
    last = None
    with open(path, encoding="utf8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("{") and "eval_won" in line:
                last = json.loads(line)
    if last is None:
        raise SystemExit(f"no hay resultados de evalckpt en {path}")
    return float(last["eval_won"])


def main(argv: list[str]) -> int:
    w = won(argv[0])
    ok = w >= float(argv[1])
    print(f"victorias {w:.3f} {'>=' if ok else '<'} {argv[1]}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
