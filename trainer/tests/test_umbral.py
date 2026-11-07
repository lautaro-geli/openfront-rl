import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import umbral


def test_umbral(tmp_path):
    f = tmp_path / "base.txt"
    f.write_text('=== linea base ===\ndispositivo=cuda\n{"eval_won": 0.45, "eval_episodes": 40.0}\n', encoding="utf8")
    assert umbral.main([str(f), "0.4"]) == 0
    assert umbral.main([str(f), "0.5"]) == 1
