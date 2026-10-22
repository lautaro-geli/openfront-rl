"""Exportar un checkpoint a pesos y volver a cargarlo da la misma política (docs/adr/0009)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openfront_rl import NodeGame  # noqa: E402
from openfront_rl.export import export_model, load_policy  # noqa: E402
from openfront_rl.model import Policy, to_tensors  # noqa: E402

CFG = {"agents": 1, "bots": 5, "maxTicks": 600}


def test_exportar_y_cargar_da_la_misma_politica(tmp_path):
    with NodeGame(CFG) as g:
        m = g.manifest
        r = g.reset("exp-1")
        torch.manual_seed(1)
        p = Policy(m)
        ck = tmp_path / "ck.pt"
        torch.save({"model": p.state_dict(), "optimizer": {}, "update": 7, "steps": 99, "episodes_done": 3, "config": {"env": CFG, "model": {}},
                    "protocol_version": m.raw["version"], "torch_version": torch.__version__}, ck)
        pt = export_model(ck, tmp_path / "m" / "demo", note="n", results="r")
        meta = json.loads(pt.with_suffix(".json").read_text(encoding="utf8"))
        assert meta["update"] == 7 and meta["episodes_done"] == 3 and meta["results"] == "r" and meta["protocol_version"] == m.raw["version"]
        q = load_policy(pt, m)
        ob, mk = to_tensors(r.obs, r.mask)
        with torch.no_grad():
            a1, _, _, v1 = p.eval()(ob, mk, greedy=True)
            a2, _, _, v2 = q(ob, mk, greedy=True)
        assert np.array_equal(a1.numpy(), a2.numpy()) and torch.allclose(v1, v2)
        assert "optimizer" not in torch.load(pt, weights_only=False)  # solo pesos


def test_cargar_un_modelo_de_otro_protocolo_falla(tmp_path):
    with NodeGame(CFG) as g:
        m = g.manifest
        torch.save({"model": Policy(m).state_dict(), "protocol_version": m.raw["version"] + 1, "config_model": {}}, tmp_path / "x.pt")
        with pytest.raises(ValueError, match="protocolo"):
            load_policy(tmp_path / "x.pt", m)
