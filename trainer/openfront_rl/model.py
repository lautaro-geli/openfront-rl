"""Red por ramas y política autorregresiva (docs/adr/0007-red-por-ramas-y-ppo.md).

Ramas: mapa (CNN) · economía · defensa/ataque · comparación → núcleo → `h`.
Política: primero `kind`; luego, según el tipo elegido, solo los campos que usa (fila, tipo, celda, fracción, emoji),
cada uno con la máscara que corresponde a ese tipo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .manifest import Manifest

NEG = -1e9
OBS_KEYS = (
    "grid", "globals", "econ_table", "econ_valid", "econ_scalars", "mil_units", "mil_units_valid",
    "mil_attacks", "mil_attacks_valid", "mil_scalars", "players", "players_valid", "cmp_scalars",
)
MASK_KEYS = (
    "spawn", "expand", "attack_slots", "kind", "player", "unit", "attack_row", "build_type", "nuke_type", "cell",
)


@dataclass(frozen=True)
class ModelConfig:
    core: int = 512
    map_out: int = 256
    branch_out: int = 128
    key_dim: int = 32
    act_dim: int = 256
    kind_emb: int = 32

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "ModelConfig":
        return ModelConfig(**(d or {}))


class ActionSpec:
    """Qué campos usa cada tipo de acción y qué bloque de la máscara le corresponde. Sale del manifiesto."""

    FIELDS = ("prow", "urow", "arow", "btype", "ntype", "cell", "frac", "aux")

    def __init__(self, space: dict[str, Any]):
        kinds: list[str] = space["kinds"]
        self.kinds = kinds
        self.n_kinds = len(kinds)
        pk, uk, ck = space["playerKinds"], space["unitKinds"], space["cellKinds"]
        self.n_fractions = len(space["fractions"])
        self.n_emojis = space["emojis"]
        self.n_build = len(space["buildTypes"])
        self.n_nuke = len(space["nukeKinds"])
        self.ship_type = space["buildTypes"].index("warship")
        self.cells_kinds = len(ck)
        K = self.n_kinds
        used = {f: [False] * K for f in self.FIELDS}
        player_idx, unit_idx, cell_idx = [-1] * K, [-1] * K, [-1] * K
        for i, name in enumerate(kinds):
            if name in pk:
                player_idx[i] = pk.index(name)
                used["prow"][i] = True
            if name in uk:
                unit_idx[i] = uk.index(name)
                used["urow"][i] = True
            if name in ("spawn", "boat_attack", "move_warship", "nuke"):
                cell_idx[i] = ck.index(name)
                used["cell"][i] = True
            used["frac"][i] = name in ("expand", "attack", "boat_attack", "donate_gold", "donate_troops")
            used["aux"][i] = name == "emoji"
        b = kinds.index("build")
        cell_idx[b] = ck.index("build")
        used["cell"][b] = used["btype"][b] = True
        used["ntype"][kinds.index("nuke")] = True
        used["arow"][kinds.index("cancel_attack")] = True
        self.build_kind, self.nuke_kind = b, kinds.index("nuke")
        self.cell_build, self.cell_build_ship = ck.index("build"), ck.index("build_ship")
        self.used = {f: torch.tensor(v) for f, v in used.items()}
        self.player_idx = torch.tensor(player_idx)
        self.unit_idx = torch.tensor(unit_idx)
        self.cell_idx = torch.tensor(cell_idx)

    def to(self, device: torch.device) -> "ActionSpec":
        self.used = {f: v.to(device) for f, v in self.used.items()}
        self.player_idx, self.unit_idx, self.cell_idx = (t.to(device) for t in (self.player_idx, self.unit_idx, self.cell_idx))
        return self


def _masked_mean_max(x: Tensor, valid: Tensor) -> Tensor:
    """x: [B,R,D], valid: [B,R] (0/1). Media y máximo sobre las filas válidas; 0 si no hay ninguna."""
    v = valid.unsqueeze(-1)
    n = v.sum(1).clamp(min=1.0)
    mean = (x * v).sum(1) / n
    mx = x.masked_fill(v == 0, -1e4).max(1).values
    mx = torch.where(valid.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], -1)


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o), nn.GELU())


class Encoder(nn.Module):
    """Las cuatro ramas y el núcleo. Devuelve `h` y las representaciones por fila/celda que usan las cabezas."""

    def __init__(self, m: Manifest, cfg: ModelConfig):
        super().__init__()
        o = m.obs
        self.cfg = cfg
        c, gh, gw = o["grid"].shape
        self.gh, self.gw = gh, gw
        # Rama mapa: dos conv a resolución completa (alimentan las cabezas de celda), luego dos con stride 2.
        self.conv1 = nn.Conv2d(c, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 48, 3, padding=1)
        self.conv3 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        flat = 64 * ((gh + 3) // 4) * ((gw + 3) // 4)
        self.map_fc = nn.Sequential(nn.Linear(flat, cfg.map_out), nn.LayerNorm(cfg.map_out), nn.GELU())
        # Rama economía.
        ne, ce = o["econ_table"].shape
        self.econ_row = _mlp(ce, 32, 32)
        self.econ_out = nn.Sequential(nn.Linear(64 + o["econ_scalars"].shape[0], cfg.branch_out), nn.GELU())
        # Rama defensa/ataque.
        nm, cm = o["mil_units"].shape
        na, ca = o["mil_attacks"].shape
        self.mil_row = _mlp(cm, 64, 64)
        self.att_row = _mlp(ca, 32, 32)
        self.mil_out = nn.Sequential(nn.Linear(128 + 64 + o["mil_scalars"].shape[0], cfg.branch_out), nn.GELU())
        # Rama comparación.
        npl, cp = o["players"].shape
        self.n_players = npl
        self.cmp_row = _mlp(cp, 64, 64)
        self.cmp_out = nn.Sequential(nn.Linear(npl * 64 + o["cmp_scalars"].shape[0], cfg.branch_out), nn.GELU())
        # Núcleo.
        n_glob = o["globals"].shape[0]
        self.core = nn.Sequential(
            nn.Linear(cfg.map_out + 3 * cfg.branch_out + n_glob, cfg.core), nn.GELU(),
            nn.Linear(cfg.core, cfg.core), nn.LayerNorm(cfg.core), nn.GELU(),
        )
        # Proyecciones a claves de atención (una por tipo de objetivo).
        k = cfg.key_dim
        self.key_cell = nn.Conv2d(48, k, 1)
        self.bias_cell = nn.Conv2d(48, len(m.space["cellKinds"]), 1)
        self.key_player = nn.Linear(64, k)
        self.key_econ = nn.Linear(32, k)
        self.key_mil = nn.Linear(64, k)
        self.key_att = nn.Linear(32, k)

    def forward(self, ob: dict[str, Tensor]) -> dict[str, Tensor]:
        x = F.gelu(self.conv1(ob["grid"]))
        f1 = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(f1))
        x = F.gelu(self.conv4(x))
        map_h = self.map_fc(x.flatten(1))

        er = self.econ_row(ob["econ_table"])
        econ_h = self.econ_out(torch.cat([_masked_mean_max(er, ob["econ_valid"]), ob["econ_scalars"]], -1))

        mr = self.mil_row(ob["mil_units"])
        ar = self.att_row(ob["mil_attacks"])
        mil_h = self.mil_out(torch.cat([
            _masked_mean_max(mr, ob["mil_units_valid"]), _masked_mean_max(ar, ob["mil_attacks_valid"]), ob["mil_scalars"],
        ], -1))

        pr = self.cmp_row(ob["players"]) * ob["players_valid"].unsqueeze(-1)
        cmp_h = self.cmp_out(torch.cat([pr.flatten(1), ob["cmp_scalars"]], -1))

        h = self.core(torch.cat([map_h, econ_h, mil_h, cmp_h, ob["globals"]], -1))
        B = h.shape[0]
        return {
            "h": h,
            "cell_keys": self.key_cell(f1).flatten(2),  # [B,k,cells]
            "cell_bias": self.bias_cell(f1).flatten(2),  # [B,cell_kinds,cells]
            "player_keys": self.key_player(pr),  # [B,16,k]
            "unit_keys": torch.cat([self.key_econ(er), self.key_mil(mr)], 1),  # [B,144,k]
            "attack_keys": self.key_att(ar[:, :8]),  # [B,8,k]
        }


def _field(logits: Tensor, mask: Tensor, used: Tensor, given: Tensor | None, greedy: bool) -> tuple[Tensor, Tensor, Tensor]:
    """Una distribución enmascarada. Devuelve (valor, log-prob, entropía); los campos no usados valen 0."""
    use = used & mask.any(-1)
    m = torch.where(use.unsqueeze(-1), mask, torch.ones_like(mask))
    lg = logits.masked_fill(~m, NEG)
    lp = F.log_softmax(lg, -1)
    if given is not None:
        a = given.long().clamp(0, logits.shape[-1] - 1)  # un campo no usado puede traer un valor de otro (p. ej. `row`)
    elif greedy:
        a = lp.argmax(-1)
    else:
        a = torch.multinomial(lp.exp(), 1)[:, 0]
    logp = lp.gather(1, a.unsqueeze(1))[:, 0]
    p = lp.exp()
    ent = -(p * lp.masked_fill(~m, 0.0)).sum(-1)
    z = torch.zeros_like(logp)
    return torch.where(use, a, torch.zeros_like(a)), torch.where(use, logp, z), torch.where(use, ent, z)


class Policy(nn.Module):
    """Red completa: codificador + cabezas autorregresivas + valor."""

    def __init__(self, m: Manifest, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.spec = ActionSpec(m.space)
        c = self.cfg
        self.enc = Encoder(m, c)
        self.kind_head = nn.Linear(c.core, self.spec.n_kinds)
        self.kind_emb = nn.Embedding(self.spec.n_kinds, c.kind_emb)
        self.act = nn.Sequential(nn.Linear(c.core + c.kind_emb, c.act_dim), nn.GELU(), nn.Linear(c.act_dim, c.act_dim), nn.GELU())
        k = c.key_dim
        self.q_player, self.q_unit, self.q_att, self.q_cell = (nn.Linear(c.act_dim, k) for _ in range(4))
        self.type_emb = nn.Embedding(self.spec.n_build + self.spec.n_nuke, c.act_dim)
        self.frac_head = nn.Linear(c.act_dim, self.spec.n_fractions)
        self.btype_head = nn.Linear(c.act_dim, self.spec.n_build)
        self.ntype_head = nn.Linear(c.act_dim, self.spec.n_nuke)
        self.aux_head = nn.Linear(c.act_dim, self.spec.n_emojis)
        self.value = nn.Sequential(nn.Linear(c.core, 256), nn.GELU(), nn.Linear(256, 1))
        self.scale = k ** -0.5
        nn.init.orthogonal_(self.kind_head.weight, 0.01)
        nn.init.zeros_(self.kind_head.bias)
        nn.init.orthogonal_(self.value[-1].weight, 1.0)

    # -- utilidades -----------------------------------------------------------------------
    def _sel(self, mask: Tensor, idx: Tensor) -> Tensor:
        """mask [B,G,N]; idx [B] (−1 = ninguno) → [B,N]."""
        return mask[torch.arange(mask.shape[0], device=mask.device), idx.clamp(min=0)]

    def values(self, ob: dict[str, Tensor]) -> Tensor:
        return self.value(self.enc(ob)["h"])[:, 0]

    def forward(self, ob: dict[str, Tensor], mask: dict[str, Tensor], action: Tensor | None = None,
                greedy: bool = False) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Devuelve (acción [B,6], log-prob [B], entropía [B], valor [B]).

        Con `action` no muestrea: calcula la log-prob de esa acción (se usa en la actualización de PPO).
        """
        sp = self.spec
        e = self.enc(ob)
        h = e["h"]
        B = h.shape[0]
        dev = h.device
        if sp.used["prow"].device != dev:
            sp.to(dev)
        g_ = lambda i: None if action is None else action[:, i]  # noqa: E731

        kind, lp_k, en_k = _field(self.kind_head(h), mask["kind"], torch.ones(B, dtype=torch.bool, device=dev), g_(0), greedy)
        g = self.act(torch.cat([h, self.kind_emb(kind)], -1))
        u = {f: sp.used[f][kind] for f in sp.FIELDS}

        # Filas: jugador / unidad / ataque saliente.
        lg = torch.einsum("bk,bnk->bn", self.q_player(g), e["player_keys"]) * self.scale
        r_p, lp_p, en_p = _field(lg, self._sel(mask["player"], sp.player_idx[kind]), u["prow"], g_(1), greedy)
        lg = torch.einsum("bk,bnk->bn", self.q_unit(g), e["unit_keys"]) * self.scale
        r_u, lp_u, en_u = _field(lg, self._sel(mask["unit"], sp.unit_idx[kind]), u["urow"], g_(1), greedy)
        lg = torch.einsum("bk,bnk->bn", self.q_att(g), e["attack_keys"]) * self.scale
        r_a, lp_a, en_a = _field(lg, mask["attack_row"], u["arow"], g_(1), greedy)
        row = r_p + r_u + r_a  # a lo sumo uno está activo

        # Tipo de estructura / de bomba (ocupan el mismo campo `type`).
        t_b, lp_b, en_b = _field(self.btype_head(g), mask["build_type"], u["btype"], g_(4), greedy)
        t_n, lp_n, en_n = _field(self.ntype_head(g), mask["nuke_type"], u["ntype"], g_(4), greedy)
        typ = t_b + t_n

        # Celda, condicionada al tipo elegido.
        is_b, is_n = u["btype"], u["ntype"]
        tidx = torch.where(is_b, t_b, torch.where(is_n, sp.n_build + t_n, torch.zeros_like(t_b)))
        gc = g + self.type_emb(tidx) * (is_b | is_n).unsqueeze(-1)
        ck = sp.cell_idx[kind]
        ck = torch.where(is_b & (t_b == sp.ship_type), torch.full_like(ck, sp.cell_build_ship), ck)
        lg = torch.einsum("bk,bkc->bc", self.q_cell(gc), e["cell_keys"]) * self.scale
        lg = lg + e["cell_bias"][torch.arange(B, device=dev), ck.clamp(min=0)]
        cell, lp_c, en_c = _field(lg, self._sel(mask["cell"], ck), u["cell"], g_(2), greedy)

        frac, lp_f, en_f = _field(self.frac_head(g), torch.ones(B, sp.n_fractions, dtype=torch.bool, device=dev), u["frac"], g_(3), greedy)
        aux, lp_x, en_x = _field(self.aux_head(g), torch.ones(B, sp.n_emojis, dtype=torch.bool, device=dev), u["aux"], g_(5), greedy)

        act = torch.stack([kind, row, cell, frac, typ, aux], -1)
        logp = lp_k + lp_p + lp_u + lp_a + lp_b + lp_n + lp_c + lp_f + lp_x
        ent = en_k + en_p + en_u + en_a + en_b + en_n + en_c + en_f + en_x
        return act, logp, ent, self.value(h)[:, 0]


def to_tensors(obs: dict[str, Any], mask: dict[str, Any], device: torch.device | str = "cpu") -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Arreglos numpy [B,...] → tensores. Las máscaras pasan a bool."""
    ob = {k: torch.as_tensor(np.array(obs[k])).to(device=device, dtype=torch.float32) for k in OBS_KEYS}
    mk = {k: torch.as_tensor(np.array(mask[k])).to(device=device, dtype=torch.bool) for k in MASK_KEYS}
    return ob, mk
