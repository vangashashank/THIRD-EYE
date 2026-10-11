"""Forecaster architectures.

* ``OneStepMLP``       H=1 baseline (the TuneAhead-style comparison).
* ``ThirdEyeDirectV2`` H=2 multi-head forecaster with a *masked* history GRU.
* ``ThirdEyeDynamics`` latent transition model z_{t+1}=F(z_t,u_t), rolled to H=2.

All three consume the same normalised batch (state, cand, hist, mask) and
return a dict with ``h2`` ([B,3], target/ood/retention) and/or ``h1``
([B,n_h1]).  Outputs are in *standardised* label units; ``Standardizer``
converts back.  Every model stays far below the plan's 1M-parameter cap.

Fixes relative to Sprint 1's ``ThirdEyeDirect`` (see the Sprint 2 audit):
padded history steps are skipped exactly (hidden state carried through), so
output does not depend on padded values, padding position or batch mixing;
empty history is handled by the same code path; and dimensions come from the
layout rather than being hard-coded to (3, 4, 3).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .layout import FeatureLayout


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class _Enc(nn.Module):
    """Linear+GELU that tolerates zero input width (ablated group) by emitting zeros."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim) if in_dim > 0 else None

    def forward(self, x):
        if self.lin is None:
            return x.new_zeros(x.shape[0], self.out_dim)
        return torch.nn.functional.gelu(self.lin(x))


class MaskedGRU(nn.Module):
    """GRU over [B,H,C] that skips steps where mask==0 (hidden state is carried).

    Because skipped steps neither read their input nor change the state, the
    result is invariant to the values stored in padded slots, and a fully
    masked history yields the all-zero state.
    """

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.cell = nn.GRUCell(in_dim, hidden)

    def forward(self, x, mask):
        B, H, _ = x.shape
        h = x.new_zeros(B, self.hidden)
        for t in range(H):
            nxt = self.cell(x[:, t], h)
            h = torch.where(mask[:, t:t + 1] > 0.5, nxt, h)
        return h


class _HistoryEncoder(nn.Module):
    def __init__(self, channels: int, hist_len: int, hidden: int):
        super().__init__()
        self.active = channels > 0 and hist_len > 0
        self.hist_len = hist_len
        self.gru = MaskedGRU(channels, hidden) if self.active else None
        self.hidden = hidden

    def forward(self, hist, mask):
        """Returns (summary [B,hidden], fraction of real entries [B,1])."""
        if not self.active:
            B = hist.shape[0]
            return hist.new_zeros(B, self.hidden), hist.new_zeros(B, 1)
        return self.gru(hist, mask), mask.sum(1, keepdim=True) / float(self.hist_len)


class OneStepMLP(nn.Module):
    """2-layer MLP over state + candidate (+ flattened masked history) -> t+1 deltas."""

    def __init__(self, state_dim, cand_dim, hist_len, hist_channels, n_h1,
                 hidden=32, dropout=0.1, use_history=True):
        super().__init__()
        self.config = dict(kind="one_step", state_dim=state_dim, cand_dim=cand_dim,
                           hist_len=hist_len, hist_channels=hist_channels, n_h1=n_h1,
                           hidden=hidden, dropout=dropout, use_history=use_history)
        self.use_history = bool(use_history and hist_len > 0 and hist_channels > 0)
        in_dim = state_dim + cand_dim
        if self.use_history:
            in_dim += hist_len * hist_channels + hist_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_h1))

    def forward(self, state, cand, hist, mask) -> Dict[str, torch.Tensor]:
        parts = [state, cand]
        if self.use_history:
            parts += [(hist * mask.unsqueeze(-1)).flatten(1), mask]
        return {"h1": self.net(torch.cat(parts, dim=-1))}


class ThirdEyeDirectV2(nn.Module):
    """Direct H=2 forecaster: state/candidate/history encoders -> trunk -> heads.

    ``h2`` = [target, ood, retention] change after the candidate plus one
    standardised continuation update.  An optional auxiliary ``h1`` head
    (same labels the Dynamics model uses at z_{t+1}) keeps the E13 comparison
    matched.
    """

    def __init__(self, state_dim, cand_dim, hist_len, hist_channels, n_h1,
                 hidden=32, dropout=0.1, use_h1_aux=True):
        super().__init__()
        self.config = dict(kind="direct", state_dim=state_dim, cand_dim=cand_dim,
                           hist_len=hist_len, hist_channels=hist_channels, n_h1=n_h1,
                           hidden=hidden, dropout=dropout, use_h1_aux=use_h1_aux)
        self.state_enc = _Enc(state_dim, hidden)
        self.cand_enc = _Enc(cand_dim, hidden)
        self.hist_enc = _HistoryEncoder(hist_channels, hist_len, hidden)
        trunk_in = 3 * hidden + 1
        self.trunk = nn.Sequential(nn.LayerNorm(trunk_in), nn.Linear(trunk_in, hidden),
                                   nn.GELU(), nn.Dropout(dropout))
        self.head_h2 = nn.Linear(hidden, 3)
        self.head_h1 = nn.Linear(hidden, n_h1) if (use_h1_aux and n_h1 > 0) else None

    def forward(self, state, cand, hist, mask) -> Dict[str, torch.Tensor]:
        hs, frac = self.hist_enc(hist, mask)
        z = self.trunk(torch.cat([self.state_enc(state), self.cand_enc(cand), hs, frac], -1))
        out = {"h2": self.head_h2(z)}
        if self.head_h1 is not None:
            out["h1"] = self.head_h1(z)
        return out


class ThirdEyeDynamics(nn.Module):
    """Latent self-dynamics: z_t -> z_{t+1} -> z_{t+2}, decoded to consequences.

        z_t      = E(state, history)
        u_t      = U(candidate features)
        z_{t+1}  = z_t     + F(z_t,     u_t)        (residual transition)
        z_{t+2}  = z_{t+1} + F(z_{t+1}, u_std)      (same F, learned standardised
                                                     continuation embedding u_std)
        h1 = D1(z_{t+1}),   h2 = D2(z_{t+2})

    Uses exactly the same inputs and labels as ``ThirdEyeDirectV2``.  It earns a
    place as a headline method only if it wins on ranking, trajectories or
    transfer (Dynamics value gate); otherwise it is reported as an ablation.
    """

    def __init__(self, state_dim, cand_dim, hist_len, hist_channels, n_h1,
                 hidden=32, latent=32, update_dim=16, dropout=0.1):
        super().__init__()
        self.config = dict(kind="dynamics", state_dim=state_dim, cand_dim=cand_dim,
                           hist_len=hist_len, hist_channels=hist_channels, n_h1=n_h1,
                           hidden=hidden, latent=latent, update_dim=update_dim, dropout=dropout)
        self.state_enc = _Enc(state_dim, hidden)
        self.hist_enc = _HistoryEncoder(hist_channels, hist_len, hidden)
        self.encoder = nn.Sequential(nn.Linear(2 * hidden + 1, latent), nn.GELU(),
                                     nn.Linear(latent, latent))
        self.update_enc = _Enc(cand_dim, update_dim)
        self.u_std = nn.Parameter(torch.zeros(update_dim))
        self.transition = nn.Sequential(
            nn.Linear(latent + update_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent))
        self.dec_h1 = nn.Linear(latent, n_h1) if n_h1 > 0 else None
        self.dec_h2 = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def rollout(self, state, cand, hist, mask):
        hs, frac = self.hist_enc(hist, mask)
        z0 = self.encoder(torch.cat([self.state_enc(state), hs, frac], -1))
        u = self.update_enc(cand)
        z1 = z0 + self.transition(torch.cat([z0, u], -1))
        u_std = self.u_std.unsqueeze(0).expand(z1.shape[0], -1)
        z2 = z1 + self.transition(torch.cat([z1, u_std], -1))
        return z0, z1, z2

    def forward(self, state, cand, hist, mask) -> Dict[str, torch.Tensor]:
        z0, z1, z2 = self.rollout(state, cand, hist, mask)
        out = {"h2": self.dec_h2(z2), "z": torch.stack([z0, z1, z2], dim=0)}
        if self.dec_h1 is not None:
            out["h1"] = self.dec_h1(z1)
        return out


KINDS = ("one_step", "direct", "dynamics")


def build_model(kind: str, layout: FeatureLayout, *, hidden: int = 32, dropout: float = 0.1,
                latent: int = 32, update_dim: int = 16, use_h1_aux: bool = True,
                use_history: bool = True) -> nn.Module:
    if kind not in KINDS:
        raise ValueError("kind must be one of %s" % (KINDS,))
    n_h1 = len(layout.h1_heads)
    H = layout.history_len if layout.has_history else 0
    C = len(layout.history_channels) if layout.has_history else 0
    common = dict(state_dim=layout.dim("state"), cand_dim=layout.dim("candidate"),
                  hist_len=H, hist_channels=C, n_h1=n_h1, hidden=hidden, dropout=dropout)
    if kind == "one_step":
        return OneStepMLP(use_history=use_history, **common)
    if kind == "direct":
        return ThirdEyeDirectV2(use_h1_aux=use_h1_aux, **common)
    return ThirdEyeDynamics(latent=latent, update_dim=update_dim, **common)


def build_from_config(config: Dict) -> nn.Module:
    cfg = dict(config)
    kind = cfg.pop("kind")
    if kind == "one_step":
        return OneStepMLP(**cfg)
    if kind == "direct":
        return ThirdEyeDirectV2(**cfg)
    if kind == "dynamics":
        return ThirdEyeDynamics(**cfg)
    raise ValueError("unknown model kind %r" % (kind,))
