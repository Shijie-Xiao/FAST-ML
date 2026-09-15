#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-stream CNN that diagnoses the FAST ventilation terms chi and S.

This module contains the *inference* definition of the released model. The
architecture is split into two streams that mirror the physical decomposition
of ventilation:

* **S stream** -- from the wind and geopotential fields (U, V, Z). S measures
  the shear-driven ventilation efficiency, so it is a purely dynamical
  quantity and the thermodynamic fields are deliberately withheld from it.
* **chi stream** -- from the thermodynamic fields (T, Q, SST, MSLP) plus the
  FAST scalars. chi is the non-dimensional entropy deficit of the air being
  ventilated into the core.

Both are instantaneous diagnostics rather than prognostic variables, so they
are predicted per timestep with no recurrence; this keeps the model from
smuggling in a memory of the observed intensity and forces the temporal
evolution to come from the physics in :mod:`fastml.fast_physics`.

The state dict shipped in ``ckpt/twostream_final_d2.pth`` matches
``TwoStreamFASTModel()`` with its default hyper-parameters exactly (237
tensors, strict load). Training code is not part of this release.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .config import CNN_CHUNK


def _chunk_enc(encoder, x):
    """Apply an encoder in chunks, to bound peak GPU memory.

    A single storm expands to ``T`` (up to 480) independent 72x72x7 samples,
    which is too much to push through the 3-D convolutions at once.
    """
    if x.shape[0] <= CNN_CHUNK:
        return encoder(x)
    return torch.cat([encoder(c) for c in x.split(CNN_CHUNK, 0)], 0)


class Enc3D(nn.Module):
    """Encode one 3-D field ``[N, 1, L, H, W]`` to a feature vector ``[N, out_dim]``.

    Ventilation depends on both the bulk state and its extremes -- a single dry
    intrusion in one quadrant matters more than the annulus mean -- so the
    pooled representation keeps four complementary statistics: average, maximum,
    a high-tail term ``max - avg``, and a low-tail term ``avg - min``. A channel
    attention gate rescales features residually within roughly [0.5, 1.5], which
    lets the network emphasise structure without suppressing weak signals.
    ``LayerNorm`` on the output keeps feature magnitudes bounded so the
    downstream ODE cannot be driven into saturation.

    Args:
        out_dim: Width of the output feature vector.
        use_vert_diff: Add an explicit mid-minus-low-level difference branch.
            Enabled for T and Q, where the vertical structure of the entropy
            deficit is what distinguishes a ventilating layer from a moist one.
    """

    def __init__(self, out_dim=64, use_vert_diff=False):
        super().__init__()
        self.use_vert_diff = use_vert_diff
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, (2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), nn.BatchNorm3d(16), nn.GELU(),
            nn.Conv3d(16, 32, (2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), nn.BatchNorm3d(32), nn.GELU(),
            nn.Conv3d(32, 64, (2, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), nn.BatchNorm3d(64), nn.GELU(),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(64, 16), nn.GELU(),
            nn.Linear(16, 64), nn.Tanh(),
        )
        self.pool_avg = nn.AdaptiveAvgPool3d((4, 2, 2))
        self.pool_max = nn.AdaptiveMaxPool3d((4, 2, 2))

        diff_dim = 16 * 2 * 2 if use_vert_diff else 0
        if use_vert_diff:
            self.diff_proj = nn.Sequential(
                nn.Conv2d(1, 8, 3, 2, 1), nn.BatchNorm2d(8), nn.GELU(),
                nn.Conv2d(8, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.GELU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 2 * 2 * 2 + 64 * 2 + diff_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        h = self.conv(x)
        g_avg = F.adaptive_avg_pool3d(h, (1, 1, 1)).flatten(1)
        g_max = F.adaptive_max_pool3d(h, (1, 1, 1)).flatten(1)
        g_min = torch.amin(h, dim=(2, 3, 4))

        gate = self.attn_mlp(0.5 * (g_avg + g_max)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h * (1.0 + 0.5 * gate)

        feats = [
            self.pool_avg(h).flatten(1),
            self.pool_max(h).flatten(1),
            g_max - g_avg,
            g_avg - g_min,
        ]
        if self.use_vert_diff:
            # Levels 0-1 are 1000/850 hPa; 2-4 are 700/600/500 hPa.
            low = x[:, :, 0:2].mean(dim=2)
            mid = x[:, :, 2:5].mean(dim=2)
            feats.append(self.diff_proj(mid - low).flatten(1))
        return self.fc(torch.cat(feats, dim=1))


class Enc2D(nn.Module):
    """Encode one 2-D field ``[N, 1, H, W]`` to a feature vector ``[N, out_dim]``.

    Same four-statistic pooling as :class:`Enc3D`, without the vertical axis.
    """

    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.pool_avg = nn.AdaptiveAvgPool2d((2, 2))
        self.pool_max = nn.AdaptiveMaxPool2d((2, 2))
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2 * 2 + 64 * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        h = self.conv(x)
        g_avg = F.adaptive_avg_pool2d(h, (1, 1)).flatten(1)
        g_max = F.adaptive_max_pool2d(h, (1, 1)).flatten(1)
        g_min = torch.amin(h, dim=(2, 3))
        return self.fc(torch.cat([
            self.pool_avg(h).flatten(1),
            self.pool_max(h).flatten(1),
            g_max - g_avg,
            g_avg - g_min,
        ], dim=1))


class TwoStreamFASTModel(nn.Module):
    """Predict the FAST ventilation terms (chi, S) from gridded ERA5 fields.

    Args:
        feat_dim: Per-encoder feature width.
        scalar_dim: Width of the auxiliary vector fed to the chi head: the four
            FAST scalars (alpha, beta, gamma, vp) plus the entropy-deficit proxy.
    """

    def __init__(self, feat_dim=64, scalar_dim=5):
        super().__init__()
        # S stream: dynamics only.
        self.enc_u = Enc3D(feat_dim)
        self.enc_v = Enc3D(feat_dim)
        self.enc_z = Enc3D(feat_dim)
        # S is predicted as base + delta: a positive trunk fixes the main trend,
        # and a small bounded correction adapts the poorly-constrained low range.
        self.head_s_backbone = nn.Sequential(
            nn.Linear(feat_dim * 3, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.head_s_base = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        self.head_s_delta = nn.Linear(64, 1)

        # chi stream: thermodynamics only. T/Q additionally get the vertical
        # difference branch that resolves the ventilating layer.
        self.enc_t = Enc3D(feat_dim, use_vert_diff=True)
        self.enc_q = Enc3D(feat_dim, use_vert_diff=True)
        self.enc_sst = Enc2D(feat_dim)
        self.enc_mslp = Enc2D(feat_dim)
        self.head_chi = nn.Sequential(
            nn.Linear(feat_dim * 4 + scalar_dim, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x3d, x2d, sc):
        """Diagnose (chi, S) for every timestep of a storm.

        Args:
            x3d: ``[B, T, 5, 7, 72, 72]`` normalised 3-D fields, channel order
                (T, Q, U, V, Z).
            x2d: ``[B, T, 2, 72, 72]`` normalised 2-D fields, order (SST, MSLP).
            sc: ``[B, T, 4]`` FAST scalars (alpha, beta, gamma, vp), vp in m/s.

        Returns:
            Dict with ``s`` (clamped to [0.1, 50]), ``chi`` (the calibrated
            effective value on [0, 4]), and their product ``xs``, each
            ``[B, T, 1]``. ``xs`` is what :func:`fastml.fast_physics.run_fast_with_init`
            consumes.
        """
        B = x3d.shape[0]
        f3 = rearrange(x3d, "b t c l h w -> (b t) c l h w")
        f2 = rearrange(x2d, "b t c h w -> (b t) c h w")

        fu = _chunk_enc(self.enc_u, f3[:, 2:3])
        fv = _chunk_enc(self.enc_v, f3[:, 3:4])
        fz = _chunk_enc(self.enc_z, f3[:, 4:5])
        ft = _chunk_enc(self.enc_t, f3[:, 0:1])
        fq = _chunk_enc(self.enc_q, f3[:, 1:2])
        fs = _chunk_enc(self.enc_sst, f2[:, 0:1])
        fm = _chunk_enc(self.enc_mslp, f2[:, 1:2])

        dyn = rearrange(torch.cat([fu, fv, fz], 1), "(b t) d -> b t d", b=B)
        thermo = rearrange(torch.cat([ft, fq, fs, fm], 1), "(b t) d -> b t d", b=B)

        # Entropy-deficit proxy: SST minus a mid-level saturation surrogate.
        # This hands the chi head a dimensional quantity it would otherwise have
        # to reconstruct from normalised fields.
        sst_mean = x2d[:, :, 0:1].mean(dim=(-2, -1))
        mid_t = x3d[:, :, 0, 2:5].mean(dim=(2, 3, 4)).unsqueeze(-1)
        mid_q = x3d[:, :, 1, 2:5].mean(dim=(2, 3, 4)).unsqueeze(-1)
        delta_s = torch.tanh(sst_mean - (mid_t - 2.5 * mid_q))

        hs = self.head_s_backbone(dyn)
        s_base = self.head_s_base(hs)
        s_delta = 0.25 * torch.tanh(self.head_s_delta(hs))
        pred_s = torch.clamp(s_base + s_delta, 0.1, 50.0)

        chi_aux = torch.cat([sc, delta_s], dim=-1)
        pred_chi_raw = torch.clamp(self.head_chi(torch.cat([thermo, chi_aux], dim=-1)), 0.0, 1.0)
        # Same mean-to-90th-percentile calibration as the ERA5 reference path
        # (fast_physics.chi_calibrated_multiply), so FAST and FAST_ML feed the
        # ODE ventilation indices on one common scale.
        pred_chi_eff = torch.clamp(5.0 * pred_chi_raw, 0.0, 4.0)

        return {"s": pred_s, "chi": pred_chi_eff, "xs": pred_chi_eff * pred_s}


def load_model(ckpt_path, device="cpu"):
    """Build :class:`TwoStreamFASTModel` and strictly load released weights.

    Raises:
        FileNotFoundError: If the checkpoint is missing.
        RuntimeError: If the checkpoint does not match the architecture, which
            would silently change the predictions.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "It ships with the repository under ckpt/."
        )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model = TwoStreamFASTModel().to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The released forward path drops the training-time ODE coupling, but every
    # parameter tensor must still be accounted for.
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match the architecture: "
            f"{len(missing)} missing, {len(unexpected)} unexpected tensors.\n"
            f"missing={sorted(missing)[:5]}\nunexpected={sorted(unexpected)[:5]}"
        )
    model.eval()
    return model
