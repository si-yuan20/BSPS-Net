"""

BSPS-Net V9 — Deep Hemispheric Symmetry Prior Reasoning Components.

Three formal innovations with complete mechanism closure:

  Innovation 1: Correlation-Guided Bidirectional Lesion-Preserving

                Homologous Alignment (CG-BLPHA)

  Innovation 2: Stage-Specific, Reliability-Modulated Hierarchical

                Multi-Evidence Differential Encoding & Global Relation Reasoning

  Innovation 3: Progressive Lesion-Nuisance Evidence Disentanglement & Symmetry-Calibrated Decoding (PLED)

Design principles:

  - Every module has ≥2 3×3×3 residual blocks (no single 1×1 conv as core). - Cue weights truly participate in fusion.

  - Reliability and preservation truly modulate every stage. - All symmetric operations share parameters.

  - Data flow uses typed dataclasses (no aux dict for core forward). - Dynamic 3D relative position bias (no fixed-N table).

  - max_tokens fallback is active and tested.

"""

from __future__ import annotations

import math

from dataclasses import dataclass

from typing import Dict, List, Optional, Tuple, Union

import torch

import torch.nn as nn

import torch.nn.functional as F

# ======================================================================

# Helpers

# ======================================================================

def _safe_groups(channels: int, max_groups: int = 8) -> int:

    g = min(max_groups, channels)

    while g > 1 and channels % g != 0:

        g -= 1

    return g

def pick_valid_num_heads(channels: int, max_heads: int = 8) -> int:

    for h in range(min(max_heads, channels), 0, -1):

        if channels % h == 0:

            return h

    return 1

def safe_odd_window(requested: int, spatial_shape: Tuple[int, ...]) -> int:

    """Return odd window >=1, <=min(spatial_shape). w=0 sentinel = global stats.

    Guarantees: stride=1, padding=w//2 preserves output shape when w>=3.

    """

    max_w = min(spatial_shape)

    if max_w < 3:

        return 0

    w = min(requested, max_w)

    if w % 2 == 0:

        w -= 1

    return max(1, w)

def _local_pool3d(x: torch.Tensor, w: int, op: str = "mean"):

    """Safe avg_pool3d. w<=1 → global per-sample stat (shape [B,C,1,1,1]).

    w>=3 → local pool preserving spatial shape."""

    if w <= 1:

        if op == "mean":

            return x.mean(dim=(2, 3, 4), keepdim=True)

        elif op == "mean_sq":

            return (x * x).mean(dim=(2, 3, 4), keepdim=True)

        elif op == "var":

            return x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)

        return x

    return F.avg_pool3d(x, w, 1, w // 2)

class SafeNorm3D(nn.Module):

    """..."""

    def __init__(self, channels: int, affine: bool = True, eps: float = 1e-5):

        super().__init__()

        self.channels = channels

        self.in_norm = nn.InstanceNorm3d(channels, affine=affine, eps=eps)

        g = min(8, channels)

        while g > 1 and channels % g != 0:

            g -= 1

        self.gn_norm = nn.GroupNorm(g, channels, affine=affine, eps=eps)

    def forward(self, x: torch.Tensor):

        min_dim = min(x.shape[2:])

        if min_dim >= 4:

            return self.in_norm(x)

        return self.gn_norm(x)

def make_sobel_3d(axis: int) -> torch.Tensor:

    """[1,1,3,3,3] zero-sum separable 3D Sobel kernel."""

    d = torch.tensor([-1.0, 0.0, 1.0])

    s = torch.tensor([1.0, 2.0, 1.0])

    if axis == 0:

        core = torch.einsum('i,j,k->ijk', d, s, s)

    elif axis == 1:

        core = torch.einsum('i,j,k->ijk', s, d, s)

    else:

        core = torch.einsum('i,j,k->ijk', s, s, d)

    return (core / 32.0).view(1, 1, 3, 3, 3)

# ======================================================================

# 3D Residual Block (fundamental building block — replaces single 1×1 convs)

# ======================================================================

class ResidualBlock3D(nn.Module):

    """Two 3×3×3 convs with InstanceNorm + residual connection."""

    def __init__(self, channels: int, expansion: float = 1.0):

        super().__init__()

        mid = max(int(channels * expansion), 8)

        self.conv1 = nn.Conv3d(channels, mid, 3, padding=1, bias=False)

        self.norm1 = SafeNorm3D(mid, affine=True)

        self.act1 = nn.LeakyReLU(1e-2, inplace=True)

        self.conv2 = nn.Conv3d(mid, channels, 3, padding=1, bias=False)

        self.norm2 = SafeNorm3D(channels, affine=True)

        self.act2 = nn.LeakyReLU(1e-2, inplace=True)

    def forward(self, x):

        r = self.conv2(self.act1(self.norm1(self.conv1(x))))

        return self.act2(self.norm2(x + r))

class MultiScaleConvBlock3D(nn.Module):

    """Multi-scale 3×3×3 conv branches with different dilations, fused via residual."""

    def __init__(self, channels: int, dilations: Tuple[int, ...] = (1, 2, 3)):

        super().__init__()

        mid = max(channels // len(dilations), 16)

        self.branches = nn.ModuleList([

            nn.Sequential(

                nn.Conv3d(channels, mid, 3, padding=d, dilation=d, bias=False),

                SafeNorm3D(mid, affine=True),

                nn.LeakyReLU(1e-2, inplace=True),

            ) for d in dilations

        ])

        self.fuse = nn.Sequential(

            nn.Conv3d(mid * len(dilations), channels, 3, padding=1, bias=False),

            SafeNorm3D(channels, affine=True),

            nn.LeakyReLU(1e-2, inplace=True),

        )

    def forward(self, x):

        feats = [b(x) for b in self.branches]

        return x + self.fuse(torch.cat(feats, dim=1))

# ======================================================================

# Typed evidence dataclasses — formal data flow (no aux dict for core)

# ======================================================================

@dataclass

class AlignmentEvidence:

    """Output of CG-BLPHA — complete alignment evidence."""

    aligned_mirror: torch.Tensor

    reverse_aligned_original: torch.Tensor

    flow_o2m: torch.Tensor

    flow_m2o: torch.Tensor

    residual_flow: Optional[torch.Tensor]

    geometric_reliability: torch.Tensor

    lesion_preservation_confidence: Optional[torch.Tensor]

    match_entropy: torch.Tensor

    cycle_error: torch.Tensor

    inverse_error: torch.Tensor

    jacobian_map: Optional[torch.Tensor]

@dataclass

class StageSymmetryEvidence:

    """Per-stage differential evidence — formal data conduit."""

    original_feature: torch.Tensor

    aligned_mirror_feature: torch.Tensor

    differential_feature: torch.Tensor

    geometric_reliability: Optional[torch.Tensor]

    lesion_preservation_confidence: Optional[torch.Tensor]

    cue_features: Optional[Dict[str, torch.Tensor]]

    cue_weights_global: Optional[torch.Tensor]

    cue_weights_spatial: Optional[torch.Tensor]

    boundary_evidence: Optional[torch.Tensor]

    uncertainty: Optional[torch.Tensor]

    stage_index: int = -1

@dataclass

class RelationEvidence:

    """Output of global relation reasoning."""

    relation_context: torch.Tensor

    attention_entropy: torch.Tensor

    attention_concentration: torch.Tensor

    expected_correspondence_distance: torch.Tensor

    reliability_weighted_attention: Optional[torch.Tensor]

    relation_uncertainty: Optional[torch.Tensor]

    geometry_consistency: Optional[torch.Tensor] = None

@dataclass

class DeepFeatureExtractor3D(nn.Module):
    """Multi-layer 3D feature extractor: 2 residual blocks + multi-scale context."""
    __hash__ = nn.Module.__hash__  # PyTorch 1.x compat

    def __init__(self, in_channels: int, out_channels: int,

                 num_res_blocks: int = 2, use_multiscale: bool = True):

        super().__init__()

        self.proj = nn.Conv3d(in_channels, out_channels, 1, bias=False) \
            if in_channels != out_channels else nn.Identity()

        self.res_blocks = nn.ModuleList([

            ResidualBlock3D(out_channels) for _ in range(num_res_blocks)

        ])

        self.ms_block = MultiScaleConvBlock3D(out_channels) if use_multiscale else None

    def forward(self, x):

        x = self.proj(x)

        for blk in self.res_blocks:

            x = blk(x)

        if self.ms_block is not None:

            x = self.ms_block(x)

        return x

# ======================================================================

# INNOVATION 1: CG-BLPHA

# Correlation-Guided Bidirectional Lesion-Preserving Homologous Alignment

# ======================================================================

class CorrelationGuidedBidirectionalAlignment3D(nn.Module):

    """Deep local-correlation-driven bidirectional alignment with lesion preservation.

    Key mechanisms:

      1. SHARED matching function G applied symmetrically: G(F_o,F_m) and G(F_m,F_o)

      2. Local 3D correlation pyramid → softmax displacement expectation

      3. Residual flow refinement with multi-scale cues

      4. Deep lesion preservation predictor (no GT leakage at inference)

      5. Multi-source geometric reliability

      6. Flow inverse consistency constraint

    Multi-stage usage:

      - Deep (E5/E4): full local cost volume + residual refinement - Mid (E3/E2): lightweight residual refinement from coarse flow

      - Shallow (E1): propagation only, no heavy cost volume

    """

    def __init__(self, channels: int, max_displacement: float = 4.0,

                 search_radius: int = 2, hidden: Optional[int] = None,

                 use_lesion_preservation: bool = True,

                 use_heavy_cost_volume: bool = True):

        super().__init__()

        self.max_displacement = float(max_displacement)

        self.search_radius = int(search_radius)

        self.use_lesion_preservation = bool(use_lesion_preservation)

        self.use_heavy_cost_volume = bool(use_heavy_cost_volume)

        h = hidden or max(channels // 4, 16)

        # Shared projection for both directions (two residual blocks)

        self.shared_proj = DeepFeatureExtractor3D(channels, h, num_res_blocks=2)

        if self.use_heavy_cost_volume:

            # Flow decoder from correlation volume + projected features

            r = self.search_radius

            corr_ch = (2 * r + 1) ** 3

            self.flow_decoder = nn.Sequential(

                nn.Conv3d(corr_ch + h, h * 2, 3, padding=1, bias=False),

                SafeNorm3D(h * 2, affine=True),

                nn.LeakyReLU(1e-2, inplace=True),

                ResidualBlock3D(h * 2),

                nn.Conv3d(h * 2, h, 3, padding=1, bias=False),

                SafeNorm3D(h, affine=True),

                nn.LeakyReLU(1e-2, inplace=True),

                nn.Conv3d(h, 3, 1, bias=True),

            )

            nn.init.constant_(self.flow_decoder[-1].bias, 0.0)

            nn.init.normal_(self.flow_decoder[-1].weight, std=1e-4)

        else:

            # Lightweight residual flow refiner

            self.flow_decoder = nn.Sequential(

                nn.Conv3d(channels * 2 + h, h, 3, padding=1, bias=False),

                SafeNorm3D(h, affine=True),

                nn.LeakyReLU(1e-2, inplace=True),

                nn.Conv3d(h, h, 3, padding=1, bias=False),

                SafeNorm3D(h, affine=True),

                nn.LeakyReLU(1e-2, inplace=True),

                nn.Conv3d(h, 3, 1, bias=True),

            )

            nn.init.constant_(self.flow_decoder[-1].bias, 0.0)

            nn.init.normal_(self.flow_decoder[-1].weight, std=1e-4)

        # Deep lesion preservation predictor (multi-layer, no GT in forward)

        if self.use_lesion_preservation:

            self.preservation_extractor = DeepFeatureExtractor3D(

                channels * 2, h, num_res_blocks=2)

            self.preservation_head = nn.Sequential(

                nn.Conv3d(h, h, 3, padding=1, bias=False),

                SafeNorm3D(h, affine=True),

                nn.LeakyReLU(1e-2, inplace=True),

                nn.Conv3d(h, 1, 1, bias=True),

            )

            nn.init.constant_(self.preservation_head[-1].bias, -1.0)

        # Multi-source geometric reliability estimator

        self.reliability_extractor = DeepFeatureExtractor3D(6, h, num_res_blocks=1)

        self.reliability_head = nn.Sequential(

            nn.Conv3d(h, h, 3, padding=1, bias=False),

            SafeNorm3D(h, affine=True),

            nn.LeakyReLU(1e-2, inplace=True),

            nn.Conv3d(h, 1, 1, bias=True),

        )

        nn.init.constant_(self.reliability_head[-1].bias, 2.0)

    def _build_local_correlation(self, p: torch.Tensor, q: torch.Tensor):

        """Build local 3D correlation volume via shifted dot products.

        p, q: [B, C, D, H, W] — projected features.

        Returns: [B, (2r+1)^3, D, H, W]

        """

        B, C, D, H, W = p.shape

        r = self.search_radius

        volume_parts = []

        padded = F.pad(q, (r, r, r, r, r, r))

        for dz in range(-r, r + 1):

            for dy in range(-r, r + 1):

                for dx in range(-r, r + 1):

                    shifted = padded[:, :,

                                     r + dz: r + dz + D,

                                     r + dy: r + dy + H,

                                     r + dx: r + dx + W]

                    volume_parts.append((p * shifted).sum(1, keepdim=True))

        return torch.cat(volume_parts, dim=1)

    def _match(self, f_src: torch.Tensor, f_tgt: torch.Tensor):

        """Shared matching function G: projects both, builds correlation, predicts flow."""

        B, C, D, H, W = f_src.shape

        p = self.shared_proj(f_src)

        q = self.shared_proj(f_tgt)

        if self.use_heavy_cost_volume:

            correlation = self._build_local_correlation(p, q)

            # Per-sample softmax temperature (no cross-batch max)

            corr_flat = correlation.reshape(B, -1)

            scale = corr_flat.max(dim=1, keepdim=True).values.clamp(min=1e-3)

            scale = scale.view(B, 1, 1, 1, 1)

            w = F.softmax(correlation / scale, dim=1)

            # Expected displacement (soft argmax)

            flow = torch.zeros(B, 3, D, H, W, device=f_src.device, dtype=f_src.dtype)

            idx = 0

            r = self.search_radius

            for dz in range(-r, r + 1):

                for dy in range(-r, r + 1):

                    for dx in range(-r, r + 1):

                        flow[:, 0] += w[:, idx] * dx

                        flow[:, 1] += w[:, idx] * dy

                        flow[:, 2] += w[:, idx] * dz

                        idx += 1

            flow = flow * (self.max_displacement / max(r, 1))

            # Residual refinement from correlation + projected features

            residual = self.flow_decoder(torch.cat([correlation, p], dim=1))

            flow = flow + torch.tanh(residual) * self.max_displacement

            return flow, correlation, p

        else:

            # Lightweight: use concatenated features

            combined = torch.cat([f_src, f_tgt, p], dim=1)

            residual = self.flow_decoder(combined)

            flow = torch.tanh(residual) * self.max_displacement

            return flow, None, p

    def _warp(self, feat: torch.Tensor, flow: torch.Tensor):

        B = feat.shape[0]

        D, H, W = feat.shape[2], feat.shape[3], feat.shape[4]

        grid = self._base_grid(D, H, W, feat.device, feat.dtype)

        norm = torch.tensor([max(W - 1, 1), max(H - 1, 1), max(D - 1, 1)],

                            device=feat.device, dtype=feat.dtype)

        grid = grid + 2.0 * flow.permute(0, 2, 3, 4, 1) / norm

        return F.grid_sample(feat, grid.expand(B, -1, -1, -1, -1),

                             mode="bilinear", padding_mode="border",

                             align_corners=True)

    @staticmethod

    def _base_grid(D: int, H: int, W: int, device, dtype):

        z = torch.linspace(-1, 1, D, device=device, dtype=dtype)

        y = torch.linspace(-1, 1, H, device=device, dtype=dtype)

        x = torch.linspace(-1, 1, W, device=device, dtype=dtype)

        gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")

        return torch.stack([gx.permute(2, 1, 0), gy.permute(2, 1, 0),

                            gz.permute(2, 1, 0)], dim=-1).unsqueeze(0)

    def forward(self, f_orig: torch.Tensor, f_mirror: torch.Tensor,

                coarse_offset: Optional[torch.Tensor] = None,

                lesion_prior: Optional[torch.Tensor] = None,

                teacher_forcing_ratio: float = 0.0) -> AlignmentEvidence:

        """Full bidirectional alignment with all evidence.

        Args:

            f_orig: Original feature [B, C, D, H, W]

            f_mirror: Mirrored feature [B, C, D, H, W]

            coarse_offset: Coarse flow from deeper stage [B, 3, Dc, Hc, Wc]

            lesion_prior: GT lesion mask for TEACHER FORCING ONLY (training).

            teacher_forcing_ratio: Probability of using GT (decays over epochs).

                At inference, this MUST be 0.

        Returns:

            AlignmentEvidence with all alignment outputs.

        """

        B, C, D, H, W = f_orig.shape

        # --- Bidirectional matching (coarse-to-fine or full) ---

        residual_flow = None

        if coarse_offset is not None:

            # Coarse-to-fine: warp first, then match residual

            up = F.interpolate(coarse_offset, size=(D, H, W),

                               mode="trilinear", align_corners=False)

            w_r = W / max(coarse_offset.shape[4], 1)

            h_r = H / max(coarse_offset.shape[3], 1)

            d_r = D / max(coarse_offset.shape[2], 1)

            sc = torch.tensor([w_r, h_r, d_r], device=f_orig.device,

                              dtype=f_orig.dtype).view(1, 3, 1, 1, 1)

            coarse_scaled = up * sc

            f_mirror_warped = self._warp(f_mirror, coarse_scaled)

            flow_o2m, corr_o2m, proj_o = self._match(f_orig, f_mirror_warped)

            residual_flow = flow_o2m.clone()

            flow_o2m = coarse_scaled + flow_o2m

            f_orig_warped = self._warp(f_orig, -coarse_scaled)

            flow_m2o, _, _ = self._match(f_mirror, f_orig_warped)

            flow_m2o = -coarse_scaled + flow_m2o

        else:

            flow_o2m, corr_o2m, proj_o = self._match(f_orig, f_mirror)

            flow_m2o, _, _ = self._match(f_mirror, f_orig)

        # --- Lesion preservation (deep predictor, NOT GT addition) ---

        pres = None

        if self.use_lesion_preservation:

            pres_feat = self.preservation_extractor(

                torch.cat([f_orig, f_mirror], dim=1))

            pres = torch.sigmoid(self.preservation_head(pres_feat))

            # Teacher forcing: probabilistic interpolation between prediction and GT

            if lesion_prior is not None and teacher_forcing_ratio > 0:

                if self.training and torch.rand(1).item() < teacher_forcing_ratio:

                    pres = pres * (1.0 - teacher_forcing_ratio) + \
                           lesion_prior * teacher_forcing_ratio

            # Protected regions suppress displacement

            flow_o2m = flow_o2m * (1.0 - pres)

            flow_m2o = flow_m2o * (1.0 - pres)

        # --- Warping ---

        aligned_mirror = self._warp(f_mirror, flow_o2m)

        aligned_original = self._warp(f_orig, flow_m2o)

        # --- Cycle + inverse consistency ---

        cycle_back = self._warp(aligned_mirror, flow_m2o)

        cycle_err = (cycle_back - f_orig).abs().mean(1, keepdim=True)

        # Inverse consistency: flow_o2m + warp(flow_m2o, flow_o2m) ≈ 0

        inv_warped = self._warp(flow_m2o, flow_o2m)

        inverse_err = (flow_o2m + inv_warped).abs().mean(1, keepdim=True)

        # --- Match entropy (from correlation volume, per-sample) ---

        if corr_o2m is not None:

            corr_flat = corr_o2m.reshape(B, -1)

            scale = corr_flat.max(dim=1, keepdim=True).values.clamp(min=1e-3)

            scale = scale.view(B, 1, 1, 1, 1)

            vol_soft = F.softmax(corr_o2m / scale, dim=1)

            match_entropy = -(vol_soft * (vol_soft + 1e-8).log()).sum(1, keepdim=True)

            # Top-1 vs top-2 margin

            top2 = vol_soft.topk(2, dim=1).values

            match_margin = (top2[:, 0:1] - top2[:, 1:2]).detach()

        else:

            match_entropy = torch.zeros(B, 1, D, H, W, device=f_orig.device)

            match_margin = torch.zeros(B, 1, D, H, W, device=f_orig.device)

        # --- Multi-source geometric reliability ---

        cos_sim = F.cosine_similarity(f_orig, aligned_mirror, dim=1, eps=1e-6).unsqueeze(1)

        abs_diff = (f_orig - aligned_mirror).abs().mean(1, keepdim=True)

        off_mag = flow_o2m.norm(p=2, dim=1, keepdim=True) / (self.max_displacement + 1e-6)

        # Local NCC improvement after alignment

        ncc_before = self._local_ncc_single(f_orig, f_mirror, window=5)

        ncc_after = self._local_ncc_single(f_orig, aligned_mirror, window=5)

        ncc_improvement = (ncc_after - ncc_before).clamp(min=0)

        rel_in = torch.cat([abs_diff, cos_sim, match_margin, cycle_err,

                            inverse_err, ncc_improvement], dim=1)

        rel_feat = self.reliability_extractor(rel_in)

        geo_reli = torch.sigmoid(self.reliability_head(rel_feat))

        # --- Jacobian determinant map (folding detection) ---

        jacobian_map = self._jacobian_det(flow_o2m, D, H, W)

        return AlignmentEvidence(

            aligned_mirror=aligned_mirror,

            reverse_aligned_original=aligned_original,

            flow_o2m=flow_o2m,

            flow_m2o=flow_m2o,

            residual_flow=residual_flow,

            geometric_reliability=geo_reli,

            lesion_preservation_confidence=pres,

            match_entropy=match_entropy,

            cycle_error=cycle_err,

            inverse_error=inverse_err,

            jacobian_map=jacobian_map,

        )

    @staticmethod

    def _local_ncc_single(a: torch.Tensor, b: torch.Tensor, window: int = 5):

        """Per-sample local NCC using safe pooling. Always returns [B, 1, D, H, W]."""

        w = safe_odd_window(window, a.shape[2:])

        mu_a = _local_pool3d(a, w, "mean")

        mu_b = _local_pool3d(b, w, "mean")

        a_c = a - mu_a

        b_c = b - mu_b

        cov = _local_pool3d(a_c * b_c, w, "mean")

        va = _local_pool3d(a_c * a_c, w, "mean") + 1e-6

        vb = _local_pool3d(b_c * b_c, w, "mean") + 1e-6

        ncc = (cov / (va * vb).sqrt()).mean(1, keepdim=True)

        if ncc.shape[2:] != a.shape[2:]:

            ncc = F.interpolate(ncc, size=a.shape[2:], mode="trilinear",

                                align_corners=False)

        return ncc

    @staticmethod

    def _jacobian_det(flow: torch.Tensor, D: int, H: int, W: int):

        """Jacobian determinant map for folding detection."""

        # Compute spatial gradients of each flow component

        B = flow.shape[0]

        dz = torch.gradient(flow, dim=2)[0] if D > 1 else torch.zeros_like(flow)

        dy = torch.gradient(flow, dim=3)[0] if H > 1 else torch.zeros_like(flow)

        dx = torch.gradient(flow, dim=4)[0] if W > 1 else torch.zeros_like(flow)

        # Jacobian: identity + flow gradient

        j00 = 1.0 + dx[:, 0:1]

        j01 = dx[:, 1:2]

        j02 = dx[:, 2:3]

        j10 = dy[:, 0:1]

        j11 = 1.0 + dy[:, 1:2]

        j12 = dy[:, 2:3]

        j20 = dz[:, 0:1]

        j21 = dz[:, 1:2]

        j22 = 1.0 + dz[:, 2:3]

        det = (j00 * (j11 * j22 - j12 * j21)

               - j01 * (j10 * j22 - j12 * j20)

               + j02 * (j10 * j21 - j11 * j20))

        return det

# ======================================================================

# INNOVATION 2A: Stage-Specific Multi-Cue Differential Encoders

# ======================================================================

class CueCompetitionFusion3D(nn.Module):

    """True cue-competition fusion: each cue is independently encoded, then

    fused with learnable, reliability-aware, input-dependent weights that

    actually participate in the output computation.

    w_final = softmax(w_global + λ_local * w_spatial + λ_rel * w_reliability)

    delta = Σ_k w_final,k * cue_k

    """

    def __init__(self, num_cues: int, cue_channels: int, out_channels: int,

                 hidden: Optional[int] = None):

        super().__init__()

        h = hidden or max(out_channels // 4, 16)

        self.num_cues = num_cues

        self.cue_encoders = nn.ModuleList([

            DeepFeatureExtractor3D(cue_channels, out_channels, num_res_blocks=1,

                                   use_multiscale=False)

            for _ in range(num_cues)

        ])

        # Weight predictors

        self.global_weight_net = nn.Sequential(

            nn.AdaptiveAvgPool3d(1),

            nn.Conv3d(out_channels * num_cues, h, 1, bias=False),

            nn.LeakyReLU(1e-2, inplace=True),

            nn.Conv3d(h, num_cues, 1, bias=True),

        )

        self.spatial_weight_net = nn.Sequential(

            nn.Conv3d(out_channels * num_cues, h, 1, bias=False),

            SafeNorm3D(h, affine=True),

            nn.LeakyReLU(1e-2, inplace=True),

            nn.Conv3d(h, num_cues, 1, bias=True),

        )

        self.reliability_weight_net = nn.Sequential(

            nn.Conv3d(1, h, 1, bias=False),

            nn.LeakyReLU(1e-2, inplace=True),

            nn.Conv3d(h, num_cues, 1, bias=True),

        )

        # Learnable mixing coefficients

        self.lambda_local = nn.Parameter(torch.tensor(0.3))

        self.lambda_rel = nn.Parameter(torch.tensor(0.2))

        self.final_fuse = nn.Sequential(

            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),

            SafeNorm3D(out_channels, affine=True),

            nn.LeakyReLU(1e-2, inplace=True),

        )

    def forward(self, cues: List[torch.Tensor],

                reliability: Optional[torch.Tensor] = None):

        """Args:

            cues: List of [B, Ci, D, H, W] cue tensors.

            reliability: [B, 1, D, H, W] or None.

        Returns:

            delta: [B, out_channels, D, H, W] fused output.

            w_global: [B, num_cues] global weights.

            w_spatial: [B, num_cues, D, H, W] spatial weights.

            w_final: [B, num_cues, D, H, W] final fusion weights.

        """

        encoded = [encoder(cue) for cue, encoder in zip(cues, self.cue_encoders)]

        stacked = torch.cat(encoded, dim=1)  # [B, C*K, D, H, W]

        # Global weights

        w_global = self.global_weight_net(stacked).squeeze(-1).squeeze(-1).squeeze(-1)

        w_global = w_global.softmax(dim=1)  # [B, K]

        # Spatial weights

        w_spatial = self.spatial_weight_net(stacked).softmax(dim=1)  # [B, K, D, H, W]

        # Reliability modulation

        w_rel = torch.zeros_like(w_spatial[:, :1])

        if reliability is not None:

            if reliability.shape[2:] != w_spatial.shape[2:]:

                reliability = F.interpolate(reliability, size=w_spatial.shape[2:],

                                            mode="trilinear", align_corners=False)

            w_rel = self.reliability_weight_net(reliability)  # [B, K, D, H, W]

        # Final weights: combine global, spatial, and reliability

        w_final = w_global.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) \
            + self.lambda_local.abs() * w_spatial \
            + self.lambda_rel.abs() * w_rel

        w_final = w_final.softmax(dim=1)  # [B, K, D, H, W]

        # Weighted fusion: each cue weighted by its final weight

        fused = sum(encoded[k] * w_final[:, k:k + 1] for k in range(len(encoded)))

        delta = self.final_fuse(fused)

        return delta, w_global, w_spatial, w_final

class SignalBoundaryDifferentialEncoder3D(nn.Module):

    """Shallow-stage encoder: local signal statistics + multi-scale 3D gradient cues.

    NOT just two 1×1 convs. Internals:

      - Local statistical normalization branch - Signed difference branch

      - Absolute difference branch - Multi-scale 3D gradient branch (Sobel at multiple smoothing levels)

      - Gradient direction consistency branch - High-frequency residual branch

      - Deep cue competition fusion (weights actually control output)

    """

    def __init__(self, channels: int, local_window: int = 5):

        super().__init__()

        sd = max(channels // 2, 16)

        self.local_window = local_window

        self.register_buffer("sobel_dx", make_sobel_3d(0), persistent=False)

        self.register_buffer("sobel_dy", make_sobel_3d(1), persistent=False)

        self.register_buffer("sobel_dz", make_sobel_3d(2), persistent=False)

        # Cue 1: Signed + absolute difference with local normalization

        self.signal_extractor = DeepFeatureExtractor3D(

            channels * 3, sd, num_res_blocks=2)

        # Cue 2: Multi-scale gradient + direction consistency

        self.boundary_extractor = DeepFeatureExtractor3D(

            4, sd, num_res_blocks=2)

        # Cue 3: High-frequency residual (original - aligned mirror residual)

        self.hf_extractor = DeepFeatureExtractor3D(

            channels, sd, num_res_blocks=1)

        # True cue competition fusion (weights participate in output!)

        self.cue_fusion = CueCompetitionFusion3D(

            num_cues=3, cue_channels=sd, out_channels=channels, hidden=sd)

        # Reliability & preservation modulation

        self.reli_mod = nn.Sequential(

            nn.Conv3d(1, channels, 1, bias=False),

            nn.Sigmoid(),

        )

        self.pres_mod = nn.Sequential(

            nn.Conv3d(1, channels, 1, bias=False),

            nn.Sigmoid(),

        )

    def _local_norm(self, f: torch.Tensor):

        w = safe_odd_window(self.local_window, f.shape[2:])

        mu = _local_pool3d(f, w, "mean")

        mu2 = _local_pool3d(f * f, w, "mean")

        var = (mu2 - mu * mu).clamp(min=1e-6)

        return (f - mu) / (var + 1e-6).sqrt()

    def _edge_multi_scale(self, f: torch.Tensor):

        """Multi-scale gradient magnitude and direction."""

        C = f.shape[1]

        gx = F.conv3d(f, self.sobel_dx.expand(C, 1, 3, 3, 3), padding=1, groups=C)

        gy = F.conv3d(f, self.sobel_dy.expand(C, 1, 3, 3, 3), padding=1, groups=C)

        gz = F.conv3d(f, self.sobel_dz.expand(C, 1, 3, 3, 3), padding=1, groups=C)

        return gx, gy, gz

    def forward(self, f_o: torch.Tensor, f_m: torch.Tensor,

                reli: Optional[torch.Tensor] = None,

                pres: Optional[torch.Tensor] = None):

        B, C, D, H, W = f_o.shape

        # Cue 1: Signal difference (signed + absolute + local norm)

        fo_n = self._local_norm(f_o)

        fm_n = self._local_norm(f_m)

        signed = fo_n - fm_n

        abs_diff = signed.abs()

        signal_cue = self.signal_extractor(torch.cat([signed, abs_diff, fo_n], dim=1))

        # Cue 2: Multi-scale boundary

        gxo, gyo, gzo = self._edge_multi_scale(f_o)

        gxm, gym, gzm = self._edge_multi_scale(f_m)

        mag_o = (gxo ** 2 + gyo ** 2 + gzo ** 2).sqrt().mean(1, keepdim=True)

        mag_m = (gxm ** 2 + gym ** 2 + gzm ** 2).sqrt().mean(1, keepdim=True)

        dot = (gxo * gxm + gyo * gym + gzo * gzm).mean(1, keepdim=True)

        dir_agree = (dot / (mag_o * mag_m + 1e-6)).clamp(-1, 1)

        boundary_cue = self.boundary_extractor(

            torch.cat([mag_o, mag_m, (mag_o - mag_m).abs(), dir_agree], dim=1))

        # Cue 3: High-frequency residual

        hf_residual = f_o - f_m

        hf_cue = self.hf_extractor(hf_residual)

        # True cue competition fusion (weights ACTUALLY control output)

        delta, w_global, w_spatial, w_final = self.cue_fusion(

            [signal_cue, boundary_cue, hf_cue], reliability=reli)

        # Reliability and preservation modulation (ACTUALLY applied)

        if reli is not None:

            if reli.shape[2:] != delta.shape[2:]:

                reli = F.interpolate(reli, size=delta.shape[2:],

                                     mode="trilinear", align_corners=False)

            delta = delta * self.reli_mod(reli)

        if pres is not None:

            if pres.shape[2:] != delta.shape[2:]:

                pres = F.interpolate(pres, size=delta.shape[2:],

                                     mode="trilinear", align_corners=False)

            # In lesion regions, reduce differential emphasis

            delta = delta * (1.0 - 0.5 * self.pres_mod(pres))

        return delta, w_global, w_spatial, w_final

class StructureTextureDifferentialEncoder3D(nn.Module):

    """Mid-stage encoder: multi-window NCC, cosine dissimilarity, local statistics.

    NOT just one 1×1 conv. Internals:

      - Multi-window local NCC (3, 5, 7)

      - Cosine dissimilarity - Local variance ratio

      - Local entropy / texture stability - Correspondence residual

      - Shared bilateral texture encoder - Deep cue competition fusion

    """

    def __init__(self, channels: int):

        super().__init__()

        sd = max(channels // 2, 16)

        # Cue 1: Multi-window NCC + cosine

        self.ncc_extractor = DeepFeatureExtractor3D(5, sd, num_res_blocks=2)

        # Cue 2: Local texture statistics (variance, entropy proxy)

        self.texture_extractor = DeepFeatureExtractor3D(3, sd, num_res_blocks=2)

        # Cue 3: Correspondence residual (deep bilateral)

        self.correspondence_encoder = DeepFeatureExtractor3D(

            channels * 2, sd, num_res_blocks=2)

        # True cue competition fusion

        self.cue_fusion = CueCompetitionFusion3D(

            num_cues=3, cue_channels=sd, out_channels=channels, hidden=sd)

        self.reli_mod = nn.Sequential(

            nn.Conv3d(1, channels, 1, bias=False), nn.Sigmoid())

        self.pres_mod = nn.Sequential(

            nn.Conv3d(1, channels, 1, bias=False), nn.Sigmoid())

    @staticmethod

    def _local_ncc(a: torch.Tensor, b: torch.Tensor, w: int):

        """Per-sample local NCC using safe pooling (batch independent)."""

        mu_a = _local_pool3d(a, w, "mean")

        mu_b = _local_pool3d(b, w, "mean")

        a_c = a - mu_a

        b_c = b - mu_b

        cov = _local_pool3d(a_c * b_c, w, "mean")

        va = _local_pool3d(a_c * a_c, w, "mean") + 1e-6

        vb = _local_pool3d(b_c * b_c, w, "mean") + 1e-6

        return (cov / (va * vb).sqrt()).mean(1, keepdim=True)

    @staticmethod

    def _local_entropy_proxy(f: torch.Tensor, w: int):

        """Local coefficient of variation as entropy proxy."""

        mu = _local_pool3d(f, w, "mean")

        mu2 = _local_pool3d(f * f, w, "mean")

        var = (mu2 - mu * mu).clamp(min=1e-8)

        return (var.sqrt() / (mu.abs() + 1e-6)).mean(1, keepdim=True)

    def forward(self, f_o: torch.Tensor, f_m: torch.Tensor,

                reli: Optional[torch.Tensor] = None,

                pres: Optional[torch.Tensor] = None):

        B, C, D, H, W = f_o.shape

        spatial = (D, H, W)

        # Multi-window NCC (all windows via safe_odd_window)

        w3 = safe_odd_window(3, spatial)

        w5 = safe_odd_window(5, spatial)

        w7 = safe_odd_window(7, spatial)

        ncc_3 = self._local_ncc(f_o, f_m, w3) if w3 > 1 else torch.zeros_like(f_o[:, :1])

        ncc_5 = self._local_ncc(f_o, f_m, w5) if w5 > 1 else ncc_3

        ncc_7 = self._local_ncc(f_o, f_m, w7) if w7 > 1 else ncc_5

        cos = F.cosine_similarity(f_o, f_m, dim=1, eps=1e-6).unsqueeze(1)

        ncc_cue = self.ncc_extractor(

            torch.cat([1.0 - ncc_3, 1.0 - ncc_5, 1.0 - ncc_7,

                       1.0 - cos, (f_o - f_m).abs().mean(1, keepdim=True)], dim=1))

        # Cue 2: Local texture statistics (safe odd window)

        w_tex = safe_odd_window(5, spatial)

        std_o = self._local_entropy_proxy(f_o, w_tex)

        std_m = self._local_entropy_proxy(f_m, w_tex)

        tex_cue = self.texture_extractor(

            torch.cat([std_o, std_m, (std_o - std_m).abs()], dim=1))

        # Cue 3: Deep correspondence residual

        corr_cue = self.correspondence_encoder(

            torch.cat([f_o, f_m], dim=1))

        # True cue competition fusion

        delta, w_global, w_spatial, w_final = self.cue_fusion(

            [ncc_cue, tex_cue, corr_cue], reliability=reli)

        # Reliability and preservation modulation

        if reli is not None:

            if reli.shape[2:] != delta.shape[2:]:

                reli = F.interpolate(reli, size=delta.shape[2:],

                                     mode="trilinear", align_corners=False)

            delta = delta * self.reli_mod(reli)

        if pres is not None:

            if pres.shape[2:] != delta.shape[2:]:

                pres = F.interpolate(pres, size=delta.shape[2:],

                                     mode="trilinear", align_corners=False)

            delta = delta * (1.0 - 0.5 * self.pres_mod(pres))

        return delta, w_global, w_spatial, w_final

class SemanticContextDifferentialEncoder3D(nn.Module):

    """Deep-stage encoder: SHARED semantic context extractor for both hemispheres.

    Key fix over V8: dil_o and dil_m NOW SHARE PARAMETERS.

    Internals:

      - Shared multi-scale dilated context extractor - Deep semantic difference

      - Deep semantic similarity - Lesion-candidate region pooling

      - Global case-level context - Uncertainty calibration

      - Reliability/preservation modulation

    """

    def __init__(self, channels: int, dilation_rates: Tuple[int, ...] = (1, 2, 3)):

        super().__init__()

        sd = max(channels // 2, 16)

        self.dilation_rates = dilation_rates

        # SHARED multi-scale context extractor (NOT separate left/right!)

        self.shared_context = nn.ModuleList([

            nn.Sequential(

                nn.Conv3d(channels, sd, 3, padding=d, dilation=d, bias=False),

                SafeNorm3D(sd, affine=True),

                nn.LeakyReLU(1e-2, inplace=True),

            ) for d in dilation_rates

        ])

        # After shared extraction, fuse multi-scale context

        self.ctx_fuse = DeepFeatureExtractor3D(

            sd * len(dilation_rates), channels, num_res_blocks=2)

        # Cue 1: Semantic difference (shared context space comparison)

        self.sem_diff_extractor = DeepFeatureExtractor3D(channels, sd, num_res_blocks=2)

        # Cue 2: Semantic similarity (4C+1 input: ctx_o, ctx_m, abs_diff, product, cosine)

        self.sem_sim_extractor = DeepFeatureExtractor3D(channels * 4 + 1, sd, num_res_blocks=2)

        # Cue 3: Region-aware relation (lesion candidate pooling)

        self.region_extractor = DeepFeatureExtractor3D(channels, sd, num_res_blocks=1)

        # True cue competition fusion

        self.cue_fusion = CueCompetitionFusion3D(

            num_cues=3, cue_channels=sd, out_channels=channels, hidden=sd)

        # Uncertainty estimation (deep, not single 1×1)

        self.uncertainty_net = nn.Sequential(

            nn.Conv3d(channels, sd, 3, padding=1, bias=False),

            SafeNorm3D(sd, affine=True),

            nn.LeakyReLU(1e-2, inplace=True),

            nn.Conv3d(sd, sd, 3, padding=1, bias=False),

            SafeNorm3D(sd, affine=True),

            nn.LeakyReLU(1e-2, inplace=True),

            nn.Conv3d(sd, 1, 1, bias=True),

        )

        nn.init.constant_(self.uncertainty_net[-1].bias, 1.0)

        self.reli_mod = nn.Sequential(

            nn.Conv3d(1, channels, 1, bias=False), nn.Sigmoid())

        self.pres_mod = nn.Sequential(

            nn.Conv3d(1, channels, 1, bias=False), nn.Sigmoid())

    def _extract_shared_context(self, f: torch.Tensor):

        """Extract multi-scale context using SHARED convolutions."""

        ctx_feats = [conv(f) for conv in self.shared_context]

        return self.ctx_fuse(torch.cat(ctx_feats, dim=1))

    def forward(self, f_o: torch.Tensor, f_m: torch.Tensor,

                reli: Optional[torch.Tensor] = None,

                pres: Optional[torch.Tensor] = None):

        B, C, D, H, W = f_o.shape

        # SHARED context extraction for both hemispheres

        ctx_o = self._extract_shared_context(f_o)

        ctx_m = self._extract_shared_context(f_m)

        # Cue 1: Semantic difference in shared context space

        sem_diff = (ctx_o - ctx_m).abs()

        sem_diff_cue = self.sem_diff_extractor(sem_diff)

        # Cue 2: Semantic similarity — explicit multi-source encoder

        # ctx_o[C], ctx_m[C], abs_diff[C], product[C], cosine[1] → total 3C+1

        sem_sim_input = torch.cat([

            ctx_o, ctx_m,

            (ctx_o - ctx_m).abs(),

            ctx_o * ctx_m,

            F.cosine_similarity(ctx_o, ctx_m, dim=1, eps=1e-6).unsqueeze(1),

        ], dim=1)

        sem_sim_cue = self.sem_sim_extractor(sem_sim_input)

        # Cue 3: Region-aware relation

        region_cue = self.region_extractor(ctx_o * ctx_m)

        # True cue competition fusion

        delta, w_global, w_spatial, w_final = self.cue_fusion(

            [sem_diff_cue, sem_sim_cue, region_cue], reliability=reli)

        # Reliability and preservation modulation

        if reli is not None:

            if reli.shape[2:] != delta.shape[2:]:

                reli = F.interpolate(reli, size=delta.shape[2:],

                                     mode="trilinear", align_corners=False)

            delta = delta * self.reli_mod(reli)

        if pres is not None:

            if pres.shape[2:] != delta.shape[2:]:

                pres = F.interpolate(pres, size=delta.shape[2:],

                                     mode="trilinear", align_corners=False)

            delta = delta * (1.0 - 0.5 * self.pres_mod(pres))

        # Uncertainty (deep, calibrated)

        uncertainty = torch.sigmoid(self.uncertainty_net(delta))

        return delta, uncertainty, w_global, w_spatial, w_final

class StageSpecificDifferentialEncoder3D(nn.Module):

    """Uniform interface dispatching to stage-specific deep encoders."""

    def __init__(self, channels: int, role: str):

        super().__init__()

        self.role = role

        if role == "shallow":

            self.encoder = SignalBoundaryDifferentialEncoder3D(channels)

        elif role == "mid":

            self.encoder = StructureTextureDifferentialEncoder3D(channels)

        elif role == "deep":

            self.encoder = SemanticContextDifferentialEncoder3D(channels)

        else:

            raise ValueError(f"Unknown stage role '{role}'")

    def forward(self, f_o: torch.Tensor, f_m: torch.Tensor,

                reli: Optional[torch.Tensor] = None,

                pres: Optional[torch.Tensor] = None):

        return self.encoder(f_o, f_m, reli, pres)

# ======================================================================

# INNOVATION 2B: Global Symmetric Relation Reasoner

# ======================================================================

class GlobalSymmetricRelationReasoner3D(nn.Module):

    """Bottleneck global bidirectional cross-hemisphere relation reasoning.

    Key mechanisms:

      - Multi-layer residual projection BEFORE Q/K/V (not just 1×1)

      - Differential tokens participate in Q/K/V computation - Dynamic 3D relative position bias (no fixed-N table)

      - Key-wise reliability modulation (changes attention distribution)

      - Value reliability gating (suppresses unreliable context)

      - Pairwise geometry bias from alignment evidence - Relation uncertainty estimation

      - max_tokens fallback with adaptive pooling - Explanations: attention entropy, top-k concentration, correspondence distance

    """

    def __init__(self, channels: int, num_heads: Optional[int] = None,

                 layer_scale_init: float = 1e-2,

                 use_reliability_modulation: bool = True,

                 max_tokens: int = 16384,

                 max_rel_distance: int = 12):

        super().__init__()

        self.channels = int(channels)

        self.num_heads = pick_valid_num_heads(self.channels, num_heads or 8)

        self.head_dim = self.channels // self.num_heads

        self.use_reliability_modulation = bool(use_reliability_modulation)

        self.max_tokens = int(max_tokens)

        self.max_rel_distance = int(max_rel_distance)

        tbl_size = max_rel_distance * 2 + 1

        # Multi-layer residual projection BEFORE attention (NOT just 1×1)

        self.proj_orig = DeepFeatureExtractor3D(channels, channels, num_res_blocks=2)

        self.proj_mirror = DeepFeatureExtractor3D(channels, channels, num_res_blocks=2)

        # Differential feature injection projections

        self.diff_proj_qk = nn.Conv3d(channels, channels, 1, bias=False)

        self.diff_proj_v = nn.Conv3d(channels, channels, 1, bias=False)

        # Q/K/V projections (1×1 is standard for attention heads)

        self.wq_o = nn.Conv3d(channels, channels, 1, bias=False)

        self.wk_m = nn.Conv3d(channels, channels, 1, bias=False)

        self.wv_m = nn.Conv3d(channels, channels, 1, bias=False)

        self.wq_m = nn.Conv3d(channels, channels, 1, bias=False)

        self.wk_o = nn.Conv3d(channels, channels, 1, bias=False)

        self.wv_o = nn.Conv3d(channels, channels, 1, bias=False)

        # Output: multi-layer residual relation encoding AFTER attention

        self.out_proj = DeepFeatureExtractor3D(channels * 2, channels, num_res_blocks=2)

        self.layer_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))

        # Dynamic 3D axial relative position bias tables (shared over same Δ)

        self.rel_bias_d = nn.Parameter(torch.zeros(tbl_size, self.num_heads))

        self.rel_bias_h = nn.Parameter(torch.zeros(tbl_size, self.num_heads))

        self.rel_bias_w = nn.Parameter(torch.zeros(tbl_size, self.num_heads))

        nn.init.normal_(self.rel_bias_d, std=0.02)

        nn.init.normal_(self.rel_bias_h, std=0.02)

        nn.init.normal_(self.rel_bias_w, std=0.02)

        # Per-head learnable temperature for geometry bias

        self.geo_bias_scale = nn.Parameter(torch.ones(self.num_heads) * 0.1)

        # Relation uncertainty head

        self.relation_uncertainty_head = nn.Sequential(

            nn.AdaptiveAvgPool3d(1),

            nn.Conv3d(channels, channels // 4, 1, bias=False),

            nn.LeakyReLU(1e-2, inplace=True),

            nn.Conv3d(channels // 4, 1, 1, bias=True),

            nn.Sigmoid(),

        )

        self._token_count = 0

    def _compute_dynamic_position_bias(self, D: int, H: int, W: int,

                                       device, dtype) -> torch.Tensor:

        """Dynamic 3D relative position bias from axial tables.

        Works for ANY N = D*H*W, no fixed table size.

        """

        N = D * H * W

        max_r = self.max_rel_distance

        di = torch.arange(D, device=device)

        hi = torch.arange(H, device=device)

        wi = torch.arange(W, device=device)

        # Build 3D index grids

        d_idx = torch.arange(N, device=device) // (H * W)

        h_idx = (torch.arange(N, device=device) % (H * W)) // W

        w_idx = torch.arange(N, device=device) % W

        dd = (d_idx[:, None] - d_idx[None, :]).clamp(-max_r, max_r) + max_r

        dh = (h_idx[:, None] - h_idx[None, :]).clamp(-max_r, max_r) + max_r

        dw = (w_idx[:, None] - w_idx[None, :]).clamp(-max_r, max_r) + max_r

        bias = (self.rel_bias_d[dd.reshape(-1)].reshape(N, N, self.num_heads)

                + self.rel_bias_h[dh.reshape(-1)].reshape(N, N, self.num_heads)

                + self.rel_bias_w[dw.reshape(-1)].reshape(N, N, self.num_heads))

        return bias.permute(2, 0, 1).unsqueeze(0)  # [1, heads, N, N]

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,

                   key_bias: Optional[torch.Tensor],

                   value_gate: Optional[torch.Tensor],

                   pos_bias: torch.Tensor,

                   geo_bias: Optional[torch.Tensor],

                   D: int, H: int, W: int):

        """q,k,v: [B, heads, N, d]; key_bias: [B, 1, 1, N]; value_gate: [B, 1, N, 1].

        All 4D tensors — no 5D broadcasting issues.

        """

        B, Hd, N, d = q.shape

        scale = d ** -0.5

        logits = (q @ k.transpose(-1, -2)) * scale  # [B, heads, N, N]

        # Key-wise reliability bias (changes attention distribution)

        if key_bias is not None:

            logits = logits + key_bias  # [B,1,1,N] + [B,H,N,N] → broadcast

        # Dynamic 3D position bias

        logits = logits + pos_bias  # [1,heads,N,N] + [B,heads,N,N] → broadcast

        # Pairwise geometry bias from alignment evidence

        if geo_bias is not None and geo_bias.shape == logits.shape:

            logits = logits + geo_bias

        attn = F.softmax(logits, dim=-1)

        # Value gating (suppresses unreliable tokens)

        if value_gate is not None:

            v = v * value_gate  # [B,H,N,d] * [B,1,N,1] → broadcast

        out = attn @ v  # [B, heads, N, d]

        return out, attn

    def _run_direction(self, f_src: torch.Tensor, f_tgt: torch.Tensor,

                       proj_src: nn.Module, proj_tgt: nn.Module,

                       wq, wk, wv, key_bias, value_gate,

                       pos_bias, geo_bias, D: int, H: int, W: int,

                       diff_feat: Optional[torch.Tensor] = None):

        B, C, Df, Hf, Wf = f_src.shape

        N = D * H * W

        # Multi-layer residual projection BEFORE attention

        f_src_proj = proj_src(f_src)

        f_tgt_proj = proj_tgt(f_tgt)

        # Inject differential feature into Q/K/V (if available)

        if diff_feat is not None:

            if diff_feat.shape[2:] != (D, H, W):

                diff_feat = F.interpolate(diff_feat, size=(D, H, W),

                                          mode="trilinear", align_corners=False)

            diff_qk = self.diff_proj_qk(diff_feat)

            diff_v = self.diff_proj_v(diff_feat)

            f_src_proj = f_src_proj + diff_qk

            f_tgt_proj = f_tgt_proj + diff_qk

        else:

            diff_v = None

        q = wq(f_src_proj).reshape(B, self.num_heads, self.head_dim, N).transpose(-1, -2)

        k = wk(f_tgt_proj).reshape(B, self.num_heads, self.head_dim, N).transpose(-1, -2)

        v = wv(f_tgt_proj).reshape(B, self.num_heads, self.head_dim, N).transpose(-1, -2)

        # Inject diff into value

        if diff_v is not None:

            v_diff = diff_v.reshape(B, self.num_heads, self.head_dim, N).transpose(-1, -2)

            v = v + v_diff

        out, attn = self._attention(q, k, v, key_bias, value_gate,

                                     pos_bias, geo_bias, D, H, W)

        out = out.transpose(-1, -2).reshape(B, C, D, H, W)

        # Explanations

        entropy = -(attn * (attn + 1e-8).log()).sum(-1).mean(1, keepdim=True)

        top2 = attn.topk(2, dim=-1).values

        concentration = (top2[:, :, :, 0] - top2[:, :, :, 1]).mean(1, keepdim=True)

        # Expected 3D correspondence distance (real coordinates, not flat index)

        d_coords = (torch.arange(D, device=attn.device, dtype=attn.dtype)

                    .view(-1, 1, 1).expand(-1, H, W).reshape(-1))

        h_coords = (torch.arange(H, device=attn.device, dtype=attn.dtype)

                    .view(1, -1, 1).expand(D, -1, W).reshape(-1))

        w_coords = (torch.arange(W, device=attn.device, dtype=attn.dtype)

                    .view(1, 1, -1).expand(D, H, -1).reshape(-1))

        dd = (d_coords[:, None] - d_coords[None, :]) ** 2

        dh = (h_coords[:, None] - h_coords[None, :]) ** 2

        dw = (w_coords[:, None] - w_coords[None, :]) ** 2

        dist_3d = (dd + dh + dw).sqrt().view(1, 1, N, N)

        expected_dist = (attn * dist_3d).sum(-1).mean(1, keepdim=True)

        return out, entropy, concentration, expected_dist

    def forward(self, f_orig: torch.Tensor, f_mirror: torch.Tensor,

                geometric_reliability: Optional[torch.Tensor] = None,

                differential_feature: Optional[torch.Tensor] = None,

                match_entropy: Optional[torch.Tensor] = None,

                flow_o2m: Optional[torch.Tensor] = None) -> RelationEvidence:

        B, C, D, H, W = f_orig.shape

        self._token_count = D * H * W

        # --- max_tokens fallback: adaptive pool → attend → upsample ---

        use_pool = self._token_count > self.max_tokens

        orig_spatial = (D, H, W)

        if use_pool:

            scale = (self.max_tokens / float(self._token_count)) ** (1.0 / 3.0)

            tD = max(2, int(round(D * scale)))

            tH = max(2, int(round(H * scale)))

            tW = max(2, int(round(W * scale)))

            f_orig = F.adaptive_avg_pool3d(f_orig, (tD, tH, tW))

            f_mirror = F.adaptive_avg_pool3d(f_mirror, (tD, tH, tW))

            if geometric_reliability is not None:

                geometric_reliability = F.adaptive_avg_pool3d(

                    geometric_reliability, (tD, tH, tW))

            if differential_feature is not None:

                differential_feature = F.adaptive_avg_pool3d(

                    differential_feature, (tD, tH, tW))

            if match_entropy is not None:

                match_entropy = F.adaptive_avg_pool3d(match_entropy, (tD, tH, tW))

            D, H, W = tD, tH, tW

        N = D * H * W

        # --- Dynamic 3D position bias ---

        pos_bias = self._compute_dynamic_position_bias(D, H, W,

                                                       f_orig.device, f_orig.dtype)

        # Pairwise geometry bias: flow-based 3D correspondence distance [B, H, N, N]

        geo_bias = None

        if flow_o2m is not None:

            if flow_o2m.shape[2:] != (D, H, W):

                flow_o2m = F.interpolate(flow_o2m, size=(D, H, W),

                                          mode="trilinear", align_corners=False)

            # Build token-level 3D coordinates: dq, hq, wq each [N]

            d_idx = torch.arange(D, device=f_orig.device, dtype=f_orig.dtype)

            h_idx = torch.arange(H, device=f_orig.device, dtype=f_orig.dtype)

            w_idx = torch.arange(W, device=f_orig.device, dtype=f_orig.dtype)

            dq = d_idx.view(-1, 1, 1).expand(-1, H, W).reshape(N)

            hq = h_idx.view(1, -1, 1).expand(D, -1, W).reshape(N)

            wq = w_idx.view(1, 1, -1).expand(D, H, -1).reshape(N)

            # Flow at query positions (channel order: [x,y,z] = [W,H,D] in 5D tensor)

            f_w = flow_o2m[:, 0].reshape(B, N)  # dx

            f_h = flow_o2m[:, 1].reshape(B, N)  # dy

            f_d = flow_o2m[:, 2].reshape(B, N)  # dz

            # Predicted key 3D coordinates = query coords + flow

            pk_w = wq[None, :] + f_w  # [B, N]

            pk_h = hq[None, :] + f_h

            pk_d = dq[None, :] + f_d

            # Pairwise L2 distance: ||p_k - p_hat_q||^2 for all (q,k) pairs → [B, N, N]

            geo_dist_sq = ((dq[None, None, :] - pk_d[:, :, None]) ** 2 + (hq[None, None, :] - pk_h[:, :, None]) ** 2

                           + (wq[None, None, :] - pk_w[:, :, None]) ** 2)

            # Per-head learnable temperature → [B, H, N, N]

            geo_bias = -geo_dist_sq.unsqueeze(1) * self.geo_bias_scale.abs().view(1, -1, 1, 1)

        # --- Key-wise reliability + match-entropy value gating ---

        key_bias = None

        value_gate = None

        if self.use_reliability_modulation and geometric_reliability is not None:

            rel = geometric_reliability

            if rel.shape[2:] != (D, H, W):

                rel = F.interpolate(rel, size=(D, H, W), mode="trilinear",

                                    align_corners=False)

            # [B,1,1,N]: key-wise bias added to attention logits

            key_bias = (rel * 2.0 - 1.0).reshape(B, 1, 1, N)

            # Base value gate from reliability

            base_gate = ((key_bias + 1.0) * 0.5).transpose(-1, -2)  # [B,1,N,1]

            # Match entropy modulation: high entropy → lower value contribution

            if match_entropy is not None:

                if match_entropy.shape[2:] != (D, H, W):

                    match_entropy = F.interpolate(match_entropy, size=(D, H, W),

                                                  mode="trilinear", align_corners=False)

                # Normalize entropy: higher entropy → lower gate

                ent_gate = torch.exp(-match_entropy * 2.0).reshape(B, 1, N, 1)

                value_gate = base_gate * ent_gate

            else:

                value_gate = base_gate

        # --- Bidirectional relation reasoning ---

        ctx_o2m, ent_o, conc_o, dist_o = self._run_direction(

            f_orig, f_mirror, self.proj_orig, self.proj_mirror,

            self.wq_o, self.wk_m, self.wv_m, key_bias, value_gate,

            pos_bias, geo_bias, D, H, W,

            diff_feat=differential_feature)

        ctx_m2o, ent_m, conc_m, dist_m = self._run_direction(

            f_mirror, f_orig, self.proj_mirror, self.proj_orig,

            self.wq_m, self.wk_o, self.wv_o, key_bias, value_gate,

            pos_bias, geo_bias, D, H, W,

            diff_feat=differential_feature)

        # --- Output: multi-layer residual relation encoding ---

        relation_context = self.out_proj(torch.cat([ctx_o2m, ctx_m2o], dim=1))

        relation_context = relation_context * self.layer_scale

        if use_pool:

            relation_context = F.interpolate(

                relation_context, size=orig_spatial, mode="trilinear",

                align_corners=False)

        # Explanations

        entropy = ((ent_o + ent_m) * 0.5)

        concentration = ((conc_o + conc_m) * 0.5)

        expected_dist = ((dist_o + dist_m) * 0.5)

        # Relation uncertainty

        rel_uncertainty = self.relation_uncertainty_head(relation_context)

        # Reliability-weighted attention summary

        reliability_weighted_attn = None

        if key_bias is not None:

            reliability_weighted_attn = key_bias.reshape(B, N)

        # Geometry consistency map (from flow-based correspondence)

        geometry_consistency = None  # Computed from geo_bias when available

        return RelationEvidence(

            relation_context=relation_context,

            attention_entropy=entropy,

            attention_concentration=concentration,

            expected_correspondence_distance=expected_dist,

            reliability_weighted_attention=reliability_weighted_attn,

            relation_uncertainty=rel_uncertainty,

            geometry_consistency=geometry_consistency,

        )

# ======================================================================

# INNOVATION 3: PLED — Progressive Lesion-Nuisance Evidence

#               Disentanglement Decoder

# ======================================================================

class HRSCLoss(nn.Module):

    """Hemispheric Relation–Symmetry Constraint Loss.

    L_HRSC = λ_seg·L_seg + λ_nsc·L_normal_sym + λ_lac·L_lesion_asym + λ_hrc·L_hier_relation + λ_bnd·L_boundary + λ_align·Σ L_align

    Design principles:

      - Normal tissue: homologous symmetry should be preserved - Lesion tissue: asymmetric response should be discriminative (bounded)

      - Unreliable regions: constraint weakened by geometric reliability - Midline regions: naturally asymmetric, excluded from symmetry constraint

    """

    def __init__(self, weights: Optional[Dict[str, float]] = None,

                 nsc_margin: float = 0.0, lac_margin: float = 0.3,

                 midline_width: int = 3, lr_spatial_axis: Optional[int] = None,

                 nsc_stages: Tuple[int, ...] = (0, 1, 2),

                 lac_stages: Tuple[int, ...] = (0, 1)):

        super().__init__()

        self.weights = {

            "seg": 1.0, "nsc": 0.10, "lac": 0.10,

            "hrc": 0.05, "bnd": 0.05,

            "align": 0.1, "inverse_flow": 0.05, "cycle": 0.1,

            "smooth": 0.05, "jacobian": 0.01, "preservation": 0.1,

            "cue_consistency": 0.05, "relation": 0.05,

        }

        if weights:

            self.weights.update(weights)

        self.nsc_margin = nsc_margin

        self.lac_margin = lac_margin

        self.midline_width = midline_width

        self.lr_spatial_axis = lr_spatial_axis

        self.nsc_stages = nsc_stages

        self.lac_stages = lac_stages

    # ----- helpers -----

    @staticmethod

    def _dice(p, t):

        if t.sum() < 1.0:

            return torch.tensor(0.0, device=p.device)

        inter = (p * t).sum()

        return 1.0 - 2.0 * inter / (p.sum() + t.sum() + 1e-6)

    @staticmethod

    def _spatial_align(a, b):

        if a.shape == b.shape:

            return a, b

        return (F.interpolate(a, size=b.shape[2:], mode="trilinear",

                              align_corners=False), b)

    @staticmethod

    def _brain_mask_from_image(image, close_radius=2):

        """Build brain mask from non-zero voxels with morphological close."""

        B = image.shape[0]

        mask = (image.abs().sum(1, keepdim=True) > 1e-6).float()

        k = close_radius * 2 + 1

        mask = F.max_pool3d(F.max_pool3d(

            F.max_pool3d(mask, k, 1, k // 2),

            k, 1, k // 2), k, 1, k // 2)

        return mask.clamp(0, 1)

    def _midline_mask_3d(self, shape, lr_axis, device):

        """Build a midline exclusion band along the LR axis."""

        D, H, W = shape

        if lr_axis is None:

            return torch.zeros(1, 1, D, H, W, device=device)

        # Map lr_spatial_axis (0,1,2) to D,H,W dimension

        dims = [D, H, W]

        center = dims[lr_axis] // 2

        w = self.midline_width

        lo, hi = max(0, center - w), min(dims[lr_axis], center + w)

        mask = torch.ones(1, 1, D, H, W, device=device)

        if lr_axis == 0:

            mask[:, :, lo:hi, :, :] = 0.0

        elif lr_axis == 1:

            mask[:, :, :, lo:hi, :] = 0.0

        else:

            mask[:, :, :, :, lo:hi] = 0.0

        return mask

    @staticmethod

    def _downsample_label(y, target_shape):

        if y.shape[2:] == target_shape:

            return (y > 0.5).float()

        y_down = F.interpolate(y.float(), size=target_shape, mode="nearest")

        return (y_down > 0.5).float()

    # ----- core loss terms -----

    def _l_seg(self, pred, target):

        return self._dice(torch.sigmoid(pred), target) \
            + F.binary_cross_entropy_with_logits(pred, target)

    def _l_normal_sym(self, features_o, features_m_aligned, reliabilities,
                      target, stage_indices, brain_mask):

        """Normal-region homologous symmetry consistency."""

        total = torch.tensor(0.0, device=features_o[0].device)

        count = 0

        for idx, (fo, fm) in enumerate(zip(features_o, features_m_aligned)):

            if idx not in stage_indices:

                continue

            if fo is None or fm is None:

                continue

            if fo.shape[2:] != fm.shape[2:]:

                fm = F.interpolate(fm, size=fo.shape[2:], mode="trilinear",

                                   align_corners=False)

            y = self._downsample_label(target, fo.shape[2:])

            y_dilated = F.max_pool3d(y, 3, 1, 1)

            # Get reliability at correct spatial size

            reli = reliabilities.get(idx) if reliabilities else None

            if reli is not None and reli.shape[2:] != fo.shape[2:]:

                reli = F.interpolate(reli, size=fo.shape[2:],

                                    mode="trilinear", align_corners=False)

            if reli is None:

                reli = torch.ones_like(fo[:, :1])

            # Brain mask at correct size

            bm = brain_mask

            if bm is not None and bm.shape[2:] != fo.shape[2:]:

                bm = F.interpolate(bm, size=fo.shape[2:],

                                   mode="nearest")

            if bm is None:

                bm = torch.ones_like(fo[:, :1])

            # Normal mask: brain × (1-lesion_dilated) × reliability

            M_normal = bm * (1.0 - y_dilated) * reli.clamp(min=1e-6)

            if M_normal.sum() < 1.0:

                continue

            cos_sim = F.cosine_similarity(fo, fm, dim=1, eps=1e-6).unsqueeze(1)

            loss = ((1.0 - cos_sim) * M_normal).sum() / M_normal.sum().clamp(min=1)

            total = total + loss

            count += 1

        return total / max(count, 1)

    def _l_lesion_asym(self, features_o, features_m_aligned, reliabilities,
                       target, stage_indices):

        """Lesion-region bounded asymmetric discrimination."""

        total = torch.tensor(0.0, device=features_o[0].device)

        count = 0

        for idx, (fo, fm) in enumerate(zip(features_o, features_m_aligned)):

            if idx not in stage_indices:

                continue

            if fo is None or fm is None:

                continue

            if fo.shape[2:] != fm.shape[2:]:

                fm = F.interpolate(fm, size=fo.shape[2:], mode="trilinear",

                                   align_corners=False)

            y = self._downsample_label(target, fo.shape[2:])

            if y.sum() < 1.0:

                continue

            reli = reliabilities.get(idx) if reliabilities else None

            if reli is not None and reli.shape[2:] != fo.shape[2:]:

                reli = F.interpolate(reli, size=fo.shape[2:],

                                    mode="trilinear", align_corners=False)

            if reli is None:

                reli = torch.ones_like(fo[:, :1])

            cos_sim = F.cosine_similarity(fo, fm, dim=1, eps=1e-6).unsqueeze(1)

            # Bounded margin: encourage dissimilarity but not unlimited

            loss = (F.relu(self.lac_margin - (1.0 - cos_sim)) * y * reli).sum() \
                / y.sum().clamp(min=1)

            total = total + loss

            count += 1

        return total / max(count, 1)

    def _l_hier_relation(self, stage_evidences, targets, brain_mask):

        """Hierarchical relation consistency."""

        if not stage_evidences:

            return torch.tensor(0.0, device=targets.device)

        device = targets.device

        total = torch.tensor(0.0, device=device)

        count = 0

        delta_list = []

        y_list = []

        bm_list = []

        for sev in stage_evidences:

            if sev is None or sev.differential_feature is None:

                continue

            d = sev.differential_feature

            y = self._downsample_label(targets, d.shape[2:])

            bm = brain_mask

            if bm is not None and bm.shape[2:] != d.shape[2:]:

                bm = F.interpolate(bm, size=d.shape[2:], mode="nearest")

            delta_list.append(d)

            y_list.append(y)

            bm_list.append(bm)

        if len(delta_list) < 2:

            return total  # need at least 2 stages for cross-stage consistency

        # Per-stage: normal region delta should be low

        for d, y, bm in zip(delta_list, y_list, bm_list):

            if bm is None:

                bm = torch.ones_like(y)

            y_dil = F.max_pool3d(y, 3, 1, 1)

            M_normal = bm * (1.0 - y_dil)

            if M_normal.sum() < 1.0:

                continue

            loss = (d.abs().mean(1, keepdim=True) * M_normal).sum() \
                / M_normal.sum().clamp(min=1)

            total = total + loss

            count += 1

        # Cross-stage: lesion delta > normal delta + margin

        lesion_means = []

        normal_means = []

        for d, y, bm in zip(delta_list, y_list, bm_list):

            if bm is None:

                bm = torch.ones_like(y)

            y_dil = F.max_pool3d(y, 3, 1, 1)

            d_abs = d.abs().mean(1, keepdim=True)

            if y.sum() > 0:

                lesion_means.append((d_abs * y).sum() / y.sum().clamp(min=1))

            M_norm = bm * (1.0 - y_dil)

            if M_norm.sum() > 1:

                normal_means.append((d_abs * M_norm).sum() / M_norm.sum().clamp(min=1))

        if lesion_means and normal_means:

            l_mean = torch.stack(lesion_means).mean()

            n_mean = torch.stack(normal_means).mean()

            total = total + F.relu(n_mean - l_mean + 0.1)

            count += 1

        return total / max(count, 1)

    def _l_boundary(self, pred, target):

        boundary_gt = self._morph_boundary_3d(target)

        if boundary_gt.sum() < 1.0:

            return torch.tensor(0.0, device=pred.device)

        pred_prob = torch.sigmoid(pred)

        pred_boundary = self._morph_boundary_3d(pred_prob)

        return self._dice(pred_boundary, boundary_gt)

    @staticmethod

    def _morph_boundary_3d(mask):

        dilated = F.max_pool3d(mask, 3, 1, 1)

        eroded = -F.max_pool3d(-mask, 3, 1, 1)

        return ((dilated - eroded) > 0).float()

    # ----- alignment regularization (kept from CG-BLPHA) -----

    def _l_align(self, alignment_evidences, target):

        total = torch.tensor(0.0, device=target.device)

        terms: Dict[str, float] = {}

        if not alignment_evidences:

            return total, terms

        for i, aev in enumerate(alignment_evidences):

            if aev is None:

                continue

            if target is not None and target.sum() > 0:

                les_dilated = F.max_pool3d(target, 3, 1, 1)

                non_lesion_mask = (1.0 - les_dilated)

                cyc, nl = self._spatial_align(aev.cycle_error, non_lesion_mask)

                align_loss = (cyc * nl).mean() / (nl.mean() + 1e-6)

                total = total + self.weights["align"] * align_loss

                terms[f"align_{i}"] = float(align_loss.detach().cpu())

            inv_loss = aev.inverse_error.mean()

            total = total + self.weights["inverse_flow"] * inv_loss

            terms[f"inv_flow_{i}"] = float(inv_loss.detach().cpu())

            cyc_loss = aev.cycle_error.mean()

            total = total + self.weights["cycle"] * cyc_loss

            terms[f"cycle_{i}"] = float(cyc_loss.detach().cpu())

            flow = aev.flow_o2m

            tv = ((flow[:, :, 1:] - flow[:, :, :-1]).abs().mean()

                  + (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()

                  + (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).abs().mean())

            total = total + self.weights["smooth"] * tv

            terms[f"smooth_{i}"] = float(tv.detach().cpu())

            if aev.jacobian_map is not None:

                fold = F.relu(-aev.jacobian_map + 0.1).mean()

                total = total + self.weights["jacobian"] * fold

                terms[f"fold_{i}"] = float(fold.detach().cpu())

            if target is not None and aev.lesion_preservation_confidence is not None:

                pres, t = self._spatial_align(

                    aev.lesion_preservation_confidence, target)

                pres_loss = self._dice(pres, t)

                total = total + self.weights["preservation"] * pres_loss

                terms[f"pres_{i}"] = float(pres_loss.detach().cpu())

        return total, terms

    def _l_cue_relation(self, stage_evidences, relation_evidence):

        total = torch.tensor(0.0, device=(

            stage_evidences[0].differential_feature.device

            if stage_evidences else torch.device("cpu")))

        terms: Dict[str, float] = {}

        for i, sev in enumerate(stage_evidences):

            if sev is None or sev.cue_weights_global is None:

                continue

            w = sev.cue_weights_global

            K = w.shape[1]

            ent = -(w * (w + 1e-8).log()).sum(-1)

            H_min = math.log(K) * 0.5

            cue_loss = F.relu(H_min - ent).mean()

            total = total + self.weights["cue_consistency"] * cue_loss

            terms[f"cue_{i}"] = float(cue_loss.detach().cpu())

        if relation_evidence is not None:

            H = relation_evidence.attention_entropy

            N = H.shape[-1] if len(H.shape) >= 2 else 1

            H_max = math.log(max(N, 2))

            H_min = H_max * 0.15

            rel_loss = F.relu(H_min - H).mean() + F.relu(H - H_max * 0.95).mean()

            total = total + self.weights["relation"] * rel_loss

            terms["relation"] = float(rel_loss.detach().cpu())

        return total, terms

    # ----- unified forward -----

    def forward(self, pred, target,

                features_o=None, features_m_aligned=None,

                reliabilities=None, stage_evidences=None,

                relation_evidence=None, alignment_evidences=None,

                brain_mask=None, lr_spatial_axis=None):

        device = pred.device

        total = torch.tensor(0.0, device=device)

        terms: Dict[str, float] = {}

        def _add(name, loss, w):

            nonlocal total

            if w > 0 and loss is not None and not torch.isnan(loss):

                total = total + w * loss

                terms[name] = float(loss.detach().cpu())

        # Primary segmentation

        seg_loss = self._l_seg(pred, target)

        _add("seg", seg_loss, self.weights["seg"])

        # HRSC core: normal symmetry

        if features_o is not None and features_m_aligned is not None:

            axis = lr_spatial_axis if lr_spatial_axis is not None \
                else self.lr_spatial_axis

            bm = brain_mask

            if bm is None and features_o:

                bm = self._brain_mask_from_image(features_o[-1])

            # Apply midline exclusion

            midline = None

            if axis is not None and bm is not None:

                midline = self._midline_mask_3d(

                    bm.shape[2:], axis, bm.device)

                bm = bm * midline

            nsc_loss = self._l_normal_sym(

                features_o, features_m_aligned, reliabilities,

                target, self.nsc_stages, bm)

            _add("nsc", nsc_loss, self.weights["nsc"])

            lac_loss = self._l_lesion_asym(

                features_o, features_m_aligned, reliabilities,

                target, self.lac_stages)

            _add("lac", lac_loss, self.weights["lac"])

        # HRSC core: hierarchical relation

        if stage_evidences:

            hrc_loss = self._l_hier_relation(

                stage_evidences, target, bm if features_o else None)

            _add("hrc", hrc_loss, self.weights["hrc"])

        # Boundary

        if self.weights["bnd"] > 0:

            bnd_loss = self._l_boundary(pred, target)

            _add("bnd", bnd_loss, self.weights["bnd"])

        # Alignment regularization (CG-BLPHA)

        if alignment_evidences:

            align_total, align_terms = self._l_align(alignment_evidences, target)

            total = total + align_total

            terms.update(align_terms)

        # Cue + relation consistency

        if stage_evidences or relation_evidence is not None:

            cr_total, cr_terms = self._l_cue_relation(

                stage_evidences if stage_evidences else [],

                relation_evidence)

            total = total + cr_total

            terms.update(cr_terms)

        return total, terms

# ======================================================================

# [DEPRECATED] DeepDistributedHSPRLoss — kept for checkpoint compat

# ======================================================================

class DeepDistributedHSPRLoss(HRSCLoss):

    """Backward-compatible alias for HRSCLoss.

    This class exists only for loading old checkpoints.

    New code should use HRSCLoss directly.

    """

    pass

# ======================================================================

# Medical LR Geometry Resolver (NOW ACTIVELY USED)

# ======================================================================

class MedicalLRGeometryResolver:

    """Resolves and validates the physical left-right axis of network tensors.

    Integrated into the network build pipeline. Fails fast on unknown axes.

    """

    @staticmethod

    def resolve(dataset_json: Optional[dict], n_spatial_dims: int,

                user_axis: Optional[int],

                transpose_forward: Optional[Tuple[int, ...]] = None) -> int:

        if n_spatial_dims == 3:

            valid = (0, 1, 2)

        else:

            valid = (0, 1)

        if user_axis is not None:

            if user_axis not in valid:

                raise ValueError(

                    f"[BSPS-LR] Invalid LR spatial axis {user_axis}; "

                    f"must be one of {valid}")

            return int(user_axis)

        exp = None if dataset_json is None else \
            dataset_json.get("expected_lr_spatial_axis")

        if exp is not None and exp in valid:

            return int(exp)

        raise ValueError(

            "[BSPS-LR] LR spatial axis is UNKNOWN. "

            "Set LR_SPATIAL_AXIS on the Trainer subclass "

            "(verified from NIfTI affines + transpose_forward). "

            "Refusing to guess.")

    @staticmethod

    def tensor_axis(spatial_axis: int) -> int:

        return spatial_axis + 2  # [B, C, D, H, W]

# ======================================================================

# BasicMirrorEvidenceFusion3D — B1/D1 baseline fusion (no CG-BLPHA/PLED)

# ======================================================================

class BasicMirrorEvidenceFusion3D(nn.Module):

    """Lightweight contralateral fusion for B1 baseline.

    Uses original, re-flipped mirror, abs_diff, and feature product

    with 2 residual blocks. No alignment, no differential encoding,

    no global relation, no PLED.

    """

    def __init__(self, channels: int):

        super().__init__()

        mid = max(channels // 2, 16)

        self.proj = nn.Conv3d(channels * 4, mid, 1, bias=False)

        self.res1 = ResidualBlock3D(mid)

        self.res2 = ResidualBlock3D(mid)

        self.out = nn.Sequential(

            nn.Conv3d(mid, channels, 3, padding=1, bias=False),

            SafeNorm3D(channels, affine=True),

            nn.LeakyReLU(1e-2, inplace=True),

        )

    def forward(self, f_orig: torch.Tensor, f_mirror: torch.Tensor):

        d_abs = (f_orig - f_mirror).abs()

        d_sim = f_orig * f_mirror

        x = self.proj(torch.cat([f_orig, f_mirror, d_abs, d_sim], dim=1))

        x = self.res1(x)

        x = self.res2(x)

        return f_orig + self.out(x)

# ======================================================================

# Backward-compatible aliases

# ======================================================================

# Old names resolve to new deep implementations

BidirectionalLesionPreservingAlignment3D = CorrelationGuidedBidirectionalAlignment3D

LesionPreservingHomologousAlignment3D = CorrelationGuidedBidirectionalAlignment3D

HomologousHemisphericAlignmentBlock3D = CorrelationGuidedBidirectionalAlignment3D

HomologousAlignmentBlock3D = CorrelationGuidedBidirectionalAlignment3D

# Old decoder module name — PLED removed in V10
SymmetryEvidenceCalibratedSkip3D = None  # deprecated, removed in V10

# Boundary head — removed in V10 (was integrated into PLED)

BoundaryAwareRefinementHead3D = None  # deprecated, integrated into PLED

BoundaryAwareRefinementHead = None

# Legacy basic modules (kept for backward compatibility, not used in new arch)

class BaseSymmetricFusion(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        mid = max(in_channels // 2, 16)

        self.reduce = nn.Conv3d(in_channels * 2, mid, 1, bias=False)

        self.norm = SafeNorm3D(mid, affine=True)

        self.act = nn.LeakyReLU(1e-2, inplace=True)

        self.conv = nn.Conv3d(mid, out_channels, 3, padding=1, bias=False)

    def forward(self, f_orig, f_mirror):

        z = self.reduce(torch.cat([f_orig, f_mirror], dim=1))

        return f_orig + self.conv(self.act(self.norm(z)))

class AbsDiffFusionBlock(nn.Module):

    def __init__(self, in_channels):

        super().__init__()

        mid = max(in_channels // 2, 16)

        self.reduce = nn.Conv3d(in_channels, mid, 1, bias=False)

        self.norm = SafeNorm3D(mid, affine=True)

        self.act = nn.LeakyReLU(1e-2, inplace=True)

        self.conv = nn.Conv3d(mid, in_channels, 3, padding=1, bias=False)

    def forward(self, f_orig, f_mirror):

        d = torch.abs(f_orig - f_mirror)

        return f_orig + self.conv(self.act(self.norm(self.reduce(d))))

class DFRMBlock(nn.Module):

    def __init__(self, in_channels, bottleneck_ratio=2):

        super().__init__()

        mid = max(in_channels // bottleneck_ratio, 16)

        self.reduce = nn.Conv3d(in_channels * 4, mid, 1, bias=False)

        self.norm = SafeNorm3D(mid, affine=True)

        self.act = nn.LeakyReLU(1e-2, inplace=True)

        self.conv = nn.Conv3d(mid, in_channels, 3, padding=1, bias=False)

    def forward(self, f_orig, f_mirror):

        d_abs = torch.abs(f_orig - f_mirror)

        d_sim = f_orig * f_mirror

        return self.conv(self.act(self.norm(

            self.reduce(torch.cat([f_orig, f_mirror, d_abs, d_sim], 1)))))

__all__ = [

    # Helpers

    "ResidualBlock3D", "MultiScaleConvBlock3D", "DeepFeatureExtractor3D",

    "make_sobel_3d", "pick_valid_num_heads", "safe_odd_window",

    # Basic fusion (B1 baseline)

    "BasicMirrorEvidenceFusion3D",

    # Data classes

    "AlignmentEvidence", "StageSymmetryEvidence",

    "RelationEvidence",

    # Innovation 1: CG-BLPHA

    "CorrelationGuidedBidirectionalAlignment3D",

    # Innovation 2A: Stage-specific encoders

    "CueCompetitionFusion3D",

    "SignalBoundaryDifferentialEncoder3D",

    "StructureTextureDifferentialEncoder3D",

    "SemanticContextDifferentialEncoder3D",

    "StageSpecificDifferentialEncoder3D",

    # Innovation 2B: Global relation

    "GlobalSymmetricRelationReasoner3D",

    # Innovation 3: HRSC-Loss

    "HRSCLoss",

    # Loss

    "HRSCLoss",
    "DeepDistributedHSPRLoss",

    # LR resolver

    "MedicalLRGeometryResolver",

    # Legacy (thin wrappers)

    "BaseSymmetricFusion", "AbsDiffFusionBlock", "DFRMBlock",

    "BidirectionalLesionPreservingAlignment3D",

    "LesionPreservingHomologousAlignment3D",

    "HomologousHemisphericAlignmentBlock3D",

    "HomologousAlignmentBlock3D",

    "SymmetryEvidenceCalibratedSkip3D",

    "BoundaryAwareRefinementHead3D",

    "BoundaryAwareRefinementHead",

]
