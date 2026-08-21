"""
BSPS-Net V10 — Distributed Hemispheric Symmetry Prior Reasoning Network.

Complete data flow:

  X, Flip_LR(X)
    → Shared twin encoder → {F_o^l, F_m^l}
    → CG-BLPHA bidirectional alignment (deep-driven, coarse-to-fine)
    → H-DRE: Stage-specific differential encoding (every stage)
    → H-DRE: Global symmetric relation reasoning (bottleneck, dynamic 3D pos bias)
    → Standard multi-scale decoder + deep supervision
    → HRSC-Loss (training only)

Three innovations:
  1. CG-BLPHA: Correlation-Guided Bidirectional Lesion-Preserving Homologous Alignment
  2. H-DRE: Hierarchical Differential Representation & Global Relation Reasoning
  3. HRSC-Loss: Hemispheric Relation–Symmetry Constraint Loss

Formal variants (V10):
  B0 : official nnU-Net baseline (plain conv UNet, no mirror stream)
  B1 : shared mirror twin encoder
  B2 : B1 + CG-BLPHA alignment
  B3 : B1 + H-DRE (stage diff encoders + global relation)
  B4 : B1 + CG-BLPHA + H-DRE (full architecture, no auxiliary loss)
  B5 : B4 + HRSC-Loss (complete BSPS-Net, training with full HRSC supervision)
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.building_blocks.helper import (
    convert_dim_to_conv_op, get_matching_convtransp,
    get_matching_instancenorm, get_matching_pool_op)
from dynamic_network_architectures.building_blocks.residual import (
    BasicBlockD, StackedResidualBlocks)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    StackedConvBlocks)
from dynamic_network_architectures.initialization.weight_init import (
    init_last_bn_before_add_to_0)

def _safe_init_last_bn(module):
    """Safe version that only touches BasicBlockD-style modules."""
    try:
        init_last_bn_before_add_to_0(module)
    except (AttributeError, TypeError, RuntimeError):
        pass

from nnunetv2.nets.bsps_components import (
    AlignmentEvidence, StageSymmetryEvidence,
    RelationEvidence,
    CorrelationGuidedBidirectionalAlignment3D,
    StageSpecificDifferentialEncoder3D,
    GlobalSymmetricRelationReasoner3D,
    BasicMirrorEvidenceFusion3D,
    HRSCLoss,
    MedicalLRGeometryResolver,
    DeepFeatureExtractor3D,
)


# ======================================================================
# DistributedHSPRConfig — immutable network topology descriptor
# ======================================================================
@dataclass(frozen=True)
class DistributedHSPRConfig:
    shallow_offsets: Tuple[int, ...] = (3, 4)
    mid_offsets: Tuple[int, ...] = (1, 2)
    deep_offsets: Tuple[int, ...] = (0,)
    alignment_offsets: Tuple[int, ...] = (0, 1)       # deep stages for alignment
    difference_offsets: Tuple[int, ...] = (0, 1, 2, 3, 4)  # all stages for diff
    relation_stage: int = 0                            # bottleneck only
    decoder_evidence_stages: Tuple[int, ...] = (0, 1, 2, 3)  # all decoder stages
    # Ablation toggles — each controls exactly one mechanism
    use_alignment: bool = True
    use_lesion_preservation: bool = True
    use_bidirectional: bool = True
    use_heavy_cost_volume: bool = True          # deep stages use full cost volume
    use_stage_encoders: bool = True
    use_cue_competition: bool = True            # cue weights actually fuse
    use_reliability_modulation: bool = True
    use_global_relation: bool = True
    use_hrsc_loss: bool = False               # Innovation 3: HRSC-Loss (training only)
    max_displacement: float = 4.0
    search_radius: int = 2
    max_tokens: int = 16384

    def validate(self, n_stages: int) -> None:
        max_off = max(0, n_stages - 1)
        for name, offs in [("shallow_offsets", self.shallow_offsets),
                           ("mid_offsets", self.mid_offsets),
                           ("deep_offsets", self.deep_offsets),
                           ("alignment_offsets", self.alignment_offsets),
                           ("difference_offsets", self.difference_offsets)]:
            for off in offs:
                if not 0 <= off <= max_off:
                    raise ValueError(
                        f"DistributedHSPRConfig.{name}: offset {off} "
                        f"out of range [0,{max_off}] (n_stages={n_stages})")
        if not 0 <= self.relation_stage <= max_off:
            raise ValueError(f"relation_stage={self.relation_stage} out of range")

    def role_of(self, offset: int) -> str:
        if offset in self.deep_offsets:
            return "deep"
        if offset in self.mid_offsets:
            return "mid"
        return "shallow"

    def describe(self) -> str:
        return (f"dist-hspr-v9[align@{self.alignment_offsets} "
                f"diff@{self.difference_offsets} rel@{self.relation_stage} "
                f"dec@{self.decoder_evidence_stages} "
                f"bi={self.use_bidirectional} pres={self.use_lesion_preservation} "
                f"reli={self.use_reliability_modulation} "
                f"prog={self.use_progressive_evidence} "
                f"cf={self.use_counterfactual}]")


# ======================================================================
# BSPSConfig — immutable variant descriptor
# ======================================================================
@dataclass(frozen=True)
class BSPSConfig:
    variant: str = "B0"
    use_mirror_stream: bool = False
    # Innovation toggles (each controls exactly one mechanism)
    use_alignment: bool = False            # Innovation 1: CG-BLPHA
    use_stage_encoders: bool = False       # Innovation 2: H-DRE (diff encoding)
    use_global_relation: bool = False      # Innovation 2: H-DRE (relation reasoning)
    use_hrsc_loss: bool = False            # Innovation 3: HRSC-Loss (training only)
    # Sub-mechanism toggles (for fine-grained ablation)
    use_lesion_preservation: bool = True
    use_heavy_cost_volume: bool = True
    use_cue_competition: bool = True
    use_reliability_modulation: bool = True
    # Alignment params
    max_displacement: float = 4.0
    search_radius: int = 2
    max_tokens: int = 16384
    # Loss weights (for HRSC-Loss)
    loss_weights: Optional[Dict[str, float]] = None
    # Teacher forcing (decays over epochs, 0 at inference)
    teacher_forcing_ratio: float = 0.0
    # Misc
    deep_supervision: bool = True

    # ---- V10 Formal variants B0-B5 ----
    @classmethod
    def preset_b0(cls):
        return cls(variant="B0", deep_supervision=True)

    @classmethod
    def preset_b1(cls):
        return cls(variant="B1", use_mirror_stream=True, deep_supervision=True)

    @classmethod
    def preset_b2(cls):
        """B2: B1 + CG-BLPHA alignment (Innovation 1)."""
        return cls(variant="B2", use_mirror_stream=True,
                   use_alignment=True, deep_supervision=True)

    @classmethod
    def preset_b3(cls):
        """B3: B1 + H-DRE (Innovation 2: diff encoding + global relation)."""
        return cls(variant="B3", use_mirror_stream=True,
                   use_stage_encoders=True, use_global_relation=True,
                   deep_supervision=True)

    @classmethod
    def preset_b4(cls):
        """B4: B1 + CG-BLPHA + H-DRE (Innovations 1+2, no auxiliary loss)."""
        return cls(variant="B4", use_mirror_stream=True,
                   use_alignment=True, use_stage_encoders=True,
                   use_global_relation=True, deep_supervision=True)

    @classmethod
    def preset_b5(cls):
        """B5: B4 + HRSC-Loss (Innovation 3, complete BSPS-Net)."""
        return cls(variant="B5", use_mirror_stream=True,
                   use_alignment=True, use_stage_encoders=True,
                   use_global_relation=True, use_hrsc_loss=True,
                   deep_supervision=True)
        return cls(variant="D5", use_mirror_stream=True,
                   use_alignment=True, use_stage_encoders=True,
                   use_global_relation=True, use_evidence_decoder=True,
                   use_auxiliary_supervision=True,
                   deep_supervision=True)

    # ---- Legacy mappings (thin aliases for old checkpoint loading) ----
    @classmethod
    def preset_a0(cls): return cls.preset_b0()
    @classmethod
    def preset_a1(cls): return cls.preset_b1()

    @classmethod
    def preset_d1(cls): return cls.preset_b2()
    @classmethod
    def preset_d2(cls): return cls.preset_b3()
    @classmethod
    def preset_d3(cls): return cls.preset_b4()
    @classmethod
    def preset_d4(cls): return cls.preset_b4()
    @classmethod
    def preset_d5(cls): return cls.preset_b5()

    @classmethod
    def preset_e0(cls): return cls.preset_b1()
    @classmethod
    def preset_e1(cls): return cls.preset_b2()
    @classmethod
    def preset_e2(cls): return cls.preset_b3()
    @classmethod
    def preset_e3(cls): return cls.preset_b4()
    @classmethod
    def preset_e4(cls): return cls.preset_b4()
    @classmethod
    def preset_e5(cls): return cls.preset_b4()
    @classmethod
    def preset_e6(cls): return cls.preset_b5()

    @classmethod
    def from_variant(cls, variant: str):
        v = variant.upper()
        method = getattr(cls, f"preset_{v.lower()}", None)
        if method is None:
            raise ValueError(f"Unknown variant '{variant}'")
        return method()

    def to_dist_config(self, n_stages: int) -> DistributedHSPRConfig:
        """Build DistributedHSPRConfig from BSPSConfig, deriving stage offsets."""
        max_off = max(0, n_stages - 1)
        return DistributedHSPRConfig(
            shallow_offsets=tuple(i for i in range(n_stages)
                                  if i >= max_off - 1),
            mid_offsets=tuple(i for i in range(n_stages)
                              if 1 <= i <= max_off - 2),
            deep_offsets=(0, 1)[:min(2, n_stages)],
            alignment_offsets=(0, 1)[:min(2, n_stages)],
            difference_offsets=tuple(range(n_stages)),
            relation_stage=0,
            decoder_evidence_stages=tuple(range(max(1, n_stages - 1))),
            use_alignment=self.use_alignment,
            use_lesion_preservation=self.use_lesion_preservation,
            use_bidirectional=True,
            use_heavy_cost_volume=self.use_heavy_cost_volume,
            use_stage_encoders=self.use_stage_encoders,
            use_cue_competition=self.use_cue_competition,
            use_reliability_modulation=self.use_reliability_modulation,
            use_global_relation=self.use_global_relation,
            use_hrsc_loss=self.use_hrsc_loss,
            max_displacement=self.max_displacement,
            search_radius=self.search_radius,
            max_tokens=self.max_tokens,
        )


# ======================================================================
# BSPSNet V9 — Complete Distributed HSPR Network
# ======================================================================
class BSPSNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: type, kernel_sizes, strides, n_conv_per_stage,
                 n_conv_per_stage_decoder, conv_bias: bool = False,
                 norm_op=None, norm_op_kwargs=None, dropout_op=None,
                 dropout_op_kwargs=None, nonlin=None, nonlin_kwargs=None,
                 lr_tensor_axis: int = -1, config: Optional[BSPSConfig] = None,
                 pool_type: str = "conv",
                 lesion_prior_provider: Optional[Callable] = None):
        super().__init__()
        # Normalize args
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)

        self.config = config or BSPSConfig.preset_b0()
        cfg = self.config
        self.lr_tensor_axis = int(lr_tensor_axis)
        if cfg.use_mirror_stream and self.lr_tensor_axis < 2:
            raise ValueError(
                f"lr_tensor_axis must be >= 2 (got {self.lr_tensor_axis})")
        self.n_stages = int(n_stages)
        self.features_per_stage = list(features_per_stage)
        self.strides = list(strides)
        self.kernel_sizes = list(kernel_sizes)
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs or {}
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs or {}
        self.deep_supervision = cfg.deep_supervision
        self.lesion_prior_provider = lesion_prior_provider
        self.num_classes = num_classes

        # Build distributed config
        dist = cfg.to_dist_config(n_stages)
        dist.validate(n_stages)

        # ---- Shared twin encoder ----
        pool_op = get_matching_pool_op(conv_op, pool_type) if pool_type != "conv" else None
        stem_ch = features_per_stage[0]
        self.stem = StackedConvBlocks(
            1, conv_op, input_channels, stem_ch, kernel_sizes[0], 1,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs)
        self.encoder_stages = nn.ModuleList()
        enc_in = stem_ch
        for s in range(n_stages):
            sf = strides[s] if pool_op is None else 1
            stage = StackedResidualBlocks(
                n_conv_per_stage[s], conv_op, enc_in, features_per_stage[s],
                kernel_sizes[s], sf, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=BasicBlockD)
            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)
            self.encoder_stages.append(stage)
            enc_in = features_per_stage[s]

        # ---- Build distributed modules ----
        self._build_distributed_modules(features_per_stage, dist, num_classes)

        # ---- Decoder (standard nnU-Net style) ----
        tp_op = get_matching_convtransp(conv_op=conv_op)
        self.decoder_transpconvs = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for s in range(1, n_stages):
            ib, isk = features_per_stage[-s], features_per_stage[-(s + 1)]
            self.decoder_transpconvs.append(
                tp_op(ib, isk, strides[-s], strides[-s], bias=conv_bias))
            self.decoder_stages.append(StackedResidualBlocks(
                n_conv_per_stage_decoder[s - 1], conv_op, 2 * isk, isk,
                kernel_sizes[-(s + 1)], 1, conv_bias, norm_op, norm_op_kwargs,
                dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))
            self.seg_layers.append(conv_op(isk, num_classes, 1, 1, 0, bias=True))

        # ---- HRSC-Loss module (training only, Innovation 3) ----
        if cfg.use_hrsc_loss:
            lr_spatial = self.lr_tensor_axis - 2 if self.lr_tensor_axis >= 2 else None
            self.hrsc_loss = HRSCLoss(cfg.loss_weights, lr_spatial_axis=lr_spatial)
        else:
            self.hrsc_loss = None

        self.apply(_safe_init_last_bn)

    # ------------------------------------------------------------------
    def _build_distributed_modules(self, features: List[int],
                                   dist: DistributedHSPRConfig,
                                   num_classes: int):
        """Build all distributed modules with clean ablation: if a mechanism
        is off, its module is None and no parameters exist."""
        n = len(features)
        ap = [features[0]] + list(features)  # [stem_ch, s0, ..., s_{n-1}]

        # --- Stage-specific differential encoders ---
        self.stage_encoders = nn.ModuleList()
        for i, ch in enumerate(ap):
            if i == 0:  # stem has no encoder
                self.stage_encoders.append(None)
                continue
            offset = (n - 1) - (i - 1)
            role = dist.role_of(max(0, offset))
            if dist.use_stage_encoders:
                self.stage_encoders.append(
                    StageSpecificDifferentialEncoder3D(ch, role))
            else:
                self.stage_encoders.append(None)

        # --- Alignment modules (CG-BLPHA) ---
        self.align_modules = nn.ModuleList()
        for i, ch in enumerate(ap):
            offset = (n - 1) - (i - 1) if i >= 1 else n - 1
            if dist.use_alignment and max(0, offset) in dist.alignment_offsets \
                    and i >= 1:
                use_heavy = dist.use_heavy_cost_volume and offset <= 1
                self.align_modules.append(
                    CorrelationGuidedBidirectionalAlignment3D(
                        ch, max_displacement=dist.max_displacement,
                        search_radius=dist.search_radius,
                        use_lesion_preservation=dist.use_lesion_preservation,
                        use_heavy_cost_volume=use_heavy))
            else:
                self.align_modules.append(None)

        # --- Global relation reasoner ---
        if dist.use_global_relation:
            self.relation_reasoner = GlobalSymmetricRelationReasoner3D(
                features[-1],
                use_reliability_modulation=dist.use_reliability_modulation,
                max_tokens=dist.max_tokens)
        else:
            self.relation_reasoner = None

        # --- Basic mirror fusion (B1: simple mirror, D1: aligned mirror → same fusion) ---
        self.mirror_fusions = nn.ModuleList()
        for i, ch in enumerate(ap):
            # B1/D1 both use mirror_fusions on deep+mid stages when mirror stream is active
            # and no stage_encoders present (stage_encoders handle fusion at D2+)
            use_fusion = (self.config.use_mirror_stream
                          and not dist.use_stage_encoders
                          and i >= 1
                          and max(0, (n - 1) - (i - 1)) in (0, 1, 2))
            if use_fusion:
                self.mirror_fusions.append(BasicMirrorEvidenceFusion3D(ch))
            else:
                self.mirror_fusions.append(None)

        # --- Bottleneck projection for decoder ---
        self.bn_proj = DeepFeatureExtractor3D(
            features[-1], features[-1], num_res_blocks=1) \
            if dist.use_global_relation or dist.use_stage_encoders else None

        # --- Per-stage delta projections (skip enhancement) ---
        self.stage_delta_projs = nn.ModuleList()
        for i, ch in enumerate(ap):
            if 0 < i < len(ap) - 1 and dist.use_stage_encoders:
                self.stage_delta_projs.append(
                    nn.Conv3d(ch, ch, 1, bias=False))
            else:
                self.stage_delta_projs.append(nn.Identity())

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = [self.stem(x)]
        for s in self.encoder_stages:
            skips.append(s(skips[-1]))
        return skips  # [stem, s0, ..., bn]

    def _encode_original_and_mirror(self, x: torch.Tensor
                                    ) -> Tuple[List[torch.Tensor],
    List[torch.Tensor]]:
        x_m = torch.flip(x, dims=[self.lr_tensor_axis])
        skips_o = self._encode(x)
        skips_m = [torch.flip(f, dims=[self.lr_tensor_axis])
                   for f in self._encode(x_m)]
        return skips_o, skips_m

    @staticmethod
    def _get_or_fallback(d, key, fallback_key, fallback_dict):
        """Safely get value from dict with fallback — handles tensor values."""
        v = d.get(key)
        if v is not None:
            return v
        return fallback_dict.get(fallback_key)

    # ------------------------------------------------------------------
    def _run_distributed(self, skips_o: List[torch.Tensor],
                         skips_m: List[torch.Tensor],
                         collect_aux: bool, detach_aux: bool):
        """Run the full distributed symmetry reasoning pipeline.

        Returns typed evidence objects (NOT aux dict for core flow).
        Aux dict is built separately for logging/visualization only.
        """
        cfg = self.config
        dist = cfg.to_dist_config(self.n_stages)
        n_enc = self.n_stages
        n_skip = len(skips_o)

        def _maybe(t):
            return t.detach() if (t is not None and detach_aux) else t

        # --- Phase 1: Deep-driven bidirectional alignment (coarse-to-fine) ---
        bn_idx = n_skip - 1
        aligned_mirrors = list(skips_m)
        alignment_evidences: List[Optional[AlignmentEvidence]] = [None] * n_skip
        reli_by_stage: Dict[int, torch.Tensor] = {}
        pres_by_stage: Dict[int, torch.Tensor] = {}
        coarse_flow = None
        aux: Dict[str, torch.Tensor] = {}

        for i in range(n_skip - 1, 0, -1):
            mod = self.align_modules[i]
            if mod is None:
                continue
            offset = (n_enc - 1) - (i - 1)
            lesion_prior = None
            if self.lesion_prior_provider is not None:
                lesion_prior = self.lesion_prior_provider(offset)
            coarse_up = coarse_flow
            aev = mod(skips_o[i], skips_m[i],
                      coarse_offset=coarse_up,
                      lesion_prior=lesion_prior,
                      teacher_forcing_ratio=cfg.teacher_forcing_ratio)
            alignment_evidences[i] = aev
            aligned_mirrors[i] = aev.aligned_mirror
            reli_by_stage[i] = aev.geometric_reliability
            pres_by_stage[i] = aev.lesion_preservation_confidence
            if i == bn_idx and coarse_flow is None:
                coarse_flow = aev.flow_o2m
            if collect_aux:
                aux[f"stg{i}_flow_o2m"] = _maybe(aev.flow_o2m)
                aux[f"stg{i}_geometric_reliability"] = _maybe(
                    aev.geometric_reliability)
                aux[f"stg{i}_lesion_preservation"] = _maybe(
                    aev.lesion_preservation_confidence)
                aux[f"stg{i}_cycle_error"] = _maybe(aev.cycle_error)
                aux[f"stg{i}_inverse_error"] = _maybe(aev.inverse_error)
                aux[f"stg{i}_match_entropy"] = _maybe(aev.match_entropy)

        # --- Phase 2: Stage-specific differential encoding ---
        stage_evidences: List[Optional[StageSymmetryEvidence]] = [None] * n_skip
        for i in range(1, n_skip):  # skip stem
            mod = self.stage_encoders[i]
            reli = self._get_or_fallback(reli_by_stage, i, bn_idx, reli_by_stage)
            pres = self._get_or_fallback(pres_by_stage, i, bn_idx, pres_by_stage)
            if mod is not None:
                result = mod(skips_o[i], aligned_mirrors[i], reli, pres)
                if mod.role == "deep":
                    delta, uncertainty, wg, ws, wf = result
                else:
                    delta, wg, ws, wf = result
                    uncertainty = None
                stage_evidences[i] = StageSymmetryEvidence(
                    original_feature=skips_o[i],
                    aligned_mirror_feature=aligned_mirrors[i],
                    differential_feature=delta,
                    geometric_reliability=reli,
                    lesion_preservation_confidence=pres,
                    cue_features=None,
                    cue_weights_global=wg,
                    cue_weights_spatial=ws,
                    boundary_evidence=None,
                    uncertainty=uncertainty,
                    stage_index=i,
                )
                if collect_aux:
                    aux[f"stg{i}_differential_feature"] = _maybe(delta)
                    aux[f"stg{i}_cue_weights"] = _maybe(wg)
                    if uncertainty is not None:
                        aux[f"stg{i}_uncertainty"] = _maybe(uncertainty)

        # --- Phase 3: Bottleneck global relation reasoning ---
        relation_evidence = None
        if self.relation_reasoner is not None:
            bn_delta = None
            if stage_evidences[bn_idx] is not None:
                bn_delta = stage_evidences[bn_idx].differential_feature
            bn_reli = reli_by_stage.get(bn_idx)
            bn_match_entropy = None
            bn_flow = None
            if alignment_evidences[bn_idx] is not None:
                bn_match_entropy = alignment_evidences[bn_idx].match_entropy
                bn_flow = alignment_evidences[bn_idx].flow_o2m
            relation_evidence = self.relation_reasoner(
                skips_o[bn_idx], aligned_mirrors[bn_idx],
                geometric_reliability=bn_reli,
                differential_feature=bn_delta,
                match_entropy=bn_match_entropy,
                flow_o2m=bn_flow)
            if collect_aux:
                aux["bn_relation_context"] = _maybe(
                    relation_evidence.relation_context)
                aux["bn_attention_entropy"] = _maybe(
                    relation_evidence.attention_entropy)
                aux["bn_attention_concentration"] = _maybe(
                    relation_evidence.attention_concentration)

        # --- Phase 4: Build enhanced skips for decoder ---
        enhanced_skips = list(skips_o)
        for i in range(1, n_skip - 1):
            sev = stage_evidences[i]
            if sev is not None and sev.differential_feature is not None:
                proj = self.stage_delta_projs[i]
                delta_proj = (proj(sev.differential_feature)
                              if not isinstance(proj, nn.Identity)
                              else sev.differential_feature)
                enhanced_skips[i] = skips_o[i] + delta_proj
            elif self.mirror_fusions[i] is not None:
                # B1: basic mirror fusion (no alignment, no differential)
                enhanced_skips[i] = self.mirror_fusions[i](
                    skips_o[i], aligned_mirrors[i])

        # Enhanced bottleneck
        enhanced_bn = skips_o[bn_idx]
        if relation_evidence is not None:
            enhanced_bn = enhanced_bn + relation_evidence.relation_context
        if stage_evidences[bn_idx] is not None:
            enhanced_bn = enhanced_bn + stage_evidences[bn_idx].differential_feature
        if self.bn_proj is not None:
            enhanced_bn = self.bn_proj(enhanced_bn)
        enhanced_skips[bn_idx] = enhanced_bn

        return (enhanced_skips, aligned_mirrors, alignment_evidences,
                stage_evidences, relation_evidence, reli_by_stage,
                pres_by_stage, aux)

    # ------------------------------------------------------------------
    def _decode_standard(self, enhanced: List[torch.Tensor],
                         collect_aux: bool, detach_aux: bool):
        """Standard nnU-Net style decoder with H-DRE enhanced skip connections.

        Skip features are enhanced by adding the H-DRE differential feature
        (when available) to the original skip, providing richer symmetry-aware
        features without the complexity of PLED evidence disentanglement.
        """
        def _maybe(t):
            return t.detach() if (t is not None and detach_aux) else t

        seg = []
        dec_feats = []
        x = enhanced[-1]
        aux: Dict[str, torch.Tensor] = {}

        for s in range(len(self.decoder_transpconvs)):
            xu = self.decoder_transpconvs[s](x)
            skip_idx = len(enhanced) - 2 - s
            skip_orig = enhanced[skip_idx]
            xu = self._match_spatial(xu, skip_orig)
            x = torch.cat([xu, skip_orig], dim=1)
            x = self.decoder_stages[s](x)
            dec_feats.append(x)
            if self.deep_supervision:
                seg.append(self.seg_layers[s](x))
            elif s == len(self.decoder_transpconvs) - 1:
                seg.append(self.seg_layers[-1](x))

        seg = seg[::-1]
        return seg, dec_feats[-1] if dec_feats else x, aux

    # ------------------------------------------------------------------
    def _decode_plain(self, skips: List[torch.Tensor]):
        """Plain decoder for B0 (no mirror, no symmetry modules)."""
        seg = []
        dec_feats = []
        x = skips[-1]
        for s in range(len(self.decoder_transpconvs)):
            xu = self.decoder_transpconvs[s](x)
            sk = skips[-(s + 2)]
            xu = self._match_spatial(xu, sk)
            x = torch.cat([xu, sk], dim=1)
            x = self.decoder_stages[s](x)
            dec_feats.append(x)
            if self.deep_supervision:
                seg.append(self.seg_layers[s](x))
            elif s == len(self.decoder_transpconvs) - 1:
                seg.append(self.seg_layers[-1](x))
        return seg[::-1], dec_feats[-1] if dec_feats else x

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        seg, _ = self._forward_impl(x, collect_aux=False)
        return seg

    def forward_train(self, x: torch.Tensor):
        return self._forward_impl(x, collect_aux=True, detach_aux=False)

    def forward_with_aux(self, x: torch.Tensor, detach: bool = True):
        return self._forward_impl(x, collect_aux=True, detach_aux=detach)

    def _forward_impl(self, x: torch.Tensor, collect_aux: bool = False,
                      detach_aux: bool = True):
        cfg = self.config

        # B0 path: plain nnU-Net (no mirror stream)
        if not cfg.use_mirror_stream:
            seg, _ = self._decode_plain(self._encode(x))
            if self.deep_supervision:
                return seg, {}
            return seg, {}

        # Full distributed HSPR path
        skips_o, skips_m = self._encode_original_and_mirror(x)
        (enhanced, aligned_mirrors, alignment_evidences,
         stage_evidences, relation_evidence, reli_by_stage,
         pres_by_stage, aux) = self._run_distributed(
            skips_o, skips_m, collect_aux, detach_aux)

        seg, _, dec_aux = self._decode_standard(
            enhanced, collect_aux, detach_aux)
        aux.update(dec_aux)

        # Store evidence for HRSC-Loss computation (training only)
        if collect_aux:
            aux["_alignment_evidences"] = alignment_evidences
            aux["_stage_evidences"] = stage_evidences
            aux["_relation_evidence"] = relation_evidence
            # Build feature lists for HRSC-Loss
            n_skip = len(enhanced)
            features_o = []
            features_m_aligned = []
            for i in range(1, n_skip):
                sev = stage_evidences[i] if i < len(stage_evidences) else None
                features_o.append(enhanced[i])
                fm = aligned_mirrors[i] if i < len(aligned_mirrors) else enhanced[i]
                features_m_aligned.append(fm)

        if self.deep_supervision:
            return seg, aux
        return seg[0] if isinstance(seg, list) else seg, aux

    # ------------------------------------------------------------------
    @staticmethod
    def _match_spatial(src: torch.Tensor, ref: torch.Tensor):
        if src.shape[2:] == ref.shape[2:]:
            return src
        sl = [slice(None), slice(None)]
        for ss, rs in zip(src.shape[2:], ref.shape[2:]):
            sl.append(slice((ss - rs) // 2, (ss + rs) // 2) if ss > rs
                      else slice(None))
        src = src[tuple(sl)]
        pads = []
        for ss, rs in zip(reversed(src.shape[2:]), reversed(ref.shape[2:])):
            if ss < rs:
                diff = rs - ss
                pads.extend([diff // 2, diff - diff // 2])
            else:
                pads.extend([0, 0])
        return F.pad(src, pads, mode="reflect") if any(pads) else src

    def _finalise(self, seg):
        return seg if self.deep_supervision else seg[0]

    def set_deep_supervision(self, enabled: bool):
        self.deep_supervision = bool(enabled)

    def set_teacher_forcing_ratio(self, ratio: float):
        """Update teacher forcing ratio (call each epoch)."""
        self.config = dataclasses.replace(
            self.config, teacher_forcing_ratio=ratio)

    # ------------------------------------------------------------------
    def get_architecture_manifest(self) -> Dict:
        cfg = self.config
        dist = cfg.to_dist_config(self.n_stages)
        total = sum(p.numel() for p in self.parameters())
        active_modules = {}
        if cfg.use_mirror_stream:
            active_modules["SharedMirrorEncoder"] = True
        if cfg.use_alignment:
            active_modules["CG_BLPHA_Alignment"] = True
        if cfg.use_stage_encoders:
            active_modules["H_DRE_DiffEncoders"] = True
        if cfg.use_global_relation:
            active_modules["H_DRE_GlobalRelation"] = True
        if cfg.use_hrsc_loss:
            active_modules["HRSC_Loss"] = True
        return {
            "network_name": "BSPS-Net V10",
            "variant": cfg.variant,
            "innovations": [
                "CG-BLPHA: Correlation-Guided Bidirectional Lesion-Preserving "
                "Homologous Alignment",
                "H-DRE: Hierarchical Differential Representation & "
                "Global Symmetric Relation Reasoning",
                "HRSC-Loss: Hemispheric Relation–Symmetry Constraint Loss",
            ],
            "encoder_stages": self.n_stages,
            "encoder_feature_channels": list(self.features_per_stage),
            "stage_roles": {str(off): dist.role_of(off)
                            for off in range(self.n_stages)},
            "alignment_offsets": list(dist.alignment_offsets),
            "difference_offsets": list(dist.difference_offsets),
            "relation_stage": dist.relation_stage,
            "decoder_evidence_stages": list(dist.decoder_evidence_stages),
            "active_modules": active_modules,
            "deep_supervision_outputs": len(self.seg_layers),
            "deep_supervision": self.deep_supervision,
            "config_summary": {
                "alignment": cfg.use_alignment,
                "stage_encoders": cfg.use_stage_encoders,
                "global_relation": cfg.use_global_relation,
                "evidence_decoder": cfg.use_evidence_decoder,
                "auxiliary_supervision": cfg.use_auxiliary_supervision,
                "cue_competition": cfg.use_cue_competition,
                "reliability_modulation": cfg.use_reliability_modulation,
                "progressive_evidence": cfg.use_progressive_evidence,
                "counterfactual": cfg.use_counterfactual,
                "orthogonality": cfg.use_orthogonality,
                "boundary_integrated": cfg.use_boundary_integrated,
                "nuisance_augmentation": cfg.use_nuisance_augmentation,
                "heavy_cost_volume": cfg.use_heavy_cost_volume,
                "teacher_forcing_ratio": cfg.teacher_forcing_ratio,
            },
            "parameter_counts": {
                "total": total,
                "trainable": sum(p.numel() for p in self.parameters()
                                 if p.requires_grad),
            },
        }

    def describe_architecture(self) -> str:
        m = self.get_architecture_manifest()
        lines = [
            f"BSPS-Net V9 variant {m['variant']}",
            f"  Innovations:",
        ]
        for inn in m["innovations"]:
            lines.append(f"    - {inn}")
        lines += [
            f"  encoder stages: {m['encoder_stages']} "
            f"(channels {m['encoder_feature_channels']})",
            f"  stage roles: {m['stage_roles']}",
            f"  alignment@{m['alignment_offsets']} "
            f"diff@{m['difference_offsets']} "
            f"relation@{m['relation_stage']} "
            f"decoder-evidence@{m['decoder_evidence_stages']}",
            "  active: " + ", ".join(
                f"{k}={v}" for k, v in m["active_modules"].items()),
            f"  DS outputs: {m['deep_supervision_outputs']}",
            f"  parameters: {m['parameter_counts']['total']:,} total  "
            f"({m['parameter_counts']['trainable']:,} trainable)",
        ]
        return "\n".join(lines)


# ======================================================================
# build_bsps_network — factory function
# ======================================================================
def build_bsps_network(plans_manager, dataset_json, configuration_manager,
                       num_input_channels, num_output_channels, lr_tensor_axis,
                       config, deep_supervision=True,
                       lesion_prior_provider=None):
    dim = len(configuration_manager.patch_size)
    if dim != 3:
        raise ValueError(f"BSPS-Net requires 3D, got {dim}D")

    # --- LR axis resolution (ACTIVELY USED, not dead code) ---
    if config.use_mirror_stream:
        lr_spatial = lr_tensor_axis - 2 if lr_tensor_axis >= 2 else None
        lr_resolved = MedicalLRGeometryResolver.resolve(
            dataset_json=dataset_json,
            n_spatial_dims=3,
            user_axis=lr_spatial,
            transpose_forward=configuration_manager.transpose_forward
            if hasattr(configuration_manager, 'transpose_forward') else None)
        if lr_resolved != lr_spatial and lr_spatial is not None:
            raise ValueError(
                f"[BSPS-LR] Mismatch: trainer says LR spatial axis "
                f"={lr_spatial}, resolver says {lr_resolved}")
        if lr_tensor_axis != lr_resolved + 2:
            raise ValueError(
                f"[BSPS-LR] lr_tensor_axis={lr_tensor_axis} inconsistent "
                f"with resolved spatial axis={lr_resolved}")

    conv_op = convert_dim_to_conv_op(dim)
    instnorm = get_matching_instancenorm(dimension=dim)
    _conf = getattr(configuration_manager, "configuration", {})

    def _cm(k, d=None):
        return getattr(configuration_manager, k, _conf.get(k, d))

    def _cml(k, d):
        v = _cm(k, None)
        return v if (v is not None and len(v) > 0) else d

    pool = [tuple(int(x) for x in s)
            for s in _cml("pool_op_kernel_sizes",
                          [[1] * 3, [2] * 3, [2] * 3, [2] * 3, [2] * 3, [2] * 3])]
    ck = [tuple(int(x) for x in s)
          for s in _cml("conv_kernel_sizes", [[3] * 3] * 6)]
    ns = len(ck)
    bf = _cm("UNet_base_num_features", 32)
    mf = _cm("unet_max_num_features", 320)

    return BSPSNet(
        input_channels=num_input_channels,
        num_classes=num_output_channels, n_stages=ns,
        features_per_stage=[min(bf * (2 ** i), mf) for i in range(ns)],
        conv_op=conv_op, kernel_sizes=ck, strides=pool,
        n_conv_per_stage=list(_cml("n_conv_per_stage_encoder", [2] * ns)),
        n_conv_per_stage_decoder=list(
            _cml("n_conv_per_stage_decoder", [2] * (ns - 1))),
        conv_bias=True, norm_op=instnorm,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None, dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"negative_slope": 1e-2, "inplace": True},
        lr_tensor_axis=lr_tensor_axis, config=config,
        pool_type="conv",
        lesion_prior_provider=lesion_prior_provider)


# Backward-compatible aliases
DistributedHSPRNetwork = BSPSNet
BSPSAuxiliaryLoss = HRSCLoss
DistributedHSPRLoss = HRSCLoss
