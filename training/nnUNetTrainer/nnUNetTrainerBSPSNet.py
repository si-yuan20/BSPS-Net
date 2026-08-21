"""
BSPS-Net V10 Trainer — B0-B5 ablation trainers.

Usage::

    nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerBSPSB4
    nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerBSPSB5

LR_SPATIAL_AXIS MUST be set per dataset. B0 is exempt.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.nets.bsps_net import (
    BSPSConfig, build_bsps_network,
)
from nnunetv2.nets.bsps_components import (
    HRSCLoss,
)
from nnunetv2.training.nnUNetTrainer.custom_base_chain_utils import (
    propagate_deep_supervision_toggle,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context


class nnUNetTrainerBSPSNet(nnUNetTrainer):
    """Base trainer for all BSPS-Net V10 variants.

    ``LR_SPATIAL_AXIS`` is the index of the left-right axis in the
    spatial dimensions (0, 1, or 2).  The tensor axis used for flip is
    ``lr_spatial_axis + 2``.

    Default is ``None`` — fail-fast if not set for mirror variants.
    """

    BSPS_VARIANT: str = "B4"
    LR_SPATIAL_AXIS: Optional[int] = None

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.bsps_variant = self.BSPS_VARIANT
        self.lr_spatial_axis = self.LR_SPATIAL_AXIS
        self.enable_deep_supervision = True

        cfg = BSPSConfig.from_variant(self.bsps_variant)

        # HRSC-Loss module (B5 — Innovation 3, training only)
        self.hrsc_loss = HRSCLoss(cfg.loss_weights,
                                  lr_spatial_axis=self.lr_spatial_axis) \
            if cfg.use_hrsc_loss else None

        # LR axis validation
        if cfg.use_mirror_stream:
            if self.lr_spatial_axis is None or self.lr_spatial_axis not in (0, 1, 2):
                raise ValueError(
                    f"[BSPS] LR_SPATIAL_AXIS must be 0, 1, or 2 for variant "
                    f"'{self.bsps_variant}'.\n"
                    f"  Current value: {self.lr_spatial_axis}\n"
                    f"  Trainer class: {self.__class__.__name__}\n"
                    f"  Dataset ID: {dataset_json.get('dataset_id', 'unknown')}\n"
                    f"  Fix: set LR_SPATIAL_AXIS on your trainer subclass."
                )
            self.lr_tensor_axis = self.lr_spatial_axis + 2
        else:
            self.lr_tensor_axis = 2

        self.print_to_log_file(
            f"[BSPS-V9] variant={self.bsps_variant}  "
            f"lr_spatial_axis={self.lr_spatial_axis}  "
            f"lr_tensor_axis={self.lr_tensor_axis}  "
            f"aux_loss={'enabled' if self.aux_loss is not None else 'disabled'}  "
            f"nuisance_aug={'enabled' if self.nuisance_generator is not None else 'disabled'}"
        )

        # ---- LR axis direction audit ----
        exp_axis = dataset_json.get("expected_lr_spatial_axis")
        if cfg.use_mirror_stream:
            if exp_axis is not None and exp_axis != self.lr_spatial_axis:
                raise ValueError(
                    f"[BSPS] LR axis conflict for variant '{self.bsps_variant}': "
                    f"dataset.json declares expected_lr_spatial_axis={exp_axis} "
                    f"but trainer LR_SPATIAL_AXIS={self.lr_spatial_axis}.\n"
                    f"  Trainer: {self.__class__.__name__}\n"
                    f"  Dataset ID: {dataset_json.get('dataset_id', 'unknown')}\n"
                    f"  Fix: align LR_SPATIAL_AXIS with the verified physical axis."
                )
            self.print_to_log_file(
                f"[BSPS-LR] LR axis AUDIT: spatial_axis={self.lr_spatial_axis} "
                f"tensor_axis={self.lr_tensor_axis} "
                f"dataset_expected={exp_axis} "
                f"mode='user-confirmed-from-NIfTI-affine+transpose_forward'"
            )

    def build_network_architecture(self, architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True) -> nn.Module:
        bsps_cfg = BSPSConfig.from_variant(self.bsps_variant)
        self.print_to_log_file(
            f"[BSPS-V9] Building variant {bsps_cfg.variant}: "
            f"mirror={bsps_cfg.use_mirror_stream} "
            f"align={bsps_cfg.use_alignment} "
            f"stage_enc={bsps_cfg.use_stage_encoders} "
            f"relation={bsps_cfg.use_global_relation} "
            f"evidence_dec={bsps_cfg.use_evidence_decoder} "
            f"aux_sup={bsps_cfg.use_auxiliary_supervision} "
            f"in={num_input_channels} out={num_output_channels}"
        )

        if bsps_cfg.variant == "B0":
            self.print_to_log_file("[BSPS-B0] Official nnU-Net PlainConvUNet baseline.")
            return nnUNetTrainer.build_network_architecture(
                self, architecture_class_name, arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels, num_output_channels, enable_deep_supervision)

        return build_bsps_network(
            plans_manager=self.plans_manager, dataset_json=self.dataset_json,
            configuration_manager=self.configuration_manager,
            num_input_channels=num_input_channels,
            num_output_channels=num_output_channels,
            lr_tensor_axis=self.lr_tensor_axis, config=bsps_cfg,
            deep_supervision=enable_deep_supervision)

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = enabled
        if self.network is not None:
            propagate_deep_supervision_toggle(self.network, enabled)

    # ------------------------------------------------------------------
    def train_step(self, batch: dict) -> dict:
        """Training step with HRSC-Loss (Innovation 3, training only)."""
        if self.hrsc_loss is None:
            return super().train_step(batch)

        data = batch["data"]
        target = batch["target"]
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        seg_target = target[0] if isinstance(target, list) else target
        seg_bin = (seg_target > 0).float()

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True) \
                if self.device.type == "cuda" else dummy_context():
            if hasattr(self.network, "forward_train"):
                output, aux = self.network.forward_train(data)
            else:
                output = self.network(data)
                aux = {}

            # Primary segmentation loss (nnU-Net default)
            l = self.loss(output, target)

            # HRSC-Loss (Innovation 3: hemispheric symmetry supervision)
            pred_main = output[0] if isinstance(output, list) else output
            alignment_evidences = aux.get("_alignment_evidences", [])
            alignment_evidences = [a for a in alignment_evidences if a is not None]
            stage_evidences = aux.get("_stage_evidences", [])
            stage_evidences = [s for s in stage_evidences if s is not None]
            relation_evidence = aux.get("_relation_evidence")

            l_hrsc, terms = self.hrsc_loss(
                pred=pred_main, target=seg_bin,
                features_o=None,  # populated from stage_evidences inside loss
                features_m_aligned=None,
                reliabilities=None,
                stage_evidences=stage_evidences if stage_evidences else None,
                relation_evidence=relation_evidence,
                alignment_evidences=alignment_evidences if alignment_evidences else None,
                lr_spatial_axis=self.lr_spatial_axis)
            l = l + l_hrsc
            if terms:
                self.print_to_log_file(
                    "[BSPS-HRSC] " + " ".join(
                        f"{k}={v:.4f}" for k, v in terms.items()))

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {"loss": l.detach().cpu().numpy()}

    # ------------------------------------------------------------------
    def on_epoch_end(self):
        """Decay teacher forcing ratio over epochs."""
        super().on_epoch_end()
        if hasattr(self.network, "set_teacher_forcing_ratio"):
            cfg = BSPSConfig.from_variant(self.bsps_variant)
            if cfg.use_alignment:
                total_epochs = self.num_epochs
                current_epoch = self.current_epoch
                if current_epoch < total_epochs * 0.8:
                    ratio = 0.5 * (1.0 - current_epoch / (total_epochs * 0.8))
                else:
                    ratio = 0.0
                self.network.set_teacher_forcing_ratio(ratio)
                if current_epoch % 10 == 0:
                    self.print_to_log_file(
                        f"[BSPS] teacher_forcing_ratio = {ratio:.3f}")


# ======================================================================
# V10 Formal B-Series Trainers
# ======================================================================

class nnUNetTrainerBSPSB0(nnUNetTrainerBSPSNet):
    """B0: official nnU-Net baseline."""
    BSPS_VARIANT = "B0"
    LR_SPATIAL_AXIS = None


class nnUNetTrainerBSPSB1(nnUNetTrainerBSPSNet):
    """B1: shared mirror twin encoder."""
    BSPS_VARIANT = "B1"


class nnUNetTrainerBSPSB2(nnUNetTrainerBSPSNet):
    """B2: B1 + CG-BLPHA (Innovation 1)."""
    BSPS_VARIANT = "B2"


class nnUNetTrainerBSPSB3(nnUNetTrainerBSPSNet):
    """B3: B1 + H-DRE (Innovation 2)."""
    BSPS_VARIANT = "B3"


class nnUNetTrainerBSPSB4(nnUNetTrainerBSPSNet):
    """B4: B1 + CG-BLPHA + H-DRE (Innovations 1+2, no auxiliary loss)."""
    BSPS_VARIANT = "B4"


class nnUNetTrainerBSPSB5(nnUNetTrainerBSPSNet):
    """B5: B4 + HRSC-Loss (Innovation 3, complete BSPS-Net)."""
    BSPS_VARIANT = "B5"


# ======================================================================
# Legacy Trainer Aliases (thin wrappers for old checkpoint loading)
# ======================================================================

class nnUNetTrainerBSPSA0(nnUNetTrainerBSPSB0):
    """Legacy: maps to B0."""

class nnUNetTrainerBSPSA1(nnUNetTrainerBSPSB1):
    """Legacy: maps to B1."""

class nnUNetTrainerBSPSA2(nnUNetTrainerBSPSB1):
    """Legacy: maps to B1."""

class nnUNetTrainerBSPSA3(nnUNetTrainerBSPSB1):
    """Legacy: maps to B1."""

class nnUNetTrainerBSPSD1(nnUNetTrainerBSPSB2):
    """Legacy: maps to B2 (CG-BLPHA)."""

class nnUNetTrainerBSPSD2(nnUNetTrainerBSPSB3):
    """Legacy: maps to B3 (H-DRE)."""

class nnUNetTrainerBSPSD3(nnUNetTrainerBSPSB4):
    """Legacy: maps to B4 (CG-BLPHA + H-DRE)."""

class nnUNetTrainerBSPSD4(nnUNetTrainerBSPSB4):
    """Legacy: maps to B4 (CG-BLPHA + H-DRE)."""

class nnUNetTrainerBSPSD5(nnUNetTrainerBSPSB5):
    """Legacy: maps to B5 (full BSPS-Net with HRSC-Loss)."""

class nnUNetTrainerBSPSE0(nnUNetTrainerBSPSB1):
    """Legacy: maps to B1."""

class nnUNetTrainerBSPSE1(nnUNetTrainerBSPSB2):
    """Legacy: maps to B2."""

class nnUNetTrainerBSPSE2(nnUNetTrainerBSPSB3):
    """Legacy: maps to B3."""

class nnUNetTrainerBSPSE3(nnUNetTrainerBSPSB4):
    """Legacy: maps to B4."""

class nnUNetTrainerBSPSE4(nnUNetTrainerBSPSB4):
    """Legacy: maps to B4."""

class nnUNetTrainerBSPSE5(nnUNetTrainerBSPSB4):
    """Legacy: maps to B4."""

class nnUNetTrainerBSPSE6(nnUNetTrainerBSPSB5):
    """Legacy: maps to B5."""
