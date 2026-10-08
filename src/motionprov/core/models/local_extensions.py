"""Opt-in P3 architecture arms; the default LocalEncoder remains unchanged.

Parts are contextual because the shared backbone already mixes joints. Refinement
acts only on assembled histories within one output window, preserving RF28 for
the standard four-layer/radius-two encoder. Neither feature adds separate clocks
or alignment paths for individual body parts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .local_encoder import NJ, HistoryMix, LocalEncoder

# Original SMPL24 joint IDs (pelvis 0 omitted), not post-removal array indices.
PART_JOINT_IDS = {
    "torso": (3, 6, 9, 12, 15),
    "left_arm": (13, 16, 18, 20, 22),
    "right_arm": (14, 17, 19, 21, 23),
    "left_leg": (1, 4, 7, 10),
    "right_leg": (2, 5, 8, 11),
}
PART_NAMES = tuple(PART_JOINT_IDS)
assert sorted(j for group in PART_JOINT_IDS.values() for j in group) == list(range(1, 24))


class PostHistoryRefinement(nn.Module):
    """Fixed-depth shared/unshared HistoryMix with an explicit FFN control."""

    def __init__(self, dim=128, heads=4, ffn_mult=4, dropout=0.1, steps=1, mode="attn", shared=True):
        super().__init__()
        if steps not in (0, 1, 2, 4) or mode not in ("attn", "ffn"):
            raise ValueError("Refinement uses depth 0/1/2/4 and attn/ffn mode")
        self.steps, self.mode, self.shared = steps, mode, bool(shared)
        count = 0 if steps == 0 else (1 if shared else steps)
        self.layers = nn.ModuleList([HistoryMix(dim, heads, ffn_mult, dropout, mode=mode) for _ in range(count)])

    def forward(self, histories, valid, *, steps=None):
        n = self.steps if steps is None else steps
        if n not in (0, 1, 2, 4) or n > self.steps:
            raise ValueError("Requested fixed depth exceeds configured refinement")
        for i in range(n):
            histories = self.layers[0 if self.shared else i](histories, valid)
        return histories

    def budget(self, *, steps=None):
        n = self.steps if steps is None else steps
        if n not in (0, 1, 2, 4) or n > self.steps:
            raise ValueError("Requested fixed depth exceeds configured refinement")
        return {
            "mode": self.mode,
            "configured_steps": self.steps,
            "executed_operator_calls": n,
            "shared_weights": self.shared,
            "independent_modules": len(self.layers),
            "parameters": sum(p.numel() for p in self.parameters()),
            "temporal_support_added": 0,
            "adaptive_halting": False,
        }


class ContextualPartReadout(nn.Module):
    """Five independently parameterized contextual heads and exact visibility."""

    def __init__(self, dim=128, output_dim=64, min_coverage=0.6):
        super().__init__()
        if dim < 1 or output_dim < 1 or not 0 <= min_coverage <= 1:
            raise ValueError("Positive feature sizes and coverage in [0,1] required")
        self.dim, self.output_dim, self.min_coverage = dim, output_dim, min_coverage
        self.adapters = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        for name, joints in PART_JOINT_IDS.items():
            self.adapters[name] = nn.Sequential(nn.Linear(len(joints) * (dim + 1), dim), nn.GELU(), nn.Linear(dim, dim))
            self.projections[name] = nn.Sequential(nn.Linear(dim, output_dim), nn.LayerNorm(output_dim))

    def forward(self, histories, joint_coverage):
        if histories.shape[-2:] != (NJ, self.dim) or joint_coverage.shape != histories.shape[:-1]:
            raise ValueError("Parts require 23 non-root histories and per-joint core coverage")
        descriptors, coverages, validity = [], [], []
        for name, original_ids in PART_JOINT_IDS.items():
            slots = [j - 1 for j in original_ids]
            coverage = joint_coverage[..., slots]
            h = histories[..., slots, :]
            h = torch.where((coverage > 0).unsqueeze(-1), h, torch.zeros_like(h))
            features = torch.cat([h.flatten(-2), coverage.to(h.dtype)], -1)
            part = self.projections[name](self.adapters[name](features))
            mean_coverage = coverage.mean(-1)
            valid = (mean_coverage >= self.min_coverage) & (mean_coverage > 0)
            part = F.normalize(part, dim=-1, eps=1e-12)
            part = torch.where(valid.unsqueeze(-1), part, torch.zeros_like(part))
            descriptors.append(part)
            coverages.append(mean_coverage)
            validity.append(valid)
        return (torch.stack(descriptors, -2), torch.stack(coverages, -1), torch.stack(validity, -1))

    def contract(self):
        return {
            "part_names": list(PART_NAMES),
            "original_smpl_joint_ids": {k: list(v) for k, v in PART_JOINT_IDS.items()},
            "input_slot_offset": -1,
            "output_dim": self.output_dim,
            "min_core_coverage": self.min_coverage,
            "information_scope": "contextual_after_shared_spatial_attention",
            "primary_alignment": "one_shared_time_path_for_all_parts",
            "float32_descriptor_bytes_per_window": 5 * self.output_dim * 4,
            "five_uint16_codes_payload_bytes_per_window": 10,
            "part_quantizers_fitted": False,
        }


@dataclass(frozen=True)
class LocalExtensionOutput:
    descriptors: torch.Tensor
    centers: torch.Tensor
    coverage: torch.Tensor
    joint_coverage: torch.Tensor
    part_descriptors: torch.Tensor | None
    part_coverage: torch.Tensor | None
    part_valid: torch.Tensor | None


class ExtendedLocalEncoder(LocalEncoder):
    """R1-only opt-in parts/refinement with backward-compatible global forward.

    ``forward_features`` exposes retained parts. Fitting a global product
    quantizer does not automatically fit part books: each 64-D part needs its
    own train-only book/identity and the matcher must aggregate before one path.
    """

    def __init__(
        self,
        d=128,
        heads=4,
        layers=4,
        window=2,
        ffn_mult=4,
        dropout=0.1,
        stride=10,
        variant="A",
        readout="R1",
        pelvis_dim=0,
        *,
        contextual_parts=False,
        part_output_dim=64,
        min_part_coverage=0.6,
        refinement_steps=0,
        refinement_mode="attn",
        share_refinement=True,
    ):
        super().__init__(
            d=d,
            heads=heads,
            layers=layers,
            window=window,
            ffn_mult=ffn_mult,
            dropout=dropout,
            stride=stride,
            variant=variant,
            readout=readout,
            pelvis_dim=pelvis_dim,
        )
        if self.readout != "R1" or self.pelvis_dim:
            raise ValueError("P3 extension is an explicit R1/no-pelvis arm; use original R2/R1F separately")
        self.refinement = PostHistoryRefinement(
            self.d, heads, ffn_mult, dropout, refinement_steps, refinement_mode, share_refinement
        )
        self.parts = ContextualPartReadout(self.d, part_output_dim, min_part_coverage) if contextual_parts else None
        self.extension_config = {
            "contextual_parts": bool(contextual_parts),
            "part_output_dim": part_output_dim,
            "min_part_coverage": min_part_coverage,
            "refinement_steps": refinement_steps,
            "refinement_mode": refinement_mode,
            "share_refinement": bool(share_refinement),
        }

    def forward_features(self, pos, observed=None, frame_valid=None, *, refinement_steps=None):
        b, t, j, _ = pos.shape
        if j != NJ:
            raise ValueError("P3 encoder expects 23 non-root joint slots")
        if observed is None:
            observed = torch.ones((b, t, j), dtype=torch.bool, device=pos.device)
        if frame_valid is not None:
            observed = observed & frame_valid.unsqueeze(-1)
        h = self.tokens(pos, observed)
        h = h * observed.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, observed)
            h = h * observed.unsqueeze(-1)
        n = t // self.stride
        h = h[:, : n * self.stride].reshape(b, n, self.stride, j, self.d)
        v = observed[:, : n * self.stride].reshape(b, n, self.stride, j)
        h = h * v.unsqueeze(-1)
        histories = h.permute(0, 1, 3, 2, 4).reshape(b, n, j, self.stride * self.d)
        histories = self.act(self.proj_time(histories))
        histories = self.refinement(
            histories.reshape(b * n, j, self.d), v.any(2).reshape(b * n, j), steps=refinement_steps
        )
        histories = histories.reshape(b, n, j, self.d)
        global_z = self.out_norm(self.proj_joint(histories.reshape(b, n, j * self.d)))
        joint_coverage = v.float().mean(2)
        coverage = v.float().mean((2, 3))
        centers = torch.arange(n, device=pos.device) * self.stride + self.stride // 2
        parts = (None, None, None) if self.parts is None else self.parts(histories, joint_coverage)
        return LocalExtensionOutput(global_z, centers, coverage, joint_coverage, *parts)

    def forward(self, pos, observed=None, frame_valid=None, pelvis=None):
        if pelvis is not None:
            raise ValueError("Root input is a separate P3 arm")
        out = self.forward_features(pos, observed, frame_valid)
        return out.descriptors, out.centers, out.coverage

    def extension_contract(self):
        return {
            "configuration": dict(self.extension_config),
            "receptive_field_frames": self.receptive_field,
            "refinement": self.refinement.budget(),
            "parts": None if self.parts is None else self.parts.contract(),
            "adoption_status": "controlled_optional_arm_without_quality_claim",
        }
