"""PAD-LLM Phase 3 / B4: B3 features + supervised contrastive + class prototypes.

Prerequisites (place all in the SAME phase3/ directory):
    main_phase3_B1&B2.py  (B2 user-dependent vector gate)
    main_phase3_B3.py     (explicit semantic/behaviour descriptors)
    main_phase3_B4.py    (this file)

B4 changes representation TRAINING, not its final prediction interface:
    ratings [B,I] -> CNN/Performer + B2 gate + B3 descriptors -> features [B,64]
    features + TRUE BINARY labels -> supervised contrastive loss + prototype loss
    the original classifier still predicts normal(0) versus malicious(1).

Research note: prototype labels are NORMAL/MALICIOUS, NOT attack-family names.
No held-out family labels enter B4 training. Full attack-family-aware prototypes
would require a separate data/label pipeline and must not leak held-out families.

Manual: Section 10.1 (Supervised Contrastive Learning), 11.3 (Prototype
Regularization). We KEEP the existing focal-loss implementation unchanged
for a controlled comparison to B3; we do not silently replace it with a
new effective-number class-balanced loss from Section 11.1.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_b3():
    source = Path(__file__).with_name("main_phase3_B3.py")
    if not source.is_file():
        raise FileNotFoundError(f"B4 requires the existing B3 file: {source}")
    spec = importlib.util.spec_from_file_location("phase3_b3_for_b4", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


B3 = _load_b3()


class CNN3DRepresentationB4(B3.CNN3DSemanticB3):
    """B3 with two *learnable class prototypes* in its existing 64D space.

    Prototypes are only used by B4 loss during CNN training. At inference,
    `forward_features` still returns [B,64], and `forward` still returns [B,1].
    This preserves the existing separate DetectionModel and evaluation code.
    """

    def __init__(self, device, item_number, item_label_number, performer_heads,
                 gate_variant="B4", **kwargs):
        if gate_variant.upper() != "B4":
            raise ValueError("B4 requires gate_variant='B4'")
        super().__init__(device, item_number, item_label_number, performer_heads,
                         gate_variant="B3", **kwargs)
        self.gate_variant = "B4"
        # One 64D center for normal profiles and one for malicious profiles.
        # nn.Parameter is deliberately used: the existing init_weights() reset
        # only nn.Linear / nn.Conv3d, so our prototypes remain intact.
        self.class_prototypes = nn.Parameter(
            F.normalize(torch.randn(2, 64, device=device), dim=1)
        )


class SupervisedContrastiveLoss(nn.Module):
    """Two-class supervised contrastive loss using cosine similarities.

    Each anchor has as positives OTHER profiles with the same binary label,
    and as negatives profiles with the opposite label in its mini-batch.
    Invalid anchors (no positive or no negative) are excluded safely.
    The loss uses *unmixed* genuine 0/1 labels; don't use Mixup's soft labels.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def forward(self, features, labels):
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError(f"Expected features [B,64], got {tuple(features.shape)}")
        labels = labels.reshape(-1).long()
        n = features.shape[0]
        if n != labels.numel():
            raise ValueError("features and labels must have matching batch size")
        if n < 3:
            return features.sum() * 0.0
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("Supervised contrastive labels must be hard binary labels")

        z = F.normalize(features.float(), dim=1)
        sim = (z @ z.T) / self.temperature                       # [B,B]
        diag = torch.eye(n, device=features.device, dtype=torch.bool)
        same = labels[:, None].eq(labels[None, :])
        positive = same & ~diag
        negative = ~same
        valid = positive.any(dim=1) & negative.any(dim=1)
        if not valid.any():
            # All-same-class batches, or no positive pairs, provide no valid
            # supervised contrastive comparison. Return connected zero.
            return features.sum() * 0.0

        # Stable log-softmax where the anchor is excluded from denominator.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()
        sim = sim.masked_fill(diag, -1e9)
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        per_anchor = -(
            log_prob * positive.float()
        ).sum(dim=1) / positive.sum(dim=1).clamp_min(1)
        return per_anchor[valid].mean()


class PrototypeRegularizationLoss(nn.Module):
    """Compact binary-class embeddings; prevent both class centers collapsing.

    Alignment makes normalized sample features approach their true class
    prototype. Separation discourages the two prototypes from pointing in
    nearly the same direction. This is an auxiliary representation objective;
    it is NOT yet an open-set or three-class detector (that is B5).
    """

    def __init__(self, max_proto_cosine=0.25):
        super().__init__()
        self.max_proto_cosine = float(max_proto_cosine)

    def forward(self, features, labels, prototypes):
        if prototypes.shape != (2, 64):
            raise ValueError(f"Expected two 64D prototypes, got {prototypes.shape}")
        labels = labels.reshape(-1).long()
        if labels.numel() != features.shape[0]:
            raise ValueError("Feature/label batch size mismatch")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("Prototype labels must be 0/1, not Mixup labels")
        z = F.normalize(features.float(), dim=1)
        centers = F.normalize(prototypes.float(), dim=1)
        # Squared L2 distance between normalized sample and correct center.
        compactness = (z - centers[labels]).square().sum(dim=1).mean()
        # Penalize overly similar normal and malicious prototype directions.
        cosine_between = (centers[0] * centers[1]).sum()
        separation = F.relu(cosine_between - self.max_proto_cosine)
        return compactness + separation


__all__ = ["CNN3DRepresentationB4", "SupervisedContrastiveLoss",
           "PrototypeRegularizationLoss"]
