"""PAD-LLM Phase 3 B5: post-training prototype-distance open-set detector.

B5 follows official manual Sections 9.3 (prototype-distance rejection) and
9.5 (normal / known malicious / unknown-suspicious decisions).

It REUSES the *trained* B4 CNN (with B2 gating, B3 descriptors, B4 loss)
and the *trained* B4 binary FCNN classifier. B5 adds NO new CNN training.

Training protocol and safeguards:
- B4 training excludes the held-out attack family.
- B5 class-distance thresholds are fitted on genuine + KNOWN attack
  validation profiles that were never used to train the B4 models.
- The held-out attack is NEVER used to fit B5 thresholds.
- Thresholds are fixed BEFORE evaluating final known/unknown attack sets.

The unknown decision flags out-of-distribution / suspicious embeddings; it
DOES NOT prove an attack is malicious, and stealthy unfamiliar attacks can
remain close to the learned class prototypes and escape rejection.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_b4_model_module():
    """Load your B4 module as a source file; no weights are loaded here."""
    path = HERE / "main_phase3_B4.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing B4 prerequisite: {path}")
    spec = importlib.util.spec_from_file_location("pad_phase3_b4_for_b5", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load B4 model module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrototypeOpenSetHead(nn.Module):
    """Use B4's learned binary-class prototypes for three-state inference.

    B4 features:      [batch, 64]
    B4 class centers: [2, 64]  (0=normal, 1=known malicious)
    B4 detector:      one logit per user (binary known-class score)

    For feature z, measure Euclidean distance between L2-normalized z and
    each normalized class prototype. A profile is flagged UNKNOWN only when
    its distance exceeds BOTH class-specific calibration thresholds.
    Otherwise, the existing binary classifier decides normal vs malicious.

    This is a simple open-set BASELINE, not a guarantee of zero-day detection.
    """

    NORMAL = 0
    KNOWN_MALICIOUS = 1
    UNKNOWN_SUSPICIOUS = 2

    def __init__(self, class_prototypes: torch.Tensor, binary_threshold: float = 0.95):
        super().__init__()
        if tuple(class_prototypes.shape) != (2, 64):
            raise ValueError(f"Expected B4 class prototypes (2,64), got {tuple(class_prototypes.shape)}")
        if not (0.0 < binary_threshold < 1.0):
            raise ValueError("binary_threshold must be strictly between 0 and 1")
        if not torch.isfinite(class_prototypes).all():
            raise ValueError("Class prototypes contain NaN or infinity")
        self.binary_threshold = float(binary_threshold)
        # Freeze a snapshot of B4's existing learned class centers.
        centers = F.normalize(class_prototypes.detach().float().clone(), p=2, dim=1)
        self.register_buffer("class_centers", centers)
        # B5 thresholds are not learned through gradients; calibration sets them.
        self.register_buffer("distance_thresholds", torch.full((2,), float("nan"), device=centers.device))

    @torch.no_grad()
    def distances(self, features: torch.Tensor) -> torch.Tensor:
        """Return [batch,2] normalized L2 distances to B4's class centers."""
        if features.ndim != 2 or features.shape[1] != 64:
            raise ValueError(f"Expected features (batch,64), got {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise ValueError("Feature batch contains NaN or infinity")
        if (features.norm(dim=1) < 1e-8).any():
            raise ValueError("Zero-length user feature vector: cannot compute distance")
        z = F.normalize(features.detach().float().to(self.class_centers.device), p=2, dim=1)
        return torch.cdist(z, self.class_centers, p=2).clamp_min(0.0)

    @torch.no_grad()
    def calibrate(self, validation_features: torch.Tensor, validation_labels: torch.Tensor,
                  coverage: float = 0.95) -> torch.Tensor:
        """Set class-conditional distance cutoffs on *known-only* validation.

        `coverage=.95`: the 95th percentile of the distance to the actual
        class prototype is accepted for each known class on validation.
        This does NOT guarantee 95% acceptance on future/new samples.

        The caller must ensure validation contains only genuine(0) and
        KNOWN malicious(1) users; no held-out attack family is allowed.
        """
        if not (0.5 <= coverage < 1.0):
            raise ValueError("coverage must lie in [0.5, 1.0)")
        labels = torch.as_tensor(validation_labels, dtype=torch.long,
                                 device=self.class_centers.device).view(-1)
        if labels.numel() != validation_features.shape[0]:
            raise ValueError("Validation features and labels have different lengths")
        if not ((labels == 0) | (labels == 1)).all():
            raise ValueError("Calibration must use ONLY 0=normal or 1=known attack; never unknown")
        d = self.distances(validation_features)
        thresholds = []
        for cls in (0, 1):
            correct_class_dist = d[labels == cls, cls]
            if correct_class_dist.numel() < 8:
                raise ValueError(f"Need at least 8 known validation examples of class {cls}")
            thresholds.append(torch.quantile(correct_class_dist, coverage))
        thresholds = torch.stack(thresholds).to(self.distance_thresholds.device)
        self.distance_thresholds.copy_(thresholds)
        return self.distance_thresholds.detach().cpu().clone()

    @torch.no_grad()
    def predict(self, features: torch.Tensor, binary_logits: torch.Tensor) -> dict:
        """Return three-state prediction and interpretable rejection evidence."""
        if not torch.isfinite(self.distance_thresholds).all():
            raise RuntimeError("B5 needs calibrate() before predict(); never use arbitrary thresholds")
        logits = torch.as_tensor(binary_logits, dtype=torch.float32,
                                 device=self.class_centers.device).view(-1)
        if features.shape[0] != logits.numel():
            raise ValueError("Feature count and binary logit count differ")
        if not torch.isfinite(logits).all():
            raise ValueError("Binary logits contain NaN or infinity")
        d = self.distances(features)
        prob = torch.sigmoid(logits)
        pred = (prob > self.binary_threshold).long()
        outside_both = (d > self.distance_thresholds.unsqueeze(0)).all(dim=1)
        pred[outside_both] = self.UNKNOWN_SUSPICIOUS
        return {
            "predictions": pred,                        # 0, 1, or 2
            "binary_probability": prob,               # B4 binary confidence
            "distance_to_normal": d[:, 0],
            "distance_to_known_attack": d[:, 1],
            "unknown_suspicious": outside_both,
        }

    def config(self) -> dict:
        """Small JSON-compatible configuration, not an untrusted pickle."""
        if not torch.isfinite(self.distance_thresholds).all():
            raise RuntimeError("Uncalibrated B5 has no final configuration")
        return {
            "method": "B4 binary class prototypes + calibrated distance rejection",
            "normal_class": 0,
            "known_malicious_class": 1,
            "unknown_suspicious_class": 2,
            "binary_threshold": self.binary_threshold,
            "distance_thresholds": self.distance_thresholds.detach().cpu().tolist(),
            "feature_size": 64,
            "distance_metric": "L2 on unit-normalized 64D features",
        }


__all__ = ["PrototypeOpenSetHead", "load_b4_model_module"]
