"""Phase 3 B1 and B2 — adaptive gate model; original cnn_model.py unchanged.

Source architecture: the user's attention-pooling CNN3DModel in cnn_model.py.

B1: sample-specific SCALAR gate: one local/global mixture per user.
B2: sample-specific VECTOR gate: one local/global mixture per feature per user.

All original convolution, Performer, attention pooling, FC1/FC2, and losses
are reused. This file changes only HOW x_local and x_global are fused.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cnn_model import CNN3DModel  # Existing local CNN + Performer global branch


class CNN3DAdaptiveGating(CNN3DModel):
    """Drop-in CNN3DModel with one of the two new gate variants.

    Input/output shapes remain identical to the existing pipeline:
        ratings:       [B, 3952]
        item_labels:   [3952, 16]
        local vector:  [B, 32]  (attention pooling enabled)
        global vector: [B, 32]
        fused vector:  [B, 32]
        features:      [B, 64]
        classifier:    [B, 1]

    Hence existing train_cnn_3d(), feature extraction, and DetectionModel work.
    """

    def __init__(
        self,
        device,
        item_number: int,
        item_label_number: int,
        performer_heads: int,
        gate_variant: str = "B2",
        **kwargs,
    ):
        # Construct ALL existing local/global blocks, Performer, pools and FCs.
        super().__init__(
            device, item_number, item_label_number, performer_heads, **kwargs
        )
        gate_variant = gate_variant.upper()
        if gate_variant not in ("B1", "B2"):
            raise ValueError("gate_variant must be 'B1' or 'B2'")
        self.gate_variant = gate_variant

        # The parent had a single scalar `self.gate` shared by all users.
        # Delete that UNUSED parameter so it cannot be optimized or saved.
        del self.gate

        # D is the dimension of each branch AFTER pooling (currently D=32).
        d = self.conv_output_size
        output_dim = 1 if self.gate_variant == "B1" else d

        # The gate sees [local || global], i.e. 2D features for every user.
        # W has shape [1, 2D] for B1, [D, 2D] for B2.
        # b has shape [1] for B1, [D] for B2.
        #
        # We use Parameters rather than nn.Linear so the existing
        # train_cnn_3d() -> model.apply(init_weights) does not overwrite
        # this intentional initialization.
        self.gate_weight = nn.Parameter(
            torch.zeros(output_dim, 2 * d, device=device, dtype=torch.float32)
        )
        self.gate_bias = nn.Parameter(
            torch.full((output_dim,), 0.1, device=device, dtype=torch.float32)
        )

        # At initialization, sigmoid(0.1) = 0.5250... for BOTH variants.
        # That matches the original scalar gate's initial value and provides
        # a controlled starting point. The gate then learns from data.

    def branch_features(self, rating_matrix, item_labels_matrix):
        """Produce the two learned branch vectors; same logic as Phase 2.

        Local branch: rating+item semantics -> 3D ConvNeXt stages -> pooling.
        Global branch: Performer-attended semantics -> 3D ConvNeXt -> pooling.
        No additional branch/library is introduced here.
        """
        x_local_in = self._build_concat_tensor(
            rating_matrix, item_labels_matrix, use_attended_labels=False
        )
        x_local = self.stage2(self.stage1(x_local_in))
        x_local = (
            self.local_pool(x_local)
            if self.use_attn_pool
            else x_local.reshape(x_local.size(0), -1)
        )

        x_global_in = self._build_concat_tensor(
            rating_matrix, item_labels_matrix, use_attended_labels=True
        )
        x_global = self.attn_stage2(self.attn_stage1(x_global_in))
        x_global = (
            self.global_pool(x_global)
            if self.use_attn_pool
            else x_global.reshape(x_global.size(0), -1)
        )
        return x_local, x_global

    def compute_gate(self, x_local, x_global):
        """Calculate the gate from both branch outputs.

        B1: sigmoid(W [local || global] + b) -> [B, 1]
        B2: sigmoid(W [local || global] + b) -> [B, D]

        The B1 gate broadcasts across all D features. For B2 each feature
        receives its own learned mixture for that particular user.
        """
        if x_local.shape != x_global.shape:
            raise ValueError("Local and global feature shapes must match.")
        both = torch.cat([x_local, x_global], dim=1)
        return torch.sigmoid(F.linear(both, self.gate_weight, self.gate_bias))

    def forward_features(self, rating_matrix, item_labels_matrix):
        """Return the SAME 64D representation expected by the detector."""
        x_local, x_global = self.branch_features(rating_matrix, item_labels_matrix)
        gate = self.compute_gate(x_local, x_global)

        # Element-wise fusion. B1's [B,1] value broadcasts across D features.
        # B2 has [B,D], so every feature mixes local/global differently.
        fused = (1.0 - gate) * x_local + gate * x_global
        z_u = F.relu(self.fc1(self.dropout(fused)))
        return z_u

    # Parent CNN3DModel.forward() applies FC2 to this 64D representation.
    # Parent forward_features_for_contrastive() also calls this method.
    # So BOTH the classifier and contrastive loss train the adaptive gate.

    @torch.no_grad()
    def inspect_gate(self, rating_matrix, item_labels_matrix):
        """Optional analysis: return gate values to inspect after training.

        Useful for Phase 3 interpretability / ablation, NOT part of the loss.
        """
        was_training = self.training
        try:
            self.eval()
            x_local, x_global = self.branch_features(rating_matrix, item_labels_matrix)
            return self.compute_gate(x_local, x_global).detach().cpu()
        finally:
            self.train(was_training)
