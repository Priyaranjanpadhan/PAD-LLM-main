"""Phase 3 B3: explicit semantic inconsistency + behavioural category entropy.

B3 = B2 (sample-specific vector gate) + four transparent profile descriptors:
    1. mean cosine-semantic inconsistency across interacted items
    2. maximum cosine-semantic inconsistency
    3. variance of semantic inconsistency
    4. Shannon entropy of a user's movie-genre distribution

This implementation uses the EXISTING 16D T5/SVD item embeddings and the
existing MovieLens movies.dat genre metadata. It does not alter the original
CNN, Performer, loss functions, detector, or 64D feature-output interface.

IMPORTANT DATA NOTES:
- Our current MovieLens input uses 0/1 interaction indicators, not 1..5 ratings.
  Therefore w_ui is the observed interaction weight, NOT rating preference.
- The convention assumed by this pipeline is column=(MovieLens movieId - 1).
  Verify that this matches your preprocessing before a scientific full run.
- Genre entropy is based on MULTI-LABEL genre memberships, not on numeric T5
  dimensions. An item with multiple genres contributes to each of those genres.
- Descriptors alone do not prove maliciousness or causal target-filler mismatch.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use the B2 model you already smoke-tested. Its filename contains '&', so
# loading by a literal file path is safer than attempting an ordinary import.
B2_FILE = Path(__file__).with_name("main_phase3_B1&B2.py")
if not B2_FILE.exists():
    raise FileNotFoundError(f"B3 requires the existing B1/B2 file: {B2_FILE}")
_spec = importlib.util.spec_from_file_location("pad_phase3_b1b2_for_b3", B2_FILE)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {B2_FILE}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
B2Model = _module.CNN3DAdaptiveGating


def load_movie_genres(movies_path: str | Path, item_number: int) -> torch.Tensor:
    """Parse MovieLens-1M `movieId::title::genre1|genre2` into [I, C].

    MovieLens movie IDs are 1-based; columns in the current prepared matrix
    are assumed to represent movieId-1. Missing IDs remain all-zero rows.
    """
    path = Path(movies_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot read movie genres: {path}")

    records = []
    genre_names = set()
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = line.split("::", 2)
        if len(parts) != 3:
            continue
        try:
            movie_id = int(parts[0])
        except ValueError:
            continue
        genres = [g.strip() for g in parts[2].split("|")
                  if g.strip() and g.strip() != "(no genres listed)"]
        col = movie_id - 1
        if 0 <= col < item_number and genres:
            records.append((col, genres))
            genre_names.update(genres)

    if not records:
        raise ValueError("No usable movie genres. Check movies.dat and index convention.")

    ordered = sorted(genre_names)
    genre_to_col = {name: j for j, name in enumerate(ordered)}
    matrix = torch.zeros((item_number, len(ordered)), dtype=torch.float32)
    for movie_col, genres in records:
        for genre in genres:
            matrix[movie_col, genre_to_col[genre]] = 1.0
    return matrix


class CNN3DSemanticB3(B2Model):
    """B2 vector gating + explicit four-value semantic/behaviour descriptor.

    B2 creates a 64D learned user representation. B3 adds a small trainable
    projection of four descriptive statistics INTO that same 64D space.
    Output remains [batch,64], so the original DetectionModel still works.
    """
    DESCRIPTOR_NAMES = (
        "semantic_mean", "semantic_max", "semantic_variance", "genre_entropy"
    )

    def __init__(
        self,
        device,
        item_number: int,
        item_label_number: int,
        performer_heads: int,
        gate_variant: str = "B3",
        genre_matrix: torch.Tensor | None = None,
        movies_path: str | Path | None = None,
        **kwargs,
    ):
        if gate_variant.upper() != "B3":
            raise ValueError("B3 model requires gate_variant='B3'")
        super().__init__(
            device, item_number, item_label_number, performer_heads,
            gate_variant="B2", **kwargs
        )
        self.gate_variant = "B3"  # Report the new architecture accurately.

        if genre_matrix is None:
            if movies_path is None:
                movies_path = PROJECT_ROOT / "data/ml-1m/movies.dat"
            genre_matrix = load_movie_genres(movies_path, item_number)

        genre_matrix = torch.as_tensor(genre_matrix, dtype=torch.float32)
        if (genre_matrix.ndim != 2 or genre_matrix.shape[0] != item_number
                or genre_matrix.shape[1] < 1):
            raise ValueError(f"Expected genre matrix [{item_number}, C>=1], "
                             f"got {tuple(genre_matrix.shape)}")
        # Genre metadata is fixed, not trainable. Reconstructed from movies.dat
        # whenever a model is created; not stored repeatedly in every checkpoint.
        self.register_buffer("genre_multihot", genre_matrix.to(device), persistent=False)

        # These parameters project four scores into the existing 64D feature
        # vector. Zero initialization makes the B3 output initially identical
        # to B2. The existing train_cnn_3d() calls `model.apply(init_weights)`
        # on nn.Linear, so use nn.Parameter to preserve this careful zero start.
        self.anomaly_weight = nn.Parameter(torch.zeros(64, 4, device=device))
        self.anomaly_bias = nn.Parameter(torch.zeros(64, device=device))

    def anomaly_descriptors(self, ratings: torch.Tensor,
                            item_labels: torch.Tensor) -> torch.Tensor:
        """Return [B,4] explicit scores, with no trainable parameters.

        Use a 0.5 interaction threshold to suppress Gaussian noise introduced
        on previously missing ratings during augmentation. Nonzero selected
        ratings retain their continuous weights when computing the centroid.
        For original binary data, this is exactly w_ui=1 if interacted, 0 else.
        No interactions => all four descriptors are defined to be zero.
        """
        with torch.no_grad():
            ratings = ratings.float()
            item_labels = item_labels.float()
            if ratings.ndim != 2 or item_labels.ndim != 2:
                raise ValueError("ratings=[B,I] and item_labels=[I,K] required")
            if (ratings.shape[1] != item_labels.shape[0] or
                    ratings.shape[1] != self.genre_multihot.shape[0]):
                raise ValueError("Inconsistent movie/item counts")

            # Entries below 0.5 are not interpreted as real interactions.
            mask = ratings >= 0.5
            weights = ratings.clamp(0.0, 1.0) * mask.float()
            total = weights.sum(dim=1, keepdim=True)  # [B,1]
            has_items = (total.squeeze(1) > 0)
            safe_total = total.clamp_min(1e-8)

            # Semantic centroid p_u: weighted average of embeddings for
            # the movies that this user interacted with.
            taste = weights @ item_labels / safe_total  # [B,K]
            taste_norm = F.normalize(taste, p=2, dim=1, eps=1e-8)
            movie_norm = F.normalize(item_labels, p=2, dim=1, eps=1e-8)
            cosine = (taste_norm @ movie_norm.T).clamp(-1.0, 1.0)
            mismatch = 1.0 - cosine  # [B,I], between 0 and 2

            sem_mean = (weights * mismatch).sum(dim=1) / safe_total.squeeze(1)
            sem_var = (weights * (mismatch - sem_mean[:, None]).square()).sum(dim=1) / safe_total.squeeze(1)
            sem_max = mismatch.masked_fill(~mask, -1e6).max(dim=1).values
            sem_max = torch.where(has_items, sem_max, torch.zeros_like(sem_max))

            # For items with multiple genres, each listed genre receives its
            # corresponding share in the aggregated category distribution.
            category_mass = weights @ self.genre_multihot  # [B,C]
            genre_total = category_mass.sum(dim=1, keepdim=True)
            probabilities = category_mass / genre_total.clamp_min(1e-8)
            entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1)
            entropy = torch.where(genre_total.squeeze(1) > 0, entropy, torch.zeros_like(entropy))

            # Keep descriptor magnitudes comparable with [0,1] inputs.
            # 1-cos <= 2, variance <= 1, entropy <= log(C).
            scaled = torch.stack([
                sem_mean / 2.0,
                sem_max / 2.0,
                sem_var,  # already <= 1 for quantities in [0,2]
                entropy / max(float(torch.log(torch.tensor(float(max(2, self.genre_multihot.shape[1]))))), 1e-8),
            ], dim=1)
            if not torch.isfinite(scaled).all():
                raise FloatingPointError("B3 descriptors contained NaN/Inf")
            return scaled

    def forward_features(self, rating_matrix: torch.Tensor,
                         item_labels_matrix: torch.Tensor) -> torch.Tensor:
        # B2 features include the vector gate and original CNN/Performer paths.
        base_features = super().forward_features(rating_matrix, item_labels_matrix)
        descriptors = self.anomaly_descriptors(rating_matrix, item_labels_matrix)
        # Learned residual projection: initially zero, then fitted by training.
        anomaly_update = F.linear(descriptors, self.anomaly_weight, self.anomaly_bias)
        return F.relu(base_features + anomaly_update)

    # Existing inherited forward() uses self.forward_features() then self.fc2().
    # Existing inherited contrastive_features also uses forward_features().
