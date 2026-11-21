"""Phase 2.2 + 2.3: evaluation-only robustness study for saved PAD-LLM LOAO models.

IMPORTANT: THIS SCRIPT NEVER CALLS train_cnn_3d() OR train_detection_model().

Phase 2.2: Test the saved six LOAO detectors at attack prevalence ratios
           0.1%, 0.25%, 0.5%, 1%, and 5%.
Phase 2.3: Using the exact same predictions, report results separately for
           Q1 very sparse, Q2 sparse, Q3 medium, and Q4 dense users.

All folds, attack families, attack ratios and test-subsampling repeats are
handled automatically in ONE RUN. Feature caches and per-fold CSVs are saved
in Google Drive so the experiment can be restarted without training models.

Run from the PAD-LLM-main repository root (after mounting Google Drive):
    python main_phase2_2_2_3.py --check-only
    python main_phase2_2_2_3.py

Methodology:
- All attack profiles come from each family's held-out TEST pool, never
  the attack training pool. Only the test-set prevalence changes.
- We call create_loao_split(proportion=0.05) ONCE to create a fixed maximum
  evaluation pool; smaller ratios sample from that same pool.
- The original Phase 2.1 training checkpoints are never modified.
- Repeats vary *test-sample selection*, NOT the independently trained seed.
- Quartile cutoffs are determined from GENUINE TEST USER interaction counts,
  then applied unchanged to genuine and attack test profiles.
- A quartile with no malicious profiles has undefined Recall / F1 (blank CSV
  cells, not misleading zeros); FPR remains computable.
- The 1%-ratio sample is freshly drawn from the 5% pool, so it need not match
  precisely the 18 attack users used in the earlier LOAO evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from cnn_model import CNN3DModel
from detection_model import DetectionModel
from data_clean.ml_1m_attack_data_clean import create_loao_split
from utils import load_item_label_data


# ============================================================================
# 1. EXPERIMENT SETTINGS. Do not change these after results have been saved.
# ============================================================================

ATTACK_FOLDS = ("GOAT", "ItemAE", "WRMF_SGD", "Random", "Average", "Bandwagon")
RATIOS = (0.001, 0.0025, 0.005, 0.01, 0.05)
TEST_SUBSAMPLE_REPEATS = 5
SEED = 42
THRESHOLD = 0.95  # Preserve the same threshold used in Phase 2.1.
ITEM_COUNT = 3952
LABEL_DIM = 16
PERFORMER_HEADS = 4
FEATURE_BATCH_SIZE = 32  # Use 16 if Colab reports GPU out-of-memory.

NORMAL_FILE = Path("data/ml-1m/user_item_rating.txt")
FEATURES_FILE = Path("data/ml-1m/movies.dat")
ATTACK_FILE_ITEMAE = Path("attack_data/ml-1m/Sur-ItemAE_fake_data_best.npz")
ATTACK_FILE_WRMF = Path("attack_data/ml-1m/Sur-WeightedMF-sgd_fake_data_best.npz")
ATTACK_FILE_GOAT = Path("attack_data/ml-1m/posion_data.txt")

# In Colab, results/checkpoints persist in Google Drive (after mounting it).
# The input directory contains the six CNN and classifier weights from 2.1.
if Path("/content").is_dir():
    DRIVE_DIR = Path("/content/drive/MyDrive/PAD_LLM_main")
    INPUT_ROOT = DRIVE_DIR / "LOAO_ML1M"
    OUTPUT_ROOT = DRIVE_DIR / "phase2_2.2and2.3"
else:
    INPUT_ROOT = Path("outputs/LOAO_ML1M")
    OUTPUT_ROOT = Path("outputs/phase2_2.2and2.3")


# ============================================================================
# 2. SMALL FILE UTILITIES AND SANITY CHECKS
# ============================================================================

def atomic_json(path: Path, payload: dict) -> None:
    """Write a complete JSON; avoid leaving a half-written final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Save CSV only after all its rows are ready."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Cache features on CPU, so future Colab sessions can reuse them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().cpu(), temporary)
    os.replace(temporary, path)


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_existing_models_and_inputs() -> None:
    """Never start the analysis if ANY of the six model pairs is missing."""
    required = [
        NORMAL_FILE, FEATURES_FILE, ATTACK_FILE_ITEMAE,
        ATTACK_FILE_WRMF, ATTACK_FILE_GOAT,
        INPUT_ROOT / "run_config.json",
    ]
    for fold in ATTACK_FOLDS:
        required += [
            INPUT_ROOT / fold / "cnn" / "cnn_final.pth",
            INPUT_ROOT / fold / "detector" / "detector_final.pth",
        ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot start evaluation. These required files are missing:\n" +
            "\n".join("  - " + path for path in missing) +
            "\nMake sure Google Drive is mounted and the repository data files exist."
        )

    # Double-check that these models were trained with our current dimensions
    # and that they are from the same experiment configuration we intend to test.
    with (INPUT_ROOT / "run_config.json").open("r", encoding="utf-8") as file:
        old = json.load(file)
    required_settings = {
        "seed": SEED,
        "item_number": ITEM_COUNT,
        "item_label_number": LABEL_DIM,
        "performer_heads": PERFORMER_HEADS,
        "threshold": THRESHOLD,
    }
    for name, expected in required_settings.items():
        if old.get(name) != expected:
            raise RuntimeError(
                f"The saved LOAO models have {name}={old.get(name)!r}; "
                f"this script expects {expected!r}. Stop rather than mixing runs."
            )
    print("[CHECK] All six saved CNNs and detectors are present.", flush=True)
    print(f"[CHECK] Input checkpoint directory: {INPUT_ROOT}", flush=True)
    print("[CHECK] No training will be performed by this script.", flush=True)


def check_saved_analysis_config() -> None:
    """Prevent accidentally reusing caches from different experiment settings."""
    config = {
        "phase": "2.2+2.3",
        "seed": SEED,
        "ratios": list(RATIOS),
        "test_subsample_repeats": TEST_SUBSAMPLE_REPEATS,
        "threshold": THRESHOLD,
        "max_test_pool_ratio": max(RATIOS),
        "feature_batch_size": FEATURE_BATCH_SIZE,
        "folds": list(ATTACK_FOLDS),
        "model_root": str(INPUT_ROOT),
        "quartile_reference": "genuine_test_users",
        "evaluation_method": "test_only_reuse_saved_loao_models",
    }
    config_file = OUTPUT_ROOT / "evaluation_config.json"
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as file:
            prior = json.load(file)
        if prior != config:
            raise RuntimeError(
                "Saved analysis configuration differs from current settings. "
                "Restore the old settings or choose a new OUTPUT_ROOT."
            )
    else:
        atomic_json(config_file, config)


# ============================================================================
# 3. LOAD MODELS, WITHOUT ANY TRAINING
# ============================================================================

def get_model_weights(path: Path, device: torch.device):
    # File is under YOUR OWN Google Drive; weights_only reduces loading risk.
    return torch.load(path, map_location=device, weights_only=True)


def load_fold_models(fold: str, device: torch.device):
    """Load exactly the CNN + detector trained for one held-out family."""
    folder = INPUT_ROOT / fold
    set_seed(SEED)
    cnn = CNN3DModel(device, ITEM_COUNT, LABEL_DIM, PERFORMER_HEADS).to(device)
    cnn.load_state_dict(get_model_weights(folder / "cnn/cnn_final.pth", device))
    cnn.eval()

    detector = DetectionModel().to(device)
    # DetectionModel uses nn.LazyLinear. A 64-feature dummy input initializes it
    # before loading its saved state dictionary, matching its training shape.
    with torch.no_grad():
        detector(torch.zeros(1, 64, device=device, dtype=torch.float32))
    detector.load_state_dict(
        get_model_weights(folder / "detector/detector_final.pth", device)
    )
    detector.eval()
    return cnn, detector


# ============================================================================
# 4. PREPARE TEST PROFILES, REUSING THE ORIGINAL LOAO SPLIT LOGIC
# ============================================================================

def create_maximum_test_pool():
    """Create 5% attack test pools once; all folds use these identical profiles.

    We do NOT use the 1% `test_..._features.pt` caches from Phase 2.1,
    because they only contain 18 selected attacks, insufficient for 5%.
    No training arrays returned from the builder are used by this script.
    """
    print("[DATA] Preparing shared 5% TEST pools with seed=42...", flush=True)
    result = create_loao_split(
        normal_file=str(NORMAL_FILE),
        attack_file1=str(ATTACK_FILE_ITEMAE),
        attack_file2=str(ATTACK_FILE_WRMF),
        attack_file3=str(ATTACK_FILE_GOAT),
        held_out_attack="GOAT",  # Changes training exclusion only, NOT test pools.
        proportion=max(RATIOS),
        seed=SEED,
    )
    genuine = result["genuine_test"][:, :-1].astype(np.float32, copy=True)
    assert np.all(result["genuine_test"][:, -1] == 0)
    assert genuine.shape[1] == ITEM_COUNT

    attacks = {}
    for name in ATTACK_FOLDS:
        test_set = result["test_by_attack"][name]
        attack_only = test_set[test_set[:, -1] == 1, :-1]
        attacks[name] = attack_only.astype(np.float32, copy=True)
        required_count = max(1, round(len(genuine) * max(RATIOS)))
        if len(attacks[name]) < required_count:
            raise ValueError(
                f"{name} only has {len(attacks[name])} test attacks, "
                f"but 5% requires {required_count}."
            )
        print(f"[DATA] {name:10s}: {len(attacks[name])} malicious test profiles", flush=True)

    print(f"[DATA] Shared genuine test profiles: {len(genuine)}", flush=True)
    del result
    gc.collect()
    return genuine, attacks


def quartile_labels(interaction_counts: np.ndarray, cutoffs: np.ndarray):
    """Return 1..4 for each profile, with the SAME cutoffs for both classes."""
    return np.searchsorted(cutoffs, interaction_counts, side="right") + 1


# ============================================================================
# 5. EXTRACT FEATURES ONCE PER FOLD; CACHE THEM IN DRIVE
# ============================================================================

def extract_user_features(
    cnn: torch.nn.Module,
    rating_profiles: np.ndarray,
    item_labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert (users, 3952) rating profiles into (users, 64) CNN features."""
    cnn.eval()
    labels_on_device = item_labels.to(device, dtype=torch.float32)
    result_batches = []
    with torch.no_grad():
        for start in range(0, len(rating_profiles), FEATURE_BATCH_SIZE):
            stop = min(start + FEATURE_BATCH_SIZE, len(rating_profiles))
            batch = torch.from_numpy(rating_profiles[start:stop]).to(device)
            output = cnn.forward_features(batch, labels_on_device)
            result_batches.append(output.detach().cpu())
    result = torch.cat(result_batches, dim=0)
    if result.shape != (len(rating_profiles), 64):
        raise RuntimeError(f"Unexpected CNN feature shape: {tuple(result.shape)}")
    return result


def get_cached_features(cnn, profiles, item_labels, device, path):
    """Load previous extraction if available. Otherwise calculate and save."""
    if path.exists():
        cached = load_tensor(path)
        if tuple(cached.shape) != (len(profiles), 64):
            raise RuntimeError(f"Wrong cache shape in {path}. Do not silently reuse it.")
        print(f"    [CACHE HIT] {path.name}", flush=True)
        return cached
    print(f"    [EXTRACT] {path.name}, {len(profiles)} profiles", flush=True)
    features = extract_user_features(cnn, profiles, item_labels, device)
    save_tensor(path, features)
    return features


def score_profiles(detector, features: torch.Tensor, device: torch.device):
    """Return one sigmoid attack score in [0,1] for each user."""
    detector.eval()
    with torch.no_grad():
        logits = detector(features.to(device, dtype=torch.float32)).view(-1)
        probability = torch.sigmoid(logits)
    return probability.detach().cpu().numpy()


# ============================================================================
# 6. CONFUSION-MATRIX METRICS
# ============================================================================

def compute_metrics(genuine_prob, malicious_prob, threshold=THRESHOLD):
    """Evaluate given genuine (0) and attack (1) scores.

    True positive: malicious and flagged.
    False positive: genuine but flagged.
    False negative: malicious but missed.
    True negative: genuine and not flagged.
    """
    fp = int((genuine_prob > threshold).sum())
    tn = int(len(genuine_prob) - fp)
    tp = int((malicious_prob > threshold).sum())
    fn = int(len(malicious_prob) - tp)

    # False-positive rate always exists if genuine users are present.
    fpr = fp / (fp + tn) if fp + tn > 0 else None
    specificity = tn / (fp + tn) if fp + tn > 0 else None

    # A quartile may have ZERO attacks (particularly with just 2 total at
    # ratio 0.1%). Report recall/F1 as undefined rather than falsely zero.
    if len(malicious_prob) == 0:
        precision = recall = f1 = None
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "genuine_count": int(len(genuine_prob)),
        "attack_count": int(len(malicious_prob)),
        "precision": precision, "recall": recall, "f1": f1,
        "fpr": fpr, "specificity": specificity,
    }


DETAIL_COLUMNS = [
    "fold_held_out", "tested_attack", "attack_role", "ratio_target", "ratio_actual",
    "sample_repeat", "density_group", "density_label", "genuine_count",
    "attack_count", "tp", "fp", "fn", "tn", "precision", "recall",
    "f1", "fpr", "specificity", "threshold",
]
SUMMARY_COLUMNS = [
    "fold_held_out", "tested_attack", "attack_role", "ratio_target", "density_group",
    "density_label", "repeats", "f1_mean", "f1_sd", "precision_mean",
    "recall_mean", "fpr_mean", "genuine_count", "attack_count_mean",
]


def metrics_row(fold, attack, role, ratio, repeat, density, genuine_count, attack_count, m):
    result = {
        "fold_held_out": fold,
        "tested_attack": attack,
        "attack_role": role,
        "ratio_target": ratio,
        "ratio_actual": attack_count / genuine_count if genuine_count else None,
        "sample_repeat": repeat,
        "density_group": density,
        "density_label": {
            "ALL": "Overall",
            "Q1": "Very sparse", "Q2": "Sparse",
            "Q3": "Medium", "Q4": "Dense",
        }[density],
        "threshold": THRESHOLD,
    }
    result.update(m)
    return result


def float_or_none(v):
    if v is None or v == "":
        return None
    return float(v)


def summarize(rows: list[dict]):
    """Aggregate five test-resampling repeats (NOT five training seeds)."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["fold_held_out"], row["tested_attack"], row["attack_role"],
            float(row["ratio_target"]), row["density_group"], row["density_label"],
        )
        groups.setdefault(key, []).append(row)

    def average_of(batch, metric):
        vals = [float_or_none(row.get(metric)) for row in batch]
        vals = [v for v in vals if v is not None]
        return mean(vals) if vals else None

    results = []
    for key, subset in sorted(groups.items(), key=lambda entry: str(entry[0])):
        f1_vals = [float_or_none(row.get("f1")) for row in subset]
        f1_vals = [val for val in f1_vals if val is not None]
        results.append({
            "fold_held_out": key[0], "tested_attack": key[1],
            "attack_role": key[2], "ratio_target": key[3],
            "density_group": key[4], "density_label": key[5],
            "repeats": len(subset),
            "f1_mean": mean(f1_vals) if f1_vals else None,
            "f1_sd": stdev(f1_vals) if len(f1_vals) > 1 else None,
            "precision_mean": average_of(subset, "precision"),
            "recall_mean": average_of(subset, "recall"),
            "fpr_mean": average_of(subset, "fpr"),
            "genuine_count": subset[0]["genuine_count"],
            "attack_count_mean": average_of(subset, "attack_count"),
        })
    return results


# ============================================================================
# 7. ONE FOLD: ONE CNN LOAD + ONE DETECTOR LOAD + ALL PHASE 2.2/2.3 TESTS
# ============================================================================

def run_fold(fold, genuine, attack_pools, cutoffs, item_labels, device):
    fold_root = OUTPUT_ROOT / fold
    overall_file = fold_root / "low_density_detail.csv"
    sparse_file = fold_root / "sparse_user_detail.csv"
    feature_root = fold_root / "feature_cache"

    # A completed fold is skipped safely when restarted after a disconnect.
    if overall_file.is_file() and sparse_file.is_file():
        print(f"[SKIP] {fold}: both result CSVs already exist.", flush=True)
        return

    print(f"\n{'='*75}\n[FOLD] {fold} HELD OUT — EVALUATION ONLY\n{'='*75}", flush=True)
    cnn, detector = load_fold_models(fold, device)

    genuine_features = get_cached_features(
        cnn, genuine, item_labels, device, feature_root / "genuine_features.pt"
    )
    genuine_scores = score_profiles(detector, genuine_features, device)
    genuine_counts = np.count_nonzero(genuine > 0, axis=1)
    genuine_groups = quartile_labels(genuine_counts, cutoffs)

    low_density_rows = []
    sparsity_rows = []

    for attack_index, attack_name in enumerate(ATTACK_FOLDS):
        print(f"  [ATTACK] Testing {attack_name}...", flush=True)
        attack_profiles = attack_pools[attack_name]
        attack_features = get_cached_features(
            cnn, attack_profiles, item_labels, device,
            feature_root / f"{attack_name}_features.pt",
        )
        attack_scores = score_profiles(detector, attack_features, device)
        attack_counts = np.count_nonzero(attack_profiles > 0, axis=1)
        attack_groups = quartile_labels(attack_counts, cutoffs)
        role = "HELD_OUT" if attack_name == fold else "KNOWN"

        for repeat in range(TEST_SUBSAMPLE_REPEATS):
            # Critically, DO NOT seed on the held-out fold. Identical attack
            # samples are evaluated by all six trained models for fair comparison.
            rng = np.random.default_rng(SEED + 1000 * repeat + 31 * attack_index)
            permutation = rng.permutation(len(attack_profiles))

            for ratio in RATIOS:
                selected_count = min(
                    len(attack_profiles),
                    max(1, round(len(genuine) * ratio)),
                )
                chosen = permutation[:selected_count]
                selected_scores = attack_scores[chosen]
                selected_groups = attack_groups[chosen]

                # PHASE 2.2: Overall detection at this exact attack ratio.
                overall_metrics = compute_metrics(genuine_scores, selected_scores)
                low_density_rows.append(metrics_row(
                    fold, attack_name, role, ratio, repeat, "ALL",
                    len(genuine_scores), len(selected_scores), overall_metrics,
                ))

                # PHASE 2.3: Reuse EXACTLY the same probability predictions,
                # but report them separately by user interaction density.
                for q in range(1, 5):
                    g_scores_q = genuine_scores[genuine_groups == q]
                    a_scores_q = selected_scores[selected_groups == q]
                    quartile_metrics = compute_metrics(g_scores_q, a_scores_q)
                    sparsity_rows.append(metrics_row(
                        fold, attack_name, role, ratio, repeat, f"Q{q}",
                        len(g_scores_q), len(a_scores_q), quartile_metrics,
                    ))

        del attack_features, attack_scores

    # Save fold's COMPLETE tables so a Colab disconnect does not erase them.
    atomic_csv(overall_file, low_density_rows, DETAIL_COLUMNS)
    atomic_csv(sparse_file, sparsity_rows, DETAIL_COLUMNS)
    print(f"[SAVED] {fold} evaluation CSVs in {fold_root}", flush=True)

    del cnn, detector, genuine_features, genuine_scores
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ============================================================================
# 8. MAIN: VALIDATE -> BUILD SHARED TEST POOLS -> RUN 6 FOLDS -> SAVE SUMMARY
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2.2 + 2.3 without CNN retraining")
    parser.add_argument(
        "--check-only", action="store_true",
        help="Only verify checkpoint paths and configuration; do not evaluate.",
    )
    args = parser.parse_args()
    ensure_existing_models_and_inputs()
    if args.check_only:
        print("[CHECK] All required files found. Safe to run the evaluation.", flush=True)
        return

    device = choose_device()
    print(f"[DEVICE] {device} — NO training, inference only.", flush=True)
    if Path("/content").is_dir() and device.type != "cuda":
        raise RuntimeError(
            "Colab has no CUDA GPU selected. Enable a GPU runtime first "
            "to avoid running expensive feature extraction on CPU."
        )

    check_saved_analysis_config()
    set_seed(SEED)

    # Rebuild the SAME source/test splits used in LOAO, but sample 5% from
    # each attack test pool. These are all test examples, never training data.
    genuine, attack_pools = create_maximum_test_pool()
    genuine_counts = np.count_nonzero(genuine > 0, axis=1)
    cutoffs = np.quantile(genuine_counts, [0.25, 0.50, 0.75])
    genuine_groups = quartile_labels(genuine_counts, cutoffs)
    print(f"[QUARTILES] Genuine-user interaction thresholds: {cutoffs.tolist()}", flush=True)
    print("[QUARTILES] Genuine test users per group:",
          {f"Q{q}": int((genuine_groups == q).sum()) for q in range(1, 5)},
          flush=True)
    if len(np.unique(cutoffs)) < len(cutoffs):
        print("[WARNING] Tied interaction quartiles; some Q groups may be uneven.", flush=True)
    atomic_json(OUTPUT_ROOT / "quartile_boundaries.json", {
        "genuine_test_users": len(genuine),
        "interaction_cutoffs_Q1_Q2_Q3": cutoffs.tolist(),
        "genuine_users_per_quartile": {
            f"Q{q}": int((genuine_groups == q).sum()) for q in range(1, 5)
        },
        "source": "genuine test profiles from MovieLens-1M",
    })

    print("[T5] Loading precomputed 16D item semantic embeddings...", flush=True)
    item_labels_array = load_item_label_data(str(FEATURES_FILE), ITEM_COUNT)
    item_labels = torch.as_tensor(item_labels_array, dtype=torch.float32)
    if tuple(item_labels.shape) != (ITEM_COUNT, LABEL_DIM):
        raise RuntimeError(f"Wrong semantic embedding shape: {tuple(item_labels.shape)}")

    for i, fold in enumerate(ATTACK_FOLDS, start=1):
        print(f"\n[PROGRESS] Fold {i}/{len(ATTACK_FOLDS)}", flush=True)
        run_fold(fold, genuine, attack_pools, cutoffs, item_labels, device)

    # Combine the per-fold CSVs to make convenient paper-ready tables.
    low_rows, sparse_rows = [], []
    for fold in ATTACK_FOLDS:
        low_rows.extend(read_csv_rows(OUTPUT_ROOT / fold / "low_density_detail.csv"))
        sparse_rows.extend(read_csv_rows(OUTPUT_ROOT / fold / "sparse_user_detail.csv"))

    atomic_csv(OUTPUT_ROOT / "low_attack_ratio_results.csv", low_rows, DETAIL_COLUMNS)
    atomic_csv(OUTPUT_ROOT / "sparse_user_results.csv", sparse_rows, DETAIL_COLUMNS)
    atomic_csv(OUTPUT_ROOT / "low_attack_ratio_summary.csv", summarize(low_rows), SUMMARY_COLUMNS)
    atomic_csv(OUTPUT_ROOT / "sparse_user_summary.csv", summarize(sparse_rows), SUMMARY_COLUMNS)

    # The quickest table to read: the HELD-OUT attack for each fold + ratio.
    zero_day = [row for row in summarize(low_rows) if row["attack_role"] == "HELD_OUT"]
    atomic_csv(OUTPUT_ROOT / "zero_day_low_density_summary.csv", zero_day, SUMMARY_COLUMNS)

    print("\n" + "=" * 75, flush=True)
    print("PHASE 2.2 AND 2.3 EVALUATION COMPLETED — NO MODEL RETRAINING", flush=True)
    print(f"Results folder: {OUTPUT_ROOT}", flush=True)
    print("Main paper tables:", flush=True)
    for name in (
        "zero_day_low_density_summary.csv",
        "low_attack_ratio_summary.csv",
        "sparse_user_summary.csv",
    ):
        print("  " + name, flush=True)
    print("=" * 75, flush=True)


if __name__ == "__main__":
    main()
