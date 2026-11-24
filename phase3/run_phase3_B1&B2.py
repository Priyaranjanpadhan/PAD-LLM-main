"""Phase 3 B1/B2 runner for PAD-LLM on MovieLens-1M.

Each held-out attack gets its OWN newly initialized CNN and detector.
Folds run sequentially in one process, but their model weights are never shared.

Resume strategy:
- cnn_checkpoint.pth: train_cnn_3d saves model/optimizer/scheduler/scaler/epoch
  after every completed epoch. Rerunning resumes that fold's CNN.
- cnn_final.pth: indicates that fold's CNN training is finished.
- features/*.pt: cached 64-D user features; missing caches are generated again.
- detector_final.pth: detector weights saved after detector training completes.
- fold_results.json: marks a fully completed fold; completed folds are skipped.

Run from the repository root:
    python "phase3/run_phase3_B1&B2.py" --gate both --smoke-test
    python "phase3/run_phase3_B1&B2.py" --gate both --check-only
    python "phase3/run_phase3_B1&B2.py" --gate B2 --max-folds 1
    python "phase3/run_phase3_B1&B2.py" --gate both

In Colab, mount Drive before running so the checkpoints persist across runtime
restarts. By default, this script writes to MyDrive/PAD-LLM/phase3/B1 or B2.
LOAO is the evaluation protocol, not an output path or Phase 2 dependency.
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import os
import random
from pathlib import Path
import importlib.util
import sys
from statistics import mean
from typing import Any

import numpy as np
import torch

# This runner is directly executable from the repository root. Add the root to
# Python's import path so existing project files (cnn_model, utils, etc.) load.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The requested filename contains an ampersand (&). Python does not allow & in
# a normal `from ... import ...` statement, so load that file by its exact path.
MODEL_FILE = Path(__file__).with_name("main_phase3_B1&B2.py")
model_spec = importlib.util.spec_from_file_location("phase3_b1b2_model", MODEL_FILE)
if model_spec is None or model_spec.loader is None:
    raise ImportError(f"Cannot load the Phase 3 model file: {MODEL_FILE}")
model_module = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_module)
CNN3DAdaptiveGating = model_module.CNN3DAdaptiveGating

from cnn_model import train_cnn_3d
from data_clean.ml_1m_attack_data_clean import create_loao_split
from detection_model import DetectionModel, train_detection_model
from utils import load_item_label_data


# ---------------------------------------------------------------------------
# 1. Configuration: keep these values fixed while resuming an existing run.
# ---------------------------------------------------------------------------
NORMAL_FILE = "data/ml-1m/user_item_rating.txt"
FEATURES_FILE = "data/ml-1m/movies.dat"
ATTACK_FILE_ITEMAE = "attack_data/ml-1m/Sur-ItemAE_fake_data_best.npz"
ATTACK_FILE_WRMF = "attack_data/ml-1m/Sur-WeightedMF-sgd_fake_data_best.npz"
ATTACK_FILE_GOAT = "attack_data/ml-1m/posion_data.txt"

ITEM_NUMBER = 3952
ITEM_LABEL_NUMBER = 16
PERFORMER_HEADS = 4
CNN_EPOCHS = 30
CNN_LEARNING_RATE = 0.0001
CNN_BATCH_SIZE = 8
ATTACK_TEST_PROPORTION = 0.01
DETECTION_THRESHOLD = 0.95
RANDOM_SEED = 42

# One script executes all six folds in this order.
ATTACK_FOLDS = ["GOAT", "ItemAE", "WRMF_SGD", "Random", "Average", "Bandwagon"]

# The active variant is set by main() for each experiment.
# B1 and B2 always use DIFFERENT output directories and checkpoints.
ACTIVE_GATE_VARIANT = "B2"
OUTPUT_ROOT = Path("phase3/results/B2")

# Optional CLI override. The override is a PARENT folder; the runner then adds B1/B2.
CUSTOM_OUTPUT_PARENT: Path | None = None


def choose_output_root(variant: str) -> Path:
    """Every artifact stays under a Phase 3 folder, never the Phase 2 directory.

    On Colab:  MyDrive/PAD-LLM/phase3/<variant>/<held_out_attack>/...
    Locally:   <repo>/phase3/results/<variant>/<held_out_attack>/...
    You can override the parent folder with --output-root.
    """
    if CUSTOM_OUTPUT_PARENT is not None:
        return CUSTOM_OUTPUT_PARENT / variant
    if Path("/content").is_dir():
        drive_dir = Path("/content/drive/MyDrive")
        if not drive_dir.is_dir():
            raise RuntimeError("Mount Google Drive before starting training.")
        return drive_dir / "PAD-LLM" / "phase3" / variant
    return Path("phase3") / "results" / variant


# ---------------------------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    """Reset random generators before each independently trained fold."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device() -> torch.device:
    """Prefer CUDA on Colab, then Apple MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Write via a temporary file to reduce partial JSON files on interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    os.replace(temp, path)


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    """Cache feature tensors on CPU so the cache is portable between devices."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor.detach().cpu(), temp)
    os.replace(temp, path)


def load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def save_model(model: torch.nn.Module, path: Path) -> None:
    """Save final model weights separately from the resumable CNN checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temp)
    os.replace(temp, path)


def load_model(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    try:
        weights = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        weights = torch.load(path, map_location=device)
    model.load_state_dict(weights)


def validate_run_configuration() -> None:
    """Stop if an existing output folder belongs to different hyperparameters."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset": "MovieLens-1M",
        "seed": RANDOM_SEED,
        "attack_test_proportion_per_family": ATTACK_TEST_PROPORTION,
        "cnn_epochs": CNN_EPOCHS,
        "cnn_learning_rate": CNN_LEARNING_RATE,
        "cnn_batch_size": CNN_BATCH_SIZE,
        "threshold": DETECTION_THRESHOLD,
        "item_number": ITEM_NUMBER,
        "item_label_number": ITEM_LABEL_NUMBER,
        "performer_heads": PERFORMER_HEADS,
        "architecture": f"CNN3DAdaptiveGating attention pooling {ACTIVE_GATE_VARIANT}",
        "gate_variant": ACTIVE_GATE_VARIANT,
    }
    config_path = OUTPUT_ROOT / "run_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            old_config = json.load(file)
        if old_config != config:
            raise RuntimeError(
                "The existing Phase 3 variant folder was created with different settings. "
                "Use a new OUTPUT_ROOT or restore its original configuration."
            )
    else:
        save_json(config_path, config)


def preflight_check() -> None:
    """Check dependencies and data files; performs NO training or data generation."""
    needed = [NORMAL_FILE, FEATURES_FILE, ATTACK_FILE_ITEMAE,
              ATTACK_FILE_WRMF, ATTACK_FILE_GOAT]
    missing = [p for p in needed if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Missing files (run from repository root): " + ", ".join(missing))
    if "checkpoint_path" not in inspect.signature(train_cnn_3d).parameters:
        raise RuntimeError("cnn_model.train_cnn_3d needs checkpoint_path support.")
    if not callable(create_loao_split):
        raise RuntimeError("Corrected data builder create_loao_split is required.")
    print("[PREFLIGHT PASS] Dataset files and training interfaces found; NO training.", flush=True)


# ---------------------------------------------------------------------------
# 3. Data integrity checks
# ---------------------------------------------------------------------------
def audit_fold_data(data: dict[str, Any], held_out: str) -> None:
    """Verify that the held-out attack is absent from all training collections."""
    expected_train_families = set(ATTACK_FOLDS) - {held_out}
    actual_train_families = set(data["train_attack_families"])

    assert held_out not in data["train_by_attack"], (
        f"Leakage risk: {held_out} appears in train_by_attack."
    )
    assert actual_train_families == expected_train_families, (
        f"Expected training attacks {expected_train_families}, got {actual_train_families}."
    )
    assert data["train"].shape[1] == ITEM_NUMBER + 1
    assert np.all(np.isin(data["train"][:, -1], [0, 1]))
    assert set(data["test_by_attack"]) == set(ATTACK_FOLDS)

    for attack_name, samples in data["test_by_attack"].items():
        assert samples.shape[1] == ITEM_NUMBER + 1
        assert np.all(np.isin(samples[:, -1], [0, 1]))
        assert np.sum(samples[:, -1] == 0) > 0
        assert np.sum(samples[:, -1] == 1) > 0


# ---------------------------------------------------------------------------
# 4. Feature extraction and cache
# ---------------------------------------------------------------------------
def extract_features(
    cnn: torch.nn.Module,
    ratings: torch.Tensor,
    item_labels: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Return the CNN's 64-D latent features, in batches, without gradients."""
    cnn.eval()
    batches = []
    labels_on_device = item_labels.to(device)
    with torch.no_grad():
        for start in range(0, len(ratings), batch_size):
            end = min(start + batch_size, len(ratings))
            batch = ratings[start:end].to(device).float()
            # forward_features() returns the 64-D vector before CNN's final logit.
            features = cnn.forward_features(batch, labels_on_device)
            batches.append(features.detach().cpu())
            if start == 0 or end == len(ratings):
                print(f"      Feature extraction: {end}/{len(ratings)}", flush=True)
    if not batches:
        raise ValueError("Cannot extract features from an empty dataset.")
    return torch.cat(batches, dim=0)


def get_features(
    cnn: torch.nn.Module,
    ratings: torch.Tensor,
    item_labels: torch.Tensor,
    device: torch.device,
    cache_path: Path,
) -> torch.Tensor:
    """Load an existing feature cache, or extract and save it exactly once."""
    if cache_path.exists():
        print(f"      Using feature cache: {cache_path.name}", flush=True)
        return load_tensor(cache_path)
    print(f"      Creating feature cache: {cache_path.name}", flush=True)
    features = extract_features(cnn, ratings, item_labels, device)
    save_tensor(cache_path, features)
    return features


# ---------------------------------------------------------------------------
# 5. Metrics (same binary rule: probability > 0.95 means malicious)
# ---------------------------------------------------------------------------
def evaluate_detector(
    detector: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    detector.eval()
    with torch.no_grad():
        logits = detector(features.to(device).float()).view(-1)
        probabilities = torch.sigmoid(logits)
        predicted = (probabilities > DETECTION_THRESHOLD).long()
        actual = labels.to(device).view(-1).long()

    tp = int(((predicted == 1) & (actual == 1)).sum().item())
    fn = int(((predicted == 0) & (actual == 1)).sum().item())
    fp = int(((predicted == 1) & (actual == 0)).sum().item())
    tn = int(((predicted == 0) & (actual == 0)).sum().item())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sample_count": int(len(actual)),
        "genuine_count": int((actual == 0).sum().item()),
        "malicious_count": int((actual == 1).sum().item()),
        "threshold": DETECTION_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# 6. Run one attack-held-out fold, using that fold's own files/checkpoints
# ---------------------------------------------------------------------------
def run_fold(held_out: str, item_labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    print("\n" + "=" * 76, flush=True)
    print(f"FOLD: {held_out} IS HELD OUT", flush=True)
    print("=" * 76, flush=True)

    fold_dir = OUTPUT_ROOT / held_out
    cnn_dir = fold_dir / "cnn"
    feature_dir = fold_dir / "features"
    detector_dir = fold_dir / "detector"
    for folder in (cnn_dir, feature_dir, detector_dir):
        folder.mkdir(parents=True, exist_ok=True)

    cnn_checkpoint = cnn_dir / "cnn_checkpoint.pth"
    cnn_final = cnn_dir / "cnn_final.pth"
    detector_final = detector_dir / "detector_final.pth"
    result_file = fold_dir / "fold_results.json"
    status_file = fold_dir / "fold_status.json"

    # A completed fold is reused on rerun; no CNN or detector training is repeated.
    if result_file.exists():
        with result_file.open("r", encoding="utf-8") as file:
            previous = json.load(file)
        if previous.get("status") == "completed":
            print(f"[SKIP] {held_out} fold already completed.", flush=True)
            return previous

    # Rebuilding data with the same seed gives the same source splits when resuming.
    print("[DATA] Building and auditing this fold's datasets...", flush=True)
    fold_data = create_loao_split(
        normal_file=NORMAL_FILE,
        attack_file1=ATTACK_FILE_ITEMAE,
        attack_file2=ATTACK_FILE_WRMF,
        attack_file3=ATTACK_FILE_GOAT,
        held_out_attack=held_out,
        proportion=ATTACK_TEST_PROPORTION,
        seed=RANDOM_SEED,
    )
    audit_fold_data(fold_data, held_out)

    train_array = fold_data["train"]
    y_train_np = train_array[:, -1].astype(np.float32)
    train_ratings = torch.tensor(train_array[:, :-1], dtype=torch.float32)
    train_labels = torch.tensor(y_train_np, dtype=torch.float32).view(-1, 1)

    print(
        f"[DATA] Training rows={len(train_array)}; genuine={int((y_train_np == 0).sum())}; "
        f"known malicious={int((y_train_np == 1).sum())}",
        flush=True,
    )
    print(f"[DATA] Training families: {fold_data['train_attack_families']}", flush=True)
    print(
        f"[DATA] Held-out {held_out} training rows excluded: "
        f"{fold_data['excluded_attack_train_rows']}",
        flush=True,
    )
    save_json(status_file, {"fold": held_out, "stage": "data_audited"})

    # A) CNN: load completed weights OR train/resume from this fold's epoch checkpoint.
    set_all_seeds(RANDOM_SEED)
    cnn = CNN3DAdaptiveGating(
        device, ITEM_NUMBER, ITEM_LABEL_NUMBER, PERFORMER_HEADS,
        gate_variant=ACTIVE_GATE_VARIANT,
    ).to(device)

    if cnn_final.exists():
        print(f"[CNN] Loading completed CNN: {cnn_final}", flush=True)
        load_model(cnn, cnn_final, device)
    else:
        if "checkpoint_path" not in inspect.signature(train_cnn_3d).parameters:
            raise RuntimeError(
                "train_cnn_3d() must support checkpoint_path. "
                "Use the checkpoint-enabled version of cnn_model.py first."
            )
        if cnn_checkpoint.exists():
            print(
                f"[CNN] Checkpoint found. Resuming {held_out} from its latest saved epoch.",
                flush=True,
            )
        else:
            print(f"[CNN] Starting fresh CNN for {held_out} fold.", flush=True)

        cnn = train_cnn_3d(
            cnn,
            train_ratings,
            item_labels,
            train_labels,
            CNN_LEARNING_RATE,
            epochs=CNN_EPOCHS,
            batch_size=CNN_BATCH_SIZE,
            checkpoint_path=str(cnn_checkpoint),
        )
        save_model(cnn, cnn_final)
        print(f"[CNN] Completed model saved: {cnn_final}", flush=True)

    save_json(status_file, {"fold": held_out, "stage": "cnn_completed"})

    # B) Extract/cache training features and six attack-specific test feature sets.
    train_features = get_features(
        cnn, train_ratings, item_labels, device, feature_dir / "train_features.pt"
    )
    if train_features.ndim != 2 or train_features.shape[1] != 64:
        raise AssertionError(f"Expected (N, 64) train features; got {tuple(train_features.shape)}")

    test_features = {}
    test_labels = {}
    for attack_name in ATTACK_FOLDS:
        test_array = fold_data["test_by_attack"][attack_name]
        ratings = torch.tensor(test_array[:, :-1], dtype=torch.float32)
        labels = torch.tensor(test_array[:, -1], dtype=torch.float32).view(-1)
        test_features[attack_name] = get_features(
            cnn,
            ratings,
            item_labels,
            device,
            feature_dir / f"test_{attack_name}_features.pt",
        )
        test_labels[attack_name] = labels
        if test_features[attack_name].ndim != 2 or test_features[attack_name].shape[1] != 64:
            raise AssertionError(f"Expected 64-D features for test attack {attack_name}.")

    save_json(
        status_file,
        {
            "fold": held_out,
            "stage": "features_completed",
            "train_feature_shape": list(train_features.shape),
        },
    )

    # C) Detector: one NEW classifier trained only on this fold's known attacks.
    detector = DetectionModel().to(device)
    # DetectionModel contains LazyLinear; one dummy forward initializes its input size
    # before we attempt to load a previously saved detector state dictionary.
    with torch.no_grad():
        _ = detector(train_features[:2].to(device).float())

    if detector_final.exists():
        print(f"[DETECTOR] Loading saved detector: {detector_final}", flush=True)
        load_model(detector, detector_final, device)
    else:
        print(f"[DETECTOR] Training fresh detector for {held_out} fold...", flush=True)
        # Preserve the existing train_detection_model implementation and its
        # original defaults (including its current class-weight calculation).
        train_detection_model(
            detector,
            train_features.to(device).float(),
            train_labels.to(device).float(),
        )
        save_model(detector, detector_final)
        print(f"[DETECTOR] Saved detector: {detector_final}", flush=True)

    save_json(status_file, {"fold": held_out, "stage": "detector_completed"})

    # D) Evaluate this ONE detector on each attack-family test set.
    metrics_by_attack = {}
    for attack_name in ATTACK_FOLDS:
        metrics = evaluate_detector(
            detector,
            test_features[attack_name],
            test_labels[attack_name],
            device,
        )
        metrics_by_attack[attack_name] = metrics
        role = "UNKNOWN / HELD-OUT" if attack_name == held_out else "KNOWN"
        print(
            f"[{role}] {attack_name:10s} | P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} | "
            f"TP={metrics['tp']} FP={metrics['fp']} "
            f"FN={metrics['fn']} TN={metrics['tn']}",
            flush=True,
        )

    # Average the five known attack families equally (macro average).
    known_names = [name for name in ATTACK_FOLDS if name != held_out]
    known_macro = {
        metric: mean(metrics_by_attack[name][metric] for name in known_names)
        for metric in ("precision", "recall", "f1", "specificity")
    }

    results = {
        "status": "completed",
        "held_out_attack": held_out,
        "gate_variant": ACTIVE_GATE_VARIANT,
        "seed": RANDOM_SEED,
        "attack_test_proportion_per_family": ATTACK_TEST_PROPORTION,
        "threshold": DETECTION_THRESHOLD,
        "training_profiles": int(len(train_array)),
        "training_genuine_profiles": int((y_train_np == 0).sum()),
        "training_known_attack_profiles": int((y_train_np == 1).sum()),
        "held_out_training_profiles_excluded": int(fold_data["excluded_attack_train_rows"]),
        "known_attack_macro_average": known_macro,
        "unknown_attack_metrics": metrics_by_attack[held_out],
        "metrics_by_attack": metrics_by_attack,
        "cnn_model": str(cnn_final),
        "detector_model": str(detector_final),
    }
    save_json(result_file, results)
    save_json(status_file, {"fold": held_out, "stage": "completed", "results": str(result_file)})

    # Release model/data objects before allocating the next fold's CNN on GPU.
    del detector, cnn, train_features, test_features, test_labels, fold_data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    print(f"[DONE] {held_out} held-out fold completed.", flush=True)
    return results


# ---------------------------------------------------------------------------
# 7. Incrementally save one CSV row after each completed fold.
# ---------------------------------------------------------------------------
def write_summary_csv(results_by_fold: dict[str, dict[str, Any]]) -> None:
    summary_path = OUTPUT_ROOT / "loao_summary.csv"
    columns = [
        "held_out_attack",
        "unknown_precision", "unknown_recall", "unknown_f1",
        "known_macro_precision", "known_macro_recall", "known_macro_f1",
        "known_macro_specificity",
        "training_profiles", "training_genuine_profiles",
        "training_known_attack_profiles", "held_out_training_profiles_excluded",
        "threshold",
    ]
    temp = summary_path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for fold in ATTACK_FOLDS:
            if fold not in results_by_fold:
                continue
            result = results_by_fold[fold]
            unknown = result["unknown_attack_metrics"]
            known = result["known_attack_macro_average"]
            writer.writerow({
                "held_out_attack": fold,
                "unknown_precision": unknown["precision"],
                "unknown_recall": unknown["recall"],
                "unknown_f1": unknown["f1"],
                "known_macro_precision": known["precision"],
                "known_macro_recall": known["recall"],
                "known_macro_f1": known["f1"],
                "known_macro_specificity": known["specificity"],
                "training_profiles": result["training_profiles"],
                "training_genuine_profiles": result["training_genuine_profiles"],
                "training_known_attack_profiles": result["training_known_attack_profiles"],
                "held_out_training_profiles_excluded": result["held_out_training_profiles_excluded"],
                "threshold": result["threshold"],
            })
    os.replace(temp, summary_path)
    print(f"[SUMMARY] Updated {summary_path}", flush=True)


# ---------------------------------------------------------------------------
# 8. Fast, data-free smoke test (NO training and NO dataset access)
# ---------------------------------------------------------------------------
def smoke_test(variant: str) -> None:
    """A very small CPU-only forward + backward test; never trains epochs.

    We intentionally use 64 *synthetic* movie positions, rather than 3,952,
    to test tensor dimensions without loading data or using the Colab GPU.
    The real training runner still uses the full 3,952-movie input.
    """
    device = torch.device("cpu")
    set_all_seeds(RANDOM_SEED)
    small_item_count = 64
    model = CNN3DAdaptiveGating(
        device, small_item_count, ITEM_LABEL_NUMBER, PERFORMER_HEADS,
        gate_variant=variant,
    ).to(device)
    model.train()
    ratings = torch.rand(2, small_item_count, device=device)
    labels = torch.rand(small_item_count, ITEM_LABEL_NUMBER, device=device)
    features = model.forward_features(ratings, labels)
    logits = model(ratings, labels)
    assert tuple(features.shape) == (2, 64), features.shape
    assert tuple(logits.shape) == (2, 1), logits.shape
    assert torch.isfinite(features).all() and torch.isfinite(logits).all()
    loss = logits.square().mean()
    loss.backward()
    assert model.gate_weight.grad is not None, "Gate weights received no gradients"
    assert model.gate_bias.grad is not None, "Gate bias received no gradients"
    assert torch.isfinite(model.gate_weight.grad).all()
    assert torch.isfinite(model.gate_bias.grad).all()
    with torch.no_grad():
        local, global_features = model.branch_features(ratings, labels)
        gates = model.compute_gate(local, global_features)
    expected = (2, 1) if variant == "B1" else (2, model.conv_output_size)
    assert local.shape == global_features.shape == (2, 32)
    assert tuple(gates.shape) == expected, (gates.shape, expected)
    print(f"[SMOKE PASS] {variant}: local/global=(2,32), features=(2,64), "
          f"logit=(2,1), gate={expected}; gradients finite; NO training.", flush=True)


# ---------------------------------------------------------------------------
# 9. Full experiment — separated models/checkpoints for each B1/B2 fold.
# ---------------------------------------------------------------------------
def run_experiment(variant: str, max_folds: int) -> None:
    global ACTIVE_GATE_VARIANT, OUTPUT_ROOT
    ACTIVE_GATE_VARIANT = variant
    OUTPUT_ROOT = choose_output_root(variant)
    print(f"\n[EXPERIMENT] {variant} held-out model training", flush=True)
    print(f"[OUTPUT] {OUTPUT_ROOT}", flush=True)
    validate_run_configuration()

    if "checkpoint_path" not in inspect.signature(train_cnn_3d).parameters:
        raise RuntimeError("Your train_cnn_3d() must support checkpoint_path.")
    device = choose_device()
    if Path("/content").is_dir() and device.type != "cuda":
        raise RuntimeError("In Colab select a CUDA GPU runtime, not a TPU or CPU.")
    print(f"[DEVICE] {device}", flush=True)

    item_labels = torch.as_tensor(
        load_item_label_data(FEATURES_FILE, ITEM_NUMBER), dtype=torch.float32
    )
    expected = (ITEM_NUMBER, ITEM_LABEL_NUMBER)
    if tuple(item_labels.shape) != expected:
        raise ValueError(f"Expected semantic embeddings {expected}.")

    # Resume completed folds if the runtime disconnected in a previous session.
    results_by_fold: dict[str, dict[str, Any]] = {}
    for fold in ATTACK_FOLDS:
        file = OUTPUT_ROOT / fold / "fold_results.json"
        if file.is_file():
            with file.open("r", encoding="utf-8") as fp:
                result = json.load(fp)
            if result.get("status") == "completed":
                results_by_fold[fold] = result

    write_summary_csv(results_by_fold)
    folds_to_run = ATTACK_FOLDS[:max_folds]
    for i, held_out in enumerate(folds_to_run, start=1):
        print(f"\n[RUN] {variant}: fold {i}/{len(folds_to_run)}: {held_out}", flush=True)
        results_by_fold[held_out] = run_fold(held_out, item_labels, device)
        write_summary_csv(results_by_fold)
    print(f"[DONE] {variant}: {len(folds_to_run)} fold(s) requested. "
          f"Summary: {OUTPUT_ROOT / 'loao_summary.csv'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAD-LLM Phase 3 B1/B2 gate evaluation")
    parser.add_argument(
        "--gate", choices=("B1", "B2", "both", "b1", "b2"), required=True,
        help="B1=sample-specific scalar, B2=sample-specific feature-wise vector",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Only test forward, backward and tensor shapes. NO dataset training.",
    )
    parser.add_argument(
        "--max-folds", type=int, default=6,
        help="Train just first N held-out folds (1 to 6); default 6.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Check dataset paths, dependencies, and output path without training.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Optional Phase 3 results parent directory; B1/B2 will be created under it.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_folds <= len(ATTACK_FOLDS):
        parser.error("--max-folds must be between 1 and 6")
    global CUSTOM_OUTPUT_PARENT
    CUSTOM_OUTPUT_PARENT = args.output_root
    variants = ("B1", "B2") if args.gate.lower() == "both" else (args.gate.upper(),)
    if args.smoke_test:
        for variant in variants:
            smoke_test(variant)
        return
    if args.check_only:
        preflight_check()
        for variant in variants:
            print(f"[OUTPUT] {variant}: {choose_output_root(variant)}", flush=True)
        return
    preflight_check()
    for variant in variants:
        run_experiment(variant, args.max_folds)


if __name__ == "__main__":
    main()
