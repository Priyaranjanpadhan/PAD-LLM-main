"""PAD-LLM Phase 3 B5: smoke test and post-B4 open-set evaluation.

SMOKE TEST (now, from repository root):
    python phase3/run_phase3_B5.py --smoke-test

AFTER YOU HAVE FINISHED B4 CNN + DETECTOR TRAINING (later):
    python phase3/run_phase3_B5.py --check-only
    python phase3/run_phase3_B5.py --evaluate --max-folds 6

B5 DOES NOT TRAIN CNNs or detectors. It loads completed B4 fold weights.
It creates a disjoint validation/evaluation partition of the held-out
B4 test populations, and calibrates using genuine + KNOWN attack profiles
ONLY. The unknown attack is NEVER included in calibration.

Note: the B5 test subsets differ from prior B4/Phase-2 test splits because
some test profiles are reserved for independent threshold calibration.
Only compare B4/B5 on MATCHED held-out evaluation subsets later.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_source(path: Path, alias: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 3 prerequisite: {path}")
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


b5 = _load_source(HERE / "main_phase3_B5.py", "pad_b5_open_set")
b4 = b5.load_b4_model_module()

FOLDS = ["GOAT", "ItemAE", "WRMF_SGD", "Random", "Average", "Bandwagon"]
ITEM_COUNT, SEMANTIC_DIM, PERFORMER_HEADS = 3952, 16, 4
SEED = 42
TEST_POOL_RATIO = 0.05
VALIDATION_FRACTION = 0.25
KNOWN_COVERAGE = 0.95
BINARY_THRESHOLD = 0.95


def b4_root() -> Path:
    if Path("/content").is_dir():
        return Path("/content/drive/MyDrive/PAD-LLM/phase3/b4")
    return ROOT / "phase3/results/b4"


def b5_root() -> Path:
    if Path("/content").is_dir():
        return Path("/content/drive/MyDrive/PAD-LLM/phase3/b5")
    return ROOT / "phase3/results/b5"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def smoke_test() -> None:
    """Test decision logic AND actual B4 feature integration. No training."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    # Small synthetic examples with obvious prototype geometry:
    # class 0 is direction e0, class 1 is e1, unseen is direction e2.
    centers = torch.zeros(2, 64)
    centers[0, 0] = 1.0
    centers[1, 1] = 1.0
    head = b5.PrototypeOpenSetHead(centers, binary_threshold=0.95)
    generator = torch.Generator().manual_seed(4)
    normal = F.normalize(centers[0] + 0.01 * torch.randn(20, 64, generator=generator), dim=1)
    known = F.normalize(centers[1] + 0.01 * torch.randn(20, 64, generator=generator), dim=1)
    cal_features = torch.cat([normal, known], dim=0)
    cal_labels = torch.tensor([0] * 20 + [1] * 20)
    thresholds = head.calibrate(cal_features, cal_labels, coverage=0.95)
    assert thresholds.shape == (2,) and torch.isfinite(thresholds).all()

    in_normal = centers[0:1].repeat(2, 1)
    in_known = centers[1:2].repeat(2, 1)
    unknown = torch.zeros(2, 64)
    unknown[:, 2] = 1.0
    query = torch.cat([in_normal, in_known, unknown], dim=0)
    logits = torch.tensor([-5., -4., 6., 7., -5., 7.])
    result = head.predict(query, logits)
    assert result["predictions"].tolist() == [0, 0, 1, 1, 2, 2]
    assert result["unknown_suspicious"].tolist() == [False]*4 + [True]*2
    assert result["distance_to_normal"].shape == (6,)
    assert result["distance_to_known_attack"].shape == (6,)
    assert len(head.config()["distance_thresholds"]) == 2

    # Verify no unknown labels or empty validation groups can calibrate.
    try:
        head.calibrate(query, torch.tensor([0, 0, 1, 1, 2, 2]))
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown labels incorrectly accepted during calibration")
    try:
        head.calibrate(normal, torch.zeros(len(normal), dtype=torch.long))
    except ValueError:
        pass
    else:
        raise AssertionError("Calibration incorrectly accepted a missing class")

    # Confirm the B5 head accepts features produced by the real B4 architecture
    # (the B4 CNN is untrained here, so predictions are NOT meaningful).
    small_items = 64
    genres = torch.eye(4).repeat(small_items // 4, 1)
    model = b4.CNN3DRepresentationB4(
        device, small_items, SEMANTIC_DIM, PERFORMER_HEADS,
        gate_variant="B4", genre_matrix=genres,
    ).eval()
    ratings = torch.zeros(3, small_items)
    ratings[0, [1, 10, 20]] = 1.0
    ratings[1, [3, 12, 24]] = 1.0
    ratings[2, [5, 21, 28]] = 1.0
    labels = torch.randn(small_items, SEMANTIC_DIM)
    with torch.no_grad():
        representations = model.forward_features(ratings, labels)
        logits = model(ratings, labels).view(-1)
    assert representations.shape == (3, 64)
    assert model.class_prototypes.shape == (2, 64)
    assert logits.shape == (3,)
    # A temporary calibration using the synthetic centers tests interfaces only.
    head.predict(representations, logits)
    print("[SMOKE PASS] B5: B4 features=(3,64), prototypes=(2,64); "
          "class-conditional calibration, 3-state decisions, unknown rejection, "
          "and leakage/edge-case guards passed; NO training.", flush=True)


def preflight(max_folds: int, source: Path):
    """Verify prerequisite B4 fold checkpoints and dataset files; no training."""
    needed = [ROOT / "data/ml-1m/user_item_rating.txt",
              ROOT / "data/ml-1m/movies.dat",
              ROOT / "attack_data/ml-1m/Sur-ItemAE_fake_data_best.npz",
              ROOT / "attack_data/ml-1m/Sur-WeightedMF-sgd_fake_data_best.npz",
              ROOT / "attack_data/ml-1m/posion_data.txt"]
    for fold in FOLDS[:max_folds]:
        needed.extend([source/fold/"cnn"/"cnn_final.pth",
                       source/fold/"detector"/"detector_final.pth"])
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required data/B4 checkpoints missing:\n" + "\n".join(missing))
    print(f"[CHECK PASS] B4 model/detector pairs found for {max_folds} fold(s); NO training.", flush=True)


def features_in_batches(cnn, ratings: np.ndarray, item_labels: torch.Tensor,
                        device: torch.device, batch_size: int = 32) -> torch.Tensor:
    """Extract 64D features on GPU if available, return CPU tensors."""
    if len(ratings) == 0:
        raise ValueError("Empty features request")
    result = []
    with torch.no_grad():
        for idx in range(0, len(ratings), batch_size):
            batch = torch.from_numpy(ratings[idx:idx+batch_size].copy()).to(device=device, dtype=torch.float32)
            features = cnn.forward_features(batch, item_labels)
            result.append(features.detach().cpu())
    return torch.cat(result, dim=0)


def detector_logits(detector, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        result = []
        for chunk in features.split(256):
            result.append(detector(chunk.to(device).float()).reshape(-1).detach().cpu())
    return torch.cat(result)


def stratified_parts(arr: np.ndarray, fraction: float, seed: int):
    """One disjoint calibration/evaluation split of a single class's profiles."""
    if len(arr) < 4:
        raise ValueError("Need at least four profiles to split")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(arr))
    n_cal = max(1, int(round(len(arr)*fraction)))
    if n_cal >= len(arr):
        raise ValueError("No profiles left for evaluation")
    return arr[perm[:n_cal]], arr[perm[n_cal:]]


def three_state_metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Class order normal / known malicious / held-out unknown attack."""
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(truth, pred):
        cm[int(t), int(p)] += 1
    def div(a,b):
        return float(a/b) if b else None
    known_attack_recall = div(cm[1,1], cm[1,:].sum())
    unknown_recall = div(cm[2,2], cm[2,:].sum())
    unknown_precision = div(cm[2,2], cm[:,2].sum())
    return {
        "confusion_matrix_rows_truth_cols_prediction": cm.tolist(),
        "class_labels": ["normal", "known malicious", "unknown/suspicious"],
        "normal_accuracy": div(cm[0,0], cm[0,:].sum()),
        "known_attack_recall": known_attack_recall,
        "unknown_attack_recall": unknown_recall,
        "unknown_flag_precision": unknown_precision,
        "false_unknown_rate_on_known_and_normal": div(cm[0,2]+cm[1,2], cm[0,:].sum()+cm[1,:].sum()),
        "samples": int(len(truth)),
        "normal_count": int((truth==0).sum()),
        "known_attack_count": int((truth==1).sum()),
        "held_out_attack_count": int((truth==2).sum()),
    }


def evaluate_fold(held_out: str, source: Path, output: Path, device: torch.device) -> dict:
    """Fit ONLY known-calibration thresholds; evaluate on disjoint examples."""
    from data_clean.ml_1m_attack_data_clean import create_loao_split
    from detection_model import DetectionModel
    from utils import load_item_label_data

    fold = create_loao_split(
        normal_file=str(ROOT/"data/ml-1m/user_item_rating.txt"),
        attack_file1=str(ROOT/"attack_data/ml-1m/Sur-ItemAE_fake_data_best.npz"),
        attack_file2=str(ROOT/"attack_data/ml-1m/Sur-WeightedMF-sgd_fake_data_best.npz"),
        attack_file3=str(ROOT/"attack_data/ml-1m/posion_data.txt"),
        held_out_attack=held_out,
        proportion=TEST_POOL_RATIO,
        seed=SEED,
    )
    if held_out in fold["train_attack_families"]:
        raise RuntimeError("Held-out leakage in training")
    genuine = fold["genuine_test"]
    if not np.all(genuine[:,-1]==0):
        raise AssertionError("Expected genuine test labels=0")
    cal_norm, eval_norm = stratified_parts(genuine, VALIDATION_FRACTION, SEED+100)
    cal_parts = [cal_norm]
    eval_parts = [eval_norm]
    truth_parts = [np.zeros(len(eval_norm),dtype=np.int64)]
    known_families = [a for a in FOLDS if a!=held_out]
    for idx, attack in enumerate(known_families):
        attack_rows = fold["test_by_attack"][attack]
        attack_rows = attack_rows[attack_rows[:,-1]==1]
        cal_attack, eval_attack = stratified_parts(attack_rows, VALIDATION_FRACTION, SEED+101+idx)
        cal_parts.append(cal_attack)
        eval_parts.append(eval_attack)
        truth_parts.append(np.ones(len(eval_attack), dtype=np.int64))
    # Held-out attack profiles are used ONLY for evaluation, never calibration.
    unknown = fold["test_by_attack"][held_out]
    unknown = unknown[unknown[:,-1]==1]
    eval_parts.append(unknown)
    truth_parts.append(np.full(len(unknown), 2, dtype=np.int64))
    calibration = np.concatenate(cal_parts, axis=0)
    evaluation = np.concatenate(eval_parts, axis=0)
    truth = np.concatenate(truth_parts)
    assert set(calibration[:,-1])=={0.0,1.0}
    assert len(unknown)>0 and len(truth)==len(evaluation)

    item_labels = torch.as_tensor(load_item_label_data(str(ROOT/"data/ml-1m/movies.dat"), ITEM_COUNT), dtype=torch.float32).to(device)
    if item_labels.shape!=(ITEM_COUNT,SEMANTIC_DIM):
        raise RuntimeError(f"Expected item embeddings (3952,16), got {tuple(item_labels.shape)}")
    genres = b4.B3.load_movie_genres(ROOT/"data/ml-1m/movies.dat", ITEM_COUNT)
    cnn = b4.CNN3DRepresentationB4(device, ITEM_COUNT, SEMANTIC_DIM,
                                   PERFORMER_HEADS, gate_variant="B4", genre_matrix=genres).to(device).eval()
    cnn_path = source/held_out/"cnn"/"cnn_final.pth"
    detector_path = source/held_out/"detector"/"detector_final.pth"
    try:
        cnn_weights = torch.load(cnn_path, map_location=device, weights_only=True)
    except TypeError:
        cnn_weights = torch.load(cnn_path, map_location=device)
    cnn.load_state_dict(cnn_weights, strict=True)

    detector = DetectionModel().to(device)
    with torch.no_grad():
        detector(torch.zeros((1,64),device=device))  # initialize nn.LazyLinear
    try:
        detector_weights = torch.load(detector_path, map_location=device, weights_only=True)
    except TypeError:
        detector_weights = torch.load(detector_path, map_location=device)
    detector.load_state_dict(detector_weights, strict=True)
    detector.eval()

    print(f"[B5 {held_out}] Extracting known-only calibration features...", flush=True)
    cal_z = features_in_batches(cnn, calibration[:,:-1], item_labels, device)
    eval_z = features_in_batches(cnn, evaluation[:,:-1], item_labels, device)
    # B5 calibration labels are binary KNOWN labels ONLY; never held-out family.
    head = b5.PrototypeOpenSetHead(cnn.class_prototypes, BINARY_THRESHOLD)
    thresholds = head.calibrate(cal_z, torch.from_numpy(calibration[:,-1]), KNOWN_COVERAGE)
    logits = detector_logits(detector, eval_z, device)
    output_predictions = head.predict(eval_z, logits)
    ypred = output_predictions["predictions"].cpu().numpy()
    metrics = three_state_metrics(truth, ypred)
    metrics.update({
        "fold_held_out": held_out,
        "known_families": known_families,
        "calibration_genuine": len(cal_norm),
        "calibration_known_attack_profiles": len(calibration)-len(cal_norm),
        "thresholds": thresholds.tolist(),
        "coverage_target": KNOWN_COVERAGE,
        "binary_threshold": BINARY_THRESHOLD,
        "calibration_profiles_disjoint_from_evaluation": True,
        "held_out_used_for_calibration": False,
        "note": "Unknown/suspicious is an embedding novelty flag; not proof of a new attack type.",
    })
    folder = output/held_out
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder/"open_set_calibration.json"
    result_path = folder/"b5_results.json"
    # Avoid creating partial JSON on disconnect.
    for path, payload in [(config_path, head.config()), (result_path, metrics)]:
        tmp = path.with_suffix(path.suffix+".tmp")
        tmp.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        os.replace(tmp,path)
    print(f"[B5 {held_out}] unknown recall={metrics['unknown_attack_recall']}, "
          f"unknown precision={metrics['unknown_flag_precision']}, "
          f"false unknown rate={metrics['false_unknown_rate_on_known_and_normal']}",flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="B5 post-training open-set scoring; no CNN training")
    parser.add_argument("--smoke-test", action="store_true", help="CPU interface/logic test only")
    parser.add_argument("--check-only", action="store_true", help="Check trained B4 files only")
    parser.add_argument("--evaluate", action="store_true", help="Post-training B5 evaluation")
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--b4-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if not (1 <= args.max_folds <= 6):
        parser.error("--max-folds must be 1..6")
    if args.smoke_test:
        smoke_test()
        return
    if not (args.check_only or args.evaluate):
        parser.error("Use --smoke-test now, or --check-only/--evaluate after B4 training")
    source = args.b4_root or b4_root()
    output = args.output_root or b5_root()
    preflight(args.max_folds, source)
    if args.check_only:
        print(f"[CHECK] B4 source: {source}\n[CHECK] B5 output: {output}")
        return
    device = get_device()
    for fold in FOLDS[:args.max_folds]:
        result_path = output/fold/"b5_results.json"
        if result_path.is_file():
            print(f"[SKIP] Existing B5 result for {fold}; preserving prior result.")
            continue
        evaluate_fold(fold, source, output, device)
    print(f"[B5] Completed; results stored under {output}")


if __name__=="__main__":
    main()
