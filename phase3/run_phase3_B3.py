"""Phase 3 B3 runner: CPU smoke-test and OPTIONAL full six-fold training.

B3 extends your already-tested B2 architecture; there is no need to copy the
full 30KB B1/B2 training code. Reuse its data audit, feature extraction, epoch
checkpoints, independent fold training and per-attack metrics unchanged.

Run from repository root (NO training):
    python phase3/run_phase3_B3.py --smoke-test

For future use only, after smoke tests and dataset index verification:
    python phase3/run_phase3_B3.py --check-only
    python phase3/run_phase3_B3.py --max-folds 1
    python phase3/run_phase3_B3.py

Checkpoints/results are under phase3/B3 or MyDrive/PAD-LLM/phase3/B3.
Never reads Phase 2-trained CNN weights or writes in Phase 2 directories.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
PHASE3_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_source(path: Path, module_name: str):
    """Load a .py file by path, including the B1&B2 filename containing '&'."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


b3 = load_source(PHASE3_DIR / "main_phase3_B3.py", "phase3_b3_main")
base_runner = load_source(PHASE3_DIR / "run_phase3_B1&B2.py", "phase3_b1b2_runner_for_b3")


def smoke_test() -> None:
    """CPU-only: test descriptor math, B2 inheritance and gradients; NO epochs."""
    device = torch.device("cpu")
    torch.manual_seed(42)
    items = 64
    genres = 4
    # Each synthetic movie has one deterministic genre for a clear entropy test.
    genre_matrix = torch.eye(genres).repeat(items // genres, 1)
    model = b3.CNN3DSemanticB3(
        device=device,
        item_number=items,
        item_label_number=16,
        performer_heads=4,
        gate_variant="B3",
        genre_matrix=genre_matrix,
    ).to(device)

    ratings = torch.zeros(3, items, dtype=torch.float32)
    # User 0: repeatedly chooses a single genre => entropy ~0.
    ratings[0, [0, 4, 8, 12]] = 1
    # User 1: chooses from four genres => positive entropy.
    ratings[1, [0, 1, 2, 3]] = 1
    # User 2: no observed interactions => zero descriptors (not NaN).
    labels = torch.randn(items, 16)

    stats = model.anomaly_descriptors(ratings, labels)
    assert stats.shape == (3, 4)
    assert torch.isfinite(stats).all()
    assert stats[0, 3].abs() < 1e-6, stats[0]
    assert stats[1, 3] > 0.9, stats[1]
    assert torch.equal(stats[2], torch.zeros(4)), stats[2]

    model.train()
    features = model.forward_features(ratings, labels)
    logits = model(ratings, labels)
    assert features.shape == (3, 64), features.shape
    assert logits.shape == (3, 1), logits.shape
    loss = logits.square().mean()
    loss.backward()
    assert model.anomaly_weight.grad is not None
    assert torch.isfinite(model.anomaly_weight.grad).all()
    assert model.gate_weight.grad is not None
    assert torch.isfinite(model.gate_weight.grad).all()
    # Ensure input-dependent anomaly update is connected to the output.
    assert model.anomaly_weight.grad.abs().sum() > 0, "B3 contribution disconnected"
    print("[SMOKE PASS] B3: descriptors=(3,4), gate=(3,32), "
          "features=(3,64), logits=(3,1); entropy/empty-user checks passed; "
          "B2+B3 gradients finite; NO training.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PAD-LLM Phase 3 B3")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Optional results parent folder, with B3 appended")
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 6:
        parser.error("--max-folds must be 1..6")

    if args.smoke_test:
        smoke_test()
        return

    # The 30-epoch training implementation is REUSED from B1/B2; we replace
    # only the model class and activate a brand-new B3 output directory.
    base_runner.CNN3DAdaptiveGating = b3.CNN3DSemanticB3
    base_runner.CUSTOM_OUTPUT_PARENT = args.output_root
    if args.check_only:
        base_runner.preflight_check()
        # Verify actual genre metadata parses into all 3,952 rows.
        genres = b3.load_movie_genres(ROOT / "data/ml-1m/movies.dat", 3952)
        fraction = (genres.sum(dim=1) > 0).float().mean().item()
        print(f"[CHECK] Movie genre coverage: {fraction:.2%} of catalog positions")
        if fraction < 0.80:
            raise RuntimeError("Suspicious movie-ID alignment/genre coverage; review before training.")
        print(f"[CHECK] B3 output: {base_runner.choose_output_root('B3')}")
        print("[CHECK] All required files found; no training performed.")
        return

    base_runner.preflight_check()
    base_runner.run_experiment("B3", args.max_folds)


if __name__ == "__main__":
    main()
