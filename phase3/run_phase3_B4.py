"""PAD-LLM Phase 3 / B4: CPU smoke test and OPTIONAL checkpointed CNN training.

Run ONLY the smoke test now (from repository root):
    python phase3/run_phase3_B4.py --smoke-test

Later, after the entire Phase 3 implementation is ready, the existing B1/B2
LOAO experiment orchestration can be reused with this B4 trainer/model:
    python phase3/run_phase3_B4.py --check-only
    python phase3/run_phase3_B4.py --max-folds 1
    python phase3/run_phase3_B4.py

There is NO GPU training in smoke-test mode. All actual B4 experiments use a
NEW output folder (phase3/B4), independent from prior B1/B2/B3/Phase 2 files.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import inspect
import os
import random
import numpy as np
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing prerequisite: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


b4 = _load(HERE / "main_phase3_B4.py", "pad_b4_model")
runner = _load(HERE / "run_phase3_B1&B2.py", "pad_b4_reused_runner")

# Reuse the existing, tested augment/mixup/adversarial/focal utilities.
from cnn_model import (
    DynamicWeightedFocalLoss, augment_data, mixup,
    generate_adversarial_samples, init_weights
)

# The same default training values as B3 / Phase 2. Only the loss changes.
LAMBDA_SUPCON = 0.2
LAMBDA_PROTO = 0.05
SUPCON_TEMPERATURE = 0.5


def train_b4_cnn(model, rating_train_tensor, item_labels_tensor, y_train_tensor,
                 learning_rate, epochs=50, batch_size=8,
                 epsilon=0.01, alpha=0.005, num_steps=1,
                 checkpoint_path=None):
    """B4 training: existing focal loss + supervised contrastive + prototypes.

    Unlike B3, the contrastive component uses UNMIXED ORIGINAL profiles and
    their unambiguous 0/1 labels; Mixup soft labels are classification-only.

    Every completed epoch saves its model/optimizer/scheduler state to the
    B4-specific checkpoint path. Rerunning this variant/fold resumes progress.
    """
    device = torch.device(model.device)
    model.to(device)
    item_labels_tensor = item_labels_tensor.to(device).float()
    # Original runner transfers all training data to GPU in the same way.
    ratings = rating_train_tensor.float().to(device)
    labels = y_train_tensor.view(-1).float().to(device)
    dataset = TensorDataset(ratings, labels)
    # Keep the mini-batch ordering recoverable after a disconnected session.
    loader_rng = torch.Generator()
    loader_rng.manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, generator=loader_rng)
    if not loader:
        raise ValueError("Training set is too small for selected batch size")

    focal = DynamicWeightedFocalLoss(gamma=2).to(device)
    supcon = b4.SupervisedContrastiveLoss(SUPCON_TEMPERATURE).to(device)
    proto = b4.PrototypeRegularizationLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate,
        steps_per_epoch=len(loader), epochs=epochs,
    )
    start_epoch = 0

    if checkpoint_path is not None and Path(checkpoint_path).is_file():
        # Only load trusted checkpoints created by your own experiment runner.
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if ckpt.get("variant") != "B4" or ckpt.get("total_epochs") != epochs:
            raise ValueError("Incompatible B4 checkpoint/configuration; do not overwrite it")
        if ckpt.get("lambda_supcon") != LAMBDA_SUPCON or ckpt.get("lambda_proto") != LAMBDA_PROTO:
            raise ValueError("B4 loss weights differ from saved checkpoint")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        # Resume random generators as well as weights/optimizer, including
        # NumPy because the existing Mixup draws its coefficient using NumPy.
        if "loader_rng_state" in ckpt:
            loader_rng.set_state(ckpt["loader_rng_state"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        if device.type == "cuda" and "cuda_rng_states" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_states"])
        start_epoch = int(ckpt["epoch"])
        print(f"[B4] Resuming at epoch {start_epoch+1}/{epochs}", flush=True)
    else:
        model.apply(init_weights)  # same initialization function as original CNN

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_sup = 0.0
        running_proto = 0.0
        for batch_idx, (raw_ratings, raw_labels) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with nullcontext():  # use float32; prior CUDA AMP produced NaNs
                # Keep original Phase 2 classification path unchanged.
                noisy_ratings, noisy_labels = augment_data(raw_ratings, raw_labels)
                mixed_ratings, mixed_labels = mixup(noisy_ratings, noisy_labels)
                adversarial = generate_adversarial_samples(
                    model, mixed_ratings, mixed_labels, item_labels_tensor,
                    epsilon=epsilon, alpha=alpha, num_steps=num_steps,
                )
                logits = model(mixed_ratings, item_labels_tensor).view(-1)
                adv_logits = model(adversarial, item_labels_tensor).view(-1)
                classification = focal(logits, mixed_labels) + focal(adv_logits, mixed_labels)

                # B4's new representation objective: unaltered original users.
                # The labels here remain exact 0/1, never Mixup soft labels.
                z = model.forward_features(raw_ratings, item_labels_tensor)
                hard_labels = (raw_labels > 0.5).long()
                contrastive = supcon(z, hard_labels)
                prototype = proto(z, hard_labels, model.class_prototypes)
                loss = classification + LAMBDA_SUPCON * contrastive + LAMBDA_PROTO * prototype

            if not torch.isfinite(loss):
                raise FloatingPointError(f"B4 loss NaN/Inf at epoch {epoch+1}, batch {batch_idx+1}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
            running_cls += float(classification.detach())
            running_sup += float(contrastive.detach())
            running_proto += float(prototype.detach())
            if batch_idx % 100 == 0:
                print(f"[B4 CNN] Epoch {epoch+1}/{epochs} batch {batch_idx+1}/{len(loader)}", flush=True)

        denom = len(loader)
        print(f"[B4 CNN] Epoch {epoch+1}/{epochs}: total={running_loss/denom:.4f}, "
              f"focal={running_cls/denom:.4f}, supcon={running_sup/denom:.4f}, "
              f"prototype={running_proto/denom:.4f}", flush=True)

        if checkpoint_path:
            path = Path(checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            torch.save({
                "variant": "B4", "epoch": epoch + 1, "total_epochs": epochs,
                "lambda_supcon": LAMBDA_SUPCON,
                "lambda_proto": LAMBDA_PROTO,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loader_rng_state": loader_rng.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "cuda_rng_states": torch.cuda.get_rng_state_all()
                if device.type == "cuda" else None,
            }, temp)
            os.replace(temp, path)
            print(f"[B4] Checkpoint saved: {path} (epoch {epoch+1})", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return model


def smoke_test():
    """Short CPU-only forward/backward and loss edge-case tests; NO training."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    items, feature_dim, batch = 64, 16, 6
    # Synthetic, deterministic genres: each movie belongs to one category.
    genre_matrix = torch.eye(4).repeat(items // 4, 1)
    model = b4.CNN3DRepresentationB4(
        device, items, feature_dim, 4, gate_variant="B4",
        genre_matrix=genre_matrix,
    ).to(device)
    model.train()
    ratings = torch.zeros(batch, items)
    for i in range(batch):
        ratings[i, [i, i+4, i+8, i+12]] = 1.0
    item_labels = torch.randn(items, feature_dim)
    binary_labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    z = model.forward_features(ratings, item_labels)
    logits = model(ratings, item_labels)
    assert z.shape == (batch, 64), z.shape
    assert logits.shape == (batch, 1), logits.shape
    assert model.class_prototypes.shape == (2, 64)
    assert torch.isfinite(z).all() and torch.isfinite(logits).all()

    sup = b4.SupervisedContrastiveLoss(temperature=0.5)
    pro = b4.PrototypeRegularizationLoss()
    contrastive = sup(z, binary_labels)
    prototype = pro(z, binary_labels, model.class_prototypes)
    assert torch.isfinite(contrastive) and torch.isfinite(prototype)
    assert contrastive.item() > 0, "Expected contrastive loss on a mixed-class batch"
    assert prototype.item() > 0, "Expected prototype regularization to be active"
    no_positives = sup(z[:2], torch.tensor([0, 1]))
    one_class = sup(z, torch.zeros(batch, dtype=torch.long))
    assert no_positives.item() == 0 and one_class.item() == 0

    # Loss gradients must reach the NEW prototypes and EXISTING B2/B3 parts.
    objective = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1), binary_labels.float()
    ) + LAMBDA_SUPCON * contrastive + LAMBDA_PROTO * prototype
    objective.backward()
    assert model.class_prototypes.grad is not None
    assert model.class_prototypes.grad.abs().sum().item() > 0
    assert torch.isfinite(model.class_prototypes.grad).all()
    assert model.gate_weight.grad is not None
    assert torch.isfinite(model.gate_weight.grad).all()
    assert model.anomaly_weight.grad is not None
    assert torch.isfinite(model.anomaly_weight.grad).all()
    assert model.anomaly_weight.grad.abs().sum().item() > 0

    print("[SMOKE PASS] B4: B2 gate=(6,32), B3 descriptors=(6,4), "
          "features=(6,64), logits=(6,1); supervised contrastive and "
          "prototype losses finite; edge cases and gradients passed; NO training.",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 B4 representation objective")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--max-folds", type=int, default=6)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    if not 1 <= args.max_folds <= 6:
        parser.error("--max-folds must be from 1 to 6")
    if args.smoke_test:
        smoke_test()
        return

    # B4 must replace BOTH the model AND CNN training loss. Merely replacing
    # the model class would silently leave B3's old contrastive objective.
    runner.CNN3DAdaptiveGating = b4.CNN3DRepresentationB4
    runner.train_cnn_3d = train_b4_cnn
    runner.CUSTOM_OUTPUT_PARENT = args.output_root
    if args.check_only:
        runner.preflight_check()
        genres = b4.B3.load_movie_genres(ROOT / "data/ml-1m/movies.dat", runner.ITEM_NUMBER)
        coverage = (genres.sum(dim=1) > 0).float().mean().item()
        if coverage < 0.8:
            raise RuntimeError(f"Suspicious genre coverage ({coverage:.1%}); review ID alignment")
        print(f"[CHECK] B4 genre metadata coverage: {coverage:.2%}")
        print(f"[CHECK] B4 outputs: {runner.choose_output_root('B4')}")
        print("[CHECK] No training performed.")
        return
    runner.preflight_check()
    runner.run_experiment("B4", args.max_folds)


if __name__ == "__main__":
    main()
