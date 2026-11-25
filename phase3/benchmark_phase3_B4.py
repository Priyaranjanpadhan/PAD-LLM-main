"""Isolated B4 batch-size throughput benchmark (NO saved checkpoints, NO research metrics).

Put this file in the PAD-LLM-main repository root, next to cnn_model.py.
Run e.g. python benchmark_phase3_B4.py --batch-size 8
Uses the REAL Phase 3 B4 architecture, loss, adversarial generation and trainer,
but 512 SYNTHETIC 3,952-item profiles rather than real research data.
This tests speed/memory, not model quality or convergence.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from phase3.run_phase3_B4 import b4, train_b4_cnn


def main():
    parser = argparse.ArgumentParser(description="Benchmark real B4 training at one batch size")
    parser.add_argument("--batch-size", type=int, required=True, choices=[8, 16, 32, 64])
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not detected. Check NVIDIA driver and CUDA-enabled PyTorch.")
    if args.samples < 2 * args.batch_size:
        raise ValueError("Use at least two complete batches for this speed test")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Batch size: {args.batch_size}, samples: {args.samples}, input items: 3952", flush=True)

    # The REAL B4 model has 3,952 movie positions and 16 item-semantic features.
    # Four synthetic genre categories are sufficient for exercising B3's path.
    n_items, semantic_dim, heads = 3952, 16, 4
    genres = torch.eye(4).repeat((n_items + 3) // 4, 1)[:n_items]
    model = b4.CNN3DRepresentationB4(
        device, n_items, semantic_dim, heads,
        gate_variant="B4", genre_matrix=genres,
    ).to(device)

    # Approximately 4.18% interactions per profile, like the attack filler setting.
    # Binary labels have both genuine (0) and malicious (1) examples.
    ratings = (torch.rand(args.samples, n_items) < 0.0418).float()
    labels = (torch.rand(args.samples) < 0.65).float()
    item_embeddings = torch.randn(n_items, semantic_dim)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()

    # Run the SAME trainer as the full experiment; only data size and epochs
    # differ. No checkpoint_path => the test writes nothing to Google Drive.
    train_b4_cnn(
        model, ratings, item_embeddings, labels,
        learning_rate=0.0001,
        epochs=1,
        batch_size=args.batch_size,
        checkpoint_path=None,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    steps = args.samples // args.batch_size
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print("\n======== BENCHMARK (NOT A RESEARCH RESULT) ========")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Full batches: {steps}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Profiles / second: {args.samples / elapsed:.2f}")
    print(f"Average seconds / batch: {elapsed / steps:.4f}")
    print(f"Peak PyTorch GPU allocation (GB): {peak:.2f}")
    print("No checkpoints saved; this run did not modify your trained models.")


if __name__ == "__main__":
    main()
