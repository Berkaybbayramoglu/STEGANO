"""
main.py — Hybrid PyO3 Steganography Benchmarking Pipeline
==========================================================

Architecture
------------
  Cover Image (H×W uint8)
      │
      ▼  [Rust]  Phase 1: Adaptive Quadtree  →  Sparse Pool K
      ▼  [Rust]  Phase 2: Discrete ABC       →  Best L coordinates
      ▼  [Rust]  Phase 3: LSB-Matching       →  Stego Image
      │
      ▼  [Python] PSNR / SSIM evaluation

Usage
-----
  1. Compile the Rust engine once:
         pip install maturin
         maturin develop --release

  2. Run this benchmark:
         python main.py

Dependencies
------------
  pip install opencv-python scikit-image numpy
"""

import os
import sys
import time
import math
import textwrap
import csv
import argparse

# Force line-buffered stdout so progress prints never block
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

# ── Import the compiled Rust engine ─────────────────────────────────────────
# stegano_core is the PyO3 extension module produced by `maturin develop`.
# If it is not found (i.e. the module has not been compiled yet), the script
# prints a clear error and exits rather than falling back to a slower path.
try:
    import stegano_core  # type: ignore
    RUST_ENGINE_AVAILABLE = True
except ModuleNotFoundError:
    # Fallback: look in the project's .venv (built with maturin develop --release)
    import glob as _glob
    _here = Path(__file__).parent
    for _sp in _glob.glob(str(_here / ".venv" / "lib" / "python3.*" / "site-packages")):
        if _sp not in sys.path:
            sys.path.insert(0, _sp)
    try:
        import stegano_core  # type: ignore
        RUST_ENGINE_AVAILABLE = True
    except ModuleNotFoundError:
        RUST_ENGINE_AVAILABLE = False
        print(
            "\n[ERROR] Rust extension 'stegano_core' not found.\n"
            "  Please compile it first with:\n"
            "      source .venv/bin/activate\n"
            "      maturin develop --release\n",
            file=sys.stderr,
        )
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY METRICS (Python-side evaluation)
# These are kept in Python so that the evaluator is independent of the
# embedding back-end and can be used for cross-framework comparisons.
# ═══════════════════════════════════════════════════════════════════════════

def compute_mse(original: np.ndarray, stego: np.ndarray) -> float:
    """
    Mean Squared Error between cover and stego images.

        MSE = (1/N) · Σ (I(i,j) − S(i,j))²

    Lower is better; MSE = 0 means the images are identical.
    """
    diff = original.astype(np.float64) - stego.astype(np.float64)
    return float(np.mean(diff ** 2))


def compute_psnr(mse: float, max_val: float = 255.0) -> float:
    """
    Peak Signal-to-Noise Ratio in decibels.

        PSNR = 10 · log₁₀(MAX² / MSE)

    Typical acceptability threshold for steganography: PSNR ≥ 40 dB
    (imperceptible distortion to the human visual system).

    Returns `inf` when MSE = 0 (cover and stego are identical).
    """
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(max_val ** 2 / mse)


def compute_ssim(original: np.ndarray, stego: np.ndarray) -> float:
    """
    Structural Similarity Index Measure (Wang et al., 2004).
    Range: [−1, 1]; higher is better. Values > 0.99 are perceptually lossless.
    """
    return float(ssim_metric(original, stego, data_range=255))


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def iter_images(dataset_dir: str, subset_size: int | None = None):
    """
    Generator that yields (filename, image_matrix) one at a time.
    Never loads more than one image into memory simultaneously.

    Parameters
    ----------
    dataset_dir  : Path to directory containing .pgm files.
    subset_size  : If given, only the first N images are yielded (dev mode).
    """
    pgm_files = sorted(Path(dataset_dir).glob("*.pgm"), key=lambda p: int(p.stem))
    if subset_size:
        pgm_files = pgm_files[:subset_size]
    for p in pgm_files:
        mat = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mat is None:
            print(f"  [WARN] Could not read {p.name}, skipping.", flush=True)
            continue
        yield p.name, mat


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARKING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_benchmark(
    dataset_dir: str,
    subset_size: int = 10,
    payload_size: int = 10_000,
    colony_size: int = 30,
    max_iter: int = 50,
    min_block: int = 4,
    csv_out: str = "deney_sonuclari.csv",
) -> None:
    """
    Full end-to-end benchmark of the Rust steganography engine.

    For each image the pipeline calls `stegano_core.run_stegano_engine`,
    which returns the stego image and the Rust-side wall-clock time.
    Python then computes PSNR and SSIM for quality evaluation.
    Results are streamed row-by-row to `csv_out` so that progress is
    never lost even if the run is interrupted mid-way.

    Parameters
    ----------
    dataset_dir  : Directory of cover images (.pgm).
    subset_size  : Number of images to process (None = all).
    payload_size : Secret message length in bits (= embedding capacity).
    colony_size  : D-ABC colony size (half employed, half onlooker).
    max_iter     : D-ABC maximum iteration count.
    min_block    : Quadtree leaf block side length in pixels.
    csv_out      : Path to the output CSV file.
    """
    pgm_files = sorted(Path(dataset_dir).glob("*.pgm"))
    if subset_size:
        pgm_files = pgm_files[:subset_size]
    total_images = len(pgm_files)
    if total_images == 0:
        print("[ERROR] No images found. Check dataset_dir.", file=sys.stderr)
        return

    # ── CSV writer (streaming — survives mid-run interruptions) ──────────────
    csv_path   = Path(csv_out)
    csv_exists = csv_path.exists()
    csv_file   = open(csv_path, "a", newline="", encoding="utf-8")
    fieldnames = [
        "Dosya_Adi", "Payload_Bit", "Toplam_Piksel",
        "Guvenli_Piksel(K)", "Daralma_Orani(%)",
        "ABC_Suresi_sn", "MSE", "PSNR_dB", "SSIM",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()
        csv_file.flush()

    # ── Header ──────────────────────────────────────────────────────────
    sep = "═" * 80
    print(sep)
    print("  ADAPTIVE QUADTREE + DISCRETE ABC + LSB-MATCHING  |  Rust Engine")
    print(sep)
    print(f"  Dataset  : {dataset_dir}")
    print(f"  Images   : {total_images}")
    print(f"  Payload  : {payload_size} bits  |  Colony: {colony_size}  |  Iter: {max_iter}")
    print(f"  CSV out  : {csv_path.resolve()}")
    print(sep)
    print(
        f"  {'Image':<18} {'Pool K':>8} {'MSE':>8} {'PSNR (dB)':>10} "
        f"{'SSIM':>8} {'Rust Time (ms)':>15}"
    )
    print("─" * 80)

    # ── Per-image evaluation (streaming — no full image list in RAM) ─────
    # Online statistics: Welford's algorithm for mean/variance
    n_ok = 0
    sum_mse = sum_psnr = sum_ssim = sum_ms = 0.0

    for i, (fname, cover) in enumerate(iter_images(dataset_dir, subset_size), 1):
        try:
            # ── RUST ENGINE CALL ────────────────────────────────────────
            stego_np, rust_ms, pool_size = stegano_core.run_stegano_engine(
                cover,
                payload_size,
                colony_size=colony_size,
                max_iter=max_iter,
                min_block=min_block,
            )

            # ── QUALITY EVALUATION (Python) ──────────────────────────────
            mse      = compute_mse(cover, stego_np)
            psnr_db  = compute_psnr(mse)
            ssim_val = compute_ssim(cover, stego_np)
            img_pix  = cover.size
            comp_pct = round(100.0 * pool_size / img_pix, 4)
            rust_s   = rust_ms / 1000.0

            # Online aggregate (no list needed)
            n_ok += 1
            sum_mse  += mse
            if math.isfinite(psnr_db):
                sum_psnr += psnr_db
            sum_ssim += ssim_val
            sum_ms   += rust_ms

            # ── Stream row to CSV immediately ────────────────────────────
            writer.writerow({
                "Dosya_Adi":          fname,
                "Payload_Bit":        payload_size,
                "Toplam_Piksel":      img_pix,
                "Guvenli_Piksel(K)":  pool_size,
                "Daralma_Orani(%)":   comp_pct,
                "ABC_Suresi_sn":      round(rust_s, 5),
                "MSE":                round(mse, 6),
                "PSNR_dB":            round(psnr_db, 4),
                "SSIM":               round(ssim_val, 6),
            })
            csv_file.flush()

            psnr_str = f"{psnr_db:10.2f}" if math.isfinite(psnr_db) else "      ∞"
            print(
                f"  [{i:>5}/{total_images}] {fname:<14} {pool_size:>8,d} {mse:>8.4f}"
                f" {psnr_str} {ssim_val:>8.5f} {rust_ms:>15.2f}",
                flush=True,
            )

        except ValueError as e:
            print(f"  [{i:>5}/{total_images}] {fname:<14} [SKIP] {e}", flush=True)

    csv_file.close()

    # ── Aggregate Statistics ─────────────────────────────────────────────
    if n_ok > 0:
        print("─" * 80)
        avg_mse  = sum_mse  / n_ok
        avg_psnr = sum_psnr / n_ok
        avg_ssim = sum_ssim / n_ok
        avg_ms   = sum_ms   / n_ok

        print(
            f"  {'AVERAGE':<18} {'':>8} {avg_mse:>8.4f} {avg_psnr:>10.2f} "
            f"{avg_ssim:>8.5f} {avg_ms:>15.2f}"
        )
        print(sep)
        print()
        print("  INTERPRETATION GUIDE")
        print("  ─────────────────────────────────────────────────────────")
        print(f"  MSE  → {avg_mse:.4f}  (target: < 0.5 for imperceptibility)")
        print(f"  PSNR → {avg_psnr:.2f} dB  (target: ≥ 40 dB)")
        print(f"  SSIM → {avg_ssim:.5f}  (target: > 0.999)")
        print(f"  Time → {avg_ms:.2f} ms/image (Rust engine wall-clock)")
        print(sep)
        print(f"\n  ✓ Results saved to: {Path(csv_out).resolve()}")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Rust-backed steganography benchmark."
    )
    parser.add_argument("--dataset-dir", default="./data/BOSSbase-1.01/cover")
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--payload-size", type=int, default=10_000)
    parser.add_argument("--colony-size", type=int, default=30)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--min-block", type=int, default=4)
    parser.add_argument("--csv-out", default="deney_sonuclari.csv")
    args = parser.parse_args()

    run_benchmark(
        dataset_dir=args.dataset_dir,
        subset_size=args.subset_size,
        payload_size=args.payload_size,
        colony_size=args.colony_size,
        max_iter=args.max_iter,
        min_block=args.min_block,
        csv_out=args.csv_out,
    )
