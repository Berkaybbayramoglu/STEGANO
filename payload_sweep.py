"""
payload_sweep.py — Multi-payload benchmark for IEEE conference paper
====================================================================
Runs the steganography engine at 6 capacity levels on a representative
subset of BOSSbase-1.01 and computes:

  Quality metrics
  ───────────────
  · PSNR   — Peak Signal-to-Noise Ratio (dB)
  · SSIM   — Structural Similarity Index
  · NCC    — Normalized Cross-Correlation
  · MSE    — Mean Squared Error

  Steganalysis resistance metrics
  ────────────────────────────────
  · RS_stat  — Regular-Singular detection statistic (Fridrich et al., 2001)
               Near 0 → hard to detect; near 1 → easily detected.
  · chi2_p   — χ² histogram pair test p-value (Westfeld & Pfitzmann, 2000)
               High p → hard to detect; p ≪ 0.05 → detectable.

Output
------
  payload_sweep_results.csv  (streamed row-by-row, safe to interrupt)

Usage
-----
  python payload_sweep.py                   # 1000 images × 6 levels (~5 min)
  python payload_sweep.py --n 200           # quick test
  python payload_sweep.py --out my.csv
"""

import sys
import math
import csv
import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from scipy import stats as sp_stats

# ── stegano_core import with .venv fallback ──────────────────────────────────
try:
    import stegano_core  # type: ignore
except ModuleNotFoundError:
    import glob as _glob
    _here = Path(__file__).parent
    for _sp in _glob.glob(str(_here / ".venv" / "lib" / "python3.*" / "site-packages")):
        if _sp not in sys.path:
            sys.path.insert(0, _sp)
    try:
        import stegano_core  # type: ignore
    except ModuleNotFoundError:
        sys.exit("[HATA] stegano_core bulunamadı. source .venv/bin/activate && maturin develop --release")

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR = Path("./data/BOSSbase-1.01/cover")
PAYLOADS    = [1_000, 2_500, 5_000, 10_000, 20_000, 40_000]
COLONY      = 30
MAX_ITER    = 50
MIN_BLOCK   = 4


# ═══════════════════════════════════════════════════════════════════════════════
# STEGANALYSIS & QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ncc(cover: np.ndarray, stego: np.ndarray) -> float:
    """
    Normalized Cross-Correlation ∈ [−1, 1].

    NCC = Σ(c̃·s̃) / (‖c̃‖·‖s̃‖)   where  c̃ = c − μ_c,  s̃ = s − μ_s

    Value of 1 means the images are structurally identical (up to brightness).
    Accepted threshold for steganography: NCC ≥ 0.9999.
    """
    c = cover.astype(np.float64).ravel()
    s = stego.astype(np.float64).ravel()
    c -= c.mean()
    s -= s.mean()
    denom = np.linalg.norm(c) * np.linalg.norm(s)
    return float(np.dot(c, s) / denom) if denom > 0 else 1.0


def compute_rs(stego: np.ndarray, group_size: int = 8) -> float:
    """
    Regular-Singular (RS) detection statistic (Fridrich, Goljan & Du, 2001).

    Groups pixels into blocks of `group_size` and computes how the
    discrimination function changes under two complementary flipping operations:
      F  — flip LSB  (0↔1, 2↔3, ...)
      F⁻ — inverse flip (+1 for even, −1 for odd, clamped to [0,255])

    RS_stat = |R_M − R_{−M}| + |S_M − S_{−M}|

    For clean (cover) images: R_M ≈ R_{−M}, S_M ≈ S_{−M} → RS_stat ≈ 0.
    LSB substitution raises R_M above R_{−M} → RS_stat increases.
    LSB matching shows significantly lower RS_stat than substitution.

    Returns a value in [0, 1]; lower = harder to detect.
    """
    flat   = stego.ravel().astype(np.int32)
    n      = (len(flat) // group_size) * group_size
    groups = flat[:n].reshape(-1, group_size)

    # Discrimination function: sum of absolute neighbour differences
    def disc(g: np.ndarray) -> np.ndarray:
        return np.abs(np.diff(g, axis=1)).sum(axis=1).astype(np.float64)

    # F: toggle LSB
    def flip_lsb(g: np.ndarray) -> np.ndarray:
        return g ^ 1

    # F⁻: inverse flip (+1 for even values, −1 for odd), clamped
    def inv_flip(g: np.ndarray) -> np.ndarray:
        r = g.copy()
        even_mask  = (g % 2 == 0)
        r[even_mask]  = np.clip(g[even_mask]  + 1, 0, 255)
        r[~even_mask] = np.clip(g[~even_mask] - 1, 0, 255)
        return r

    d0 = disc(groups)
    dF = disc(flip_lsb(groups))
    dI = disc(inv_flip(groups))

    R_M  = float(np.mean(dF > d0))
    S_M  = float(np.mean(dF < d0))
    R_nM = float(np.mean(dI > d0))
    S_nM = float(np.mean(dI < d0))

    return abs(R_M - R_nM) + abs(S_M - S_nM)


def compute_chi2_p(stego: np.ndarray) -> float:
    """
    χ² (chi-square) histogram pair test p-value (Westfeld & Pfitzmann, 2000).

    For each value pair (2k, 2k+1) the test checks whether the pixel counts
    are equally distributed — a signature of LSB substitution.

    Returns the p-value of the chi-square test:
      · p ≫ 0.05 → indistinguishable from natural image (hard to detect)
      · p ≪ 0.05 → statistically detectable embedding
    """
    hist       = np.bincount(stego.ravel(), minlength=256).astype(np.float64)
    pair_total = np.array([hist[2 * i] + hist[2 * i + 1] for i in range(128)])
    expected   = pair_total / 2.0
    observed   = np.array([hist[2 * i] for i in range(128)])

    mask = expected > 0
    chi2 = float(np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask]))
    df   = int(mask.sum()) - 1
    return float(sp_stats.chi2.sf(chi2, df)) if df > 0 else 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-payload sweep benchmark for IEEE paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python payload_sweep.py            # 1000 images × 6 levels\n"
            "  python payload_sweep.py --n 200    # quick test\n"
        ),
    )
    parser.add_argument("--n",   type=int, default=1000,
                        help="Images per payload level (default: 1000)")
    parser.add_argument("--out", default="payload_sweep_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    files    = sorted(DATASET_DIR.glob("*.pgm"),
                      key=lambda p: int(p.stem))[: args.n]
    total    = len(files)
    csv_path = Path(args.out)

    fieldnames = [
        "Dosya_Adi", "Payload_Bit", "bpp",
        "Guvenli_Piksel(K)", "Daralma_Orani(%)",
        "ABC_Suresi_ms", "MSE", "PSNR_dB", "SSIM",
        "NCC", "RS_stat", "chi2_p",
    ]

    print()
    print("═" * 72)
    print("  PAYLOAD SWEEP — Multi-capacity benchmark for IEEE paper")
    print("  Algorithm: Adaptive Quadtree + Discrete ABC + LSB-Matching")
    print(f"  Payload levels : {[f'{p:,}' for p in PAYLOADS]} bits")
    print(f"  Images/level   : {total:,}")
    print(f"  Output         : {csv_path.resolve()}")
    print("═" * 72)

    with open(csv_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for payload in PAYLOADS:
            bpp_str = f"{payload / 65_536:.4f}"
            print(f"\n── Payload = {payload:>6,} bit  ({bpp_str} bpp) "
                  + "─" * (50 - len(bpp_str)))

            n_ok = n_skip = 0
            sums = {k: 0.0 for k in ("psnr", "ssim", "ncc", "rs", "chi2p", "ms")}

            for idx, p in enumerate(files, 1):
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                try:
                    stego, rust_ms, pool_size = stegano_core.run_stegano_engine(
                        img, payload,
                        colony_size=COLONY,
                        max_iter=MAX_ITER,
                        min_block=MIN_BLOCK,
                    )
                except ValueError:
                    n_skip += 1
                    continue

                # ── Quality metrics ──────────────────────────────────────────
                f64  = img.astype(np.float64)
                s64  = stego.astype(np.float64)
                mse  = float(np.mean((f64 - s64) ** 2))
                psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else math.inf
                ssim = float(ssim_metric(img, stego, data_range=255))
                ncc  = compute_ncc(img, stego)

                # ── Steganalysis metrics ─────────────────────────────────────
                rs   = compute_rs(stego)
                chi2 = compute_chi2_p(stego)

                bpp  = payload / img.size

                writer.writerow({
                    "Dosya_Adi":          p.name,
                    "Payload_Bit":        payload,
                    "bpp":                round(bpp, 6),
                    "Guvenli_Piksel(K)":  pool_size,
                    "Daralma_Orani(%)":   round(100 * pool_size / img.size, 4),
                    "ABC_Suresi_ms":      round(rust_ms, 3),
                    "MSE":                round(mse, 6),
                    "PSNR_dB":            round(psnr, 4),
                    "SSIM":               round(ssim, 6),
                    "NCC":                round(ncc, 7),
                    "RS_stat":            round(rs, 6),
                    "chi2_p":             round(chi2, 6),
                })
                fout.flush()

                n_ok += 1
                sums["psnr"]  += psnr if math.isfinite(psnr) else 0
                sums["ssim"]  += ssim
                sums["ncc"]   += ncc
                sums["rs"]    += rs
                sums["chi2p"] += chi2
                sums["ms"]    += rust_ms

                if idx % 200 == 0 or idx == total:
                    avgs = {k: v / max(n_ok, 1) for k, v in sums.items()}
                    print(
                        f"  [{idx:>4}/{total}]  "
                        f"PSNR={avgs['psnr']:>6.2f} dB  "
                        f"SSIM={avgs['ssim']:.5f}  "
                        f"NCC={avgs['ncc']:.7f}  "
                        f"RS={avgs['rs']:.4f}  "
                        f"χ²p={avgs['chi2p']:.4f}  "
                        f"skip={n_skip}",
                        flush=True,
                    )

            if n_ok:
                avgs = {k: v / n_ok for k, v in sums.items()}
                print(f"\n  ✓  {n_ok:,} OK  |  skip={n_skip}")
                print(f"     PSNR  = {avgs['psnr']:.3f} dB")
                print(f"     SSIM  = {avgs['ssim']:.6f}")
                print(f"     NCC   = {avgs['ncc']:.7f}")
                print(f"     RS    = {avgs['rs']:.5f}  (target: < 0.05)")
                print(f"     χ²_p  = {avgs['chi2p']:.5f}  (target: > 0.05)")
                print(f"     t̄_Rust= {avgs['ms']:.2f} ms")

    print()
    print("═" * 72)
    print(f"  ✓  Sweep tamamlandı → {csv_path.resolve()}")
    print("═" * 72)
    print()


if __name__ == "__main__":
    main()
