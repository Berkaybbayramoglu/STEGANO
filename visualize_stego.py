"""
visualize_stego.py
==================
Dataset'in ilk 5 görüntüsü üzerinde Rust steganografi motorunu çalıştırır ve
her görüntü için 4 panelden oluşan yayın kalitesinde bir figür üretir:

  [Orijinal]  |  [Stego]  |  [Fark Haritası ×20]  |  [Quadtree Havuzu]

Çıktı: stego_visual_comparison.png  (300 DPI)

Kullanım:
    source .venv/bin/activate
    python visualize_stego.py
"""

from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import cv2

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Rust motoru ──────────────────────────────────────────────────────────────
try:
    import stegano_core
except ModuleNotFoundError:
    raise SystemExit(
        "\n[HATA] 'stegano_core' bulunamadı.\n"
        "  → source .venv/bin/activate && maturin develop --release"
    )

# ── Parametreler ─────────────────────────────────────────────────────────────
DATASET_DIR  = Path("/Users/betulyedek/Downloads/BOSSbase_1.01")
N_IMAGES     = 5
PAYLOAD_BITS = 10_000
COLONY_SIZE  = 30
MAX_ITER     = 50
MIN_BLOCK    = 4
OUT_FILE     = Path("results_demo/stego_visual_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────

def load_first_n(directory: Path, n: int) -> list[tuple[str, np.ndarray]]:
    """İlk n adet .pgm dosyasını (sayısal sırayla) gri tonlamalı yükler."""
    pgm_files = sorted(directory.glob("*.pgm"),
                       key=lambda p: int(p.stem))   # 1,2,3… sırasıyla
    result = []
    for p in pgm_files[:n]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            result.append((p.name, img))
    return result


def compute_psnr(orig: np.ndarray, stego: np.ndarray) -> float:
    mse = np.mean((orig.astype(np.float64) - stego.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255**2 / mse)


def compute_ssim_simple(orig: np.ndarray, stego: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim
    return ssim(orig, stego, data_range=255)


def make_quadtree_overlay(image: np.ndarray, pool_coords) -> np.ndarray:
    """
    Quadtree'nin seçtiği pikselleri görselleştiren RGB overlay oluşturur.
    Seçili pikseller sarı, geri kalanlar orijinal gri tondadır.
    """
    rgb = np.stack([image, image, image], axis=-1).copy()
    coords = stegano_core.get_quadtree_sparse_map(image, MIN_BLOCK)
    mask   = np.zeros(image.shape, dtype=bool)
    for (y, x) in coords:
        mask[y, x] = True
    # Sarı overlay (R=255, G=255, B=0) — yarı şeffaf görünüm için blend
    rgb[mask, 0] = np.clip(rgb[mask, 0].astype(int) // 2 + 128, 0, 255).astype(np.uint8)
    rgb[mask, 1] = np.clip(rgb[mask, 1].astype(int) // 2 + 128, 0, 255).astype(np.uint8)
    rgb[mask, 2] = (rgb[mask, 2] * 0).astype(np.uint8)
    return rgb, mask.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Ana çizim rutini
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    images = load_first_n(DATASET_DIR, N_IMAGES)
    if not images:
        raise SystemExit(f"[HATA] {DATASET_DIR} içinde görüntü bulunamadı.")

    print(f"İlk {len(images)} görüntü işleniyor  (payload={PAYLOAD_BITS:,} bit)…\n")

    n_rows  = len(images)
    n_cols  = 4          # Orijinal | Stego | Fark ×20 | Quadtree

    fig = plt.figure(figsize=(22, 5.2 * n_rows))
    fig.patch.set_facecolor("#0f0f0f")   # koyu arka plan — akademik poster tarzı

    col_titles = [
        "Orijinal (Cover)",
        f"Stego  (L = {PAYLOAD_BITS:,} bit)",
        "Fark Haritası  (×20 amplifikasyon)",
        "Quadtree Havuzu  (sarı = seçili)",
    ]

    outer = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.06, wspace=0.04,
        left=0.01, right=0.99,
        top=0.96,   bottom=0.01,
    )

    # ── Sütun başlıkları ────────────────────────────────────────────────────
    for col, title in enumerate(col_titles):
        ax_t = fig.add_subplot(outer[0, col])
        ax_t.set_visible(False)
        fig.text(
            (col + 0.5) / n_cols, 0.975, title,
            ha="center", va="top", fontsize=13, fontweight="bold",
            color="white",
        )

    for row_idx, (fname, cover) in enumerate(images):
        print(f"  [{row_idx+1}/{n_rows}]  {fname}", end="  ", flush=True)

        # ── Rust engine çağrısı ─────────────────────────────────────────────
        stego_np, rust_ms, pool_size = stegano_core.run_stegano_engine(
            cover, PAYLOAD_BITS,
            colony_size=COLONY_SIZE,
            max_iter=MAX_ITER,
            min_block=MIN_BLOCK,
        )

        psnr    = compute_psnr(cover, stego_np)
        ssim_v  = compute_ssim_simple(cover, stego_np)

        # Fark haritası (×20 amplified, mutlak değer)
        diff_raw = np.abs(cover.astype(np.int16) - stego_np.astype(np.int16))
        diff_amp = np.clip(diff_raw * 20, 0, 255).astype(np.uint8)

        # Quadtree overlay
        qt_rgb, n_selected = make_quadtree_overlay(cover, pool_size)

        print(f"PSNR={psnr:.2f} dB  SSIM={ssim_v:.5f}  K={pool_size:,}  t={rust_ms:.1f} ms")

        # ── Metrik etiketi (satır sol tarafı) ───────────────────────────────
        fig.text(
            0.002, 1 - (row_idx + 0.5) / n_rows,
            f"#{row_idx+1}\n{fname}",
            ha="left", va="center", fontsize=9.5, color="#cccccc",
            rotation=90, fontfamily="monospace",
        )

        panels = [
            (cover,    "gray",       False, None),
            (stego_np, "gray",       False, None),
            (diff_amp, "hot",        True,  diff_amp),
            (qt_rgb,   None,         False, None),
        ]

        for col_idx, (data, cmap, add_cbar, cbar_data) in enumerate(panels):
            ax = fig.add_subplot(outer[row_idx, col_idx])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

            if cmap is None:
                # RGB görüntü (Quadtree overlay)
                im = ax.imshow(data, interpolation="nearest")
            else:
                im = ax.imshow(data, cmap=cmap, vmin=0, vmax=255,
                               interpolation="nearest")

            # ── Bilgi kutucukları ────────────────────────────────────────────
            if col_idx == 0:
                # Orijinal: dosya adı + boyut
                ax.set_xlabel(
                    f"{cover.shape[1]}×{cover.shape[0]} px",
                    fontsize=8, color="#aaaaaa", labelpad=2,
                )
                ax.text(
                    0.02, 0.97, fname,
                    transform=ax.transAxes, fontsize=7.5, color="white",
                    va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#222222",
                              ec="none", alpha=0.85),
                )

            elif col_idx == 1:
                # Stego: PSNR + SSIM
                ax.text(
                    0.02, 0.97,
                    f"PSNR: {psnr:.2f} dB\nSSIM: {ssim_v:.5f}",
                    transform=ax.transAxes, fontsize=8, color="lime",
                    va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#111111",
                              ec="#00ff00", lw=0.8, alpha=0.88),
                )

            elif col_idx == 2:
                # Fark: değiştirilen piksel sayısı
                n_changed = int((diff_raw > 0).sum())
                ax.text(
                    0.02, 0.97,
                    f"Değiştirilen: {n_changed:,} px\n"
                    f"({100*n_changed/cover.size:.2f}% of image)",
                    transform=ax.transAxes, fontsize=7.5, color="yellow",
                    va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#111111",
                              ec="#ffff00", lw=0.8, alpha=0.88),
                )
                # Küçük renk çubuğu
                cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
                cbar.ax.tick_params(labelsize=6, colors="white")
                cbar.outline.set_edgecolor("#555555")

            elif col_idx == 3:
                # Quadtree: havuz büyüklüğü ve daralma oranı
                comp = 100.0 * pool_size / cover.size
                ax.text(
                    0.02, 0.97,
                    f"Havuz K: {pool_size:,}\n"
                    f"Daralma: %{comp:.1f}",
                    transform=ax.transAxes, fontsize=7.5, color="cyan",
                    va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", fc="#111111",
                              ec="#00ffff", lw=0.8, alpha=0.88),
                )
                # Lejant
                patch_y = mpatches.Patch(color=(1, 1, 0), label="Seçili piksel")
                patch_g = mpatches.Patch(color=(0.45, 0.45, 0.45), label="Homojen bölge")
                ax.legend(handles=[patch_y, patch_g],
                          loc="lower right", fontsize=6.5,
                          facecolor="#111111", edgecolor="#555555",
                          labelcolor="white", framealpha=0.88)

            ax.set_facecolor("black")

    # ── Genel başlık ────────────────────────────────────────────────────────
    fig.suptitle(
        "Steganografi Görsel Karşılaştırması  —  "
        "Adaptive Quadtree + Discrete ABC + LSB-Matching\n"
        f"BOSSbase-1.01 Dataset  |  İlk {N_IMAGES} Görüntü  |  "
        f"Payload = {PAYLOAD_BITS:,} bit",
        fontsize=14, fontweight="bold", color="white", y=0.999,
    )

    fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n✓ Figür kaydedildi → {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
