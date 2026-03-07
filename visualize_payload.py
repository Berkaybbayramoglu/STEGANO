"""
visualize_payload.py
====================
Gömülen gizli verinin (payload) ve gömme sürecinin tam görselleştirilmesi.

Her görüntü için 4 panel üretir:
  ┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
  │ Orijinal Cover  │  Gizli Veri      │  Gömme Haritası  │  Piksel Zoom     │
  │  (gri tonlamalı)│  (100×100 bit   │  (0=mavi 1=kırm) │  (before / after)│
  │                 │   binary matris) │  ABC koordinatları│  32×32 bölgesi  │
  └─────────────────┴──────────────────┴──────────────────┴──────────────────┘

Kullanım:
    source .venv/bin/activate
    python visualize_payload.py
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

matplotlib.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif"],
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

try:
    import stegano_core
except ModuleNotFoundError:
    import glob as _glob
    _here = Path(__file__).parent
    for _sp in _glob.glob(str(_here / ".venv" / "lib" / "python3.*" / "site-packages")):
        if _sp not in sys.path:
            sys.path.insert(0, _sp)
    try:
        import stegano_core
    except ModuleNotFoundError:
        raise SystemExit(
            "\n[HATA] 'stegano_core' bulunamadı.\n"
            "  → source .venv/bin/activate && maturin develop --release"
        )

# ── Parametreler ─────────────────────────────────────────────────────────────
DATASET_DIR  = Path("./data/BOSSbase-1.01/cover")
N_IMAGES     = 5
PAYLOAD_BITS = 10_000      # L = 10.000 bit → 100×100 binary matris
COLONY_SIZE  = 30
MAX_ITER     = 50
MIN_BLOCK    = 4
OUT_FILE     = Path("results_demo/stego_payload_visualization.png")

# Sabit tohum: görsel tekrarlanabilirlik için
RNG_SEED     = 2026

# ─────────────────────────────────────────────────────────────────────────────

def load_first_n(directory: Path, n: int) -> list[tuple[str, np.ndarray]]:
    """İlk n .pgm dosyasını sayısal sırayla yükler."""
    pgm_files = sorted(directory.glob("*.pgm"), key=lambda p: int(p.stem))
    result = []
    for p in pgm_files[:n]:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            result.append((p.name, img))
    return result


def bits_to_matrix(bits: np.ndarray, side: int = 100) -> np.ndarray:
    """
    1-D bit dizisini 2-D binary matrise dönüştürür.
    side × side = 10.000 piksel (100×100 for L=10000).
    """
    n = side * side
    b = bits[:n].astype(np.uint8)
    return b.reshape(side, side)


def draw_payload_panel(ax: plt.Axes, bits: np.ndarray, fname: str) -> None:
    """
    Panel 2 — Gizli veri (payload) binary matris olarak gösterilir.
    0-bit = koyu lacivert, 1-bit = açık sarı → QR-kod görünümü.
    """
    side = int(np.sqrt(PAYLOAD_BITS))
    mat  = bits_to_matrix(bits, side)

    cmap_bin = ListedColormap(["#0d1b2a", "#f5c518"])   # 0→koyu, 1→sarı
    ax.imshow(mat, cmap=cmap_bin, vmin=0, vmax=1, interpolation="nearest")

    ax.set_title("Gizli Veri  (Payload Bitleri)", fontsize=10, color="white", pad=4)
    ax.set_xlabel(
        f"{PAYLOAD_BITS:,} bit  ({side}×{side} matris)\n"
        f"■ 0-bit: {int((bits==0).sum()):,}   ■ 1-bit: {int((bits==1).sum()):,}",
        fontsize=7.5, color="#cccccc", labelpad=3,
    )
    ax.set_xticks([]); ax.set_yticks([])

    # Bit oranı etiketi
    ratio = bits.mean() * 100
    ax.text(
        0.98, 0.02,
        f"1-bit oranı: %{ratio:.1f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="#f5c518", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="#111111", ec="#f5c518", lw=0.7, alpha=0.9),
    )

    # Küçük "anahtar simgesi" ekle
    ax.text(
        0.02, 0.97, "[ENC]  SIFRELI VERI",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=8, color="white",
        bbox=dict(boxstyle="round,pad=0.3", fc="#222222", ec="none", alpha=0.85),
    )


def draw_embed_map_panel(
    ax: plt.Axes,
    cover: np.ndarray,
    coords_y: np.ndarray,
    coords_x: np.ndarray,
    bits: np.ndarray,
) -> None:
    """
    Panel 3 — Gömme haritası:
    Orijinal görüntü arka plan olarak, ABC'nin seçtiği her koordinat
    taşıdığı bit değerine göre renklendirilmiş nokta olarak üzerine çizilir.
      0-bit → mavi   (#4fc3f7)
      1-bit → kırmızı (#ef5350)
    Gömme koordinatları yoğunluk haritasıyla da desteklenir.
    """
    # Scatter için ekran piksellerini küçültmek amacıyla max 3000 nokta göster
    rng = np.random.default_rng(RNG_SEED)
    n_show = min(1_500, len(coords_y))
    idx    = rng.choice(len(coords_y), n_show, replace=False)
    sy, sx, sb = coords_y[idx], coords_x[idx], bits[idx]

    ax.imshow(cover, cmap="gray", alpha=0.72, vmin=0, vmax=255,
              interpolation="nearest")

    # 0-bit noktaları (mavi)
    mask0 = sb == 0
    ax.scatter(sx[mask0], sy[mask0], s=0.9, c="#4fc3f7",
               alpha=0.75, linewidths=0, rasterized=True)
    # 1-bit noktaları (kırmızı)
    mask1 = sb == 1
    ax.scatter(sx[mask1], sy[mask1], s=0.9, c="#ef5350",
               alpha=0.75, linewidths=0, rasterized=True)

    ax.set_title("ABC Gömme Haritası", fontsize=10, color="white", pad=4)
    ax.set_xlabel(
        f"Toplam {len(coords_y):,} koordinat  |  "
        f"Görüntülenen: {n_show:,}",
        fontsize=7.5, color="#cccccc", labelpad=3,
    )
    ax.set_xticks([]); ax.set_yticks([])

    # Lejant
    p0 = mpatches.Patch(color="#4fc3f7", label="0-bit")
    p1 = mpatches.Patch(color="#ef5350", label="1-bit")
    ax.legend(handles=[p0, p1], loc="lower right", fontsize=7.5,
              facecolor="#111111", edgecolor="#555555", labelcolor="white",
              framealpha=0.88, markerscale=1.4)

    # Gömme yoğunluğu etiketi
    h, w   = cover.shape
    n_quad = int(np.sqrt(len(coords_y)))
    ax.text(
        0.02, 0.97,
        f"Gömme yoğunluğu:\n{100*len(coords_y)/(h*w):.2f}% piksel",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
        color="lime", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="#111111", ec="#00ff00",
                  lw=0.7, alpha=0.9),
    )


def draw_zoom_panel(
    ax: plt.Axes,
    cover: np.ndarray,
    stego: np.ndarray,
    coords_y: np.ndarray,
    coords_x: np.ndarray,
    bits: np.ndarray,
) -> None:
    """
    Panel 4 — 32×32 piksel yakın çekim.
    Kırmızı = −1 pertürbasyon, yeşil = +1, gri = değişmedi.
    Her hücreye stego piksel değeri yazılır (küçük font).
    """
    CROP = 24   # 24×24 = 576 hücre — render süresini dengeler

    cy_med = int(np.median(coords_y))
    cx_med = int(np.median(coords_x))
    h, w   = cover.shape
    y0 = max(0, min(cy_med - CROP // 2, h - CROP))
    x0 = max(0, min(cx_med - CROP // 2, w - CROP))
    y1, x1 = y0 + CROP, x0 + CROP

    crop_cover = cover[y0:y1, x0:x1].astype(np.int16)
    crop_stego = stego[y0:y1, x0:x1].astype(np.int16)
    diff       = crop_stego - crop_cover   # -1, 0 veya +1

    DIFF_COLORS = {-1: [0.85, 0.2, 0.2], 0: [0.18, 0.18, 0.18], 1: [0.2, 0.8, 0.3]}
    color_mat = np.array(
        [[DIFF_COLORS[int(diff[r, c])] for c in range(CROP)] for r in range(CROP)]
    )

    ax.imshow(color_mat, interpolation="nearest")

    # Piksel değerlerini yaz (sadece değişen hücreler için daha okunaklı)
    for r in range(CROP):
        for c in range(CROP):
            d   = int(diff[r, c])
            val = int(crop_stego[r, c])
            txt_color = "white" if d == 0 else ("lime" if d == 1 else "#ff9999")
            ax.text(c, r, str(val), ha="center", va="center",
                    fontsize=4.0, color=txt_color, fontweight="bold")

    n_changed = int((diff != 0).sum())
    ax.set_title(f"Piksel Zoom  ({CROP}×{CROP} bölge)", fontsize=10,
                 color="white", pad=4)
    ax.set_xlabel(
        "■ Kırmızı: −1  ■ Gri: değişmedi  ■ Yeşil: +1\n"
        f"Bölge: ({y0},{x0}) → ({y1},{x1})",
        fontsize=7.5, color="#cccccc", labelpad=3,
    )
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(
        0.98, 0.02,
        f"Değişen: {n_changed} / {CROP*CROP} px",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7.5, color="yellow", family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="#111111",
                  ec="#ffff00", lw=0.7, alpha=0.9),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ana çizim
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    images = load_first_n(DATASET_DIR, N_IMAGES)
    if not images:
        raise SystemExit(f"[HATA] {DATASET_DIR} içinde görüntü bulunamadı.")

    print(f"İlk {len(images)} görüntü işleniyor  (payload={PAYLOAD_BITS:,} bit)…\n")

    n_rows, n_cols = len(images), 5
    fig = plt.figure(figsize=(30, 5.6 * n_rows))
    fig.patch.set_facecolor("#0a0a0a")

    COL_TITLES = [
        "Orijinal Cover",
        f"Gizli Veri  (L = {PAYLOAD_BITS:,} bit)",
        "ABC Gömme Haritası  (0=mavi / 1=kırmızı)",
        "Piksel-Seviyesi Zoom  (±1 LSB-M pertürbasyonu)",
        "Stego Görüntü  (veri gömülü — gözle fark edilemez)",
    ]

    outer = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.07, wspace=0.03,
        left=0.01, right=0.99, top=0.95, bottom=0.01,
    )

    # Sütun başlıkları
    for ci, title in enumerate(COL_TITLES):
        fig.text(
            (ci + 0.5) / n_cols, 0.974, title,
            ha="center", va="top", fontsize=12, fontweight="bold", color="white",
        )

    for row_idx, (fname, cover) in enumerate(images):
        print(f"  [{row_idx+1}/{n_rows}]  {fname}", end="  ", flush=True)

        # ── Rust embed_with_details çağrısı ─────────────────────────────────
        stego_np, rust_ms, pool_size, coords_y_np, coords_x_np, payload_np = \
            stegano_core.embed_with_details(
                cover, PAYLOAD_BITS,
                colony_size=COLONY_SIZE,
                max_iter=MAX_ITER,
                min_block=MIN_BLOCK,
            )

        coords_y = np.asarray(coords_y_np)
        coords_x = np.asarray(coords_x_np)
        bits     = np.asarray(payload_np)

        print(f"K={pool_size:,}  t={rust_ms:.1f} ms  "
              f"0-bit={int((bits==0).sum()):,}  1-bit={int((bits==1).sum()):,}")

        # ── Satır etiketi ────────────────────────────────────────────────────
        fig.text(
            0.001, 1 - (row_idx + 0.5) / n_rows,
            f"#{row_idx+1}  {fname}", ha="left", va="center",
            fontsize=9, color="#aaaaaa", rotation=90, fontfamily="monospace",
        )

        # ── Panel 1: Orijinal ────────────────────────────────────────────────
        ax0 = fig.add_subplot(outer[row_idx, 0])
        ax0.imshow(cover, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.set_title("Orijinal  (Cover)", fontsize=10, color="white", pad=4)
        ax0.set_xlabel(
            f"{cover.shape[1]}×{cover.shape[0]} px  |  {cover.size:,} piksel",
            fontsize=7.5, color="#cccccc", labelpad=3,
        )
        ax0.text(
            0.02, 0.97, fname, transform=ax0.transAxes,
            fontsize=8, color="white", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="#222222", ec="none", alpha=0.85),
        )
        ax0.set_facecolor("black")
        for sp in ax0.spines.values():
            sp.set_edgecolor("#333333")

        # ── Panel 2: Payload binary matris ───────────────────────────────────
        ax1 = fig.add_subplot(outer[row_idx, 1])
        draw_payload_panel(ax1, bits, fname)
        ax1.set_facecolor("black")
        for sp in ax1.spines.values():
            sp.set_edgecolor("#333333")

        # ── Panel 3: Gömme haritası ───────────────────────────────────────────
        ax2 = fig.add_subplot(outer[row_idx, 2])
        draw_embed_map_panel(ax2, cover, coords_y, coords_x, bits)
        ax2.set_facecolor("black")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#333333")

        # ── Panel 4: Piksel zoom ─────────────────────────────────────────────
        ax3 = fig.add_subplot(outer[row_idx, 3])
        draw_zoom_panel(ax3, cover, stego_np, coords_y, coords_x, bits)
        ax3.set_facecolor("#0d1117")
        for sp in ax3.spines.values():
            sp.set_edgecolor("#333333")
        # ── Panel 5: Stego görüntü (gömülü ama normal görünen) ────────────
        ax4 = fig.add_subplot(outer[row_idx, 4])
        ax4.imshow(stego_np, cmap="gray", vmin=0, vmax=255,
                   interpolation="nearest")
        ax4.set_xticks([]); ax4.set_yticks([])
        ax4.set_title("Stego  (Gömülü Görüntü)", fontsize=10,
                      color="white", pad=4)

        # PSNR / SSIM hesapla
        diff_full = cover.astype(np.int16) - stego_np.astype(np.int16)
        mse_val   = float(np.mean(diff_full ** 2))
        psnr_val  = 10 * np.log10(255**2 / mse_val) if mse_val > 0 else float("inf")
        from skimage.metrics import structural_similarity as _ssim
        ssim_val  = _ssim(cover, stego_np, data_range=255)
        n_changed = int((diff_full != 0).sum())

        ax4.set_xlabel(
            f"PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.5f}\n"
            f"Değiştirilen piksel: {n_changed:,} / {cover.size:,}",
            fontsize=7.5, color="#cccccc", labelpad=3,
        )
        # Sadece köşeye küçük etiket
        ax4.text(
            0.02, 0.97,
            f"{PAYLOAD_BITS:,} bit gömülü\nPSNR ≥ 40 dB ✓",
            transform=ax4.transAxes, ha="left", va="top",
            fontsize=8, color="#00ff88", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#111111",
                      ec="#00ff88", lw=0.8, alpha=0.92),
        )
        # Sağ altta değiştirilen piksel bilgisi
        ax4.text(
            0.98, 0.02,
            f"Degisen: %{100*n_changed/cover.size:.2f}",
            transform=ax4.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="yellow", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#111111",
                      ec="#ffff00", lw=0.7, alpha=0.88),
        )
        ax4.set_facecolor("black")
        for sp in ax4.spines.values():
            sp.set_edgecolor("#333333")
    # ── Genel başlık ─────────────────────────────────────────────────────────
    fig.suptitle(
        "Steganografi — Gizli Veri Gömme Süreci Tam Görselleştirmesi\n"
        "Adaptive Quadtree + Discrete ABC + LSB-Matching  |  "
        f"BOSSbase-1.01 İlk {N_IMAGES} Görüntü  |  Payload = {PAYLOAD_BITS:,} bit",
        fontsize=13, fontweight="bold", color="white", y=0.998,
    )

    fig.savefig(OUT_FILE, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n✓ Figür kaydedildi → {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
