"""
Product inspection visualization utilities.

This module computes scalar similarity metrics (MSE, PSNR, SSIM),
builds per-pixel difference maps, and renders a multi-panel figure
that includes a quadtree-selected pixel overlay from the Rust backend.
The output is a deterministic visualization artifact plus a metrics dict
for the inspected cover/stego pair.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import struct

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

import stegano_core
from stegano_app.metrics import compute_psnr, compute_ssim, compute_mse

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MIN_BLOCK = 4
DIFF_AMPLIFY = 20



def _paint_points(rgb: np.ndarray, coords: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    h, w = rgb.shape[:2]
    for (y, x) in coords:
        y0 = max(0, y - 1)
        y1 = min(h, y + 2)
        x0 = max(0, x - 1)
        x1 = min(w, x + 2)
        rgb[y0:y1, x0:x1, :] = color


def make_quadtree_overlay(
    image: np.ndarray,
    pool_coords: list[tuple[int, int]],
    selected_coords: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, int, int]:
    """
    Quadtree havuzunu sari, secili D-ABC piksellerini ise kirmizi olarak gosterir.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        rgb = image.copy()
    else:
        gray = image
        rgb = np.stack([image, image, image], axis=-1).copy()

    mask = np.zeros(gray.shape, dtype=bool)
    for (y, x) in pool_coords:
        mask[y, x] = True

    rgb[mask, 0] = np.clip(rgb[mask, 0].astype(int) // 2 + 128, 0, 255).astype(np.uint8)
    rgb[mask, 1] = np.clip(rgb[mask, 1].astype(int) // 2 + 128, 0, 255).astype(np.uint8)
    rgb[mask, 2] = (rgb[mask, 2] * 0).astype(np.uint8)

    selected_count = 0
    if selected_coords:
        _paint_points(rgb, selected_coords, (255, 32, 32))
        selected_count = len(selected_coords)

    return rgb, int(mask.sum()), selected_count


def _extract_payload_len(stego_gray: np.ndarray, pool: list[tuple[int, int]]) -> int:
    if len(pool) < 32:
        raise ValueError("quadtree pool is too small for header")
    header_pool = pool[:32]
    bits = np.array([stego_gray[y, x] & 1 for (y, x) in header_pool], dtype=np.uint8)
    length_bytes = np.packbits(bits, bitorder="big").tobytes()
    return struct.unpack(">I", length_bytes)[0]


def _compute_dabc_selected_coords_gray(
    cover_gray: np.ndarray,
    stego_gray: np.ndarray,
    password: str,
    colony_size: int,
    max_iter: int,
    min_block: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    invariant_cover = cover_gray & 0xFE
    pool = stegano_core.get_quadtree_sparse_map(invariant_cover, min_block)
    pool = sorted(pool)

    payload_len = _extract_payload_len(stego_gray, pool)
    data_pool = pool[32:]
    if payload_len <= 0 or payload_len > len(data_pool):
        raise ValueError("payload length exceeds quadtree pool capacity")

    seed_bytes = hashlib.sha256(password.encode("utf-8")).digest()
    seed_int = int.from_bytes(seed_bytes, byteorder="big")
    seed_u64 = seed_int & 0xFFFFFFFFFFFFFFFF

    coords = stegano_core.dabc_select_coords(
        invariant_cover,
        data_pool,
        payload_len,
        colony_size,
        max_iter,
        seed_u64,
    )
    return pool, coords


def _compute_dabc_selected_coords_color(
    cover: np.ndarray,
    stego: np.ndarray,
    password: str,
    colony_size: int,
    max_iter: int,
    min_block: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    all_pools: list[tuple[int, int]] = []
    all_selected: list[tuple[int, int]] = []

    for ch in range(3):
        pool, coords = _compute_dabc_selected_coords_gray(
            np.ascontiguousarray(cover[:, :, ch]),
            np.ascontiguousarray(stego[:, :, ch]),
            password,
            colony_size,
            max_iter,
            min_block,
        )
        all_pools.extend(pool)
        all_selected.extend(coords)

    pool_unique = sorted(set(all_pools))
    selected_unique = sorted(set(all_selected))
    return pool_unique, selected_unique


def inspect_pair(
    cover: np.ndarray,
    stego: np.ndarray,
    output_path: Path,
    *,
    cover_name: str = "cover.pgm",
    stego_name: str = "stego.pgm",
    password: str | None = None,
    colony_size: int = 30,
    max_iter: int = 50,
    min_block: int = MIN_BLOCK,
) -> dict[str, float | int]:
    if cover.ndim not in (2, 3) or stego.ndim not in (2, 3):
        raise ValueError("expected 2D grayscale or 3D RGB images for inspection")
    if cover.shape != stego.shape:
        raise ValueError("cover and stego images must have the same dimensions")

    mse = compute_mse(cover, stego)
    psnr = compute_psnr(mse)
    if cover.ndim == 3:
        ssim_v = compute_ssim(cover, stego, channel_axis=-1)
    else:
        ssim_v = compute_ssim(cover, stego)

    diff_raw = np.abs(cover.astype(np.int16) - stego.astype(np.int16))
    if diff_raw.ndim == 3:
        diff_raw_gray = diff_raw.max(axis=-1)
        diff_amp = np.clip(diff_raw_gray * DIFF_AMPLIFY, 0, 255).astype(np.uint8)
        n_changed = int((diff_raw_gray > 0).sum())
    else:
        diff_amp = np.clip(diff_raw * DIFF_AMPLIFY, 0, 255).astype(np.uint8)
        n_changed = int((diff_raw > 0).sum())

    selected_coords: list[tuple[int, int]] | None = None
    if password:
        try:
            if cover.ndim == 3:
                pool_coords, selected_coords = _compute_dabc_selected_coords_color(
                    cover,
                    stego,
                    password,
                    colony_size,
                    max_iter,
                    min_block,
                )
            else:
                pool_coords, selected_coords = _compute_dabc_selected_coords_gray(
                    cover,
                    stego,
                    password,
                    colony_size,
                    max_iter,
                    min_block,
                )
        except Exception:
            pool_coords = stegano_core.get_quadtree_sparse_map(
                cover if cover.ndim == 2 else cv2.cvtColor(cover, cv2.COLOR_RGB2GRAY),
                min_block,
            )
            pool_coords = sorted(pool_coords)
            selected_coords = None
    else:
        pool_coords = stegano_core.get_quadtree_sparse_map(
            cover if cover.ndim == 2 else cv2.cvtColor(cover, cv2.COLOR_RGB2GRAY),
            min_block,
        )
        pool_coords = sorted(pool_coords)

    qt_rgb, pool_size, selected_count = make_quadtree_overlay(
        cover,
        pool_coords,
        selected_coords,
    )
    pool_ratio_size = cover.shape[0] * cover.shape[1]
    comp = 100.0 * pool_size / pool_ratio_size

    fig = plt.figure(figsize=(22, 6.0))
    fig.patch.set_facecolor("#0f0f0f")

    col_titles = [
        "Orijinal (Cover)",
        "Stego (Product)",
        "Fark Haritasi  (x20 amplifikasyon)",
        "Quadtree Havuzu + Secili Pikseller",
    ]

    outer = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        hspace=0.06,
        wspace=0.04,
        left=0.01,
        right=0.99,
        top=0.85,
        bottom=0.01,
    )

    fig.text(
        0.002,
        0.5,
        f"#1\n{cover_name}",
        ha="left",
        va="center",
        fontsize=9.5,
        color="#cccccc",
        rotation=90,
        fontfamily="monospace",
    )

    panels = [
        (cover, "gray", False),
        (stego, "gray", False),
        (diff_amp, "hot", True),
        (qt_rgb, None, False),
    ]

    for col_idx, (data, cmap, add_cbar) in enumerate(panels):
        ax = fig.add_subplot(outer[0, col_idx])
        ax.set_title(col_titles[col_idx], fontsize=13, fontweight="bold", color="white", pad=15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        if cmap is None:
            im = ax.imshow(data, interpolation="nearest")
        else:
            im = ax.imshow(data, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")

        if col_idx == 0:
            ax.set_xlabel(
                f"{cover.shape[1]}x{cover.shape[0]} px",
                fontsize=8,
                color="#aaaaaa",
                labelpad=2,
            )
            ax.text(
                0.02,
                0.97,
                cover_name,
                transform=ax.transAxes,
                fontsize=7.5,
                color="white",
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    fc="#222222",
                    ec="none",
                    alpha=0.85,
                ),
            )

        elif col_idx == 1:
            ax.text(
                0.02,
                0.97,
                f"PSNR: {psnr:.2f} dB\nSSIM: {ssim_v:.5f}",
                transform=ax.transAxes,
                fontsize=8,
                color="lime",
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="#111111",
                    ec="#00ff00",
                    lw=0.8,
                    alpha=0.88,
                ),
            )

        elif col_idx == 2:
            ax.text(
                0.02,
                0.97,
                f"Degistirilen: {n_changed:,} px\n"
                f"({100 * n_changed / cover.size:.2f}% of image)",
                transform=ax.transAxes,
                fontsize=7.5,
                color="yellow",
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="#111111",
                    ec="#ffff00",
                    lw=0.8,
                    alpha=0.88,
                ),
            )
            cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
            cbar.ax.tick_params(labelsize=6, colors="white")
            cbar.outline.set_edgecolor("#555555")

        elif col_idx == 3:
            ax.text(
                0.02,
                0.97,
                f"Havuz K: {pool_size:,}\n"
                f"Secili: {selected_count if selected_coords else 'n/a'}\n"
                f"Daralma: %{comp:.1f}",
                transform=ax.transAxes,
                fontsize=7.5,
                color="cyan",
                va="top",
                ha="left",
                family="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="#111111",
                    ec="#00ffff",
                    lw=0.8,
                    alpha=0.88,
                ),
            )
            patch_y = mpatches.Patch(color=(1, 1, 0), label="Quadtree havuzu")
            patch_r = mpatches.Patch(color=(1, 0.1, 0.1), label="D-ABC secimi")
            patch_g = mpatches.Patch(color=(0.45, 0.45, 0.45), label="Homojen bolge")
            ax.legend(
                handles=[patch_r, patch_y, patch_g],
                loc="lower right",
                fontsize=6.5,
                facecolor="#111111",
                edgecolor="#555555",
                labelcolor="white",
                framealpha=0.88,
            )

        ax.set_facecolor("black")

    fig.suptitle(
        "Steganografi Gorsel Karsilastirmasi - "
        "Adaptive Quadtree + Discrete ABC + LSB-Matching\n"
        f"Kullanici gorsel cift: {cover_name} | {stego_name}",
        fontsize=14,
        fontweight="bold",
        color="white",
        y=0.96,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "psnr": psnr,
        "ssim": ssim_v,
        "mse": mse,
        "changed_pixels": n_changed,
    }
