"""Python package wrapper for the Rust steganography engine.

This package is used by `maturin` to install the Rust extension as a Python
package directory, and to provide a clean location for console scripts.
"""

from pathlib import Path
import os

# The Rust extension module is placed alongside this file as `stegano_core.<ext>`.
# Import it here so `import stegano_core` works at top-level as before.
try:
    from .stegano_core import *  # type: ignore
except ModuleNotFoundError:
    # If the extension has not been built yet, keep the package importable.
    pass

__all__ = []

# Allow the package to know its root path for CLI helpers.
ROOT_DIR = Path(__file__).resolve().parent.parent

import hashlib
import numpy as np
import struct

def run_product_embed(
    cover: np.ndarray,
    payload_bits: np.ndarray,
    password: str,
    colony_size: int,
    max_iter: int,
    min_block: int,
) -> np.ndarray:
    try:
        from .stegano_core import dabc_select_coords, get_quadtree_sparse_map
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Rust extension not found. Build the core with: maturin develop --release"
        ) from exc

    # Compute quadtree on the 7 MSBs to ensure the map is invariant to LSB changes.
    invariant_cover = cover & 0xFE
    pool = get_quadtree_sparse_map(invariant_cover, min_block)
    pool = sorted(pool)

    if len(pool) < 32:
        raise ValueError("Image Quadtree pool is too small to even embed the header.")

    header_pool = pool[:32]
    data_pool = pool[32:]

    if len(payload_bits) > len(data_pool):
        raise ValueError(
            f"Payload size ({len(payload_bits)} bits) exceeds available quadtree pool capacity ({len(data_pool)})."
        )

    seed_bytes = hashlib.sha256(password.encode("utf-8")).digest()
    seed_int = int.from_bytes(seed_bytes, byteorder="big")
    seed_u64 = seed_int & 0xFFFFFFFFFFFFFFFF

    data_coords = dabc_select_coords(
        invariant_cover,
        data_pool,
        len(payload_bits),
        colony_size,
        max_iter,
        seed_u64,
    )

    length_bytes = struct.pack(">I", len(payload_bits))
    length_bits = np.unpackbits(np.frombuffer(length_bytes, dtype=np.uint8), bitorder="big")

    stego = cover.copy()
    np_rng = np.random.RandomState(seed_int & 0xFFFFFFFF)

    def embed_bit(y, x, bit):
        pixel = stego[y, x]
        if (pixel & 1) != bit:
            # Use strict LSB replacement instead of LSB-M to guarantee the 7 MSBs
            # remain completely unmodified. This ensures the quadtree map is perfectly
            # reproducible during the extraction phase.
            stego[y, x] = (pixel & 0xFE) | bit

    for i, bit in enumerate(length_bits):
        y, x = header_pool[i]
        embed_bit(y, x, bit)

    for (y, x), bit in zip(data_coords, payload_bits):
        embed_bit(y, x, bit)

    return stego


def run_product_extract(
    stego: np.ndarray,
    password: str,
    colony_size: int,
    max_iter: int,
    min_block: int,
) -> np.ndarray:
    try:
        from .stegano_core import dabc_select_coords, get_quadtree_sparse_map
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Rust extension not found. Build the core with: maturin develop --release"
        ) from exc

    # Compute quadtree on the 7 MSBs (which were preserved during embedding)
    invariant_stego = stego & 0xFE
    pool = get_quadtree_sparse_map(invariant_stego, min_block)
    pool = sorted(pool)

    if len(pool) < 32:
        raise ValueError("Image Quadtree pool is too small to contain a header.")

    header_pool = pool[:32]
    data_pool = pool[32:]

    length_bits = []
    for (y, x) in header_pool:
        length_bits.append(stego[y, x] & 1)

    length_bytes = np.packbits(length_bits, bitorder="big").tobytes()
    payload_len = struct.unpack(">I", length_bytes)[0]

    if payload_len > len(data_pool):
        raise ValueError(
            f"Extracted payload length ({payload_len}) exceeds pool capacity. Incorrect password or corrupted image."
        )

    seed_bytes = hashlib.sha256(password.encode("utf-8")).digest()
    seed_int = int.from_bytes(seed_bytes, byteorder="big")
    seed_u64 = seed_int & 0xFFFFFFFFFFFFFFFF

    data_coords = dabc_select_coords(
        invariant_stego,
        data_pool,
        payload_len,
        colony_size,
        max_iter,
        seed_u64,
    )

    payload_bits = np.zeros(payload_len, dtype=np.uint8)
    for i, (y, x) in enumerate(data_coords):
        payload_bits[i] = stego[y, x] & 1

    return payload_bits
