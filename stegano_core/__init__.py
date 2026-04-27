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
