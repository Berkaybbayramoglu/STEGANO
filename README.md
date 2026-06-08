<p align="center">
  <img src="docs/banner.svg" alt="STEGANO — Steganographic Embedding Engine" width="100%" draggable="false"/>
</p>

<h1 align="center">High-Capacity Steganographic Embedding for Classified Communications</h1>

<h2 align="center"><em>Conceal. Obscure. Persist. Obfuscate signal within noise.</em></h2>

<p align="center">
  Production-grade steganography engine architected for imperceptible, high-fidelity data embedding. Combines adaptive spatial decomposition, metaheuristic pixel selection, and cryptographic payload protection. Engineered for threat intelligence operations, secure communications, and academic cryptanalysis.
</p>

<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/Berkaybbayramoglu/STEGANO?style=flat-square&label=stars&color=f7bf02)](https://github.com/Berkaybbayramoglu/STEGANO)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab?style=flat-square)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/rust-core-db6d28?style=flat-square)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-stegano--core-blue?style=flat-square)](https://pypi.org/project/stegano-core/)
[![Version](https://img.shields.io/badge/version-0.2.0-444?style=flat-square)](https://github.com/Berkaybbayramoglu/STEGANO/releases)
[![TR](https://img.shields.io/badge/lang-TR-c0392b?style=flat-square)](README.tr.md)

</div>

<p align="center">
  <a href="https://github.com/Berkaybbayramoglu/STEGANO"><img src="docs/btn_star.svg" height="50"/></a>
  &nbsp;
  <a href="https://pypi.org/project/stegano-core/"><img src="docs/btn_pip.svg" height="50"/></a>
  &nbsp;
  <a href="https://github.com/Berkaybbayramoglu/STEGANO/issues"><img src="docs/btn_issues.svg" height="50"/></a>
</p>

<p align="center">
  <a href="https://github.com/Berkaybbayramoglu/STEGANO"><img src="docs/btn_contribute.svg" width="80%"/></a>
</p>

---

## Installation

**macOS / Linux:**
```bash
pipx install stegano-core
pipx ensurepath
```

**Windows (via WSL2):**
```bash
wsl2
pipx install stegano-core
pipx ensurepath
```

**Minimum Requirements:**
- Python 3.10 or later
- 200 MB available RAM
- Supported image format (PNG, BMP, TIFF, PGM, JPG)

---

## Interactive Demo

<p align="center">
  <img src="docs/demo.gif" alt="STEGANO interactive terminal demo" width="100%"/>
</p>

STEGANO operates as a menu-driven terminal interface. The tool guides users through a structured workflow with real-time validation, progress indication, and visual diagnostics.

```bash
stegano
```

---

## Workflows

### Command Deck

<p align="center">
  <img src="docs/menu.png" alt="STEGANO main menu — Lock, Unlock, Inspect, Help, Exit" width="80%"/>
</p>

Launching `stegano` drops you into the Command Deck — a keyboard-navigable menu with four operations:

- **Lock** — Encrypt and embed a payload into a cover image
- **Unlock** — Extract and decrypt a hidden payload from a stego image
- **Inspect** — Generate a publication-ready visual comparison of cover vs. stego
- **Help / Exit** — Usage and contact/Exit

---

### Lock — Embedding a Payload

<p align="center">
  <img src="docs/lock.png" alt="STEGANO Lock workflow — step-by-step guided embedding" width="80%"/>
</p>

The **Lock** workflow guides you through each step with inline validation:

1. **Input Cover Image** — Path to the original, untouched carrier (PNG recommended)
2. **Output Path** — Destination for the stego image
3. **Password** — Derives a 256-bit AES key via PBKDF2 (100 000 iterations)
4. **Embed** — Paste text directly or reference a file path for binary payloads
5. **Parameters** — Colony size, max iterations, min block size (defaults work well for most cases)

Embedding completes with spatial quality metrics printed to the console (PSNR, SSIM, MSE).

---

### Unlock — Recovering Hidden Data

<p align="center">
  <img src="docs/unlock.png" alt="STEGANO Unlock workflow — extraction and decryption result" width="80%"/>
</p>

The **Unlock** workflow mirrors Lock in reverse:

1. **Stego Path** — Path to the image containing the embedded payload
2. **Output Path** — Where to write the recovered payload
3. **Password** — Must match the password used at embed time
4. **Parameters** — Replay the same colony / iteration settings used during embedding

On success, readable text content is optionally printed to stdout and the raw payload is saved to disk. The GCM authentication tag is verified before any data is returned — a corrupted or tampered image is rejected outright.

---

### Inspect — Visual Steganalysis

<p align="center">
  <img src="docs/inspect.png" alt="STEGANO Inspect workflow — metrics output" width="80%"/>
</p>

The **Inspect** operation accepts a cover / stego pair and writes a publication-ready PNG figure. Metrics reported:

| Metric | Description |
|--------|-------------|
| PSNR (dB) | Peak Signal-to-Noise Ratio — higher is more imperceptible |
| SSIM | Structural Similarity Index — 1.0 = perceptually identical |
| MSE | Mean Squared Error — pixel-level distortion |
| Changed Pixels | Absolute count of modified pixels |

<p align="center">
  <img src="docs/inspection_result.png" alt="Inspection output — Original, Stego, Difference map, Quadtree pixel pool" width="100%"/>
</p>

The generated figure displays four panels side by side: the original cover, the stego product, a ×20 amplified difference map, and the Quadtree pixel pool (yellow = selected safe pixels). This layout is suitable for direct inclusion in academic papers or security reports.

---

## Command-Line Mode

For scripted or non-interactive operation, use the `stegano-product` command directly.

**Lock — embed a text message:**
```bash
stegano-product lock -i cover.png -o stego.png -p "password" -m "secret message"
```

**Lock — embed a binary file:**
```bash
stegano-product lock -i cover.png -o stego.png -p "password" -f path/to/secret.bin
```

**Unlock — extract payload:**
```bash
stegano-product unlock -i stego.png -o extracted.bin -p "password"
```

**Unlock — extract and print text to stdout:**
```bash
stegano-product unlock -i stego.png -o extracted.bin -p "password" --print
```

**Inspect — generate visual report:**
```bash
stegano-product inspect --cover cover.png --stego stego.png --output report.png
```

### Full Flag Reference

| Command | Flag | Description | Default |
|---------|------|-------------|---------|
| `lock` | `-i` / `--input` | Cover image path | required |
| `lock` | `-o` / `--output` | Output stego image path | required |
| `lock` | `-p` / `--password` | AES-GCM encryption password | required |
| `lock` | `-m` / `--message` | Text payload to embed | — |
| `lock` | `-f` / `--file` | Binary file path to embed | — |
| `lock` / `unlock` | `--colony-size` | D-ABC colony size | `30` |
| `lock` / `unlock` | `--max-iter` | D-ABC max iterations | `50` |
| `lock` / `unlock` | `--min-block` | Quadtree min block size | `4` |
| `lock` / `unlock` / `inspect` | `--force` | Overwrite existing output | `false` |
| `unlock` | `--print` | Print recovered text to stdout | `false` |
| `inspect` | `--cover` | Original cover image path | required |
| `inspect` | `--stego` | Stego image path | required |
| `inspect` | `--output` | Output figure path | `inspection_result.png` |

> Either `-m` or `-f` must be provided for `lock`, but not both. Interactive mode (`stegano`) is recommended for guided workflows and error recovery.

---

## Supported Formats

| Format | Grayscale | RGB | Recommended | Notes |
|--------|-----------|-----|-------------|-------|
| PNG | Yes | Yes | **Primary** | Lossless; no data loss post-embedding |
| BMP | Yes | Yes | Yes | Uncompressed; large file size acceptable |
| TIFF | Yes | Yes | Yes | Flexible codec support |
| PGM | Yes | — | Grayscale-only | Raw/ASCII grayscale baseline |
| JPG/JPEG | Yes | Yes | Avoid | Lossy compression destroys LSB layer |

Use PNG or BMP for maximum imperceptibility and reproducibility. Deploy RGB over grayscale for 3× payload capacity. Avoid JPEG unless capacity takes priority over security margin.

---

## Technical Specifications

### Embedding Capacity

- **Grayscale:** ~0.125 bits/pixel (single LSB layer, post-quadtree filtering)
- **RGB:** ~0.375 bits/pixel (LSB across 3 channels, post-quadtree filtering)

Exact capacity depends on image complexity and D-ABC parameter tuning.

### Memory Footprint

- Base overhead: ~50 MB
- Per-operation peak (1024×1024): 150–200 MB

---

## Dependencies

```
opencv-python      >=4.8.0     Image I/O and basic CV operations
numpy              >=1.24.0    Numerical arrays and linear algebra
scikit-image       >=0.21.0    SSIM, MSE, and image metrics
rich               >=13.5.2    Terminal UI, colors, progress rendering
psutil             >=5.9.5     Process monitoring and resource tracking
cryptography       >=41.0.3    AES-256-GCM encryption and key derivation
matplotlib         >=3.7.2     Visualization and figure generation
questionary        >=2.0.0     Interactive terminal prompts and menus
```

**System Requirements:** Python 3.10+, Linux (x86\_64, ARM64), macOS (Intel, Apple Silicon), Windows via WSL2. Rust optional — pre-compiled wheels provided for common platforms.

---

## Advanced Parameter Tuning

```
--colony-size     [int, default: 30]   Number of bees in optimization colony
                                       Quality ↑ exponentially; runtime ↑ linearly
                                       Recommended range: 20–60

--max-iter        [int, default: 50]   Maximum optimization iterations
                                       Convergence ↑; time ↑ linearly
                                       Recommended range: 20–100; diminishing returns >100

--min-block       [int, default: 4]    Minimum quadtree leaf block size
                                       Safe pixels ↑; complexity ↓
                                       Recommended range: 4–8
```

---

## Core Technologies

### Adaptive Quadtree Spatial Decomposition

<p align="center">
  <img src="docs/quadtree.svg" alt="Adaptive Quadtree decomposition — recursive subdivision and bit embedding" width="80%"/>
</p>

Image regions exhibit heterogeneous statistical properties. The adaptive quadtree partitions the spatial domain into blocks of varying granularity, selecting only complex regions (high variance) as candidates for payload embedding.

**Algorithm:**

1. Initialize: Recursively divide image into quadrants
2. Compute variance σ²(B) for each candidate block B
3. If σ²(B) > threshold T and block dimensions > min\_block: subdivide into 4 child quadrants; repeat
4. Else if σ²(B) > T and dimensions ≤ min\_block: mark all pixels in block as "safe"
5. Else: discard block (insufficient complexity)

**Result:** Sparse coordinate matrix of safe pixels exploitable for embedding without detectability risk.

- Variance threshold T balances capacity vs. imperceptibility
- Minimum block size min\_block prevents over-granular partitioning
- Complexity O(N log N) where N = image dimensions

---

### Discrete Artificial Bee Colony Optimization

<p align="center">
  <img src="docs/dabc.svg" alt="D-ABC — bees search the safe pixel pool, selecting high-fitness pixels for embedding" width="80%"/>
</p>

Pixel selection in a discrete combinatorial space requires stochastic exploration. The discrete ABC algorithm treats safe-pixel subset selection as an optimization problem, minimizing detectability while maximizing fidelity preservation.

**Algorithm Phases:**

1. **Initialization:** Generate random L-subsets of the K safe-pixel pool, initializing food sources
2. **Employed Bee Phase:** Each employed bee explores via single-element swaps; retain improvements, increment failure counter otherwise
3. **Onlooker Bee Phase:** Roulette-wheel selection based on fitness distribution; high-fitness solutions selected with higher probability
4. **Scout Phase:** Discard exhausted food sources (stagnated for `limit` iterations); discover new random L-subsets
5. **Global Best Tracking:** Maintain archive of best solution across all generations

**Fitness Function:**

```
f(indices) = Σ image_intensity[j] for j ∈ selected_pixels
```

Higher-intensity pixels tolerate LSB modification with lower detectability.

**Convergence:** O(colony\_size × max\_iter × L) per image; typical runtime 2–3 seconds on modern hardware.

---

### LSB-Matching with Payload Encryption

**Embedding Protocol:**

1. Serialize payload (message or file binary)
2. Derive 256-bit AES key via PBKDF2(password, salt, 100 000 iterations)
3. Encrypt payload using AES-256-GCM; produces ciphertext + authentication tag
4. For each safe pixel selected by D-ABC: replace LSB with next payload bit — distortion per pixel ≤ 1

**Extraction Protocol:**

1. Recover bitstream from stego image via selected pixel LSB extraction
2. Decrypt ciphertext using derived key and embedded nonce
3. Verify authentication tag; abort if corrupted
4. Return plaintext or binary payload

**Security Properties:**

- AES-256-GCM ensures semantic security (IND-CCA2)
- Authentication prevents tampering and confirms integrity
- LSB modifications undetectable to human vision
- Requires cryptanalysis in complement to steganalysis for compromise

---

## Security Architecture

### Threat Model

STEGANO provides defense against passive observation and standard frequency-domain detection methods. Assumes:

- Attacker has access to stego image but not cover image
- Attacker lacks knowledge of embedding parameters
- Authentication is maintained separately from stego channel

### Guarantees

- **Semantic Security:** AES-256-GCM encryption prevents payload reconstruction without key
- **Imperceptibility:** Quadtree + D-ABC minimize perceptual detectability in safe regions
- **Integrity:** GCM authentication tag prevents tampering
- **Reproducibility:** Deterministic embedding with fixed parameters yields identical stego images

---

## References

### Steganography & Steganalysis

- Fridrich, J., Goljan, M., & Hogea, D. (2003). "Steganalysis of LSB embedding in grayscale images." *IEEE Trans. on Signal Processing*, 51(5), 1413–1422.
- Holub, V., Fridrich, J., & Denemark, T. (2014). "Universal Distortion Function for Steganography in an Arbitrary Domain." *EURASIP Journal on Information Security*.

### Metaheuristic Optimization

- Karaboga, D., & Basturk, B. (2007). "A powerful and efficient algorithm for numerical function optimization: Artificial Bee Colony (ABC) algorithm." *Journal of Global Optimization*, 39(3), 459–471.

### Quadtree Spatial Decomposition

- Finkel, R. A., & Bentley, J. L. (1974). "Quad Trees: A Data Structure for Retrieval on Composite Keys." *Acta Informatica*, 4(1), 1–9.

---

## Contributors

[Berkay Bayramoğlu](https://github.com/Berkaybbayramoglu) · [Betül Göksu](https://github.com/betulgoksu98) · [Gülnur Durukan](https://github.com/glnur-d) · [İsmail Erol](https://github.com/ismailerol61)

---

## Contributing

Contributions are solicited for algorithm refinement, platform expansion, and cryptanalytic hardening.

- Open an issue for architectural proposals before implementing changes
- Benchmark impact on embedding speed and imperceptibility metrics
- Provide test coverage for new functionality
- Follow Rust/Python style standards outlined in CONTRIBUTING.md

---

## License

Released under the **MIT License**. See [LICENSE](LICENSE) for complete terms.

---

<p align="center">
  <strong>Version 0.2.0 — Production Ready</strong><br/>
  <sub>Adaptive Quadtree + Discrete ABC + LSB-Matching</sub><br/>
  <sub>Rust-accelerated. Python-compatible. Cryptographically-sound.</sub>
</p>

<p align="center">
  <a href="https://github.com/Berkaybbayramoglu/STEGANO">Repository</a> ·
  <a href="https://github.com/Berkaybbayramoglu/STEGANO/issues">Issue Tracker</a> ·
  <a href="https://github.com/Berkaybbayramoglu/STEGANO/blob/main/CONTRIBUTING.md">Contributing</a>
</p>
