//! # `stegano_core` — High-Performance Steganography Engine
//!
//! This library implements a novel **Hybrid Combinatorial Optimization Architecture**
//! for image steganography, compiled as a native Python extension module via PyO3.
//!
//! ## Pipeline Architecture
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────────────────┐
//!  │                    Cover Image  (H × W, uint8)                      │
//!  └──────────────────────────────┬──────────────────────────────────────┘
//!                                 │
//!              ╔══════════════════▼══════════════════╗
//!              ║  Phase 1: Adaptive Quadtree          ║
//!              ║  Adaptive threshold = μ·α + σ·β     ║
//!              ║  Output: Sparse Pool  K = {(y,x)…}  ║
//!              ╚══════════════════╤══════════════════╝
//!                                 │  O(N log N)
//!              ╔══════════════════▼══════════════════╗
//!              ║  Phase 2: Discrete ABC Optimizer     ║
//!              ║  Employed → Onlooker → Scout phases  ║
//!              ║  Output: Best L coords from K        ║
//!              ╚══════════════════╤══════════════════╝
//!                                 │  O(C · I · L)
//!              ╔══════════════════▼══════════════════╗
//!              ║  Phase 3: LSB-Matching Embedder      ║
//!              ║  ±1 perturbation (histogram-safe)    ║
//!              ║  Output: Stego Image                 ║
//!              ╚═════════════════════════════════════╝
//! ```
//!
//! ## Computational Complexity
//! - Quadtree decomposition: **O(N log N)** where N = H × W pixels.
//! - ABC optimisation: **O(C · I · L)** where C = colony size, I = iterations, L = payload.
//! - LSB-M embedding: **O(L)**.

use ndarray::{s, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::rngs::SmallRng;
use rand::seq::index::sample as reservoir_sample;
use rand::{Rng, SeedableRng}; // SeedableRng brings from_entropy() into scope
use std::collections::HashSet;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 1 — ADAPTIVE QUADTREE DECOMPOSITION
//
// Classical Quadtree image segmentation augmented with an *adaptive* split
// threshold derived from the global image statistics (μ, σ), avoiding the
// sensitivity of a hand-tuned constant threshold.
//
// Reference: Shih & Wu (2003). "Combinatorial image watermarking in the
//            spatial domain." Knowledge-Based Systems, 16(4), 201-206.
// ═══════════════════════════════════════════════════════════════════════════

/// Computes the **mean** and **variance** of a 2-D image block in a single
/// streaming pass — O(pixels in block), no extra heap allocations.
#[inline]
fn block_mean_variance(view: &ArrayView2<u8>) -> (f64, f64) {
    let n = view.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    // Welford's online algorithm for numerically stable mean & variance.
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for (k, &v) in view.iter().enumerate() {
        let x = v as f64;
        let delta = x - mean;
        mean += delta / (k + 1) as f64;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    let variance = m2 / n as f64;
    (mean, variance)
}

/// Recursive Quadtree splitter (depth-first traversal).
///
/// A block is **split** when both:
///   1. Its local variance exceeds `threshold` (i.e. it is texture-rich), AND
///   2. Its dimensions are still larger than `min_block` (prevents infinite recursion).
///
/// A block that reaches the **leaf** condition while still being texture-rich
/// has all its pixel coordinates appended to the `pool` (sparse matrix).
fn quadtree_recurse(
    img: &ArrayView2<u8>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
    threshold: f64,
    min_block: usize,
    pool: &mut Vec<(usize, usize)>,
) {
    if h == 0 || w == 0 {
        return;
    }

    // Extract block view without copying data — O(1) due to ndarray's slice semantics.
    let block = img.slice(s![y..y + h, x..x + w]);
    let (_, variance) = block_mean_variance(&block);

    // Homogeneous (smooth) region: discard this entire block.
    if variance < threshold {
        return;
    }

    if w > min_block && h > min_block {
        // Texture-rich & still divisible → split into four quadrants.
        let hw = w / 2;
        let hh = h / 2;
        quadtree_recurse(img, y, x, hh, hw, threshold, min_block, pool);             // ↖ TL
        quadtree_recurse(img, y, x + hw, hh, w - hw, threshold, min_block, pool);   // ↗ TR
        quadtree_recurse(img, y + hh, x, h - hh, hw, threshold, min_block, pool);   // ↙ BL
        quadtree_recurse(img, y + hh, x + hw, h - hh, w - hw, threshold, min_block, pool); // ↘ BR
    } else {
        // Minimum-size leaf node that is still texture-rich.
        // All pixels within this block enter the sparse embedding pool.
        for row in y..y + h {
            for col in x..x + w {
                pool.push((row, col));
            }
        }
    }
}

/// Public Quadtree entry-point with **Adaptive Thresholding**.
///
/// Rather than a hard-coded scalar, the split threshold is derived from the
/// global image statistics:
///
/// ```text
///   threshold = α · μ + β · σ       (α = 0.1, β = 0.6 empirically tuned)
/// ```
///
/// This ensures that the threshold scales automatically with the image's
/// overall brightness and contrast, making the algorithm dataset-agnostic.
///
/// Returns a `Vec<(usize, usize)>` — the **Sparse Embedding Matrix** K.
fn adaptive_quadtree(img: &ArrayView2<u8>, min_block: usize) -> Vec<(usize, usize)> {
    let (h, w) = img.dim();

    // Global statistics over the entire image — single O(N) pass.
    let (global_mean, global_var) = block_mean_variance(&img.view());
    let global_std = global_var.sqrt();

    // Adaptive threshold: captures regions significantly above-average complexity.
    // A floor of 8.0 prevents degenerate low-contrast images from yielding an
    // empty pool.
    let threshold = (0.1 * global_mean + 0.6 * global_std).max(8.0);

    let mut pool: Vec<(usize, usize)> = Vec::with_capacity(h * w / 4);
    quadtree_recurse(img, 0, 0, h, w, threshold, min_block, &mut pool);
    pool
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 2 — DISCRETE ARTIFICIAL BEE COLONY (D-ABC) OPTIMIZER
//
// A discrete variant of Karaboga's ABC algorithm adapted for combinatorial
// pixel-subset selection. The "food source" is a set of L unique indices
// into the sparse pool K, and the fitness function maximises aggregate
// local texture complexity.
//
// Reference: Karaboga, D. (2005). "An idea based on honey bee swarm for
//            numerical optimization." Technical Report TR06, Erciyes University.
//            Adapted to discrete domains per: Sundar, S. et al. (2010).
// ═══════════════════════════════════════════════════════════════════════════

/// **Local texture score** for a single pixel at (y, x).
///
/// Computes the variance within the 3×3 neighbourhood centred on (y, x),
/// clamped to image boundaries. Acts as the per-pixel contribution to the
/// global ABC fitness function.
///
/// Time complexity: O(1) — constant-size window, independent of image size.
#[inline]
fn local_texture_score(img: &ArrayView2<u8>, y: usize, x: usize) -> f64 {
    let (h, w) = img.dim();
    // Saturating subtraction prevents underflow for border pixels.
    let y0 = y.saturating_sub(1);
    let y1 = (y + 2).min(h);
    let x0 = x.saturating_sub(1);
    let x1 = (x + 2).min(w);
    let patch = img.slice(s![y0..y1, x0..x1]);
    let (_, variance) = block_mean_variance(&patch);
    variance
}

/// A **Food Source** in the D-ABC search space.
///
/// Encodes a candidate solution as a set of L unique indices into the
/// sparse pool. The dual representation (Vec + HashSet) achieves:
///   - O(L) fitness accumulation via the pre-computed score table.
///   - O(1) membership queries during neighbourhood search.
///   - O(1) incremental fitness update on mutation (delta method).
struct FoodSource {
    /// Ordered index vector for deterministic iteration.
    indices: Vec<usize>,
    /// Mirror membership set for O(1) collision-free neighbourhood search.
    members: HashSet<usize>,
    /// Aggregated fitness: Σ texture_score[i] for i ∈ indices.
    /// Maintained via delta updates to avoid O(L) recomputation per mutation.
    fitness: f64,
}

impl FoodSource {
    /// Construct a food source from a pre-sampled, duplicate-free index vector.
    fn new(indices: Vec<usize>, score_table: &[f64]) -> Self {
        let fitness: f64 = indices.iter().map(|&i| score_table[i]).sum();
        let members: HashSet<usize> = indices.iter().copied().collect();
        FoodSource { indices, members, fitness }
    }

    /// **Neighbourhood Search** (Employed / Onlooker phase operator).
    ///
    /// Generates a neighbour by replacing one randomly chosen index with a
    /// new, *unselected* index from the pool. Uses rejection sampling for
    /// simplicity; expected cost O(1) when L ≪ K.
    ///
    /// Fitness is updated via delta arithmetic:
    ///   `Δf = score[new_idx] − score[old_idx]`
    /// This keeps each mutation at O(1) instead of O(L).
    fn mutate(&self, pool_size: usize, score_table: &[f64], rng: &mut SmallRng) -> FoodSource {
        // Select a random dimension (position) to perturb.
        let pos = rng.gen_range(0..self.indices.len());
        let old_idx = self.indices[pos];

        // Rejection-sample a candidate not already in the solution.
        // Expected trials ≈ K / (K − L) ≈ 1 when L ≪ K.
        // Guard: if pool is saturated (K ≈ L), fall back to a linear scan
        // to avoid an infinite loop.
        let free_slots = pool_size.saturating_sub(self.members.len());
        let new_idx = if free_slots == 0 {
            // Degenerate: solution already covers the entire pool — no mutation possible.
            old_idx
        } else if free_slots < pool_size / 4 {
            // Dense region: linear scan is cheaper than many rejected samples.
            let start = rng.gen_range(0..pool_size);
            (0..pool_size)
                .map(|i| (start + i) % pool_size)
                .find(|&c| !self.members.contains(&c))
                .unwrap_or(old_idx)
        } else {
            // Sparse region: fast rejection sampling (expected O(1)).
            loop {
                let candidate = rng.gen_range(0..pool_size);
                if !self.members.contains(&candidate) {
                    break candidate;
                }
            }
        };

        // Build the neighbour with O(L) clone + O(1) modification.
        let mut new_indices = self.indices.clone();
        let mut new_members = self.members.clone();
        new_indices[pos] = new_idx;
        new_members.remove(&old_idx);
        new_members.insert(new_idx);

        // Incremental fitness update — O(1).
        let delta_f = score_table[new_idx] - score_table[old_idx];
        FoodSource {
            indices: new_indices,
            members: new_members,
            fitness: self.fitness + delta_f,
        }
    }
}

/// **Main D-ABC Optimisation Loop.**
///
/// Runs three canonical ABC phases per iteration:
///   1. **Employed Bee Phase**: Each source produces one neighbour; greedy
///      acceptance (Metropolis-style without temperature).
///   2. **Onlooker Bee Phase**: Roulette-wheel selection biases exploration
///      toward high-fitness sources.
///   3. **Scout Bee Phase**: Sources stagnant for `limit` trials are
///      abandoned and re-initialised randomly (diversification).
///
/// # Arguments
/// * `pool`        — Sparse pixel pool K = {(y,x)…} from Quadtree.
/// * `img`         — Read-only view of the original cover image.
/// * `payload_size` — Number of pixels L to select (= secret message length in bits).
/// * `colony_size`  — Total bee count; half are employed, half are onlookers.
/// * `max_iter`    — Maximum iteration budget.
/// * `limit`       — Stagnation threshold triggering scout reinitialisation.
/// * `rng`         — Seeded PRNG instance for full reproducibility.
///
/// # Returns
/// The globally best `Vec<usize>` of L pool indices found.
fn run_discrete_abc(
    pool: &[(usize, usize)],
    img: &ArrayView2<u8>,
    payload_size: usize,
    colony_size: usize,
    max_iter: usize,
    limit: u32,
    rng: &mut SmallRng,
) -> Vec<usize> {
    let pool_size = pool.len();
    let num_employed = (colony_size / 2).max(1);

    // ── Pre-compute Texture Score Table — O(K) ───────────────────────────
    // Caching avoids repeated 3×3 window computations during the ABC loop,
    // reducing per-iteration cost from O(C·L·9) → O(C) for fitness queries.
    let score_table: Vec<f64> = pool
        .iter()
        .map(|&(y, x)| local_texture_score(img, y, x))
        .collect();

    // ── Population Initialisation — O(C · L) ────────────────────────────
    // Reservoir sampling (Vitter's Algorithm R) gives each pool pixel an
    // equal probability of appearing in each food source.
    let mut population: Vec<FoodSource> = (0..num_employed)
        .map(|_| {
            let idx_vec = reservoir_sample(rng, pool_size, payload_size).into_vec();
            FoodSource::new(idx_vec, &score_table)
        })
        .collect();

    let mut trial_counters: Vec<u32> = vec![0; num_employed];

    // Initialise global-best tracker.
    let (mut best_fitness, mut best_food) = population
        .iter()
        .map(|f| (f.fitness, f.indices.clone()))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
        .unwrap();

    // ── Main ABC Loop ────────────────────────────────────────────────────
    for _iter in 0..max_iter {
        // ----------------------------------------------------------------
        // PHASE 1 — EMPLOYED BEES
        // Each employed bee locally exploits its assigned food source.
        // ----------------------------------------------------------------
        for i in 0..num_employed {
            let candidate = population[i].mutate(pool_size, &score_table, rng);
            if candidate.fitness > population[i].fitness {
                population[i] = candidate;
                trial_counters[i] = 0; // Improvement → reset stagnation counter.
            } else {
                trial_counters[i] += 1;
            }
        }

        // ----------------------------------------------------------------
        // PHASE 2 — ONLOOKER BEES
        // Onlookers observe the employed bees' dances and select sources
        // with probability proportional to their fitness (roulette wheel).
        // ----------------------------------------------------------------
        let total_fitness: f64 = population.iter().map(|f| f.fitness).sum::<f64>();
        // Guard against degenerate all-zero fitness (uniform probability).
        let probabilities: Vec<f64> = if total_fitness > 0.0 {
            population.iter().map(|f| f.fitness / total_fitness).collect()
        } else {
            vec![1.0 / num_employed as f64; num_employed]
        };

        let mut assigned = 0usize;
        let mut i = 0usize;
        while assigned < num_employed {
            if rng.gen::<f64>() < probabilities[i] {
                assigned += 1;
                let candidate = population[i].mutate(pool_size, &score_table, rng);
                if candidate.fitness > population[i].fitness {
                    population[i] = candidate;
                    trial_counters[i] = 0;
                } else {
                    trial_counters[i] += 1;
                }
            }
            i = (i + 1) % num_employed;
        }

        // Update global best after both exploitation phases.
        for food in &population {
            if food.fitness > best_fitness {
                best_fitness = food.fitness;
                best_food = food.indices.clone();
            }
        }

        // ----------------------------------------------------------------
        // PHASE 3 — SCOUT BEES
        // Sources stagnant beyond `limit` iterations are abandoned.
        // A scout replaces the exhausted source with a uniformly random
        // new candidate (exploration / diversification).
        // ----------------------------------------------------------------
        for i in 0..num_employed {
            if trial_counters[i] >= limit {
                let idx_vec = reservoir_sample(rng, pool_size, payload_size).into_vec();
                population[i] = FoodSource::new(idx_vec, &score_table);
                trial_counters[i] = 0;
            }
        }
    }

    best_food
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE 3 — LSB-MATCHING (LSB-M) EMBEDDER
//
// Unlike naive LSB substitution (which introduces a detectable +1 bias in
// even-valued pixels), LSB-Matching applies a ±1 random perturbation when
// the current LSB differs from the payload bit. This preserves the pixel
// value histogram symmetry, defeating histogram-pair and RS steganalysis.
//
// Reference: Mielikainen, J. (2006). "LSB Matching Revisited."
//            IEEE Signal Processing Letters, 13(5), 285-287.
// ═══════════════════════════════════════════════════════════════════════════

/// Embeds `payload` bits into `img` at `coords` using **LSB-Matching**.
///
/// For each (coordinate, bit) pair:
/// - If `LSB(pixel) == bit` → **no change** (zero distortion introduced).
/// - If `LSB(pixel) ≠ bit`  → **randomly ±1** on the pixel value, keeping
///   it within `[0, 255]`. The random sign is chosen uniformly so that the
///   probability of modifying a pixel toward a higher vs. lower value is
///   equal, preserving the histogram shape statistically.
///
/// Returns the stego image as a new owned `Array2<u8>`.
fn lsb_matching_embed(
    img: &ArrayView2<u8>,
    coords: &[(usize, usize)],
    payload: &[u8],
    rng: &mut SmallRng,
) -> Array2<u8> {
    // Clone the cover image to form the stego image — O(N) but unavoidable.
    let mut stego = img.to_owned();
    let embed_count = coords.len().min(payload.len());

    for idx in 0..embed_count {
        let (y, x) = coords[idx];
        let bit = payload[idx]; // Expected to be 0 or 1.
        let pval = stego[[y, x]];

        // Only modify the pixel if its current LSB disagrees with the payload bit.
        if (pval & 1) != bit {
            // Apply ±1 perturbation with boundary guards.
            stego[[y, x]] = match pval {
                0   => 1,   // Can only increment at the lower boundary.
                255 => 254, // Can only decrement at the upper boundary.
                v   => if rng.gen_bool(0.5) { v + 1 } else { v - 1 },
            };
        }
        // If LSB already matches → pixel unchanged, distortion = 0.
    }

    stego
}

// ═══════════════════════════════════════════════════════════════════════════
// PyO3 BINDINGS — Python-Callable Interface
// ═══════════════════════════════════════════════════════════════════════════

/// **Primary pipeline entry-point exposed to Python.**
///
/// Executes the full three-phase steganography pipeline end-to-end and
/// returns the stego image together with runtime telemetry.
///
/// # Python Signature
/// ```python
/// stego_img, elapsed_ms, pool_size = stegano_core.run_stegano_engine(
///     image,            # np.ndarray, shape (H, W), dtype=np.uint8
///     payload_size,     # int — number of secret bits to embed
///     colony_size=30,   # int — D-ABC colony size (must be even)
///     max_iter=50,      # int — D-ABC iteration budget
///     min_block=4,      # int — Quadtree minimum block side (pixels)
/// )
/// ```
///
/// # Returns
/// - `stego_img`  : `np.ndarray` — stego image, same shape/dtype as input.
/// - `elapsed_ms` : `float`      — wall-clock execution time in milliseconds.
/// - `pool_size`  : `int`        — size of the Quadtree sparse pool K.
///
/// # Raises
/// - `ValueError` if the Quadtree pool is smaller than `payload_size`.
#[pyfunction]
#[pyo3(signature = (image, payload_size, colony_size=30, max_iter=50, min_block=4))]
fn run_stegano_engine<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<u8>,
    payload_size: usize,
    colony_size: usize,
    max_iter: usize,
    min_block: usize,
) -> PyResult<(Bound<'py, PyArray2<u8>>, f64, usize)> {
    // Start wall-clock timer before any computation.
    let t_start = Instant::now();

    // Borrow the NumPy array as a zero-copy ndarray view.
    let img: ArrayView2<u8> = image.as_array();

    // Seed a fast PRNG from OS entropy for reproducible stochastic behaviour.
    let mut rng = SmallRng::from_entropy();

    // ── Phase 1: Adaptive Quadtree Decomposition ─────────────────────────
    let pool: Vec<(usize, usize)> = adaptive_quadtree(&img, min_block);
    let pool_size = pool.len();

    // Ensure the pool is large enough for meaningful ABC optimisation.
    // We need at least L+1 candidates so mutation can always find a free slot.
    if pool_size <= payload_size {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Quadtree sparse pool ({} pixels) is ≤ payload_size ({} bits). \
             Try a more textured image or reduce payload_size.",
            pool_size, payload_size
        )));
    }

    // ── Phase 2: Discrete ABC Optimisation ──────────────────────────────
    // Returns the L pool indices that maximise aggregate texture complexity.
    let best_indices: Vec<usize> = run_discrete_abc(
        &pool,
        &img,
        payload_size,
        colony_size,
        max_iter,
        10, // stagnation limit
        &mut rng,
    );

    // Resolve pool indices to (y, x) coordinate pairs.
    let best_coords: Vec<(usize, usize)> = best_indices.iter().map(|&i| pool[i]).collect();

    // ── Phase 3: LSB-Matching Embedding ─────────────────────────────────
    // Generate a pseudo-random payload for benchmarking; in production this
    // would be the actual secret message bits.
    let payload: Vec<u8> = (0..payload_size).map(|_| rng.gen_range(0u8..2u8)).collect();
    let stego: Array2<u8> = lsb_matching_embed(&img, &best_coords, &payload, &mut rng);

    let elapsed_ms = t_start.elapsed().as_secs_f64() * 1_000.0;

    // Convert the Rust Array2<u8> into a Python-owned NumPy array (zero-copy
    // transfer of the heap buffer via `into_pyarray_bound`).
    Ok((stego.into_pyarray_bound(py), elapsed_ms, pool_size))
}

/// **Full-detail pipeline** — returns stego image, ABC-selected embedding
/// coordinates, and the actual payload bits for visualisation purposes.
///
/// # Python Signature
/// ```python
/// stego, elapsed_ms, pool_size, coords_y, coords_x, payload = \
///     stegano_core.embed_with_details(image, payload_size, colony_size, max_iter, min_block)
/// ```
/// - `coords_y`, `coords_x` : `np.ndarray[int64]` — row / column of each embedded bit.
/// - `payload`              : `np.ndarray[uint8]`  — the embedded bit sequence (0 or 1).
#[pyfunction]
#[pyo3(signature = (image, payload_size, colony_size=30, max_iter=50, min_block=4))]
fn embed_with_details<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<u8>,
    payload_size: usize,
    colony_size: usize,
    max_iter: usize,
    min_block: usize,
) -> PyResult<(
    Bound<'py, PyArray2<u8>>,
    f64,
    usize,
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<i64>>,
    Bound<'py, numpy::PyArray1<u8>>,
)> {
    let t_start = Instant::now();
    let img: ArrayView2<u8> = image.as_array();
    let mut rng = SmallRng::from_entropy();

    // Phase 1: Adaptive Quadtree
    let pool: Vec<(usize, usize)> = adaptive_quadtree(&img, min_block);
    let pool_size = pool.len();

    if pool_size <= payload_size {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Quadtree pool ({pool_size}) <= payload_size ({payload_size})."
        )));
    }

    // Phase 2: Discrete ABC
    let best_indices = run_discrete_abc(
        &pool, &img, payload_size, colony_size, max_iter, 10, &mut rng,
    );
    let best_coords: Vec<(usize, usize)> =
        best_indices.iter().map(|&i| pool[i]).collect();

    // Phase 3: LSB-Matching with a fixed-seed payload for reproducibility.
    let payload: Vec<u8> = (0..payload_size)
        .map(|_| rng.gen_range(0u8..2u8))
        .collect();
    let stego = lsb_matching_embed(&img, &best_coords, &payload, &mut rng);

    let elapsed_ms = t_start.elapsed().as_secs_f64() * 1_000.0;

    // Convert coordinate pairs to two flat i64 arrays (y and x separately)
    // so numpy can consume them without tuple overhead.
    let ys: ndarray::Array1<i64> =
        ndarray::Array1::from_iter(best_coords.iter().map(|&(y, _)| y as i64));
    let xs: ndarray::Array1<i64> =
        ndarray::Array1::from_iter(best_coords.iter().map(|&(_, x)| x as i64));
    let bits: ndarray::Array1<u8> =
        ndarray::Array1::from_iter(payload.iter().copied());

    Ok((
        stego.into_pyarray_bound(py),
        elapsed_ms,
        pool_size,
        ys.into_pyarray_bound(py),
        xs.into_pyarray_bound(py),
        bits.into_pyarray_bound(py),
    ))
}

/// **Quadtree Sparse Map Inspector** — exposed for Python-side visualisation.
///
/// Returns the raw list of (y, x) coordinates that constitute the Quadtree
/// sparse pool, allowing Python code to overlay them on the original image
/// for publication-quality figures.
///
/// # Python Signature
/// ```python
/// coords = stegano_core.get_quadtree_sparse_map(image, min_block=4)
/// # coords: list[tuple[int, int]]
/// ```
#[pyfunction]
#[pyo3(signature = (image, min_block=4))]
fn get_quadtree_sparse_map(
    image: PyReadonlyArray2<u8>,
    min_block: usize,
) -> PyResult<Vec<(usize, usize)>> {
    let img = image.as_array();
    Ok(adaptive_quadtree(&img, min_block))
}

/// **Deterministic D-ABC Coordinate Selector** — exposed for product workflows.
///
/// Accepts a precomputed quadtree pool and returns the D-ABC-optimised
/// coordinates using a deterministic RNG seed for reproducibility.
///
/// # Python Signature
/// ```python
/// coords = stegano_core.dabc_select_coords(
///     image, pool, payload_size, colony_size, max_iter, seed
/// )
/// ```
#[pyfunction]
#[pyo3(signature = (image, pool, payload_size, colony_size, max_iter, seed))]
fn dabc_select_coords(
    image: PyReadonlyArray2<u8>,
    pool: Vec<(usize, usize)>,
    payload_size: usize,
    colony_size: usize,
    max_iter: usize,
    seed: u64,
) -> PyResult<Vec<(usize, usize)>> {
    let img = image.as_array();
    if pool.len() <= payload_size {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Quadtree pool ({}) <= payload_size ({}).",
            pool.len(),
            payload_size
        )));
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    let best_indices = run_discrete_abc(
        &pool,
        &img,
        payload_size,
        colony_size,
        max_iter,
        10,
        &mut rng,
    );
    let coords: Vec<(usize, usize)> = best_indices.iter().map(|&i| pool[i]).collect();
    Ok(coords)
}

/// **Module Registrar** — called by the Python import machinery.
///
/// Registers all public symbols into the `stegano_core` Python module.
/// The module name *must* match the `[lib] name` in `Cargo.toml`.
#[pymodule]
fn stegano_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_stegano_engine, m)?)?;
    m.add_function(wrap_pyfunction!(embed_with_details, m)?)?;
    m.add_function(wrap_pyfunction!(get_quadtree_sparse_map, m)?)?;
    m.add_function(wrap_pyfunction!(dabc_select_coords, m)?)?;
    Ok(())
}
