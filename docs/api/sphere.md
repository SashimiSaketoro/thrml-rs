# Sphere Module

The `thrml-sphere` crate provides hyperspherical embedding optimization, ROOTS indexing, multi-cone navigation, and advanced contrastive divergence training.

## Overview

```
thrml-sphere/
├── sphere_ebm.rs           # Core sphere optimization model
├── navigator.rs            # Multi-cone EBM navigation + RuntimeConfig
├── roots.rs                # H-ROOTS compressed index layer + FlatTreeNode
├── hypergraph.rs           # Graph connectivity (HypergraphEBM)
├── hypergraph_loader.rs    # Load hypergraph with edge type filtering
├── spherical_harmonics.rs  # Spherical harmonics basis and transforms
├── harmonic_navigator.rs   # SH-guided navigation refinement
├── training.rs             # Training infrastructure
├── contrastive.rs          # Advanced CD techniques
├── evaluation.rs           # Navigation metrics (recall@k, MRR, nDCG)
├── config.rs               # Scale profiles and configuration
├── hamiltonian.rs          # Energy functions
├── langevin.rs             # Sphere-specific Langevin dynamics
├── loader.rs               # SafeTensors file loading
├── compute.rs              # Hybrid CPU/GPU backend + text primitives
├── similarity.rs           # GPU similarity (re-exports from thrml-core)
├── double_float.rs         # f64 precision via f32 pairs (for SH stability)
└── lasing.rs               # Coherence-driven dynamics
```

### Re-exported Primitives

`thrml-sphere` re-exports useful primitives from core crates:

```rust
// Hardware-aware runtime
pub use thrml_core::compute::{HardwareTier, PrecisionProfile, RuntimePolicy};

// Generalized metrics (used internally by evaluation.rs)
pub use thrml_core::metrics::{evaluate_retrieval, RetrievalMetrics};

// Text similarity primitives
pub use thrml_core::text::{ngram_hashes, RollingHash, TextSimilarityConfig};

// Max-cut graph partitioning
pub use thrml_samplers::maxcut::{cut_value, maxcut_gibbs, maxcut_multistart};
```

---

## The Math (Start Here)

This section shows the actual energy functions and algorithms with small examples before diving into the API.

### Notation

A vector **x** = (x₁, x₂, ..., x_d) is just d numbers (sliders). Key operations:

| Symbol | Meaning | Code |
|--------|---------|------|
| ‖**x**‖ | Length: √(x₁² + x₂² + ... + x_d²) | `x.powf_scalar(2.0).sum().sqrt()` |
| **x** · **y** | Dot product: x₁y₁ + x₂y₂ + ... + x_dy_d | `x.clone().mul(y).sum()` |
| cos(θ) | Similarity: (**x** · **y**) / (‖**x**‖ ‖**y**‖) | `cosine_similarity(x, y)` |
| S^(d-1) | Unit sphere: all **x** with ‖**x**‖ = 1 | Normalized embeddings |

### SphereEBM Energy (3D Example)

In 3D, embeddings live on the unit sphere S². The energy penalizes:
- Being off the sphere (‖**x**‖ ≠ 1)
- Being far from similar points

```
E_sphere(x) = λ_norm · (‖x‖² - 1)²           # Stay on sphere
            + Σⱼ wⱼ · (1 - cos(θ_xⱼ))        # Attract similar points
            + λ_repel · Σⱼ max(0, cos_thresh - cos(θ_xⱼ))  # Repel dissimilar
```

**Toy example (d=3):**

```
Point A = (1, 0, 0)      # On sphere, ‖A‖ = 1
Point B = (0.6, 0.8, 0)  # On sphere, ‖B‖ = 1
Point C = (1.5, 0, 0)    # OFF sphere, ‖C‖ = 1.5

cos(A, B) = 0.6          # 53° apart
cos(A, C) = 1.0          # Same direction, but C is outside

E_norm(A) = 0            # Already on sphere
E_norm(C) = (1.5² - 1)² = 0.5625  # Penalty for being off sphere
```

### Langevin Update Rule

To minimize energy while exploring, we use Langevin dynamics:

```
x_{t+1} = x_t - η · ∇E(x_t) + √(2ηT) · ξ
          ────────────────   ────────────
          gradient step      Gaussian noise

# Then project back to sphere:
x_{t+1} = x_{t+1} / ‖x_{t+1}‖
```

**Parameters:**
- η = step size (0.01 typical)
- T = temperature (higher = more exploration)
- ξ ~ N(0, I) = random noise

**Why this works:** Gradient descent finds minima, noise lets you escape bad ones. On a sphere, we project after each step to stay on the manifold.

### NavigatorEBM Energy (Worked Example)

Navigation energy ranks candidates by combining three interpretable terms:

```
E_nav(q, x) = λ_sem · (1 - cos(q, x))     # Lower = more similar
           + λ_rad · |r_x - r_target|     # Radial alignment
           + λ_path · path_length(q → x)  # Cost to get there
```

**Toy example:** Query q, three candidates:

| Candidate | cos(q, x) | radius | path_len | E_nav (λ=1,1,1) |
|-----------|-----------|--------|----------|-----------------|
| x₁        | 0.9       | 1.0    | 2        | 0.1 + 0.0 + 2 = **2.1** |
| x₂        | 0.7       | 0.8    | 1        | 0.3 + 0.2 + 1 = **1.5** ← winner |
| x₃        | 0.95      | 1.5    | 5        | 0.05 + 0.5 + 5 = **5.55** |

x₂ wins despite lower similarity because it's cheaper to reach.

**Training:** Learn λ weights via contrastive divergence to match retrieval ground truth.

### MultiConeNavigator Algorithm

```
1. Encode query q
2. Compute ROOTS activations: act[k] = softmax(-dist(q, centroid[k]))
3. Find peaks: P = {k : act[k] > threshold ∧ local_max(k)}
4. Allocate budget per cone proportional to act[k]
5. For each cone k in P:
     - Define aperture from ROOTS partition
     - Run NavigatorEBM-guided sampling within cone
     - Collect top candidates
6. Merge candidates, deduplicate, rerank by E_nav
```

**Diagram (2D slice):**

```
         cone 2 (30% budget)
              /
             /
    ─────●──────────────  ← query on sphere
            \
             \
         cone 1 (40% budget)
              \
               cone 3 (30% budget)
```

### RootsIndex: Shell Compression (2D Example)

In 2D (circle), with 4 angular bins × 2 radial shells:

```
Outer shell (r > 0.8): 90% of points live here (high-d concentration)
Inner shell (r < 0.8): 10% of points, compressed ~10:1

     Bin 0    Bin 1
   ┌───────┬───────┐
   │ 2500  │ 2400  │  ← outer shell (4 bins × ~2500 each)
   ├───────┼───────┤
   │  50   │  48   │  ← inner shell (4 bins × ~50 each, ~50:1 compression)
   ├───────┼───────┤
   │ 2600  │ 2450  │
   └───────┴───────┘
     Bin 2    Bin 3
```

In high-D (d=768), this becomes ~3000:1 compression for inner shells because volume concentrates even more extremely near the surface.

### Precision Requirements

| Operation | Preferred Precision | Why |
|-----------|---------------------|-----|
| Langevin noise | FP64 | Accumulated error compounds |
| Energy gradient | FP64 | Numerical stability near saddles |
| Similarity matrix | FP32 OK | Bulk parallel, no accumulation |
| ROOTS routing | FP32 OK | Coarse, robust to noise |
| Path length | FP32 OK | Integer-like, discrete hops |

**Per hardware:**

| Hardware | Strategy |
|----------|----------|
| Apple Silicon | GPU FP32 for similarity/routing, CPU FP64 for Langevin/gradients |
| RTX 5090 | Same as Apple (weak FP64) |
| H100/B200/Spark | Full FP64 on GPU viable |

---

## Sphere Optimization

### `SphereEBM`

The core model for placing embeddings on a hypersphere using Langevin dynamics.

```rust
use thrml_sphere::{SphereEBM, SphereConfig, ScaleProfile};
use thrml_samplers::RngKey;

// Configure for development
let config = SphereConfig::from(ScaleProfile::Dev)
    .with_steps(100)
    .with_entropy_weighted(true);

// Create sphere EBM
let sphere_ebm = SphereEBM::new(
    embeddings,   // [N, D]
    prominence,   // [N]
    entropies,    // Option<[N]>
    config,
    &device,
);

// Optimize positions
let coords = sphere_ebm.optimize(RngKey::new(42), &device);

// Access results
let cartesian = coords.to_cartesian();  // [N, 3]
println!("Points: {}", coords.r.dims()[0]);
```

### `SphereConfig`

Configuration for sphere optimization:

```rust
pub struct SphereConfig {
    pub n_steps: usize,           // Langevin steps
    pub step_size: f32,           // Step size
    pub temperature: f32,         // Noise temperature
    pub entropy_weighted: bool,   // Use entropy weighting
    pub min_r: f32,               // Minimum radius
    pub max_r: f32,               // Maximum radius
}

// Scale profiles for different use cases
let dev = SphereConfig::from(ScaleProfile::Dev);       // Fast iteration
let small = SphereConfig::from(ScaleProfile::Small);   // Small datasets
let large = SphereConfig::from(ScaleProfile::Large);   // Production
```

---

## Navigation

### `NavigatorEBM`

Multi-cone EBM navigation through hyperspherical embeddings.

```rust
use thrml_sphere::{NavigatorEBM, NavigationWeights};

// Create navigator from sphere
let navigator = NavigatorEBM::from_sphere_ebm(sphere_ebm.clone())
    .with_weights(NavigationWeights::default()
        .with_semantic(2.0)
        .with_radial(0.5)
        .with_entropy(0.3));

// Navigate to find relevant targets
let result = navigator.navigate(query, 50.0, RngKey::new(42), 10, &device);

println!("Top target: {} with energy {:.4}", 
    result.target_indices[0], 
    result.target_energies[0]);
```

### Navigation Energy Function

The navigation energy combines multiple terms:

```text
E_nav(q, T, P) = λ_sem · E_semantic(q, T) +
                λ_rad · E_radial(q, T) +
                λ_graph · E_graph(P) +
                λ_ent · E_entropy(T) +
                λ_path · E_path(P)
```

### `NavigationWeights`

Learnable weights for energy terms:

```rust
pub struct NavigationWeights {
    pub lambda_semantic: f32,  // Embedding similarity
    pub lambda_radial: f32,    // Radial distance
    pub lambda_graph: f32,     // Graph traversal
    pub lambda_entropy: f32,   // Entropy penalty
    pub lambda_path: f32,      // Path length
    pub lambda_harmonic: f32,  // Spherical harmonics interference
    pub temperature: f32,      // LogSumExp temperature
}

// Presets
let uniform = NavigationWeights::uniform();
let semantic_only = NavigationWeights::semantic_only();
```

### `MultiConeNavigator`

ROOTS-guided multi-cone navigation with dynamic budget allocation:

```rust
use thrml_sphere::{MultiConeNavigator, BudgetConfig, RootsConfig};

// Create navigator with ROOTS
let roots_config = RootsConfig::default()
    .with_partitions(64)
    .with_default_substring_coupling();

let budget_config = BudgetConfig::new(512 * 1024 * 1024)
    .with_max_cones(8);

let mut navigator = MultiConeNavigator::from_sphere_ebm_with_bytes(
    &sphere_ebm,
    &raw_bytes,
    roots_config,
    budget_config,
    RngKey::new(42),
    &device,
);

// Navigate - cones spawn from ROOTS peaks
let result = navigator.navigate_multi_cone(
    query, 50.0, 10, RngKey::new(123), &device
);

println!("{} targets from {} cones", result.n_targets(), result.n_cones());
```

### `BudgetConfig`

Budget allocation for multi-cone navigation:

```rust
pub struct BudgetConfig {
    pub total_budget_bytes: usize,    // Total attention budget
    pub max_cones: usize,             // Maximum cones (default: 32)
    pub min_cone_budget: usize,       // Minimum per cone
    pub peak_threshold: f32,          // ROOTS peak detection
    pub min_peak_separation: f32,     // Angular separation
}

// Presets
let dev = BudgetConfig::dev();           // 128MB, 4 cones
let large = BudgetConfig::large_scale(); // 16GB, 32 cones

// Hardware-aware presets
let budget = BudgetConfig::for_tier(HardwareTier::AppleSilicon);  // 6GB, 8 cones
let budget = BudgetConfig::for_tier(HardwareTier::NvidiaHopper);  // 64GB, 32 cones
```

### `RuntimeConfig`

Unified configuration that bundles hardware detection with budget allocation:

```rust
use thrml_sphere::RuntimeConfig;

// Auto-detect everything
let config = RuntimeConfig::auto();

println!("Hardware: {:?}", config.policy.tier);      // e.g., AppleSilicon
println!("Profile: {:?}", config.policy.profile);    // e.g., CpuFp64Strict
println!("Budget: {:.1} GB", config.budget_gb());    // e.g., 6.0

// Access components
let backend = &config.backend;  // ComputeBackend for routing
let budget = &config.budget;    // BudgetConfig for navigation
let sphere = &config.sphere;    // SphereConfig for optimization

// Tier-specific presets
let hpc = RuntimeConfig::for_tier(HardwareTier::NvidiaHopper);
let spark = RuntimeConfig::for_tier(HardwareTier::NvidiaSpark);
```

**Hardware Tiers:**

| Tier | Examples | Memory | Cones | FP64 |
|------|----------|--------|-------|------|
| `AppleSilicon` | M1–M4 Pro/Max | 6 GB | 8 | CPU only |
| `NvidiaConsumer` | RTX 3080–5090 | 8 GB | 12 | Weak |
| `NvidiaHopper` | H100, H200 | 64 GB | 32 | Native |
| `NvidiaBlackwell` | B200 | 128 GB | 64 | Native |
| `NvidiaSpark` | DGX Spark / GB10 | 100 GB | 64 | Native |
| `AmdRdna` | RX 7900 | 8 GB | 12 | Weak |
| `CpuOnly` | No GPU | 4 GB | 4 | Native |

---

## H-ROOTS Index (Hierarchical ROOTS)

Compressed inner-shell index with O(log K) routing and semantic edge extraction (3000:1 compression).

### `RootsIndex`

```rust
use thrml_sphere::{RootsIndex, RootsConfig, SubstringConfig, FlatTreeNode};

// Configure with substring coupling
let config = RootsConfig::default()
    .with_partitions(64)
    .with_substring_coupling(SubstringConfig {
        enabled: true,
        weight: 0.3,    // β in J_ij = α·cos_sim + β·substring_sim
        ..Default::default()
    });

// Build HIERARCHICAL index (default in pipeline)
let roots = RootsIndex::from_sphere_ebm_with_bytes_hierarchical(
    &sphere_ebm,
    &patch_bytes,
    config,
    RngKey::new(42),
    &device,
);

// Tree-based activation (O(log K) instead of O(K))
let peaks = roots.activate_tree(&query, threshold, beam_width);

// Extract semantic edges (reuses similarity from Ising - FREE!)
let semantic_edges = roots.extract_semantic_edges(0.7);
// Returns Vec<(src_idx, dst_idx, similarity)> for sim ≥ 0.7

// Serialize tree for persistence
if let Some(flat_nodes) = roots.flatten_tree() {
    db.save_roots_tree(&flat_nodes, version)?;
}

// Reconstruct tree from database
let flat_nodes = db.load_roots_tree()?;
let root = RootsIndex::unflatten_tree(&flat_nodes, &partitions);
```

### `FlatTreeNode` (Tree Serialization)

```rust
pub struct FlatTreeNode {
    pub node_id: usize,
    pub parent_id: Option<usize>,   // None for root
    pub is_leaf: bool,
    pub partition_id: Option<usize>, // Only for leaves
    pub left_child: Option<usize>,   // Only for internals
    pub right_child: Option<usize>,
    pub centroid: Vec<f32>,
    pub point_count: usize,
    pub radius_range: (f32, f32),
    pub prom_range: (f32, f32),
}
```

### Tree Lifecycle

```rust
// After sphere optimization changes positions:
db.invalidate_roots_tree()?;

// Check before using tree:
if db.is_roots_tree_stale()? {
    let new_roots = RootsIndex::from_sphere_ebm_hierarchical(...);
    db.save_roots_tree(&new_roots.flatten_tree().unwrap(), version)?;
    db.clear_roots_tree_stale()?;
}
```

### `RootsConfig`

```rust
pub struct RootsConfig {
    pub n_partitions: usize,           // Number of partitions
    pub ising_steps: usize,            // Ising optimization steps
    pub ising_temperature: f32,        // Annealing temperature
    pub threshold: f32,                // Activation threshold
    pub substring_coupling: Option<SubstringConfig>,
    pub similarity_k: usize,           // Top-k for sparse similarity
}
```

---

## Hypergraph

Structural adjacency graph with heterogeneous edge types.

### Edge Types

| Label | Default Weight | Description |
|-------|----------------|-------------|
| `"next"` | 1.0 | Sequential edges within documents |
| `"contains"` | 0.5 | Hierarchical containment |
| `"same_source"` | 1.0 | Cross-view connections |
| `"semantic"` | 0.8 | **High similarity pairs from ROOTS (≥0.7)** |

### `HypergraphLoadConfig`

```rust
use thrml_sphere::{HypergraphLoadConfig, load_hypergraph_from_sqlite};

// Configure which edge types to include
let config = HypergraphLoadConfig {
    include_next_edges: true,
    include_contains_edges: false,
    include_same_source_edges: true,
    include_semantic_edges: true,     // From ROOTS similarity
    next_edge_weight: 1.0,
    contains_edge_weight: 0.5,
    same_source_edge_weight: 1.0,
    semantic_edge_weight: 0.8,        // Slightly weaker than sequential
    ..Default::default()
};

// Load with filtering
let hypergraph = load_hypergraph_from_sqlite(&db_path, n_patches, &config, &device)?;

// Presets
let code_config = HypergraphLoadConfig::for_code();   // semantic_weight: 0.9
let text_config = HypergraphLoadConfig::for_text();   // semantic_weight: 0.7
```

---

## Spherical Harmonics

Frequency-domain smooth interpolation using spherical harmonic (SH) basis functions. Complements ROOTS (coarse partition routing) and hypergraph (structural edges) with smooth wave-like refinement.

### `SphericalHarmonicsBasis`

Precomputed Y_l^m basis on a Driscoll-Healy grid:

```rust
use thrml_sphere::{SphericalHarmonicsBasis, SphericalHarmonicsConfig};

// Create basis with band limit L
let config = SphericalHarmonicsConfig {
    band_limit: 16,  // l = 0..16, higher = finer resolution
    use_f64: false,  // Use DoubleTensor for high L
};
let basis = SphericalHarmonicsBasis::new(config, &device);

// Forward transform: field → coefficients
let coeffs = basis.forward_sht(&density_field);

// Inverse transform: coefficients → field
let reconstructed = basis.inverse_sht(&coeffs);

// Find peak in interference pattern
let (theta, phi, value) = basis.find_peak(&intensity_field);
```

### `HarmonicNavigator`

SH-guided navigation refinement using wave superposition:

```rust
use thrml_sphere::HarmonicNavigator;

// Create navigator with pre-fitted SH coefficients
let harmonic_nav = HarmonicNavigator::from_ebm(&sphere_ebm, config, &device);

// Navigate using SH interference pattern
let result = harmonic_nav.navigate(&query_embedding, radius, &device);
// Returns: HarmonicNavigationResult { r, theta, phi, alpha, score, confidence }
```

### Integration with NavigatorEBM

The `lambda_harmonic` weight controls SH influence in the total energy:

```
E_total = λ_semantic · E_sem + λ_radial · E_rad + λ_graph · E_graph 
        + λ_entropy · E_ent + λ_path · E_path + λ_harmonic · E_harmonic
```

**Use cases:**
- **ROOTS**: Coarse partition routing (O(log K))
- **Hypergraph**: Structural adjacency (sequential, containment)
- **HarmonicNavigator**: Frequency-domain smooth interpolation

---

## Training

### Basic Training

```rust
use thrml_sphere::{
    TrainableNavigatorEBM, NavigatorTrainingConfig,
    TrainingDataset, generate_pairs_from_similarity,
};

// Generate training pairs
let examples = generate_pairs_from_similarity(&sphere_ebm, 1, 8, &device);
let dataset = TrainingDataset::from_examples(examples, 0.1, 42);

// Configure training
let config = NavigatorTrainingConfig::default()
    .with_learning_rate(0.01)
    .with_negatives(8)
    .with_momentum(0.9);

// Create trainable navigator
let mut trainable = TrainableNavigatorEBM::from_navigator(
    NavigatorEBM::from_sphere_ebm(sphere_ebm.clone()),
    config,
);

// Train
let losses = trainable.train(&dataset.train, 50, 16, RngKey::new(42), &device);
```

### Hybrid Training (Recommended)

GPU-accelerated training with CPU fallback for precision-sensitive operations.
Uses batched GPU operations for forward pass and routes gradient accumulation
based on `ComputeBackend` configuration.

```rust
use thrml_sphere::TrainableNavigatorEBM;
use thrml_core::ComputeBackend;

// Create trainable navigator (same as above)
let mut trainable = TrainableNavigatorEBM::from_navigator(
    NavigatorEBM::from_sphere_ebm(sphere_ebm.clone()),
    config,
);

// Train with hybrid CPU/GPU execution (auto-detects backend)
let losses = trainable.train_hybrid(
    &dataset.train, 
    50,             // epochs
    16,             // batch_size
    RngKey::new(42), 
    &device,
    None,           // auto-detect backend
);

// Or with explicit backend configuration
let backend = ComputeBackend::apple_silicon();
let losses = trainable.train_hybrid(
    &dataset.train, 50, 16, RngKey::new(42), &device,
    Some(&backend),
);
```

**Why use hybrid training?**

| Feature | `train()` | `train_hybrid()` |
|---------|-----------|------------------|
| Forward pass | Sequential | GPU-batched |
| Gradient accumulation | CPU f32 | Routed (CPU f64 when needed) |
| Metal f64 handling | May overflow | Auto-fallback to CPU |
| AMD RDNA support | May fail | Auto-fallback to CPU |

### Extended Training with Validation

```rust
use thrml_sphere::ExtendedTrainingConfig;

let config = ExtendedTrainingConfig::default()
    .with_val_every(5)
    .with_early_stopping(3)
    .with_top_k(10);

let report = trainable.train_with_validation(
    &dataset, 100, 16, &config, RngKey::new(42), &device,
);

println!("Best MRR: {:.4} at epoch {}", report.best_val_mrr, report.best_epoch);
```

### Hyperparameter Tuning

```rust
use thrml_sphere::{TuningGrid, TuningSession};

let grid = TuningGrid::default()
    .with_learning_rates(vec![1e-4, 1e-3, 1e-2])
    .with_temperatures(vec![0.1, 0.5, 1.0]);

let mut session = TuningSession::new(grid);
session.run_grid_search(&sphere_ebm, &dataset, 50, 16, key, &device);

if let Some(best) = session.best_result() {
    println!("Best: MRR={:.4}, lr={:.0e}", 
        best.metrics.mrr, best.config.learning_rate);
}

session.to_csv(Path::new("results.csv"))?;
```

---

## Advanced Contrastive Divergence

State-of-the-art training techniques from 2024-2025 research.

### `HardNegativeMiner`

Similarity-based hard negative mining with false negative filtering:

```rust
use thrml_sphere::HardNegativeMiner;

let miner = HardNegativeMiner::new(0.9)  // Filter top 10% similar to positive
    .with_min_negatives(4)
    .with_hardness_temperature(0.5)
    .with_hard_fraction(0.7);  // 70% hard, 30% random

let negatives = miner.mine(
    &query, 
    positive_idx, 
    &embeddings, 
    Some(&similarity_matrix),
    8,  // n_negatives
    &device,
);
```

### `PersistentParticleBuffer`

Persistent Contrastive Divergence (PCD) with fantasy particles:

```rust
use thrml_sphere::PersistentParticleBuffer;

let mut buffer = PersistentParticleBuffer::new(1000, 128)  // 1000 particles, dim 128
    .with_langevin_steps(10)
    .with_replay_prob(0.95);

buffer.initialize_from_data(&embeddings, &device);

// Update particles with energy gradient
buffer.update_langevin(
    |particles| energy_gradient(particles),
    &device,
);

// Sample negatives
let negatives = buffer.sample(batch_size, Some(&data), &device);
```

### `NegativeCurriculumSchedule`

Progressive difficulty scheduling for negatives:

```rust
use thrml_sphere::{NegativeCurriculumSchedule, NegativeDifficulty};

let schedule = NegativeCurriculumSchedule::new(10, 30, 100);
// Epoch 0-10: Easy, 10-30: Medium, 30+: Hard

let difficulty = schedule.get_difficulty(epoch);
let (easy_frac, med_frac, hard_frac) = schedule.get_fractions(progress);

// Select negatives based on curriculum
let negatives = schedule.select_negatives(&candidates, n_negatives, epoch);
```

### `AdvancedTrainingConfig`

Unified configuration for all advanced techniques:

```rust
use thrml_sphere::AdvancedTrainingConfig;

let config = AdvancedTrainingConfig::default()
    .with_hard_negative_mining()           // Enable hard negatives
    .with_pcd(100)                          // 100 persistent particles
    .with_curriculum()                      // Progressive difficulty
    .with_warmup(5)                         // LR warmup epochs
    .with_cosine_annealing(true)           // Cosine schedule
    .with_min_lr(1e-5);                    // Minimum LR

let report = trainable.train_advanced(
    &dataset, 100, 16, &config, RngKey::new(42), &device,
);

println!("{}", report);  // Shows all technique statistics
```

### `SGLDNegativeSampler`

SGLD-based negative phase sampling:

```rust
use thrml_sphere::{SGLDNegativeSampler, SGLDNegativeConfig};

let config = SGLDNegativeConfig::new(20)
    .with_step_size(0.01)
    .with_informative_init(0.5)  // 50% data, 50% noise
    .with_proximal_radius(0.5);   // Constrain near init

let mut sampler = SGLDNegativeSampler::new(config);

let negatives = sampler.sample(
    batch_size,
    &data_init,
    |x| energy_gradient(x),
    &device,
);
```

---

## Evaluation Metrics

### `NavigationMetrics`

Standard IR metrics for navigation quality:

```rust
use thrml_sphere::{evaluate_navigator, NavigationMetrics};

let metrics = evaluate_navigator(&navigator, &test_set, 10, key, &device);

println!("Recall@1:  {:.4}", metrics.recall_1);
println!("Recall@10: {:.4}", metrics.recall_10);
println!("MRR:       {:.4}", metrics.mrr);
println!("nDCG@10:   {:.4}", metrics.ndcg_10);

if metrics.is_good() {
    println!("Good performance!");
}
```

### Metric Functions

```rust
use thrml_sphere::evaluation::{recall_at_k, reciprocal_rank, ndcg_at_k};

let recall = recall_at_k(&results, ground_truth, 10);
let rr = reciprocal_rank(&results, ground_truth);
let ndcg = ndcg_at_k(&results, ground_truth, 10);
```

---

## Data Loading

### BLT SafeTensors Format

Load BLT v3 output with embeddings and raw bytes:

```rust
use thrml_sphere::load_blt_safetensors;

let (sphere_ebm, bytes) = load_blt_safetensors(
    Path::new("output.safetensors"),
    SphereConfig::default(),
    &device,
)?;

// bytes: Vec<Vec<u8>> - raw byte sequences per patch
```

### Standard SafeTensors

Load embeddings and prominence only:

```rust
use thrml_sphere::load_from_safetensors;

let sphere_ebm = load_from_safetensors(
    Path::new("embeddings.safetensors"),
    SphereConfig::default(),
    &device,
)?;
```

---

## Hypergraph Support

### `HypergraphSidecar`

Graph connectivity for path-based navigation:

```rust
use thrml_sphere::{HypergraphSidecar, HypergraphEBM};

let mut sidecar = HypergraphSidecar::new(n_nodes);
sidecar.add_edge(0, 1, 1.0);
sidecar.add_edge(1, 2, 0.5);

let hypergraph_ebm = HypergraphEBM::from_sidecar(
    &sidecar, 
    0.1,    // spring_k
    0.3,    // gravity
    &device,
);

let navigator = NavigatorEBM::from_sphere_ebm(sphere_ebm)
    .with_hypergraph(hypergraph_ebm);
```

---

## Complete Example

```rust
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{
    load_blt_safetensors, MultiConeNavigator, BudgetConfig,
    RootsConfig, SphereConfig, AdvancedTrainingConfig,
    TrainableNavigatorEBM, NavigatorEBM, TrainingDataset,
    generate_pairs_from_similarity, evaluate_navigator,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Setup
    ensure_backend();
    let device = init_gpu_device();

    // Load data
    let (sphere_ebm, bytes) = load_blt_safetensors(
        Path::new("data.safetensors"),
        SphereConfig::default(),
        &device,
    )?;

    // Generate training data
    let examples = generate_pairs_from_similarity(&sphere_ebm, 1, 8, &device);
    let dataset = TrainingDataset::from_examples(examples, 0.1, 42);

    // Train with advanced techniques
    let config = AdvancedTrainingConfig::default()
        .with_hard_negative_mining()
        .with_curriculum();

    let mut trainable = TrainableNavigatorEBM::from_navigator(
        NavigatorEBM::from_sphere_ebm(sphere_ebm.clone()),
        config.base.clone(),
    );

    let report = trainable.train_advanced(
        &dataset, 50, 16, &config, RngKey::new(42), &device,
    );
    println!("{}", report);

    // Evaluate
    let metrics = evaluate_navigator(
        &trainable.navigator, 
        &dataset.validation, 
        10, 
        RngKey::new(123), 
        &device,
    );
    println!("{}", metrics);

    // Create multi-cone navigator for production
    let roots_config = RootsConfig::default().with_partitions(64);
    let budget_config = BudgetConfig::dev();

    let mut navigator = MultiConeNavigator::from_sphere_ebm_with_bytes(
        &sphere_ebm,
        &bytes,
        roots_config,
        budget_config,
        RngKey::new(42),
        &device,
    );

    // Use trained weights
    navigator = navigator.with_weights(trainable.navigator.weights.clone());

    // Run navigation
    let query = /* your query embedding */;
    let result = navigator.navigate_multi_cone(
        query, 50.0, 10, RngKey::new(999), &device,
    );

    println!("Found {} targets from {} cones", 
        result.n_targets(), result.n_cones());

    Ok(())
}
```
