# Sphere Module

The `thrml-sphere` crate provides hyperspherical embedding optimization, ROOTS indexing, multi-cone navigation, and advanced contrastive divergence training.

## Overview

```
thrml-sphere/
├── sphere_ebm.rs      # Core sphere optimization model
├── navigator.rs       # Multi-cone EBM navigation
├── roots.rs           # ROOTS compressed index layer
├── training.rs        # Training infrastructure
├── contrastive.rs     # Advanced CD techniques
├── evaluation.rs      # Navigation metrics (recall@k, MRR, nDCG)
├── config.rs          # Scale profiles and configuration
├── hamiltonian.rs     # Energy functions
├── langevin.rs        # Sphere-specific Langevin dynamics
├── hypergraph.rs      # Graph connectivity
├── loader.rs          # SafeTensors file loading
├── compute.rs         # Hybrid CPU/GPU backend
└── lasing.rs          # Coherence-driven dynamics
```

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
```

---

## ROOTS Index

Compressed inner-shell index for coarse-grained navigation (3000:1 compression).

### `RootsIndex`

```rust
use thrml_sphere::{RootsIndex, RootsConfig, SubstringConfig};

// Configure with substring coupling
let config = RootsConfig::default()
    .with_partitions(64)
    .with_substring_coupling(SubstringConfig {
        enabled: true,
        weight: 0.3,    // β in J_ij = α·cos_sim + β·substring_sim
        ..Default::default()
    });

// Build index
let roots = RootsIndex::from_sphere_ebm_with_bytes(
    &sphere_ebm,
    &patch_bytes,
    config,
    RngKey::new(42),
    &device,
);

// Route queries to partitions
let partition_id = roots.route(&query_embedding);

// Detect activation peaks for cone spawning
let activations = roots.activate(&query, &device);
let peaks = roots.detect_peaks(&activations);
```

### `RootsConfig`

```rust
pub struct RootsConfig {
    pub n_partitions: usize,           // Number of partitions
    pub ising_steps: usize,            // Ising optimization steps
    pub ising_temperature: f32,        // Annealing temperature
    pub threshold: f32,                // Activation threshold
    pub substring_coupling: Option<SubstringConfig>,
}
```

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
