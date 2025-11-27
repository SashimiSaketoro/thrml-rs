# Examples

The `thrml-examples` crate provides runnable example programs demonstrating all features.

## Running Examples

```bash
cd thrml-rs

# Run any example with:
cargo run --example <name> --release
```

---

## Navigation & Sphere Examples

### `navigator_demo`

Demonstrates multi-cone navigation with ROOTS-based cone spawning.

```bash
cargo run --example navigator_demo --release
```

**Features shown:**
- Loading BLT embeddings
- ROOTS index construction
- Multi-cone navigation
- Result interpretation

### `train_navigator`

Trains a navigator using contrastive divergence with validation.

```bash
cargo run --example train_navigator --release
```

**Features shown:**
- Training data generation
- Basic CD training
- Validation metrics (MRR, Recall@k)
- Early stopping

### `tune_navigator`

Hyperparameter tuning via grid search.

```bash
cargo run --example tune_navigator --release
```

**Features shown:**
- TuningGrid configuration
- Grid/random search
- Result analysis
- CSV/JSON export

### `blt_sphere`

Sphere optimization with Langevin dynamics.

```bash
cargo run --example blt_sphere --release
```

**Features shown:**
- SphereEBM creation
- Langevin optimization
- Coordinate extraction
- Energy visualization

### `build_roots`

ROOTS index construction with substring coupling.

```bash
cargo run --example build_roots --release
```

**Features shown:**
- RootsConfig setup
- Substring coupling
- Partition visualization
- Query routing

---

## PGM Sampling Examples

### `ising_chain`

Simple Ising chain sampling.

```bash
cargo run --example ising_chain --release
```

**Features shown:**
- Node and Block creation
- IsingEBM model
- Two-color block Gibbs
- Sample collection

### `spin_models`

Ising model with performance benchmarking.

```bash
cargo run --example spin_models --release
```

**Features shown:**
- Lattice graph construction
- Graph coloring for blocks
- Timing measurements
- Magnetization statistics

### `categorical_sampling`

Potts model (categorical variables).

```bash
cargo run --example categorical_sampling --release
```

**Features shown:**
- Categorical nodes
- SoftmaxConditional sampler
- Multi-category sampling

### `gaussian_pgm`

Gaussian PGM with continuous variables.

```bash
cargo run --example gaussian_pgm --release
```

**Features shown:**
- GaussianSampler
- Continuous factors
- Mean/variance estimation

### `gaussian_bernoulli_ebm`

Mixed Gaussian-Bernoulli model.

```bash
cargo run --example gaussian_bernoulli_ebm --release
```

**Features shown:**
- Mixed variable types
- Hybrid sampling
- Factor composition

---

## Training Examples

### `train_mnist`

Full MNIST training with contrastive divergence.

```bash
cargo run --example train_mnist --release
```

**Features shown:**
- MNIST data loading
- DiscreteEBM training
- Gradient estimation
- Visualization

---

## Walkthrough Examples

### `full_api_walkthrough`

Comprehensive API demonstration.

```bash
cargo run --example full_api_walkthrough --release
```

**Covers:**
- All core types
- Factor construction
- Program creation
- Sampling workflow
- Observer usage

---

## Example Code Patterns

### Basic Navigation

```rust
use thrml_sphere::{
    NavigatorEBM, SphereEBM, SphereConfig, NavigationWeights,
};
use thrml_samplers::RngKey;

// Create sphere and navigator
let sphere_ebm = SphereEBM::new(embeddings, prominence, None, config, &device);
let navigator = NavigatorEBM::from_sphere_ebm(sphere_ebm)
    .with_weights(NavigationWeights::default());

// Run navigation
let result = navigator.navigate(query, 50.0, RngKey::new(42), 10, &device);

for (idx, energy) in result.target_indices.iter().zip(&result.target_energies) {
    println!("Target {}: energy {:.4}", idx, energy);
}
```

### Multi-Cone Navigation

```rust
use thrml_sphere::{
    MultiConeNavigator, RootsConfig, BudgetConfig,
};

// Setup
let roots_config = RootsConfig::default().with_partitions(64);
let budget_config = BudgetConfig::dev();

let mut navigator = MultiConeNavigator::from_sphere_ebm(
    &sphere_ebm, roots_config, budget_config, key, &device,
);

// Navigate
let result = navigator.navigate_multi_cone(query, 50.0, 10, key, &device);

println!("{} targets from {} cones", result.n_targets(), result.n_cones());
```

### Training with Advanced Techniques

```rust
use thrml_sphere::{
    AdvancedTrainingConfig, TrainableNavigatorEBM,
    TrainingDataset, generate_pairs_from_similarity,
};

// Generate data
let examples = generate_pairs_from_similarity(&sphere_ebm, 1, 8, &device);
let dataset = TrainingDataset::from_examples(examples, 0.1, 42);

// Configure
let config = AdvancedTrainingConfig::default()
    .with_hard_negative_mining()
    .with_curriculum()
    .with_warmup(5);

// Train
let mut trainable = TrainableNavigatorEBM::from_navigator(
    NavigatorEBM::from_sphere_ebm(sphere_ebm.clone()),
    config.base.clone(),
);

let report = trainable.train_advanced(
    &dataset, 100, 16, &config, RngKey::new(42), &device,
);

println!("{}", report);
```

### Evaluation

```rust
use thrml_sphere::{evaluate_navigator, NavigationMetrics};

let metrics = evaluate_navigator(
    &navigator, &test_set, 10, RngKey::new(42), &device,
);

println!("MRR: {:.4}", metrics.mrr);
println!("Recall@10: {:.4}", metrics.recall_10);
```

---

## Creating New Examples

Add new examples to `crates/thrml-examples/examples/`:

```rust
// crates/thrml-examples/examples/my_example.rs
use thrml_core::backend::init_gpu_device;
use thrml_sphere::*;

fn main() {
    let device = init_gpu_device();
    
    // Your example code...
}
```

Then run:

```bash
cargo run --example my_example --release
```
