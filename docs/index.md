# THRML-RS Documentation

**GPU-accelerated energy-based models and hyperspherical navigation in Rust**

---

`thrml-rs` is a pure Rust library for GPU-accelerated probabilistic graphical models and energy-based navigation systems. Built on the [Burn](https://burn.dev) deep learning framework with WGPU backend, it runs natively on Apple Silicon (Metal), NVIDIA GPUs, and other Vulkan-compatible platforms.

## Features

- **GPU Acceleration** via WGPU (Metal on macOS, Vulkan on Linux/Windows)
- **Block Gibbs Sampling** for efficient PGM inference
- **Hyperspherical Navigation** - Multi-cone EBM navigation with ROOTS indexing
- **Contrastive Divergence Training** - SOTA techniques including hard negative mining, PCD, and curriculum learning
- **Discrete EBM Utilities** (Ising/Boltzmann machines)
- **Gaussian PGMs** with custom factors
- **Langevin Dynamics** for continuous optimization
- **Type-Safe** Rust implementation with zero Python dependencies

## Crate Structure

| Crate | Description |
|-------|-------------|
| [`thrml-core`](api/core.md) | Core types: `Node`, `Block`, `BlockSpec`, `SphericalCoords`, similarity |
| [`thrml-samplers`](api/samplers.md) | Sampling programs, Gibbs conditionals, Langevin dynamics |
| [`thrml-models`](api/models.md) | EBM abstractions, Ising models, factor system |
| [`thrml-observers`](api/observers.md) | Observation and statistics collection |
| [`thrml-kernels`](api/kernels.md) | GPU kernels for autodiff operations (Gumbel-Softmax, batch gather) |
| [`thrml-sphere`](api/sphere.md) | **Hyperspherical embedding, ROOTS index, multi-cone navigation, training** |
| [`thrml-examples`](api/examples.md) | Example programs and demos |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
thrml-core = "0.1"
thrml-samplers = "0.1"
thrml-models = "0.1"
thrml-sphere = "0.1"  # For navigation and sphere optimization
```

Or clone and build from source:

```bash
git clone https://github.com/sashimisaketoro/thrml-rs
cd thrml-rs
cargo build --release
```

## Quick Start: Multi-Cone Navigation

The flagship feature is the multi-cone EBM navigation system with ROOTS-based cone spawning:

```rust
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{
    load_blt_safetensors, MultiConeNavigator, BudgetConfig,
    RootsConfig, SphereConfig,
};

// Initialize GPU
ensure_backend();
let device = init_gpu_device();

// Load BLT v3 data (embeddings + raw bytes)
let (sphere_ebm, bytes) = load_blt_safetensors(&path, SphereConfig::default(), &device)?;

// Configure ROOTS and budget
let roots_config = RootsConfig::default()
    .with_partitions(64)
    .with_default_substring_coupling();

let budget_config = BudgetConfig::new(512 * 1024 * 1024)  // 512MB
    .with_max_cones(8)
    .with_min_cone_budget(16 * 1024 * 1024);

// Create multi-cone navigator
let mut navigator = MultiConeNavigator::from_sphere_ebm_with_bytes(
    &sphere_ebm,
    &bytes,
    roots_config,
    budget_config,
    RngKey::new(42),
    &device,
);

// Navigate - cones spawn automatically from ROOTS peaks
let result = navigator.navigate_multi_cone(query, 50.0, 10, RngKey::new(123), &device);

println!("Found {} targets from {} cones", result.n_targets(), result.n_cones());
```

## Quick Start: Advanced Training

Train navigators with state-of-the-art contrastive divergence techniques:

```rust
use thrml_sphere::{
    AdvancedTrainingConfig, TrainableNavigatorEBM, NavigatorEBM,
    HardNegativeMiner, NegativeCurriculumSchedule, TrainingDataset,
    generate_pairs_from_similarity,
};

// Generate training data
let examples = generate_pairs_from_similarity(&sphere_ebm, 1, 8, &device);
let dataset = TrainingDataset::from_examples(examples, 0.1, 42);  // 10% validation

// Configure advanced training
let config = AdvancedTrainingConfig::default()
    .with_hard_negative_mining()           // Similarity-based hard negatives
    .with_pcd(100)                          // 100 persistent fantasy particles
    .with_curriculum()                      // Progressive difficulty
    .with_warmup(5)                         // LR warmup epochs
    .with_cosine_annealing(true);           // Cosine LR schedule

let mut trainable = TrainableNavigatorEBM::from_navigator(
    NavigatorEBM::from_sphere_ebm(sphere_ebm.clone()),
    config.base.clone(),
);

// Train with all techniques
let report = trainable.train_advanced(
    &dataset, 100, 16, &config, RngKey::new(42), &device,
);

println!("{}", report);  // Detailed stats from all techniques
```

## Quick Start: Ising Chain Sampling

Classical block Gibbs sampling for probabilistic graphical models:

```rust
use burn::tensor::Tensor;
use thrml_core::backend::{init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};
use thrml_models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
use thrml_samplers::{RngKey, SamplingSchedule, sample_states};

// Initialize GPU
let device = init_gpu_device();

// Create 5 spin nodes
let nodes: Vec<Node> = (0..5)
    .map(|_| Node::new(NodeType::Spin))
    .collect();

// Linear chain edges
let edges: Vec<_> = (0..4)
    .map(|i| (nodes[i].clone(), nodes[i + 1].clone()))
    .collect();

// Model parameters
let biases: Tensor<WgpuBackend, 1> = Tensor::zeros([5], &device);
let weights: Tensor<WgpuBackend, 1> = Tensor::full([4], 0.5, &device);
let beta: Tensor<WgpuBackend, 1> = Tensor::from_data([1.0f32], &device);

// Create model
let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);

// Two-color blocks (even/odd)
let even_nodes: Vec<_> = nodes.iter().step_by(2).cloned().collect();
let odd_nodes: Vec<_> = nodes.iter().skip(1).step_by(2).cloned().collect();
let free_blocks = vec![
    Block::new(even_nodes).unwrap(),
    Block::new(odd_nodes).unwrap(),
];

// Create sampling program
let program = IsingSamplingProgram::new(&model, free_blocks.clone(), vec![], &device)?;

// Initialize states and run sampling
let key = RngKey::new(42);
let init_state = hinton_init(key, &model, &free_blocks, &[], &device);
let schedule = SamplingSchedule::new(100, 1000, 2);
let observe_blocks = vec![Block::new(nodes).unwrap()];

let samples = sample_states(
    RngKey::new(42),
    &program.program,
    &schedule,
    init_state,
    &[],
    &observe_blocks,
    &device,
);

println!("Collected {} samples", samples.len());
```

## Running Examples

```bash
# Sphere and Navigation
cargo run --example navigator_demo --release      # Multi-cone navigation demo
cargo run --example train_navigator --release     # Train navigator with CD
cargo run --example tune_navigator --release      # Hyperparameter tuning
cargo run --example blt_sphere --release          # Sphere optimization
cargo run --example build_roots --release         # ROOTS index construction

# Classical PGM Sampling
cargo run --example ising_chain --release         # Simple Ising chain
cargo run --example spin_models --release         # Ising with benchmarking
cargo run --example categorical_sampling --release # Potts model
cargo run --example gaussian_pgm --release        # Gaussian PGM
cargo run --example gaussian_bernoulli_ebm --release # Mixed model
cargo run --example train_mnist --release         # MNIST training

# Full API walkthrough
cargo run --example full_api_walkthrough --release
```

## Documentation

- [Architecture Guide](architecture.md) - System design and internals
- [API Reference](api/) - Detailed API documentation
  - [Core Types](api/core.md) - Nodes, Blocks, BlockSpec, SphericalCoords
  - [Samplers](api/samplers.md) - Sampling programs, Langevin dynamics
  - [Models](api/models.md) - EBM and Ising implementations
  - [Observers](api/observers.md) - State and moment observers
  - [Kernels](api/kernels.md) - GPU autodiff operations
  - [Sphere](api/sphere.md) - **Navigation, ROOTS, training, contrastive learning**
  - [Examples](api/examples.md) - Runnable example programs

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../LICENSE-MIT))

at your option.
