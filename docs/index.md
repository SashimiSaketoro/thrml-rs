# THRML-RS Documentation

**GPU-accelerated probabilistic graphical models in Rust**

---

`thrml-rs` is a pure Rust implementation providing GPU-accelerated block Gibbs sampling for probabilistic graphical models. Built on the [Burn](https://burn.dev) deep learning framework with WGPU backend, it runs natively on Apple Silicon (Metal), NVIDIA GPUs, and other Vulkan-compatible platforms.

## Features

- **GPU Acceleration** via WGPU (Metal on macOS, Vulkan on Linux/Windows)
- **Hybrid Compute Backend** - CPU (f64) / GPU (f32) routing for Apple Silicon unified memory
- **Block Gibbs Sampling** for efficient PGM inference
- **Spin, Categorical, and Continuous Variables** - full mixed-type support
- **Discrete EBM Utilities** (Ising/Boltzmann machines)
- **Graph-based EBMs** - SpringEBM, NodeBiasEBM with lattice construction utilities
- **Gaussian PGMs** with custom factors
- **Training Support** (contrastive divergence, KL gradient estimation)
- **Type-Safe** Rust implementation with zero Python dependencies

## Crate Structure

| Crate | Description |
|-------|-------------|
| `thrml-core` | Core types: `Node`, `Block`, `BlockSpec`, `InteractionGroup` |
| `thrml-samplers` | Sampling programs and conditionals (Gibbs, Gaussian, Softmax) |
| `thrml-models` | EBM abstractions, Ising models, factor system |
| `thrml-observers` | Observation and statistics collection |
| `thrml-examples` | Example programs with visualizations |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
thrml-core = "0.1"
thrml-samplers = "0.1"
thrml-models = "0.1"
thrml-observers = "0.1"
```

Or clone and build from source:

```bash
git clone https://github.com/sashimisaketoro/thrml-rs
cd thrml-rs
cargo build --release --features gpu
```

## Quick Example

Sampling a small Ising chain with two-color block Gibbs:

```rust
use burn::tensor::Tensor;
use thrml_core::backend::{init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};
use thrml_models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
use thrml_samplers::{RngKey, SamplingSchedule, sample_states};

fn main() {
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
    let program = IsingSamplingProgram::new(&model, free_blocks.clone(), vec![], &device)
        .expect("Failed to create program");
    
    // Initialize states
    let key = RngKey::new(42);
    let init_state = hinton_init(key, &model, &free_blocks, &[], &device);
    
    // Run sampling
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
}
```

## Running Examples

```bash
# Simple Ising chain
cargo run --example ising_chain --release

# Ising model with performance benchmarking
cargo run --example spin_models --release

# Categorical variable sampling (Potts model)
cargo run --example categorical_sampling --release

# Full API walkthrough
cargo run --example full_api_walkthrough --release

# Gaussian PGM (continuous variables)
cargo run --example gaussian_pgm --release

# Mixed Gaussian-Bernoulli model
cargo run --example gaussian_bernoulli_ebm --release

# Full MNIST training with contrastive divergence
cargo run --example train_mnist --release
```

## Documentation

- [Architecture Guide](architecture.md) - Internal design and developer documentation
- [API Reference](api/) - Detailed API documentation
  - [Core Types](api/core.md) - Nodes, Blocks, BlockSpec
  - [Samplers](api/samplers.md) - Sampling programs and conditionals
  - [Models](api/models.md) - EBM and Ising implementations
  - [Observers](api/observers.md) - State and moment observers

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE))
- MIT License ([LICENSE-MIT](../LICENSE-MIT))

at your option.
