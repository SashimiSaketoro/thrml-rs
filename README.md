# thrml-rs

**GPU-accelerated probabilistic graphical models in Rust**

[![CI](https://github.com/sashimisaketoro/thrml-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sashimisaketoro/thrml-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/thrml.svg)](https://crates.io/crates/thrml)
[![Documentation](https://docs.rs/thrml/badge.svg)](https://docs.rs/thrml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

`thrml-rs` is a pure Rust implementation of GPU-accelerated sampling for probabilistic graphical models (PGMs), 
inspired by [Extropic's THRML](https://github.com/extropic-ai/thrml) library.

## Features

- 🚀 **GPU Acceleration**: Multiple backend support:
  - **WGPU** (default): Metal (macOS), Vulkan (Linux/Windows)
  - **CUDA**: Native NVIDIA GPU support
- 🎲 **Block Gibbs Sampling**: Efficient parallel sampling for PGMs
- 🧠 **Energy-Based Models**: Ising models, discrete EBMs, Gaussian PGMs
- 🔢 **Mixed Variable Types**: Spin, categorical, and continuous nodes
- 🔄 **Deterministic RNG**: Reproducible sampling with ChaCha8-based key splitting
- 📊 **Moment Estimation**: Built-in observers for computing statistics
- 📈 **Training Support**: Contrastive divergence, KL gradient estimation

## Quick Start

```rust
use thrml_core::{Node, NodeType, Block, backend::init_gpu_device};
use thrml_models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
use thrml_samplers::{RngKey, SamplingSchedule};
use burn::tensor::Tensor;

fn main() {
    // Initialize GPU
    let device = init_gpu_device();

    // Create a 5-node Ising chain
    let nodes: Vec<Node> = (0..5).map(|_| Node::new(NodeType::Spin)).collect();
    let edges: Vec<_> = nodes.windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect();

    // Define biases and coupling weights
    let biases = Tensor::from_data([0.1f32, 0.2, 0.0, -0.1, 0.3], &device);
    let weights = Tensor::from_data([0.5f32, -0.3, 0.4, 0.2], &device);
    let beta = Tensor::from_data([1.0f32], &device);

    // Create the Ising model
    let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);

    // Initialize using Hinton's method
    let key = RngKey::new(42);
    let blocks = vec![Block::new(nodes).unwrap()];
    let init_state = hinton_init(key, &model, &blocks, &[], &device);
    
    println!("Model initialized with {} nodes", model.nodes().len());
}
```

## Crate Structure

| Crate | Description |
|-------|-------------|
| [`thrml-core`](crates/thrml-core/) | Core types: Node, Block, BlockSpec, GPU backend |
| [`thrml-samplers`](crates/thrml-samplers/) | Sampling algorithms: Gibbs, Bernoulli, Softmax, Gaussian |
| [`thrml-models`](crates/thrml-models/) | Model implementations: Ising, Discrete EBM, Continuous factors |
| [`thrml-observers`](crates/thrml-observers/) | Observation utilities: State, Moments |
| [`thrml-examples`](crates/thrml-examples/) | Example programs and utilities |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
thrml-core = "0.1"
thrml-samplers = "0.1"
thrml-models = "0.1"
thrml-observers = "0.1"
```

### Feature Flags

- `gpu` (default): Enable WGPU backend (Metal/Vulkan/DX12)
- `cuda`: Enable CUDA backend in addition to WGPU (requires NVIDIA GPU + CUDA toolkit)

```bash
# Default: WGPU backend (Metal on macOS, Vulkan on Linux)
cargo build --release

# Enable CUDA support alongside WGPU
cargo build --release --features cuda
```

## Requirements

- Rust 1.89+ (stable) - required by Burn 0.19
- **WGPU backend**: GPU with Metal (macOS) or Vulkan (Linux/Windows) support
- **CUDA backend**: NVIDIA GPU with CUDA toolkit installed

## Examples

See the [`examples/`](crates/thrml-examples/examples/) directory:

```bash
# Simple Ising chain demonstration
cargo run --release --example ising_chain

# Spin models with performance benchmarking
cargo run --release --example spin_models

# Categorical variable sampling with visualization
cargo run --release --example categorical_sampling

# Full API walkthrough tutorial
cargo run --release --example full_api_walkthrough

# Gaussian PGM sampling (continuous nodes)
cargo run --release --example gaussian_pgm

# Mixed Gaussian-Bernoulli model
cargo run --release --example gaussian_bernoulli_ebm

# Full MNIST training with contrastive divergence
cargo run --release --example train_mnist
```

## Documentation

- [API Documentation](https://docs.rs/thrml)
- [Architecture Guide](docs/architecture.md)
- [Examples README](crates/thrml-examples/README.md)

## Performance

THRML-RS leverages the [Burn](https://burn.dev) deep learning framework for GPU acceleration:

| Backend | Platform | GPU Support |
|---------|----------|-------------|
| WGPU-Metal | macOS | Apple Silicon, AMD, Intel |
| WGPU-Vulkan | Linux/Windows | NVIDIA, AMD, Intel |
| CUDA | Linux/Windows | NVIDIA (native) |

Key optimizations:
- Native Metal acceleration on Apple Silicon
- CUDA for maximum performance on NVIDIA GPUs
- Efficient tensor operations with automatic batching
- Fused GPU kernels for sampling operations

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Acknowledgments

This project is inspired by [Extropic's THRML](https://github.com/extropic-ai/thrml) library. 
THRML-RS is an independent Rust implementation providing the same functionality with native 
GPU acceleration.
