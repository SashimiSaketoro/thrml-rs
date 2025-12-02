# thrml-rs

**GPU-accelerated probabilistic graphical models in Rust**

[![CI](https://github.com/sashimisaketoro/thrml-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/sashimisaketoro/thrml-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/thrml.svg)](https://crates.io/crates/thrml)
[![Documentation](https://docs.rs/thrml/badge.svg)](https://docs.rs/thrml)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

`thrml-rs` is a pure Rust implementation of GPU-accelerated sampling for probabilistic graphical models (PGMs), 
ported from [Extropic's THRML](https://github.com/extropic-ai/thrml) library, with a few tweaks. 

## Features

- **GPU Acceleration**: Multiple backend support:
  - **WGPU** (default): Metal (macOS), Vulkan (Linux/Windows)
  - **CUDA**: Native NVIDIA GPU support
- **Block Gibbs Sampling**: Efficient parallel sampling for PGMs
- **Energy-Based Models**: Ising models, discrete EBMs, Gaussian PGMs
- **Mixed Variable Types**: Spin, categorical, and continuous nodes
- **Deterministic RNG**: Reproducible sampling with ChaCha8-based key splitting
- **Moment Estimation**: Built-in observers for computing statistics
- **Training Support**: Contrastive divergence, KL gradient estimation

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

### From crates.io (stable, main branch)

```toml
[dependencies]
thrml-core = "0.1"
thrml-samplers = "0.1"
thrml-models = "0.1"
thrml-observers = "0.1"
```

### From Git (sphere branch, experimental)

The `sphere` branch includes additional crates and features not yet published to crates.io:

```toml
[dependencies]
thrml-core = { git = "https://github.com/SashimiSaketoro/thrml-rs", branch = "sphere" }
thrml-samplers = { git = "https://github.com/SashimiSaketoro/thrml-rs", branch = "sphere" }
thrml-models = { git = "https://github.com/SashimiSaketoro/thrml-rs", branch = "sphere" }
thrml-observers = { git = "https://github.com/SashimiSaketoro/thrml-rs", branch = "sphere" }

# Experimental: hyperspherical navigation (sphere branch only)
thrml-sphere = { git = "https://github.com/SashimiSaketoro/thrml-rs", branch = "sphere" }
```

### Feature Flags

| Feature | Backend | Use Case |
|---------|---------|----------|
| `gpu` (default) | WGPU | Metal (macOS), Vulkan (Linux), DX12 (Windows) |
| `cuda` | CUDA + WGPU | NVIDIA GPUs with native CUDA |
| `cpu` | ndarray + WGPU | Development/testing without GPU, or CPU fallback |

```bash
# Default: WGPU backend (Metal on macOS, Vulkan on Linux)
cargo build --release

# Enable CUDA support alongside WGPU
cargo build --release --features cuda

# Enable CPU backend (useful for testing or systems without GPU)
cargo build --release --features cpu
```

## Requirements

- Rust 1.89+ (stable) - required by Burn 0.19
- **WGPU backend**: GPU with Metal (macOS) or Vulkan (Linux/Windows) support
- **CUDA backend**: NVIDIA GPU with CUDA toolkit installed

## Runtime & Hardware Profiles

`thrml-rs` is designed to run from laptops to DGX-class servers. The core crates share a
common runtime abstraction:

- **`ComputeBackend`**: Selects CPU / GPU / hybrid execution
- **`PrecisionMode`**: Chooses between `GpuFast`, `CpuPrecise`, or `Adaptive` routing
- **`OpType`**: Tags operations (Ising sampling, distance, navigator steps) for precision-aware routing

### Hardware Tiers

| Tier | Examples | FP64 | Default Profile |
|------|----------|------|-----------------|
| **Apple Silicon** | M1–M4 Pro/Max/Ultra | CPU only | `CpuFp64Strict` - GPU for throughput, CPU for precision |
| **Consumer GPU** | RTX 3080–5090, RDNA3/4 | Weak | `GpuMixed` - GPU FP32, CPU f64 for corrections |
| **HPC GPU** | H100, H200, B200, DGX Spark | Native | `GpuHpcFp64` - Full f64 on GPU |
| **CPU Only** | Servers without GPU | Native | `CpuFp64Strict` - All operations on CPU |

### Usage

```rust
use thrml_core::compute::{ComputeBackend, RuntimePolicy, OpType};

// Auto-detect hardware and create appropriate backend
let policy = RuntimePolicy::detect();
let backend = ComputeBackend::from_policy(&policy);

println!("Detected: {:?}", policy.tier);  // e.g., AppleSilicon
println!("Profile: {:?}", policy.profile); // e.g., CpuFp64Strict

// Precision-aware routing
if backend.use_cpu(OpType::IsingSampling, None) {
    // High-precision CPU f64 path (Apple Silicon, consumer GPU)
} else {
    // Fast GPU path (HPC GPUs with native f64)
}
```

The default `ComputeBackend::default()` auto-detects your hardware. For explicit control:

```rust
// Force specific profiles
let apple = RuntimePolicy::apple_silicon();
let hpc = RuntimePolicy::nvidia_hopper();  // H100/H200
let spark = RuntimePolicy::nvidia_spark(); // DGX Spark / GB10
```

## `thrml-sphere` (this branch)

| Crate | Description |
|-------|-------------|
| [`thrml-sphere`](crates/thrml-sphere/) | Hyperspherical navigation, ROOTS indexing, multi-cone EBM |

**What it does:**
- **SphereEBM**: Langevin dynamics to place embeddings on a hypersphere
- **NavigatorEBM**: EBM with learnable weights for similarity, radial alignment, path length
- **MultiConeNavigator**: Spawns cones from ROOTS peaks, allocates budget per cone
- **RootsIndex**: Compresses inner shells ~3000:1 for coarse routing

Training: contrastive divergence with hard negatives, PCD, curriculum scheduling.

**BLT Integration:** Works with [`blt-burn`](https://github.com/SashimiSaketoro/blt-burn) — BLT provides embeddings, `thrml-sphere` provides the navigator.

```rust
use thrml_sphere::{RuntimeConfig, BudgetConfig, SphereConfig, RootsConfig};

let runtime = RuntimeConfig::auto();
println!("Hardware: {:?}, Budget: {:.1} GB", runtime.policy.tier, runtime.budget_gb());

let roots_cfg = RootsConfig::default().with_partitions(64);
let budget = runtime.budget.with_max_cones(8);
```

For full API, see [docs/api/sphere.md](docs/api/sphere.md).

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

- [API Documentation](https://docs.rs/thrml) - Published crates (main branch)
- [Core API](docs/api/core.md) - Nodes, blocks, compute backends, metrics, text utilities
- [Samplers API](docs/api/samplers.md) - Gibbs, Bernoulli, Gaussian, max-cut algorithms
- [Models API](docs/api/models.md) - Ising, discrete EBMs, continuous factors
- [Sphere API](docs/api/sphere.md) - Hyperspherical navigation (sphere branch)
- [BLT Pipeline](docs/blt_sphere_pipeline.md) - BLT → thrml-sphere integration
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
