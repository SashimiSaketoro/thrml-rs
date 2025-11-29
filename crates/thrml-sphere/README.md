# thrml-sphere

> ⚠️ **Experimental (v0.0.x)**: API will change without notice. This crate is maintained on the `sphere` branch and is not published to crates.io.

[![Rust](https://img.shields.io/badge/rust-1.85+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Branch](https://img.shields.io/badge/branch-sphere-green.svg)](https://github.com/SashimiSaketoro/thrml-rs/tree/sphere)
[![Version](https://img.shields.io/badge/version-0.0.1-red.svg)](Cargo.toml)

Hyperspherical navigation with ROOTS indexing and multi-cone EBM for the THRML framework.

## Installation

```toml
# Add as git dependency (sphere branch)
[dependencies]
thrml-sphere = { git = "https://github.com/SashimiSaketoro/thrml-rs", branch = "sphere" }
```

Or clone the branch directly:
```bash
git clone -b sphere https://github.com/SashimiSaketoro/thrml-rs.git
```

`thrml-sphere` provides Langevin dynamics-based sphere optimization for organizing BLT (Byte Latent Transformer) embeddings on a hyperspherical manifold using the "water-filling" algorithm.

### Key Features

- **Sphere Optimization**: Place embeddings on a hypersphere using Langevin dynamics
- **ROOTS Index**: Compressed inner-shell index for coarse-grained navigation
- **Ising Max-Cut Partitioning**: Similarity-aware partitioning using energy-based models
- **Substring Coupling**: Structural byte-level relationships enhance partitioning
- **Hybrid Compute**: Optimized for Apple Silicon unified memory systems

## Quick Start

### Sphere Optimization

```rust
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{load_from_safetensors, ScaleProfile, SphereConfig};

// Initialize GPU
ensure_backend();
let device = init_gpu_device();

// Load and configure
let config = SphereConfig::from(ScaleProfile::Dev)
    .with_entropy_weighted(true);
let ebm = load_from_safetensors(&path, config, &device)?;

// Run optimization
let key = RngKey::new(42);
let coords = ebm.optimize(key, &device);

// Access results
let cartesian = coords.to_cartesian(); // [N, 3]
```

### ROOTS Index

```rust
use thrml_sphere::{RootsConfig, RootsIndex, load_blt_safetensors, SphereConfig};
use thrml_samplers::RngKey;

// Load BLT v3 output (embeddings + raw bytes)
let (sphere_ebm, patch_bytes) = load_blt_safetensors(
    Path::new("output.safetensors"),
    SphereConfig::default(),
    &device,
)?;

// Build ROOTS with substring coupling
let config = RootsConfig::default()
    .with_partitions(256)
    .with_default_substring_coupling();  // 70% embedding, 30% substring

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

## CLI Examples

### Build ROOTS Index

```bash
# Using BLT v3 SafeTensors (preferred)
cargo run --example build_roots --release -- \
    --blt-safetensors output/item_0.safetensors \
    --substring-coupling \
    --partitions 256 \
    --output roots.bin

# Using separate files (legacy)
cargo run --example build_roots --release -- \
    --input embeddings.safetensors \
    --bytes patches.txt \
    --partitions 256
```

## Architecture

### ROOTS Layer

The ROOTS (inner-shell index) provides 3000:1 compression over full embeddings:

```
┌─────────────────────────────────────────┐
│  Stage 1: COARSE PARTITION              │
│  Sphere → K macro-regions via Ising     │
│  J_ij = α×cos_sim + β×substring_sim     │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Stage 2: ROOTS INDEX                   │
│  Each partition → centroid + stats      │
│  Ultra-compact index (~1MB for 1M pts)  │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│  Stage 3: CLASSIFIER EBM                │
│  Learned query → partition routing      │
└─────────────────────────────────────────┘
```

### Hybrid Compute (Apple Silicon)

On unified memory systems, precision-sensitive operations run on CPU while bulk operations use GPU:

| Operation | Backend | Reason |
|-----------|---------|--------|
| Ising sampling | CPU (f64) | Numerical precision |
| Similarity matrix | GPU (f32) | Parallelism |
| Spherical harmonics | CPU (f64) | Recurrence stability |
| Energy computation | GPU (f32) | Bulk parallelism |

## Modules

| Module | Description |
|--------|-------------|
| `roots` | ROOTS index and Ising max-cut partitioning |
| `compute` | Hybrid CPU/GPU backend and substring similarity |
| `loader` | SafeTensors file loading (BLT v3 format) |
| `sphere_ebm` | Main sphere optimization model |
| `hamiltonian` | Water-filling energy function |
| `langevin` | Sphere-specific Langevin sampler |
| `navigator` | Multi-cone EBM navigation |
| `training` | Contrastive divergence training |

## Configuration Presets

```rust
// For development/testing (fast, small)
let config = RootsConfig::dev();

// For terabyte-scale datasets
let config = RootsConfig::terabyte_scale();

// Custom configuration
let config = RootsConfig::default()
    .with_partitions(512)
    .with_beta(1.5)
    .with_default_substring_coupling()
    .without_member_indices();  // Save memory at scale
```

## BLT Integration

This crate integrates with `blt-burn` for processing BLT model output:

### BLT v3 SafeTensors Format

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `embeddings` | [N, D] | F32 | Patch-aggregated embeddings |
| `prominence` | [N] | F32 | Patch importance scores |
| `patch_entropies` | [N] | F32 | Aggregated entropy per patch |
| `bytes` | [total_bytes] | U8 | Concatenated raw bytes |
| `patch_lengths` | [N] | I32 | Length of each patch |

## Performance

Memory footprint for ROOTS index:

| Partitions | Embedding Dim | Index Size | Compression |
|------------|---------------|------------|-------------|
| 256 | 768 | ~1 MB | 3000:1 |
| 1024 | 768 | ~4 MB | 750:1 |
| 4096 | 768 | ~16 MB | 190:1 |

## Related Crates

- [`thrml-core`](../thrml-core) - Distance/similarity utilities, SphericalCoords
- [`thrml-samplers`](../thrml-samplers) - General Langevin sampler
- [`thrml-models`](../thrml-models) - IsingEBM, GraphSidecar, SpringEBM

## License

MIT License - see [LICENSE](../../LICENSE) for details.
