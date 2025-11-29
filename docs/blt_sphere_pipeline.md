# BLT → thrml-sphere Pipeline

This document describes how to integrate [`blt-burn`](https://github.com/SashimiSaketoro/blt-burn) embeddings with `thrml-sphere` navigation.

## Overview

```
┌─────────────────────┐
│   Raw Bytes         │
│   (text, code)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   blt-burn          │
│   Byte-Latent       │
│   Transformer       │
└──────────┬──────────┘
           │
           ▼ safetensors
┌─────────────────────┐
│   embeddings [N,D]  │
│   prominence [N]    │
│   raw_bytes [N]     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   thrml-sphere      │
│   SphereEBM         │  ← Langevin optimization
│   RootsIndex        │  ← 3000:1 compression
│   MultiConeNavigator│  ← Budget-allocated search
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   ranked results    │
│   + cone metadata   │
└─────────────────────┘
```

## Step 1: Generate Embeddings with BLT

```rust
use blt_burn::{BltConfig, BltModel, encode_bytes};

// Configure BLT encoder
let config = BltConfig::default()
    .with_patch_size(4)
    .with_dim(512);

let model = BltModel::new(&config, &device);

// Encode raw bytes → patch embeddings
let raw_bytes: Vec<Vec<u8>> = load_documents();
let embeddings = encode_bytes(&model, &raw_bytes, &device);
// embeddings: Tensor [N_patches, D]

// Save for thrml-sphere consumption
save_safetensors("embeddings.safetensors", &embeddings, &raw_bytes)?;
```

## Step 2: Load into thrml-sphere

```rust
use thrml_sphere::{load_blt_safetensors, SphereConfig};

let (sphere_ebm, bytes) = load_blt_safetensors(
    Path::new("embeddings.safetensors"),
    SphereConfig::default(),
    &device,
)?;

// sphere_ebm contains:
// - embeddings [N, D]
// - prominence [N] (from attention weights)
// - similarity [N, N] (precomputed cosine sim)
// bytes: Vec<Vec<u8>> - original byte sequences
```

## Step 3: Build ROOTS Index

```rust
use thrml_sphere::{RootsIndex, RootsConfig, SubstringConfig};

let roots_cfg = RootsConfig::default()
    .with_partitions(64)
    .with_substring_coupling(SubstringConfig {
        enabled: true,
        weight: 0.3,  // β in J_ij = α·cos_sim + β·substring_sim
        ..Default::default()
    });

let roots = RootsIndex::from_sphere_ebm_with_bytes(
    &sphere_ebm,
    &bytes,
    roots_cfg,
    RngKey::new(42),
    &device,
);

// ~3000:1 compression of inner-shell embeddings
println!("Partitions: {}", roots.n_partitions());
```

## Step 4: Create Navigator

```rust
use thrml_sphere::{MultiConeNavigator, BudgetConfig, RuntimeConfig};

let runtime = RuntimeConfig::auto();
let budget = runtime.budget
    .with_max_cones(8)
    .with_peak_threshold(0.15);

let mut navigator = MultiConeNavigator::from_sphere_ebm_with_bytes(
    &sphere_ebm,
    &bytes,
    roots_cfg,
    budget,
    RngKey::new(42),
    &device,
);

// Use trained weights if available
// navigator = navigator.with_weights(trained_weights);
```

## Step 5: Navigate

```rust
// Query embedding (from same BLT encoder)
let query = encode_query(&model, b"search query here", &device);

// Navigate - cones spawn from ROOTS activation peaks
let result = navigator.navigate_multi_cone(
    query,
    50.0,   // temperature
    10,     // top_k per cone
    RngKey::new(123),
    &device,
);

// Results
println!("Found {} targets from {} cones", result.n_targets(), result.n_cones());

for (idx, energy) in result.top_k(5) {
    println!("  {} → energy {:.4}", idx, energy);
}
```

## SafeTensors Format

The interchange format between `blt-burn` and `thrml-sphere`:

```
embeddings.safetensors
├── embeddings     [N, D] float32  # Patch embeddings
├── prominence     [N]   float32  # Attention-derived importance
├── entropies      [N]   float32  # Optional: entropy per patch
├── bytes_offsets  [N+1] int64    # Byte sequence boundaries
└── bytes_data     [M]   uint8    # Concatenated raw bytes
```

## Hardware-Aware Execution

The pipeline respects `RuntimeConfig`:

```rust
let runtime = RuntimeConfig::auto();
println!("Hardware: {:?}", runtime.policy.tier);
println!("Profile: {:?}", runtime.policy.profile);

// On Apple Silicon: GPU f32 for embeddings, CPU f64 for precision ops
// On H100/B200: Full f64 on GPU
// On RTX 5090: GPU f32, CPU f64 fallback for IsingSampling
```

## Encoder Independence

`thrml-sphere` doesn't require BLT specifically. Any encoder that produces:

- `embeddings: [N, D]` float tensor
- `prominence: [N]` float tensor (optional, defaults to uniform)
- `bytes: Vec<Vec<u8>>` (optional, for substring coupling)

...can be used. Examples:
- BLT (byte-latent transformer)
- Sentence transformers
- Custom patch encoders
- Vision encoders (treat patches as "bytes")

The navigator doesn't care how embeddings were produced.
