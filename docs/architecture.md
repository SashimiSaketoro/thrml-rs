# Architecture Guide

## Overview

THRML-RS is a pure Rust library for GPU-accelerated energy-based models and probabilistic graphical models. It consists of two main subsystems:

1. **PGM Sampling**: Block Gibbs sampling for discrete and continuous graphical models
2. **Hyperspherical Navigation**: Multi-cone EBM navigation with ROOTS indexing

## Crate Architecture

```
thrml-rs/
├── thrml-core/           # Core types and utilities
├── thrml-samplers/       # Sampling infrastructure
├── thrml-models/         # EBM abstractions
├── thrml-observers/      # Observation utilities
├── thrml-kernels/        # GPU autodiff kernels
├── thrml-sphere/         # Hyperspherical navigation (main feature)
└── thrml-examples/       # Example programs
```

---

## Part 1: PGM Sampling System

### Block Gibbs Sampling

The core sampling algorithm processes graphical models via block-wise updates:

```
Graphical Model → Factors → InteractionGroups → BlockSamplingProgram → Samples
```

### Key Abstractions

#### Blocks

A `Block` is a collection of nodes of the same type with implicit ordering:

```rust
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};

let nodes: Vec<Node> = (0..10)
    .map(|_| Node::new(NodeType::Spin))
    .collect();
let block = Block::new(nodes);
```

#### Factors

Factors define interactions between variables via `InteractionGroup`:

```rust
pub trait AbstractFactor {
    fn to_interaction_groups(&self, device: &WgpuDevice) -> Vec<InteractionGroup>;
}
```

#### Programs

Programs orchestrate sampling:

```rust
pub struct BlockSamplingProgram {
    pub gibbs_spec: BlockGibbsSpec,
    pub samplers: Vec<Box<dyn DynConditionalSampler>>,
    // Pre-computed index tensors...
}
```

### Sampling Flow

1. **Factor → InteractionGroup**: Factors produce weight tensors and node mappings
2. **BlockSamplingProgram**: Pre-computes slices for efficient indexing
3. **Sampler**: Computes conditionals and samples new states

### Type Hierarchies

#### Factors
```
AbstractFactor
├── WeightedFactor
└── EBMFactor
    └── DiscreteEBMFactor
        ├── SpinEBMFactor
        └── CategoricalEBMFactor
```

#### Samplers
```
AbstractConditionalSampler
├── BernoulliConditional → SpinGibbsConditional
└── SoftmaxConditional → CategoricalGibbsConditional
```

---

## Part 2: Hyperspherical Navigation System

The `thrml-sphere` crate implements a multi-cone EBM navigation architecture for semantic retrieval over hyperspherical embeddings.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Embedding                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ROOTS Index Layer                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ Partition 0 │ │ Partition 1 │ │ Partition N │ ...           │
│  │  centroid   │ │  centroid   │ │  centroid   │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│                    Activation Peaks                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Cone Spawning                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │  Cone 0  │  │  Cone 1  │  │  Cone N  │  ...                 │
│  │ (35% $)  │  │ (30% $)  │  │ (35% $)  │  Budget allocation   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                      │
└───────┼─────────────┼─────────────┼────────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│              Parallel Cone Navigation (EBM)                     │
│                                                                 │
│   navigate()    navigate()    navigate()                        │
│       │             │             │                             │
│       └─────────────┼─────────────┘                             │
│                     ▼                                           │
│              merge_cone_results()                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MultiConeResult                              │
│   target_indices, target_energies, per_cone_results             │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### SphereEBM

Optimizes embedding positions on a hypersphere using Langevin dynamics:

```rust
pub struct SphereEBM {
    pub embeddings: Tensor<WgpuBackend, 2>,  // [N, D]
    pub prominence: Tensor<WgpuBackend, 1>,  // [N]
    pub similarity: Tensor<WgpuBackend, 2>,  // [N, N]
    pub config: SphereConfig,
}
```

The Hamiltonian includes:
- Similarity-based attraction
- Prominence-based radial energy
- Entropy weighting (optional)

#### NavigatorEBM

Computes navigation energy combining multiple terms:

```
E_nav(q, T, P) = λ_sem · E_semantic(q, T)    // Embedding similarity
              + λ_rad · E_radial(q, T)       // Radial distance
              + λ_graph · E_graph(P)         // Graph traversal
              + λ_ent · E_entropy(T)         // Entropy penalty
              + λ_path · E_path(P)           // Path length
```

#### ROOTS Index

Compressed inner-shell index using Ising max-cut partitioning:

```rust
pub struct RootsIndex {
    pub partitions: Vec<Vec<usize>>,      // Point assignments
    pub centroids: Tensor<WgpuBackend, 2>, // [K, D]
    pub config: RootsConfig,
}
```

Features:
- 3000:1 compression ratio
- Substring coupling for code/text (J_ij = α·cos_sim + β·substring_sim)
- Activation peak detection for cone spawning

#### MultiConeNavigator

ROOTS-guided multi-cone navigation with dynamic budget allocation:

```rust
pub struct MultiConeNavigator {
    pub navigator: NavigatorEBM,
    pub roots: RootsIndex,
    pub budget_config: BudgetConfig,
    pub active_cones: Vec<ConeState>,
}
```

### Navigation Flow

1. **Query arrives** → Compute ROOTS activations
2. **Peak detection** → Identify high-activation regions
3. **Cone spawning** → Allocate budget proportional to peak strength
4. **Parallel navigation** → Run independent EBM navigation per cone
5. **Result merging** → Deduplicate and rank by energy

---

## Part 3: Training System

### Contrastive Divergence Training

The training infrastructure implements state-of-the-art CD techniques:

```
┌────────────────────────────────────────────────────────────────┐
│                   Training Configuration                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ AdvancedTrainingConfig                                   │   │
│  │  ├── HardNegativeMiner (positive-aware filtering)       │   │
│  │  ├── PersistentParticleBuffer (PCD)                     │   │
│  │  ├── NegativeCurriculumSchedule (progressive)           │   │
│  │  └── LR Schedule (warmup + cosine annealing)            │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                    Training Loop                                │
│                                                                 │
│  for epoch in epochs:                                          │
│    lr = get_learning_rate(epoch)  # Warmup + cosine            │
│    difficulty = curriculum.get_difficulty(epoch)               │
│                                                                 │
│    for batch in dataset:                                        │
│      if hard_negative_mining:                                   │
│        negatives = miner.mine(query, positive, ...)            │
│      elif curriculum:                                           │
│        negatives = curriculum.select_negatives(...)            │
│                                                                 │
│      if pcd:                                                    │
│        buffer.update_langevin(energy_gradient)                 │
│                                                                 │
│      loss, gradients = compute_contrastive_loss(batch)          │
│      weights = sgd_step(weights, gradients, lr, momentum)       │
│                                                                 │
│    if epoch % val_every == 0:                                   │
│      metrics = evaluate(validation_set)                         │
│      if metrics.mrr > best_mrr:                                │
│        best_weights = weights                                   │
│      elif patience_exceeded:                                    │
│        break  # Early stopping                                  │
└────────────────────────────────────────────────────────────────┘
```

### Loss Function

InfoNCE-style contrastive loss:

```
L = -log(exp(-E_pos/τ) / (exp(-E_pos/τ) + Σ_neg exp(-E_neg/τ)))
```

With numerical stability via log-sum-exp trick.

### Hard Negative Mining

Positive-aware filtering to avoid false negatives:

```
hardness(n) = sim(query, n) × (1 - sim(positive, n))
```

Candidates with `sim(positive, n) > threshold` are filtered as potential false negatives.

### Persistent Contrastive Divergence

Fantasy particles maintained across batches:

```rust
buffer.update_langevin(|particles| {
    // dx = -∇E(x)dt + √(2Tdt)ξ
    energy_gradient(particles)
}, &device);
```

Periodically reinitialize a fraction to prevent mode collapse.

### Curriculum Learning

Three-phase progression:

| Phase | Epochs | Negative Similarity Range |
|-------|--------|---------------------------|
| Easy | 0-10 | 0.0 - 0.3 |
| Medium | 10-30 | 0.3 - 0.7 |
| Hard | 30+ | 0.7 - 1.0 |

---

## Part 4: GPU Backend

### WGPU/Burn Integration

All tensor operations use the Burn framework with WGPU backend:

```rust
use thrml_core::backend::{init_gpu_device, ensure_backend, WgpuBackend};

ensure_backend();  // Verify Metal on macOS
let device = init_gpu_device();
```

Supported platforms:
- **Metal** on Apple Silicon (M1/M2/M3/M4)
- **Vulkan** on Linux/Windows
- **DX12** on Windows

### Custom GPU Kernels

The `thrml-kernels` crate provides optimized operations:

- **Gumbel-Softmax**: Differentiable discrete sampling
- **Batch Gather**: Efficient batched indexing
- **Sigmoid-Bernoulli**: Differentiable binary sampling

### Performance Considerations

1. **Batch Operations**: All tensor ops are batched for GPU efficiency
2. **Pre-computed Indices**: Programs pre-compute index tensors at construction
3. **Minimal Allocations**: State tensors reused across iterations
4. **Parallel Cones**: Independent cone navigation is parallelizable

---

## Design Principles

### Separation of Concerns

- **Core**: Types only, no GPU operations
- **Samplers**: Sampling logic, independent of model specifics
- **Models**: Model definitions, produce InteractionGroups
- **Sphere**: Navigation-specific components
- **Kernels**: Low-level GPU operations

### Type Safety

Rust's type system ensures:
- Correct tensor dimensions via const generics
- Valid block/node type combinations
- Safe state access patterns

### Deterministic RNG

JAX-style RNG key management:

```rust
let key = RngKey::new(42);
let (key1, key2) = key.split();
let keys = key.split(n);
```

Ensures reproducibility across runs.

---

## File Layout

```
thrml-rs/
├── crates/
│   ├── thrml-core/
│   │   ├── src/
│   │   │   ├── backend.rs      # GPU initialization
│   │   │   ├── block.rs        # Block type
│   │   │   ├── node.rs         # Node, NodeType
│   │   │   ├── blockspec.rs    # Index management
│   │   │   ├── interaction.rs  # InteractionGroup
│   │   │   ├── spherical.rs    # SphericalCoords
│   │   │   └── similarity.rs   # Similarity utilities
│   │   └── tests/
│   │
│   ├── thrml-samplers/
│   │   ├── src/
│   │   │   ├── program.rs      # BlockSamplingProgram
│   │   │   ├── sampling.rs     # sample_blocks, sample_states
│   │   │   ├── rng.rs          # RngKey
│   │   │   ├── langevin.rs     # Langevin dynamics
│   │   │   └── *.rs            # Conditional samplers
│   │   └── tests/
│   │
│   ├── thrml-sphere/           # ★ Main feature
│   │   ├── src/
│   │   │   ├── sphere_ebm.rs   # Core sphere model
│   │   │   ├── navigator.rs    # NavigatorEBM, MultiConeNavigator
│   │   │   ├── roots.rs        # ROOTS index
│   │   │   ├── training.rs     # Training infrastructure
│   │   │   ├── contrastive.rs  # Advanced CD techniques
│   │   │   ├── evaluation.rs   # Navigation metrics
│   │   │   ├── config.rs       # Configuration
│   │   │   ├── loader.rs       # SafeTensors loading
│   │   │   └── *.rs            # Supporting modules
│   │   └── tests/
│   │
│   └── ...
│
├── docs/                       # This documentation
└── Cargo.toml                  # Workspace manifest
```
