# API Reference

Complete API documentation for THRML-RS.

## Crates

| Crate | Description | Documentation |
|-------|-------------|---------------|
| **thrml-core** | Core types: nodes, blocks, spherical coords, similarity | [core.md](core.md) |
| **thrml-samplers** | Sampling infrastructure, Langevin dynamics | [samplers.md](samplers.md) |
| **thrml-models** | EBM and factor implementations | [models.md](models.md) |
| **thrml-observers** | Observation and statistics | [observers.md](observers.md) |
| **thrml-kernels** | GPU autodiff kernels (Gumbel-Softmax, batch gather) | [kernels.md](kernels.md) |
| **thrml-sphere** | **Hyperspherical navigation, ROOTS, advanced training** | [sphere.md](sphere.md) |
| **thrml-examples** | Runnable example programs | [examples.md](examples.md) |

## Quick Reference

### Core Types (`thrml-core`)

| Type | Description |
|------|-------------|
| `Node` | A node in the graphical model |
| `NodeType` | Spin or Categorical |
| `Block` | Collection of same-type nodes |
| `BlockSpec` | Global state index management |
| `InteractionGroup` | Defines sampling dependencies |
| `SphericalCoords` | Spherical coordinate representation (r, θ, φ) |

### Sampling (`thrml-samplers`)

| Function/Type | Description |
|---------------|-------------|
| `BlockGibbsSpec` | Sampling specification |
| `BlockSamplingProgram` | Core sampling program |
| `SamplingSchedule` | Warmup, samples, steps |
| `sample_states` | Main sampling function |
| `RngKey` | Deterministic RNG |
| `LangevinConfig` | Langevin dynamics configuration |

### Conditionals (`thrml-samplers`)

| Type | Description |
|------|-------------|
| `BernoulliConditional` | Spin sampling |
| `SpinGibbsConditional` | Spin Gibbs updates |
| `SoftmaxConditional` | Categorical sampling |
| `CategoricalGibbsConditional` | Categorical Gibbs updates |

### Models (`thrml-models`)

| Type | Description |
|------|-------------|
| `AbstractEBM` | Energy-based model trait |
| `AbstractFactor` | Factor trait |
| `DiscreteEBMFactor` | Discrete variable factor |
| `IsingEBM` | Ising/Boltzmann model |

### Observers (`thrml-observers`)

| Type | Description |
|------|-------------|
| `AbstractObserver` | Observer trait |
| `StateObserver` | Collect raw states |
| `MomentAccumulatorObserver` | Compute moments |

### Kernels (`thrml-kernels`)

| Function | Description |
|----------|-------------|
| `gumbel_softmax` | Differentiable categorical sampling |
| `batch_gather` | Batched gather with gradients |
| `sigmoid_bernoulli_sample` | Differentiable Bernoulli sampling |

### Sphere & Navigation (`thrml-sphere`)

| Type | Description |
|------|-------------|
| `SphereEBM` | Sphere optimization model |
| `SphereConfig` | Optimization configuration |
| `NavigatorEBM` | Single-cone navigation |
| `MultiConeNavigator` | ROOTS-guided multi-cone navigation |
| `NavigationWeights` | Learnable energy weights |
| `RootsIndex` | Compressed ROOTS index |
| `RootsConfig` | ROOTS configuration |
| `BudgetConfig` | Cone budget allocation |

### Training (`thrml-sphere`)

| Type | Description |
|------|-------------|
| `TrainableNavigatorEBM` | Trainable navigator wrapper |
| `NavigatorTrainingConfig` | Basic training config |
| `AdvancedTrainingConfig` | Advanced CD techniques |
| `TrainingDataset` | Train/validation split |
| `TrainingReport` | Training results |
| `TuningGrid` | Hyperparameter grid |
| `TuningSession` | Grid/random search |

### Contrastive Learning (`thrml-sphere`)

| Type | Description |
|------|-------------|
| `HardNegativeMiner` | Similarity-based hard negative mining |
| `PersistentParticleBuffer` | PCD fantasy particles |
| `NegativeCurriculumSchedule` | Progressive difficulty |
| `SGLDNegativeSampler` | SGLD for negative phase |

### Evaluation (`thrml-sphere`)

| Function/Type | Description |
|---------------|-------------|
| `evaluate_navigator` | Compute navigation metrics |
| `NavigationMetrics` | Recall@k, MRR, nDCG |
| `recall_at_k` | Hit rate at k |
| `reciprocal_rank` | RR for a single query |
| `ndcg_at_k` | Normalized DCG |

### Data Loading (`thrml-sphere`)

| Function | Description |
|----------|-------------|
| `load_blt_safetensors` | Load BLT v3 format (embeddings + bytes) |
| `load_from_safetensors` | Load embeddings only |

## Rustdoc

Generate full API documentation locally:

```bash
cd thrml-rs
cargo doc --workspace --open
```

This opens interactive HTML documentation with all types, traits, and functions.

## Feature Flags

Most crates use WGPU by default. Key features:

```toml
[dependencies]
thrml-sphere = "0.1"
# GPU acceleration is enabled by default via burn-wgpu
```
