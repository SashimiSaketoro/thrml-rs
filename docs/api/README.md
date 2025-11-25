# API Reference

Complete API documentation for THRML-RS.

## Crates

| Crate | Description | Documentation |
|-------|-------------|---------------|
| **thrml-core** | Core types and utilities | [core.md](core.md) |
| **thrml-samplers** | Sampling infrastructure | [samplers.md](samplers.md) |
| **thrml-models** | EBM and factor implementations | [models.md](models.md) |
| **thrml-observers** | Observation and statistics | [observers.md](observers.md) |

## Quick Reference

### Core Types

| Type | Description |
|------|-------------|
| `Node` | A node in the graphical model |
| `NodeType` | Spin or Categorical |
| `Block` | Collection of same-type nodes |
| `BlockSpec` | Global state index management |
| `InteractionGroup` | Defines sampling dependencies |
| `TensorSpec` | Shape and dtype specification |

### Sampling

| Function/Type | Description |
|---------------|-------------|
| `BlockGibbsSpec` | Sampling specification |
| `BlockSamplingProgram` | Core sampling program |
| `SamplingSchedule` | Warmup, samples, steps |
| `sample_states` | Main sampling function |
| `sample_with_observation` | Sampling with observer |
| `RngKey` | Deterministic RNG |

### Conditionals

| Type | Description |
|------|-------------|
| `AbstractConditionalSampler` | Base trait |
| `BernoulliConditional` | Spin sampling |
| `SpinGibbsConditional` | Spin Gibbs updates |
| `SoftmaxConditional` | Categorical sampling |
| `CategoricalGibbsConditional` | Categorical Gibbs updates |

### Models

| Type | Description |
|------|-------------|
| `AbstractEBM` | Energy-based model trait |
| `AbstractFactor` | Factor trait |
| `DiscreteEBMFactor` | Discrete variable factor |
| `IsingEBM` | Ising/Boltzmann model |
| `IsingSamplingProgram` | Ising-specific program |

### Observers

| Type | Description |
|------|-------------|
| `AbstractObserver` | Observer trait |
| `StateObserver` | Collect raw states |
| `MomentAccumulatorObserver` | Compute moments |

## Cargo Features

| Feature | Description |
|---------|-------------|
| `gpu` | Enable GPU acceleration via WGPU |
| `default` | CPU-only (not recommended) |

Enable GPU features in `Cargo.toml`:

```toml
[dependencies]
thrml-core = { version = "0.1", features = ["gpu"] }
thrml-samplers = { version = "0.1", features = ["gpu"] }
thrml-models = { version = "0.1", features = ["gpu"] }
thrml-observers = { version = "0.1", features = ["gpu"] }
```

## Rustdoc

Generate full API documentation locally:

```bash
cd thrml-rs
cargo doc --workspace --features gpu --open
```

This opens interactive HTML documentation with all types, traits, and functions.

