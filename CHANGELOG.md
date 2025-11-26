# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-25

### Initial Release

Complete Rust implementation of GPU-accelerated probabilistic graphical models.

### Added

#### Core (`thrml-core`)
- `Node` and `NodeType` (Spin, Categorical, Continuous)
- `Block` structure with node management
- `BlockSpec` with all mappings and validation
- `InteractionGroup` and `InteractionData` enum for flexible interactions
- `StateLeaf` trait for state management
- GPU backend initialization (WGPU/Metal)
- State conversion utilities (`block_state_to_global`, `from_global_state`)

#### Samplers (`thrml-samplers`)
- `BlockSamplingProgram` for block Gibbs sampling
- `BernoulliConditional` sampler for spin variables
- `SoftmaxConditional` sampler using Gumbel-max trick
- `SpinGibbsConditional` for Ising-type models
- `CategoricalGibbsConditional` for categorical variables
- `GaussianSampler` for continuous variables
- `SamplingSchedule` for sampling configuration
- Deterministic RNG key system (`RngKey`) with ChaCha8
- Stateful sampler support via `AbstractConditionalSampler::SamplerState`

#### Models (`thrml-models`)
- `AbstractFactor` and `AbstractEBM` traits
- `IsingEBM` with energy computation and factor generation
- `IsingSamplingProgram` for Ising model sampling
- `DiscreteEBMFactor` with multi-dimensional `batch_gather`
- `SpinEBMFactor` and `CategoricalEBMFactor`
- `LinearFactor`, `QuadraticFactor`, `CouplingFactor` for continuous models
- Square factor optimization with automatic group merging
- Training utilities: `estimate_moments`, `estimate_kl_grad`, `hinton_init`
- `IsingTrainingSpec` for training configuration

#### Observers (`thrml-observers`)
- `StateObserver` for collecting state samples
- `MomentAccumulatorObserver` for moment statistics

#### Examples (`thrml-examples`)
- `ising_chain`: Simple Ising chain model
- `spin_models`: Ising lattice with performance visualization
- `categorical_sampling`: Grid sampling with heatmap output
- `full_api_walkthrough`: Comprehensive API tutorial
- `gaussian_pgm`: Continuous variable sampling
- `gaussian_bernoulli_ebm`: Mixed-type model sampling
- `train_mnist`: Full MNIST training with contrastive divergence

#### Fused Kernels (`thrml-kernels`)
- `gumbel_argmax_fused`: Fused Gumbel-max categorical sampling
- `sigmoid_bernoulli_fused`: Fused sigmoid-Bernoulli spin sampling
- `batch_gather_fused`: Fused multi-index weight gathering
- Feature-gated integration with `fused-kernels` feature flag
- CubeCL kernel definitions for GPU fusion

### Technical Highlights

- GPU acceleration via WGPU/Metal backend (Burn 0.19)
- ChaCha8-based deterministic RNG for reproducibility
- Linear indexing for multi-dimensional tensor gather operations
- Native Apple Silicon (M1/M2/M3/M4) support via Metal
- Comprehensive test suite with GPU smoke tests

## [Unreleased]

### Added

- Full CLI configuration for `train_mnist` example with all 14+ hyperparameters
- `--epochs`, `--learning-rate`, `--batch-size`, `--warmup-neg/pos`, `--samples-neg/pos`, etc.
- Environment variable support for training params (`THRML_EPOCHS`, `THRML_LR`, `THRML_BATCH_SIZE`)
- Configurable path system (`--base-dir`, `--data-dir`, `--output-dir`, `--cache-dir`)
- Config file support (`~/.config/thrml/config.toml`)
- Extended `batch_gather` kernels for 3-6 indices with compile-time unrolling
- Dynamic `batch_gather` fallback for 7+ indices
- Gumbel-Softmax autodiff integration for differentiable categorical sampling
- Straight-Through Estimator (STE) for sigmoid-Bernoulli backward pass
- `BatchedEBM` trait for vectorized energy computation

### Changed

- Made `PathConfig::from_path_args` public for custom CLI integration
- Improved batch progress display consistency in training loop

### Planned

- Performance benchmarks
- Python bindings via PyO3
- Additional optimization passes
