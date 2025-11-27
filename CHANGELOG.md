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

#### Hyperspherical Navigation (`thrml-sphere`) - NEW CRATE
- `SphereEBM`: Langevin dynamics-based sphere optimization ("water-filling")
- `NavigatorEBM`: Multi-cone EBM navigation through hyperspherical embeddings
- `MultiConeNavigator`: ROOTS-guided navigation with dynamic budget allocation
- `RootsIndex`: Compressed inner-shell index layer (3000:1 compression)
- Ising max-cut partitioning with substring coupling for byte-level structure
- `SubstringConfig`: Byte-level substring similarity for code/text clustering
- `BudgetConfig`: Memory budget allocation across navigation cones
- Scale profiles: `Dev`, `Small`, `Large` for different use cases

#### Training Infrastructure (`thrml-sphere`)
- `TrainableNavigatorEBM`: End-to-end trainable navigator
- `TrainingDataset`: Train/validation split with similarity-based pair generation
- `ExtendedTrainingConfig`: Validation, early stopping, checkpointing
- `TuningGrid` and `TuningSession`: Hyperparameter search (grid/random)
- `NavigationMetrics`: Recall@k, MRR, nDCG evaluation

#### Advanced Contrastive Divergence (`thrml-sphere`)
- `HardNegativeMiner`: Similarity-based hard negative mining with false-negative filtering
- `PersistentParticleBuffer`: Persistent Contrastive Divergence (PCD) with fantasy particles
- `NegativeCurriculumSchedule`: Progressive difficulty scheduling (Easy→Medium→Hard)
- `SGLDNegativeSampler`: SGLD-based negative phase sampling
- `AdvancedTrainingConfig`: Unified config for all CD techniques
- Learning rate warmup and cosine annealing schedules

#### Hybrid Compute Backend (`thrml-core`)
- `ComputeBackend`: CPU/GPU/Hybrid/Adaptive backend selection
- `OpType`: Operation classification for precision routing
- `PrecisionMode`: GpuFast (f32), CpuPrecise (f64), Adaptive
- `HybridConfig`: Combined backend + precision configuration
- `ComputeBackend::apple_silicon()`: Auto-detect Apple Silicon unified memory
- `cpu_ising` module: Pure Rust f64 Ising implementations for precision-sensitive ops
- `test_both_backends` utility: Test CPU and GPU paths with appropriate tolerances

#### Graph-based Models (`thrml-models`)
- `GraphSidecar`: Graph structure with edges and node attributes
- `SpringEBM`: Spring-like energy between connected nodes
- `NodeBiasEBM`: Weighted node bias energy
- `make_lattice_graph`: 2D lattice with beyond-nearest-neighbor connections
- `make_nearest_neighbor_lattice`: Simple 4-connected 2D grid
- Torus (periodic) boundary support
- Two-color blocking for efficient block Gibbs sampling

#### Examples (`thrml-examples`)
- `navigator_demo`: Multi-cone navigation demonstration
- `train_navigator`: Train navigator with contrastive divergence
- `tune_navigator`: Hyperparameter tuning session
- `blt_sphere`: Sphere optimization from BLT embeddings
- `build_roots`: ROOTS index construction

#### CLI & Configuration
- Full CLI configuration for `train_mnist` example with all 14+ hyperparameters
- `--epochs`, `--learning-rate`, `--batch-size`, `--warmup-neg/pos`, `--samples-neg/pos`, etc.
- Environment variable support for training params (`THRML_EPOCHS`, `THRML_LR`, `THRML_BATCH_SIZE`)
- Configurable path system (`--base-dir`, `--data-dir`, `--output-dir`, `--cache-dir`)
- Config file support (`~/.config/thrml/config.toml`)

#### Kernels (`thrml-kernels`)
- Extended `batch_gather` kernels for 3-6 indices with compile-time unrolling
- Dynamic `batch_gather` fallback for 7+ indices
- Gumbel-Softmax autodiff integration for differentiable categorical sampling
- Straight-Through Estimator (STE) for sigmoid-Bernoulli backward pass
- `BatchedEBM` trait for vectorized energy computation

### Changed

- Made `PathConfig::from_path_args` public for custom CLI integration
- Improved batch progress display consistency in training loop
- Feature-gated GPU-dependent modules in `thrml-core` (`distance`, `similarity`, `spherical`, `interaction`, `state_tree`)
- Loosened GPU floating-point tolerance in tests for Metal precision characteristics

### Fixed

- Clippy lint for `is_multiple_of()` usage
- Unused `mut` warning in `available_backends()` when no features enabled
- Feature flag configuration allowing true CPU-only builds

### Planned

- Performance benchmarks
- Python bindings via PyO3
- Additional optimization passes
