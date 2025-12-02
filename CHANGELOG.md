# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-02

Code quality improvements and kernel restoration.

### Added

#### Fused Kernels (`thrml-kernels`)
- `cosine_similarity_fused` - Single-kernel cosine similarity (restored)
- `cosine_similarity_fused_batched` - Batched query-to-vectors similarity
- `cosine_similarity_prenormalized` - Optimized for pre-normalized vectors
- `l2_normalize_fused` - Single-kernel L2 row normalization (restored)
- `launch_l2_normalize_with_norms` - Returns norms alongside normalized output
- CubeCL kernels with `#[comptime]` parameters for compile-time optimization

#### API Improvements
- `FromStr` trait implementation for `ScaleProfile`
- `EdgeBatch` type alias for cleaner graph training APIs

### Changed

#### Code Quality
- Enabled `clippy::pedantic` with documented allows
- Applied `clippy::nursery` fixes (~370 warnings resolved):
  - `missing_const_for_fn` - functions marked `const` where applicable
  - `use_self` - replaced type names with `Self`
  - `option_if_let_else` - converted simple cases to `map_or`/`map_or_else`
  - `or_fun_call` - lazy evaluation with `or_else`/`unwrap_or_else`
  - `significant_drop_tightening` - earlier MutexGuard drops
- Replaced manual `as` casts with `mul_add()` for numeric accuracy
- Iterator-based loops replace index loops where appropriate
- `copy_from_slice()` replaces manual memcpy loops
- Fixed all rustdoc `invalid_html_tags` warnings (bracket escaping in doc comments)

---

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

## [0.2.0] - 2025-12-01

Major feature additions including hardware-aware routing and precision control.

### Added

#### `thrml-core::metrics` (new module)

- `recall_at_k()`, `mrr()`, `ndcg()`, `ndcg_multi()`
- `find_rank()`: 1-indexed rank lookup
- `evaluate_retrieval()`: Batch evaluation
- `RetrievalMetrics`: Aggregated struct

#### `thrml-core::text` (new module)

- `RollingHash`: O(1) sliding window hash
- `ngram_hashes()`, `ngram_hashes_with_length()`
- `jaccard_similarity()`, `contains_subsequence()`
- `TextSimilarityConfig`, `text_similarity()`

#### `thrml-samplers::maxcut` (new module)

- `maxcut_gibbs()`: Gibbs sampling for max-cut
- `maxcut_multistart()`: Multi-restart best partition
- `maxcut_greedy()`: Greedy local search
- `cut_value()`, `ising_energy()`
- `partition_to_binary()`, `binary_to_partition()`

#### Hardware-Aware Routing (`thrml-core`)
- `RuntimePolicy`: Auto-detect hardware and configure precision routing
  - `RuntimePolicy::detect()`: Automatic GPU detection via WGPU adapter info
  - `RuntimePolicy::apple_silicon()`, `nvidia_consumer()`, `nvidia_hopper()`, `nvidia_blackwell()`: Tier-specific constructors
  - `is_hpc_tier()`: Check if hardware supports native GPU f64
  - `precision_dtype()`: Get appropriate dtype based on profile
- `HardwareTier` enum: `AppleSilicon`, `NvidiaConsumer`, `AmdRdna`, `NvidiaHopper`, `NvidiaBlackwell`, `CpuOnly`, `Unknown`
- `PrecisionProfile` enum: `CpuFp64Strict`, `GpuMixed`, `GpuHpcFp64`
- `ComputeBackend::from_policy()`: Create backend from runtime policy
- `ComputeBackend::GpuHpcF64`: New variant for HPC GPUs with CUDA f64 support
- `ComputeBackend::uses_gpu_f64()`: Check if backend uses GPU-native f64
- `OpType` expanded: Added `CategoricalSampling`, `GradientCompute`, `LossReduction`, `BatchEnergyForward`
- `GpuInfo` struct and `detect_gpu_info()`: Hardware detection via WGPU
- Vendor ID constants for Apple, NVIDIA, AMD, Intel detection
- CUDA f64 feature gate for HPC GPU paths

#### Hybrid Compute Backend (`thrml-core`)
- `ComputeBackend`: CPU/GPU/Hybrid/Adaptive backend selection
- `OpType`: Operation classification for precision routing
- `PrecisionMode`: GpuFast (f32), CpuPrecise (f64), Adaptive
- `HybridConfig`: Combined backend + precision configuration
- `ComputeBackend::apple_silicon()`: Auto-detect Apple Silicon unified memory
- `cpu_ising` module: Pure Rust f64 Ising implementations for precision-sensitive ops
- `test_both_backends` utility: Test CPU and GPU paths with appropriate tolerances

#### Sampler Precision Routing (`thrml-samplers`)
- `SpinGibbsConditional::sample_routed()`: Routes Ising sampling based on ComputeBackend
  - CUDA f64 path for HPC GPUs (H100, B200)
  - CPU f64 path for Apple Silicon and consumer GPUs
  - GPU f32 fallback for bulk operations
- `CategoricalGibbsConditional::sample_routed()`: Precision routing for categorical sampling
- `GaussianSampler::sample_routed()`: Routes precision/mean accumulation to CPU f64
- `langevin_step_2d_routed()`: Routes Langevin dynamics based on precision requirements
- CPU f64 implementations extract tensors, compute in f64, convert back to f32
- Battle tests for sampler precision routing consistency

#### Model Precision Routing (`thrml-models`)
- `DiscreteEBMFactor::factor_energy_routed()`: Routes energy computation based on ComputeBackend
- Spin product and categorical indexing computed in CPU f64 for precision
- Delegates to specialized factor types: `SpinEBMFactor`, `CategoricalEBMFactor`, etc.
- Battle tests for model precision routing and backend selection

#### Graph-based Models (`thrml-models`)
- `GraphSidecar`: Graph structure with edges and node attributes
- `SpringEBM`: Spring-like energy between connected nodes
- `NodeBiasEBM`: Weighted node bias energy
- `make_lattice_graph`: 2D lattice with beyond-nearest-neighbor connections
- `make_nearest_neighbor_lattice`: Simple 4-connected 2D grid
- Torus (periodic) boundary support
- Two-color blocking for efficient block Gibbs sampling

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
