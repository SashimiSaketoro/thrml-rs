// Clippy allows for experimental/research crate
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::get_first)]
#![allow(clippy::clone_on_copy)]
#![allow(irrefutable_let_patterns)]

//! # thrml-sphere
//!
//! Hyperspherical embedding optimization, ROOTS indexing, and multi-cone navigation
//! for the THRML framework.
//!
//! This crate provides Langevin dynamics-based sphere optimization for organizing
//! BLT (Byte Latent Transformer) embeddings on a hyperspherical manifold using
//! the "water-filling" algorithm, along with the ROOTS compressed index layer
//! and multi-cone navigation system.
//!
//! ## Features
//!
//! - **Sphere Optimization**: Place embeddings on a hypersphere using Langevin dynamics
//! - **ROOTS Index**: Compressed inner-shell index for coarse-grained navigation (3000:1 compression)
//! - **Multi-Cone Navigation**: ROOTS-guided navigation with dynamic budget allocation
//! - **Ising Max-Cut**: Similarity-aware partitioning using energy-based models
//! - **Substring Coupling**: Structural byte-level relationships enhance partitioning
//! - **Hybrid Compute**: Optimized for Apple Silicon unified memory systems
//!
//! ## Quick Start
//!
//! ### Multi-Cone Navigation (Recommended)
//!
//! The most complete workflow using ROOTS peaks to guide navigation:
//!
//! ```rust,ignore
//! use thrml_core::backend::{ensure_backend, init_gpu_device};
//! use thrml_samplers::RngKey;
//! use thrml_sphere::{
//!     load_blt_safetensors, MultiConeNavigator, BudgetConfig,
//!     RootsConfig, SphereConfig,
//! };
//!
//! // Initialize GPU
//! ensure_backend();
//! let device = init_gpu_device();
//!
//! // Load BLT v3 data (embeddings + raw bytes)
//! let (sphere_ebm, bytes) = load_blt_safetensors(&path, SphereConfig::default(), &device)?;
//!
//! // Configure ROOTS and budget
//! let roots_config = RootsConfig::default()
//!     .with_partitions(64)
//!     .with_default_substring_coupling();
//!
//! let budget_config = BudgetConfig::new(512 * 1024 * 1024)  // 512MB
//!     .with_max_cones(8)
//!     .with_min_cone_budget(16 * 1024 * 1024);
//!
//! // Create multi-cone navigator
//! let mut navigator = MultiConeNavigator::from_sphere_ebm_with_bytes(
//!     &sphere_ebm,
//!     &bytes,
//!     roots_config,
//!     budget_config,
//!     RngKey::new(42),
//!     &device,
//! );
//!
//! // Navigate - cones spawn automatically from ROOTS peaks
//! let result = navigator.navigate_multi_cone(query, 50.0, 10, RngKey::new(123), &device);
//!
//! println!("Found {} targets from {} cones", result.n_targets(), result.n_cones());
//! println!("Best: index {} with energy {:.4}", result.best_target().unwrap(), result.best_energy().unwrap());
//! ```
//!
//! ### Sphere Optimization
//!
//! ```rust,ignore
//! use thrml_core::backend::{ensure_backend, init_gpu_device};
//! use thrml_samplers::RngKey;
//! use thrml_sphere::{load_from_safetensors, ScaleProfile, SphereConfig};
//!
//! // Initialize GPU
//! ensure_backend();
//! let device = init_gpu_device();
//!
//! // Load and configure
//! let config = SphereConfig::from(ScaleProfile::Dev)
//!     .with_entropy_weighted(true);
//! let ebm = load_from_safetensors(&path, config, &device)?;
//!
//! // Run optimization
//! let key = RngKey::new(42);
//! let coords = ebm.optimize(key, &device);
//!
//! // Access results
//! let cartesian = coords.to_cartesian(); // \[N, 3\]
//! ```
//!
//! ### ROOTS Index with Substring Coupling
//!
//! For byte-level structural relationships (e.g., code where "calculate_total"
//! should cluster with "function_calculate_total"):
//!
//! ```rust,ignore
//! use thrml_sphere::{
//!     load_blt_safetensors, RootsConfig, RootsIndex,
//!     SubstringConfig, SphereConfig,
//! };
//! use thrml_samplers::RngKey;
//!
//! // Load BLT v3 output (embeddings + raw bytes in one file)
//! let (sphere_ebm, patch_bytes) = load_blt_safetensors(
//!     Path::new("output.safetensors"),
//!     SphereConfig::default(),
//!     &device,
//! )?;
//!
//! // Configure ROOTS with substring-enhanced coupling
//! // J_ij = α × cosine_sim(emb) + β × substring_sim(bytes)
//! let roots_config = RootsConfig::default()
//!     .with_partitions(256)
//!     .with_default_substring_coupling();  // α=0.7, β=0.3
//!
//! // Build index
//! let roots = RootsIndex::from_sphere_ebm_with_bytes(
//!     &sphere_ebm,
//!     &patch_bytes,
//!     roots_config,
//!     RngKey::new(42),
//!     &device,
//! );
//!
//! // Route queries to partitions
//! let partition_id = roots.route(&query_embedding);
//!
//! // Detect activation peaks for cone spawning
//! let activations = roots.activate(&query, &device);
//! let peaks = roots.detect_peaks(&activations);
//! ```
//!
//! ### Training with Evaluation
//!
//! Train a navigator with validation metrics and early stopping:
//!
//! ```rust,ignore
//! use thrml_sphere::{
//!     generate_pairs_from_similarity, TrainingDataset, evaluate_navigator,
//!     ExtendedTrainingConfig, TrainableNavigatorEBM, NavigatorEBM,
//! };
//!
//! // Generate training pairs from embedding similarity
//! let examples = generate_pairs_from_similarity(&sphere_ebm, 1, 8, &device);
//!
//! // Create dataset with 10% validation split
//! let dataset = TrainingDataset::from_examples(examples, 0.1, 42);
//!
//! // Configure training with early stopping
//! let config = ExtendedTrainingConfig::default()
//!     .with_val_every(5)
//!     .with_early_stopping(3);
//!
//! // Create trainable navigator
//! let mut trainable = TrainableNavigatorEBM::from_navigator(
//!     NavigatorEBM::from_sphere_ebm(sphere_ebm.clone()),
//!     config.base.clone(),
//! );
//!
//! // Train with validation
//! let report = trainable.train_with_validation(
//!     &dataset, 50, 16, &config, RngKey::new(42), &device,
//! );
//!
//! println!("Best MRR: {:.4} at epoch {}", report.best_val_mrr, report.best_epoch);
//!
//! // Final evaluation
//! let metrics = evaluate_navigator(&trainable.navigator, &dataset.validation, 10, key, &device);
//! println!("Recall@10: {:.4}, MRR: {:.4}", metrics.recall_10, metrics.mrr);
//! ```
//!
//! ### Hybrid Training (Recommended)
//!
//! GPU-accelerated training with automatic CPU fallback for precision-sensitive ops:
//!
//! ```rust,ignore
//! use thrml_sphere::TrainableNavigatorEBM;
//! use thrml_core::ComputeBackend;
//!
//! // Train with hybrid CPU/GPU execution (auto-detects backend)
//! let losses = trainable.train_hybrid(
//!     &dataset.train,
//!     50,              // epochs
//!     16,              // batch_size
//!     RngKey::new(42),
//!     &device,
//!     None,            // auto-detect backend
//! );
//!
//! // Or with explicit Apple Silicon config
//! let backend = ComputeBackend::apple_silicon();
//! let losses = trainable.train_hybrid(
//!     &dataset.train, 50, 16, RngKey::new(42), &device,
//!     Some(&backend),
//! );
//! ```
//!
//! The hybrid training method:
//! - Uses GPU-batched energy computation for fast forward passes
//! - Routes precision-sensitive gradient ops to CPU (f64) when needed
//! - Automatically handles Metal/AMD RDNA lack of native f64 support
//!
//! ### Hyperparameter Tuning
//!
//! Run grid or random search over hyperparameters:
//!
//! ```rust,ignore
//! use thrml_sphere::{TuningGrid, TuningSession};
//!
//! // Create grid (default: 729 combinations)
//! let grid = TuningGrid::small();  // 8 combinations for quick testing
//!
//! // Create and run session
//! let mut session = TuningSession::new(grid);
//! session.run_grid_search(&sphere_ebm, &dataset, 30, 16, key, &device);
//!
//! // Get best configuration
//! if let Some(best) = session.best_result() {
//!     println!("Best MRR: {:.4}", best.metrics.mrr);
//!     println!("Best config: lr={:.0e}, λ_sem={:.1}",
//!         best.config.learning_rate, best.config.lambda_semantic);
//! }
//!
//! // Save results (both optional)
//! session.to_csv(std::path::Path::new("results.csv"))?;
//! session.to_json(std::path::Path::new("results.json"))?;
//! ```
//!
//! ## Modules
//!
//! ### Core
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`sphere_ebm`] | Main sphere optimization model |
//! | [`config`] | Scale profiles and configuration |
//! | [`hamiltonian`] | Energy function (Hamiltonian) for water-filling dynamics |
//! | [`langevin`] | Sphere-specific Langevin sampler |
//!
//! ### ROOTS & Navigation
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`roots`] | ROOTS layer - compressed inner-shell index |
//! | [`navigator`] | Multi-cone EBM navigation system |
//! | [`training`] | Training infrastructure, validation, hyperparameter tuning |
//! | [`evaluation`] | Navigation quality metrics (recall@k, MRR, nDCG) |
//!
//! ### Infrastructure
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`loader`] | SafeTensors file loading (BLT v3 format) |
//! | [`compute`] | Hybrid CPU/GPU backend for unified memory |
//! | [`compute::substring`] | Substring containment similarity |
//! | [`compute::cpu_ising`] | CPU-based f64 Ising max-cut |
//! | [`similarity`] | GPU-accelerated similarity (re-exports from thrml-core) |
//! | [`hypergraph`] | Hypergraph connectivity |
//! | [`lasing`] | Coherence-driven lasing dynamics |
//!
//! ## BLT Integration
//!
//! The crate integrates with BLT output via the `blt_patches_v3` SafeTensors format:
//!
//! | Tensor | Shape | Description |
//! |--------|-------|-------------|
//! | `embeddings` | \[N, D\] | Patch-aggregated embeddings |
//! | `prominence` | \[N\] | Patch importance scores |
//! | `bytes` | \[total_bytes\] | Concatenated raw bytes |
//! | `patch_lengths` | \[N\] | Length of each patch |
//!
//! Use [`load_blt_safetensors`] for unified loading of embeddings and raw bytes.
//!
//! ## Related Crates
//!
//! - [`thrml-core`](../thrml-core): Distance/similarity utilities, SphericalCoords
//! - [`thrml-samplers`](../thrml-samplers): General Langevin sampler
//! - [`thrml-models`](../thrml-models): IsingEBM, GraphSidecar, SpringEBM

pub mod compute;
pub mod config;
pub mod contrastive;
pub mod evaluation;
pub mod hamiltonian;
pub mod hypergraph;
pub mod langevin;
pub mod lasing;
pub mod loader;
pub mod navigator;
pub mod roots;
pub mod similarity;
pub mod sphere_ebm;
pub mod training;

pub use compute::*;
pub use config::*;
pub use contrastive::*;
pub use evaluation::*;
pub use hamiltonian::*;
pub use hypergraph::*;
pub use langevin::*;
pub use lasing::*;
pub use loader::*;
pub use navigator::*;
pub use roots::*;
pub use similarity::*;
pub use sphere_ebm::*;
pub use training::*;

// Re-export key types from nested modules for convenience
pub use compute::substring::SubstringConfig;

// Re-export hardware tier types from thrml-core for RuntimeConfig users
pub use thrml_core::compute::{HardwareTier, PrecisionProfile, RuntimePolicy};

// Re-export generalized primitives from thrml-core and thrml-samplers
// These can be used directly or as building blocks for sphere-specific APIs
pub use thrml_core::metrics::{evaluate_retrieval, RetrievalMetrics};
pub use thrml_core::text::{ngram_hashes, RollingHash, TextSimilarityConfig};
pub use thrml_samplers::maxcut::{cut_value, maxcut_gibbs, maxcut_multistart};
