//! # thrml-models
//!
//! Model implementations for the THRML probabilistic computing library.
//!
//! This crate provides implementations of various probabilistic graphical models:
//!
//! ## Ising Model
//!
//! The classic Ising model for spin systems:
//!
//! ```rust,ignore
//! use thrml_models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};
//!
//! let model = IsingEBM::new(nodes, edges, biases, weights, beta);
//! let energy = model.energy(&state, &blocks, &device);
//! ```
//!
//! ## Discrete EBM Factors
//!
//! Flexible factor types for building custom energy-based models:
//!
//! - [`SpinEBMFactor`]: Binary/spin interactions
//! - [`CategoricalEBMFactor`]: Categorical variable interactions
//! - [`DiscreteEBMFactor`]: Mixed spin + categorical
//! - [`SquareDiscreteEBMFactor`]: Symmetric interactions
//!
//! ## Continuous Variable Factors
//!
//! Factors for Gaussian PGMs and continuous-discrete hybrid models:
//!
//! - [`LinearFactor`]: Linear bias term `w * x`
//! - [`QuadraticFactor`]: Quadratic self-interaction `w * x^2`
//! - [`CouplingFactor`]: Pairwise coupling `w * x_i * x_j`
//!
//! ## Graph-based Models
//!
//! Energy-based models using graph connectivity:
//!
//! - [`GraphSidecar`]: Graph structure with edges and node attributes
//! - [`SpringEBM`]: Spring-like forces between connected nodes
//! - [`NodeBiasEBM`]: Weighted node bias energy
//!
//! ## Graph Construction Utilities
//!
//! Utilities for building common graph topologies:
//!
//! - [`make_lattice_graph`]: 2D lattice with beyond-nearest-neighbor connections
//! - [`make_nearest_neighbor_lattice`]: Simple 4-connected 2D grid
//!
//! Both support torus (periodic) boundaries and two-color blocking for block Gibbs sampling.
//!
//! ## Training Utilities
//!
//! - [`ising::estimate_moments`]: Estimate first/second moments via sampling
//! - [`ising::estimate_kl_grad`]: Estimate KL divergence gradients
//! - [`ising::hinton_init`]: Initialize states from marginal biases
//!
//! ## Precision Routing
//!
//! All factor types support precision routing via `factor_energy_routed()`:
//!
//! ```rust,ignore
//! use thrml_models::{SpinEBMFactor, RuntimePolicy, ComputeBackend};
//!
//! // Auto-detect hardware and create policy
//! let policy = RuntimePolicy::detect();
//! let backend = ComputeBackend::from_policy(&policy);
//!
//! // Compute energy with precision routing
//! let energy = factor.factor_energy_routed(&backend, &state, &spec, &device);
//!
//! // On Apple Silicon: EnergyCompute may use CPU f64 for large problems
//! // On HPC GPUs with CUDA: Uses GPU f64 for both speed and precision
//! ```
//!
//! ## Hybrid Compute Backend
//!
//! Re-exports from `thrml-core` for CPU/GPU precision routing:
//!
//! - [`RuntimePolicy`]: Auto-detects hardware and creates appropriate precision profile
//! - [`ComputeBackend`]: Backend selection (CPU, GPU, Hybrid, Adaptive, HpcF64)
//! - [`OpType`]: Operation classification for routing decisions
//! - [`HardwareTier`]: Hardware classification (Apple Silicon, NVIDIA tiers, AMD)
//! - [`PrecisionProfile`]: Precision strategy per hardware tier
//!
//! ```rust,ignore
//! use thrml_models::{RuntimePolicy, ComputeBackend, OpType};
//!
//! let policy = RuntimePolicy::detect();
//! let backend = ComputeBackend::from_policy(&policy);
//!
//! // Check if precision-sensitive ops should use CPU
//! if backend.use_cpu(OpType::EnergyCompute, Some(1000000)) {
//!     // Route large energy computations to CPU f64
//! }
//! ```

#![recursion_limit = "256"] // Required for burn-wgpu

pub mod continuous_factors;
pub mod discrete_ebm;
pub mod ebm;
pub mod factor;
pub mod graph_ebm;
pub mod graph_utils;
pub mod ising;

pub use continuous_factors::*;
pub use discrete_ebm::*;
pub use ebm::*;
pub use factor::*;
pub use graph_ebm::*;
pub use graph_utils::*;
pub use ising::*;

// Re-export compute types from thrml-core for precision routing
pub use thrml_core::compute::{
    ComputeBackend, HardwareTier, HybridConfig, OpType, PrecisionMode, PrecisionProfile,
    RuntimePolicy,
};
