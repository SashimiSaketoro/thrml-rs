//! # thrml-samplers
//!
//! Sampling algorithms for the THRML probabilistic computing library.
//!
//! This crate provides GPU-accelerated sampling algorithms for probabilistic graphical models:
//!
//! - **Block Gibbs Sampling**: Parallel sampling of independent blocks via [`BlockSamplingProgram`]
//! - **Bernoulli Sampler**: For binary/spin variables via [`BernoulliConditional`]
//! - **Softmax Sampler**: For categorical variables using Gumbel-max trick via [`SoftmaxConditional`]
//! - **Spin Gibbs Sampler**: Specialized for Ising-type models via [`SpinGibbsConditional`]
//! - **Gaussian Sampler**: For continuous variables in Gaussian PGMs via [`GaussianSampler`]
//! - **Langevin Sampler**: Overdamped Langevin dynamics for continuous EBMs via [`LangevinConfig`]
//!
//! ## Precision Routing
//!
//! All samplers support precision routing via `*_routed()` methods:
//!
//! ```rust,ignore
//! use thrml_samplers::{SpinGibbsConditional, RuntimePolicy, ComputeBackend, OpType};
//!
//! // Auto-detect hardware and create policy
//! let policy = RuntimePolicy::detect();
//! let backend = ComputeBackend::from_policy(&policy);
//!
//! // Sample with precision routing
//! let sampler = SpinGibbsConditional::new();
//! let (samples, _) = sampler.sample_routed(
//!     &backend, key, &interactions, &active, &states, &n_spin, (), &spec, &device
//! );
//!
//! // On Apple Silicon: IsingSampling routes to CPU f64 for precision
//! // On HPC GPUs with CUDA: Uses GPU f64 for both speed and precision
//! ```
//!
//! ## RNG Key System
//!
//! Deterministic RNG key management (similar to JAX):
//!
//! ```rust
//! use thrml_samplers::RngKey;
//!
//! let key = RngKey::new(42);
//! let (key1, key2) = key.split_two();
//! ```
//!
//! ## Sampling Schedule
//!
//! Control warmup, number of samples, and steps per sample:
//!
//! ```rust
//! use thrml_samplers::SamplingSchedule;
//!
//! let schedule = SamplingSchedule::new(100, 1000, 5);
//! // 100 warmup steps, 1000 samples, 5 steps between samples
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
//! use thrml_samplers::{RuntimePolicy, ComputeBackend, OpType};
//!
//! let policy = RuntimePolicy::detect();
//! let backend = ComputeBackend::from_policy(&policy);
//!
//! // Check if precision-sensitive ops should use CPU
//! if backend.use_cpu(OpType::IsingSampling, None) {
//!     // Route to CPU f64 path
//! }
//! ```

#![recursion_limit = "256"] // Required for burn-wgpu

pub mod bernoulli;
pub mod gaussian;
pub mod langevin;
pub mod maxcut;
pub mod program;
pub mod rng;
pub mod sampler;
pub mod sampling;
pub mod schedule;
pub mod softmax;
pub mod spin_gibbs;

pub use bernoulli::*;
pub use gaussian::*;
pub use langevin::*;
pub use maxcut::*;
pub use program::*;
pub use rng::*;
pub use sampler::*;
pub use sampling::*;
pub use schedule::*;
pub use softmax::*;
pub use spin_gibbs::*;

// Re-export compute types from thrml-core for precision routing
pub use thrml_core::compute::{
    ComputeBackend, HardwareTier, HybridConfig, OpType, PrecisionMode, PrecisionProfile,
    RuntimePolicy,
};
