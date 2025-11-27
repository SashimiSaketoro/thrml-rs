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

#![recursion_limit = "256"] // Required for burn-wgpu

pub mod bernoulli;
pub mod gaussian;
pub mod langevin;
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
pub use program::*;
pub use rng::*;
pub use sampler::*;
pub use sampling::*;
pub use schedule::*;
pub use softmax::*;
pub use spin_gibbs::*;
