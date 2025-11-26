//! Autodiff integration for fused kernels.
//!
//! This module provides:
//! - Burn autodiff `Backward` trait implementations
//! - Gumbel-Softmax for differentiable discrete sampling

pub mod gumbel_softmax;
pub mod ops;
pub mod sigmoid_bernoulli;

