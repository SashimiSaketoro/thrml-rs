//! # thrml-kernels
//!
//! Fused CubeCL GPU kernels for performance-critical sampling operations in THRML.
//!
//! This crate provides optimized GPU kernels that fuse multiple tensor operations into
//! single kernel launches, eliminating intermediate memory allocations and reducing
//! kernel launch overhead.
//!
//! ## Available Kernels
//!
//! - `gumbel_argmax` - Fused Gumbel-max trick for categorical sampling
//! - `sigmoid_bernoulli` - Fused sigmoid + Bernoulli for spin sampling
//! - `batch_gather` - Fused multi-index weight gathering
//!
//! ## Usage
//!
//! These kernels are automatically used when the `fused-kernels` feature is enabled
//! in `thrml-samplers` or `thrml-models`.
//!
//! ## Architecture
//!
//! This crate follows the official Burn custom kernel pattern:
//! 1. Define custom `Backend` and `AutodiffBackend` traits
//! 2. Implement for `CubeBackend` with actual GPU kernels
//! 3. Implement for `Autodiff<B>` with proper backward passes
//! 4. Provide reference implementations as fallback

#![cfg_attr(not(feature = "gpu"), allow(unused))]

use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::ops::{FloatTensor, IntTensor};

// ============================================================================
// Custom Backend Traits
// ============================================================================

/// Custom backend trait for fused kernel operations.
///
/// This trait extends Burn's `Backend` with specialized fused operations
/// that combine multiple tensor operations into single GPU kernel launches.
///
/// # Implementation
///
/// - GPU backends (e.g., `CubeBackend`) implement this with actual fused kernels
/// - Other backends can use the reference implementation fallback
pub trait FusedKernelBackend: BurnBackend {
    /// Fused sigmoid-Bernoulli sampling.
    ///
    /// Computes `uniform < sigmoid(2 * gamma) ? 1.0 : 0.0` in a single kernel.
    ///
    /// # Arguments
    /// * `gamma` - Gibbs parameters `[batch_size]`
    /// * `uniform` - Pre-generated uniform samples `[batch_size]`
    ///
    /// # Returns
    /// Bernoulli samples as floats (0.0 or 1.0) `[batch_size]`
    fn sigmoid_bernoulli_fused(
        gamma: FloatTensor<Self>,
        uniform: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    /// Fused Gumbel-max categorical sampling.
    ///
    /// Computes `argmax(logits + gumbel_noise)` in a single kernel where
    /// `gumbel_noise = -log(-log(uniform))`.
    ///
    /// # Arguments
    /// * `logits` - Log-probabilities [batch_size, n_categories]
    /// * `uniform` - Pre-generated uniform samples [batch_size, n_categories]
    ///
    /// # Returns
    /// Category indices `[batch_size]` as integers
    fn gumbel_argmax_fused(
        logits: FloatTensor<Self>,
        uniform: FloatTensor<Self>,
    ) -> IntTensor<Self>;

    /// Fused batch gather with linear indexing.
    ///
    /// Gathers values from weights using multi-dimensional indices without
    /// creating intermediate stride tensors.
    ///
    /// # Arguments
    /// * `weights` - Weight tensor [batch_size, dim1, dim2, ...]
    /// * `indices` - Index tensor [batch_size, n_indices]
    /// * `strides` - Stride values for each index dimension
    /// * `batch_stride` - Stride for the batch dimension
    ///
    /// # Returns
    /// Gathered values `[batch_size]`
    fn batch_gather_fused(
        weights: FloatTensor<Self>,
        indices: IntTensor<Self>,
        strides: &[usize],
        batch_stride: usize,
    ) -> FloatTensor<Self>;
}

/// Custom autodiff backend trait for fused kernel operations.
///
/// This trait provides autodiff-compatible versions of the fused operations.
/// The backward passes use appropriate gradient estimation strategies:
/// - Sigmoid: analytic gradient
/// - Bernoulli/Argmax: Straight-Through Estimator (STE)
/// - Gather: Scatter-add
pub trait FusedKernelAutodiffBackend:
    FusedKernelBackend + burn::tensor::backend::AutodiffBackend
{
}

// ============================================================================
// GPU Kernel Modules
// ============================================================================

#[cfg(feature = "gpu")]
pub mod gumbel_argmax;

#[cfg(feature = "gpu")]
pub mod sigmoid_bernoulli;

#[cfg(feature = "gpu")]
pub mod batch_gather;

// ============================================================================
// Reference Implementations (for non-GPU backends or fallback)
// ============================================================================

pub mod reference {
    //! Reference implementations using standard tensor operations.
    //!
    //! These can be used as fallback for backends that don't support
    //! the fused kernels, or for testing/validation purposes.

    use burn::tensor::{activation::sigmoid, backend::Backend, Int, Tensor};

    /// Reference implementation of sigmoid-Bernoulli sampling.
    pub fn sigmoid_bernoulli<B: Backend, const D: usize>(
        gamma: Tensor<B, D>,
        uniform: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let probs = sigmoid(gamma * 2.0);
        uniform.lower_equal(probs).float()
    }

    /// Reference implementation of Gumbel-max categorical sampling.
    pub fn gumbel_argmax<B: Backend>(
        logits: Tensor<B, 2>,
        uniform: Tensor<B, 2>,
    ) -> Tensor<B, 1, Int> {
        // Gumbel noise: -log(-log(u))
        let gumbel = -(-(uniform.log())).log();
        let perturbed = logits + gumbel;
        perturbed.argmax(1).squeeze::<1>()
    }

    /// Reference implementation of batch gather.
    ///
    /// Note: This is a simplified 2-index version. For full generality,
    /// use the fused kernel.
    pub fn batch_gather_2d<B: Backend>(
        weights: Tensor<B, 3>,
        indices: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        let [batch_size, n_k, n_dim] = weights.dims();
        let device = weights.device();

        // Create batch indices
        let batch_indices: Tensor<B, 1, Int> =
            Tensor::arange(0..batch_size as i64, &device).reshape([batch_size]);

        // Extract k and dim indices
        let k_indices: Tensor<B, 1, Int> =
            indices.clone().slice([0..batch_size, 0..1]).squeeze::<1>();
        let dim_indices: Tensor<B, 1, Int> = indices.slice([0..batch_size, 1..2]).squeeze::<1>();

        // Compute linear indices
        let linear_indices =
            batch_indices * (n_k * n_dim) as i64 + k_indices * n_dim as i64 + dim_indices;

        // Flatten weights and gather
        let weights_flat = weights.reshape([batch_size * n_k * n_dim]);
        weights_flat.select(0, linear_indices)
    }
}

// ============================================================================
// CubeBackend Implementation
// ============================================================================

#[cfg(feature = "gpu")]
mod cube_impl {
    use super::FusedKernelBackend;
    use burn::tensor::ops::{FloatTensor, IntTensor};
    use burn_cubecl::{element::BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};

    impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FusedKernelBackend
        for CubeBackend<R, F, I, BT>
    {
        fn sigmoid_bernoulli_fused(
            gamma: FloatTensor<Self>,
            uniform: FloatTensor<Self>,
        ) -> FloatTensor<Self> {
            crate::sigmoid_bernoulli::launch_sigmoid_bernoulli::<R, F, I, BT>(gamma, uniform)
        }

        fn gumbel_argmax_fused(
            logits: FloatTensor<Self>,
            uniform: FloatTensor<Self>,
        ) -> IntTensor<Self> {
            crate::gumbel_argmax::launch_gumbel_argmax::<R, F, I, BT>(logits, uniform)
        }

        fn batch_gather_fused(
            weights: FloatTensor<Self>,
            indices: IntTensor<Self>,
            strides: &[usize],
            batch_stride: usize,
        ) -> FloatTensor<Self> {
            crate::batch_gather::launch_batch_gather::<R, F, I, BT>(
                weights,
                indices,
                strides,
                batch_stride,
            )
        }
    }
}

// ============================================================================
// Autodiff Integration
// ============================================================================

#[cfg(feature = "gpu")]
pub mod autodiff;

#[cfg(feature = "gpu")]
pub use autodiff::gumbel_softmax::{gumbel_softmax, TemperatureSchedule};

// ============================================================================
// Public API Functions
// ============================================================================

#[cfg(feature = "gpu")]
pub use gumbel_argmax::gumbel_argmax_fused;

#[cfg(feature = "gpu")]
pub use sigmoid_bernoulli::sigmoid_bernoulli_fused;

#[cfg(feature = "gpu")]
pub use batch_gather::batch_gather_fused;
