//! Fused batch gather kernel for multi-index weight lookups.
//!
//! This module provides a GPU-fused implementation for gathering weights
//! using multiple indices simultaneously. Instead of:
//! 1. Creating stride tensors
//! 2. Computing linear indices
//! 3. Reshaping weights
//! 4. Selecting from flat weights
//!
//! All operations are fused into a single GPU kernel with no intermediate allocations.

mod backward;
mod forward;
pub mod kernel;

pub use forward::batch_gather_fused;
pub use forward::launch_batch_gather;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};
    use thrml_core::backend::{init_gpu_device, WgpuBackend};

    #[test]
    fn test_batch_gather_basic() {
        let device = init_gpu_device();

        let n_nodes = 10;
        let k = 5;
        let dim = 8;

        // Create test weights
        let weights: Tensor<WgpuBackend, 3> =
            Tensor::random([n_nodes, k, dim], Distribution::Normal(0.0, 1.0), &device);

        // Create random indices within bounds - need to floor them
        let k_indices_float: Tensor<WgpuBackend, 2> = Tensor::random(
            [n_nodes, 1],
            Distribution::Uniform(0.0, (k - 1) as f64),
            &device,
        );
        let dim_indices_float: Tensor<WgpuBackend, 2> = Tensor::random(
            [n_nodes, 1],
            Distribution::Uniform(0.0, (dim - 1) as f64),
            &device,
        );

        // Floor and convert to int
        let k_indices: Tensor<WgpuBackend, 2, burn::tensor::Int> = k_indices_float.int();
        let dim_indices: Tensor<WgpuBackend, 2, burn::tensor::Int> = dim_indices_float.int();

        let indices = Tensor::cat(vec![k_indices, dim_indices], 1);

        let result = batch_gather_fused(weights, indices, &[dim, 1], k * dim);

        // Verify output shape
        assert_eq!(
            result.dims(),
            [n_nodes],
            "Output should have shape [n_nodes]"
        );
    }
}
