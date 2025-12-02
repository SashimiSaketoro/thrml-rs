//! Fused sigmoid-Bernoulli sampling kernel for spin variables.
//!
//! This module provides a GPU-fused implementation for sampling spin variables
//! in Gibbs sampling. Instead of:
//! 1. Computing `2 * gamma`
//! 2. Applying sigmoid
//! 3. Comparing with uniform random
//! 4. Casting to float
//!
//! All operations are fused into a single GPU kernel.

mod backward;
mod forward;
pub mod kernel;

pub use forward::launch_sigmoid_bernoulli;
pub use forward::sigmoid_bernoulli_fused;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};
    use thrml_core::backend::{init_gpu_device, CubeWgpuBackend};

    /// Reference implementation for comparison
    fn sigmoid_bernoulli_reference(
        gamma: Tensor<CubeWgpuBackend, 1>,
        uniform: Tensor<CubeWgpuBackend, 1>,
    ) -> Tensor<CubeWgpuBackend, 1> {
        use burn::tensor::activation::sigmoid;
        let probs = sigmoid(gamma * 2.0);
        uniform.lower_equal(probs).float()
    }

    #[test]
    fn test_sigmoid_bernoulli_equivalence() {
        let device = init_gpu_device();

        // Create test inputs
        let gamma: Tensor<CubeWgpuBackend, 1> =
            Tensor::random([1000], Distribution::Normal(0.0, 1.0), &device);
        let uniform: Tensor<CubeWgpuBackend, 1> =
            Tensor::random([1000], Distribution::Uniform(0.0, 1.0), &device);

        let reference = sigmoid_bernoulli_reference(gamma.clone(), uniform.clone());
        let fused = sigmoid_bernoulli_fused(gamma, uniform);

        // Results should match exactly (same RNG input)
        let ref_data: Vec<f32> = reference.into_data().to_vec().expect("read ref");
        let fused_data: Vec<f32> = fused.into_data().to_vec().expect("read fused");

        assert_eq!(ref_data.len(), fused_data.len());
        for (r, f) in ref_data.iter().zip(fused_data.iter()) {
            assert_eq!(*r, *f, "Mismatch: ref={}, fused={}", r, f);
        }
    }
}
