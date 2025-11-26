//! Fused Gumbel-max categorical sampling kernel.
//!
//! This module provides a GPU-fused implementation of the Gumbel-max trick
//! for efficient categorical sampling. Instead of:
//! 1. Computing Gumbel noise: `-log(-log(uniform))`
//! 2. Adding to logits
//! 3. Taking argmax
//!
//! All three operations are fused into a single GPU kernel.

mod backward;
mod forward;
pub mod kernel;

pub use forward::gumbel_argmax_fused;
pub use forward::launch_gumbel_argmax;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Int, Tensor};
    use thrml_core::backend::{init_gpu_device, WgpuBackend};

    /// Reference implementation for comparison
    fn gumbel_argmax_reference(
        logits: Tensor<WgpuBackend, 2>,
        uniform: Tensor<WgpuBackend, 2>,
    ) -> Tensor<WgpuBackend, 1, Int> {
        // Gumbel noise: -log(-log(u))
        let gumbel = -(-(uniform.log())).log();
        let perturbed = logits + gumbel;
        perturbed.argmax(1).squeeze::<1>()
    }

    #[test]
    fn test_gumbel_argmax_equivalence() {
        let device = init_gpu_device();

        // Create test inputs
        let logits: Tensor<WgpuBackend, 2> =
            Tensor::random([100, 10], Distribution::Normal(0.0, 1.0), &device);
        let uniform: Tensor<WgpuBackend, 2> = Tensor::random(
            [100, 10],
            Distribution::Uniform(1e-10, 1.0 - 1e-10),
            &device,
        );

        let reference = gumbel_argmax_reference(logits.clone(), uniform.clone());
        let fused = gumbel_argmax_fused(logits, uniform);

        // Results should match exactly (same RNG input)
        let ref_data: Vec<i32> = reference.into_data().to_vec().expect("read ref");
        let fused_data: Vec<i32> = fused.into_data().to_vec().expect("read fused");

        assert_eq!(ref_data.len(), fused_data.len());
        for (r, f) in ref_data.iter().zip(fused_data.iter()) {
            assert_eq!(*r, *f, "Mismatch: ref={}, fused={}", r, f);
        }
    }
}
