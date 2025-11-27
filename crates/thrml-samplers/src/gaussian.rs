//! Gaussian sampler for continuous-valued random variables.
//!
//! This module provides the [`GaussianSampler`] for sampling from Gaussian conditionals
//! in probabilistic graphical models.
//!
//! ## Sampling Algorithm
//!
//! For a Gaussian conditional with linear and quadratic interactions:
//!
//! ```text
//! p(x | neighbors) ∝ exp(-0.5 * A * x^2 + b * x)
//! ```
//!
//! where:
//! - `A` is the precision (inverse variance) from quadratic interactions
//! - `b` is the mean contribution from linear interactions with neighbor states
//!
//! The sampler:
//! 1. Accumulates precision from `InteractionData::Quadratic` interactions
//! 2. Accumulates mean contribution from `InteractionData::Linear` interactions
//! 3. Samples x ~ N(b/A, sqrt(1/A))

use crate::rng::RngKey;
use crate::sampler::AbstractConditionalSampler;
use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use thrml_core::interaction::InteractionData;
use thrml_core::node::TensorSpec;

/// Gaussian conditional sampler for continuous-valued variables.
///
/// This sampler computes the conditional Gaussian distribution given:
/// - `Linear` interactions contributing to the mean
/// - `Quadratic` interactions contributing to the variance
///
/// The sampling uses GPU-accelerated normal distribution via Burn's
/// `Distribution::Normal`.
pub struct GaussianSampler;

impl GaussianSampler {
    pub fn new() -> Self {
        GaussianSampler
    }
}

impl Default for GaussianSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractConditionalSampler for GaussianSampler {
    type SamplerState = (); // Stateless sampler

    fn sample(
        &self,
        _key: RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        _n_spin_per_interaction: &[usize],
        _sampler_state: Self::SamplerState,
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, Self::SamplerState) {
        let n_nodes = output_spec.shape[0];

        // Accumulate precision (inverse variance) from quadratic interactions
        // and mean contribution from linear interactions
        let mut precision: Tensor<WgpuBackend, 1> = Tensor::zeros([n_nodes], device);
        let mut mean_contribution: Tensor<WgpuBackend, 1> = Tensor::zeros([n_nodes], device);

        for (interaction, active, states) in
            itertools::izip!(interactions, active_flags, neighbor_states)
        {
            match interaction {
                InteractionData::Quadratic { inverse_weights } => {
                    // inverse_weights: [n_nodes, k] - precision contributions
                    // Sum over the k dimension, weighted by active flags
                    let weighted: Tensor<WgpuBackend, 2> = inverse_weights.clone() * active.clone();
                    let contribution: Tensor<WgpuBackend, 1> =
                        weighted.sum_dim(1).squeeze_dim::<1>(1);
                    precision = precision + contribution;
                }
                InteractionData::Linear { weights } => {
                    // weights: [n_nodes, k] - linear contributions to mean
                    // Multiply by neighbor states and sum

                    // Compute product of all neighbor states
                    let state_prod = if states.is_empty() {
                        // No neighbors, use ones
                        let dims = weights.dims();
                        Tensor::ones([dims[0], dims[1]], device)
                    } else {
                        // Element-wise product of all neighbor states
                        let first = states[0].clone();
                        states.iter().skip(1).fold(first, |acc, s| acc * s.clone())
                    };

                    // Weighted contribution: weights * active * state_prod
                    let weighted: Tensor<WgpuBackend, 2> =
                        weights.clone() * active.clone() * state_prod;
                    let contribution: Tensor<WgpuBackend, 1> =
                        weighted.sum_dim(1).squeeze_dim::<1>(1);

                    // Linear interactions contribute to the mean via: mean += weights / precision
                    // We accumulate the numerator here and divide later
                    mean_contribution = mean_contribution + contribution;
                }
                InteractionData::Tensor(_) => {
                    // Tensor interactions are for discrete EBMs, not Gaussian
                    // Skip for GaussianSampler
                }
                InteractionData::Sphere { .. } => {
                    // Sphere interactions are for Langevin dynamics, not Gibbs sampling
                    // Skip for GaussianSampler
                }
            }
        }

        // Compute variance = 1/precision (avoid division by zero)
        let epsilon = 1e-8;
        let variance: Tensor<WgpuBackend, 1> = (precision.clone() + epsilon).recip();

        // Compute mean = mean_contribution * variance
        let mean: Tensor<WgpuBackend, 1> = mean_contribution * variance.clone();

        // Compute standard deviation
        let std: Tensor<WgpuBackend, 1> = variance.sqrt();

        // Sample from standard normal and transform
        // x = mean + std * z, where z ~ N(0, 1)
        let noise: Tensor<WgpuBackend, 1> =
            Tensor::random([n_nodes], Distribution::Normal(0.0, 1.0), device);

        let samples = mean + std * noise;

        (samples, ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gaussian_sampler_basic() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};

        ensure_backend();
        let device = init_gpu_device();

        let sampler = GaussianSampler::new();

        // Create simple test case with quadratic (variance) interaction only
        let output_spec = TensorSpec {
            shape: vec![4],
            dtype: burn::tensor::DType::F32,
        };

        // Quadratic interaction with precision = 1.0 (unit variance)
        let inverse_weights: Tensor<WgpuBackend, 2> = Tensor::ones([4, 1], &device);
        let interactions = vec![InteractionData::Quadratic { inverse_weights }];
        let active: Tensor<WgpuBackend, 2> = Tensor::ones([4, 1], &device);
        let active_flags = vec![active];
        let neighbor_states: Vec<Vec<Tensor<WgpuBackend, 2>>> = vec![vec![]];
        let n_spin_per_interaction: Vec<usize> = vec![0];

        let key = RngKey::new(42);
        let (samples, _) = sampler.sample(
            key,
            &interactions,
            &active_flags,
            &neighbor_states,
            &n_spin_per_interaction,
            (),
            &output_spec,
            &device,
        );

        assert_eq!(samples.dims(), [4], "Should produce 4 samples");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gaussian_sampler_with_linear() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};

        ensure_backend();
        let device = init_gpu_device();

        let sampler = GaussianSampler::new();
        let output_spec = TensorSpec {
            shape: vec![4],
            dtype: burn::tensor::DType::F32,
        };

        // Quadratic interaction with precision = 1.0
        let inverse_weights: Tensor<WgpuBackend, 2> = Tensor::ones([4, 1], &device);

        // Linear interaction with bias = [1, 2, 3, 4]
        let linear_weights: Tensor<WgpuBackend, 2> =
            Tensor::from_data([[1.0f32], [2.0], [3.0], [4.0]], &device);

        let interactions = vec![
            InteractionData::Quadratic { inverse_weights },
            InteractionData::Linear {
                weights: linear_weights,
            },
        ];

        let active1: Tensor<WgpuBackend, 2> = Tensor::ones([4, 1], &device);
        let active2: Tensor<WgpuBackend, 2> = Tensor::ones([4, 1], &device);
        let active_flags = vec![active1, active2];

        let neighbor_states: Vec<Vec<Tensor<WgpuBackend, 2>>> = vec![vec![], vec![]];
        let n_spin_per_interaction: Vec<usize> = vec![0, 0];

        let key = RngKey::new(42);
        let (samples, _) = sampler.sample(
            key,
            &interactions,
            &active_flags,
            &neighbor_states,
            &n_spin_per_interaction,
            (),
            &output_spec,
            &device,
        );

        assert_eq!(samples.dims(), [4], "Should produce 4 samples");

        // With precision=1 and bias=[1,2,3,4], samples should be centered around those means
        // (within statistical noise)
    }
}
