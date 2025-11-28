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
//!
//! ## Precision Routing
//!
//! When using `sample_routed` with a [`ComputeBackend`], the sampler routes
//! precision-sensitive accumulation based on hardware capabilities. The accumulation
//! of precision and mean contributions can overflow in f32 for large models.

use crate::rng::RngKey;
use crate::sampler::AbstractConditionalSampler;
use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use thrml_core::compute::{ComputeBackend, OpType};
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

impl GaussianSampler {
    /// Sample with precision routing based on ComputeBackend.
    ///
    /// Routes the precision/mean accumulation through CPU f64 or CUDA f64
    /// based on the backend configuration.
    pub fn sample_routed(
        &self,
        backend: &ComputeBackend,
        _key: RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        _n_spin_per_interaction: &[usize],
        _sampler_state: (),
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, ()) {
        let n_nodes = output_spec.shape[0];

        // Route based on backend - use GradientCompute OpType since accumulation is similar
        if backend.use_cpu(OpType::GradientCompute, Some(n_nodes)) {
            return self.sample_cpu_f64(
                interactions,
                active_flags,
                neighbor_states,
                output_spec,
                device,
            );
        }

        // Standard GPU f32 path
        self.sample(
            _key,
            interactions,
            active_flags,
            neighbor_states,
            _n_spin_per_interaction,
            _sampler_state,
            output_spec,
            device,
        )
    }

    /// Sample using CPU f64 for precision-sensitive accumulation.
    fn sample_cpu_f64(
        &self,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, ()) {
        let n_nodes = output_spec.shape[0];

        // Accumulate in f64
        let mut precision_f64: Vec<f64> = vec![0.0; n_nodes];
        let mut mean_contribution_f64: Vec<f64> = vec![0.0; n_nodes];

        for (interaction, active, states) in
            itertools::izip!(interactions, active_flags, neighbor_states)
        {
            let active_data: Vec<f32> = active.clone().into_data().to_vec().unwrap();
            let active_dims = active.dims();
            let k = active_dims[1];

            match interaction {
                InteractionData::Quadratic { inverse_weights } => {
                    let weights_data: Vec<f32> = inverse_weights.clone().into_data().to_vec().unwrap();

                    // Sum over k dimension with active flags
                    for node in 0..n_nodes {
                        for ki in 0..k {
                            let w = weights_data.get(node * k + ki).copied().unwrap_or(0.0) as f64;
                            let a = active_data.get(node * k + ki).copied().unwrap_or(0.0) as f64;
                            precision_f64[node] += w * a;
                        }
                    }
                }
                InteractionData::Linear { weights } => {
                    let weights_data: Vec<f32> = weights.clone().into_data().to_vec().unwrap();

                    // Get state product
                    let state_prod: Vec<f64> = if states.is_empty() {
                        vec![1.0; n_nodes * k]
                    } else {
                        let mut prod = vec![1.0f64; n_nodes * k];
                        for state in states {
                            let state_data: Vec<f32> = state.clone().into_data().to_vec().unwrap();
                            for (i, &s) in state_data.iter().enumerate() {
                                if i < prod.len() {
                                    prod[i] *= s as f64;
                                }
                            }
                        }
                        prod
                    };

                    // Accumulate weighted contribution
                    for node in 0..n_nodes {
                        for ki in 0..k {
                            let w = weights_data.get(node * k + ki).copied().unwrap_or(0.0) as f64;
                            let a = active_data.get(node * k + ki).copied().unwrap_or(0.0) as f64;
                            let sp = state_prod.get(node * k + ki).copied().unwrap_or(1.0);
                            mean_contribution_f64[node] += w * a * sp;
                        }
                    }
                }
                InteractionData::Tensor(_) | InteractionData::Sphere { .. } => {
                    // Skip
                }
            }
        }

        // Compute variance, mean, std in f64
        let epsilon = 1e-8f64;
        let mut samples_f64: Vec<f64> = Vec::with_capacity(n_nodes);
        
        // Use thread_rng for noise
        use rand_distr::{Distribution as RandDist, StandardNormal};
        let mut rng = rand::thread_rng();

        for i in 0..n_nodes {
            let variance = 1.0 / (precision_f64[i] + epsilon);
            let mean = mean_contribution_f64[i] * variance;
            let std = variance.sqrt();
            let noise: f64 = StandardNormal.sample(&mut rng);
            samples_f64.push(mean + std * noise);
        }

        // Convert to f32 tensor
        let samples_f32: Vec<f32> = samples_f64.iter().map(|&x| x as f32).collect();
        let samples = Tensor::from_floats(samples_f32.as_slice(), device);

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
