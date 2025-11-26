//! SpinGibbsConditional - Gibbs updates for spin-valued variables in discrete EBMs.
//!
//! This sampler performs Gibbs sampling updates for spin (binary) variables,
//! computing the conditional distribution given neighboring states.
//!
//! The conditional probability is: P(S=1) = sigmoid(2*γ)
//! where γ = Σ_i s_1^i ... s_K^i * W^i[x_1^i, ..., x_M^i]
//!
//! When the `fused-kernels` feature is enabled, uses a GPU-fused kernel that
//! combines the sigmoid and Bernoulli sampling into a single kernel launch.

use crate::rng::RngKey;
use crate::sampler::AbstractConditionalSampler;
use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use thrml_core::interaction::InteractionData;
use thrml_core::node::TensorSpec;

#[cfg(feature = "fused-kernels")]
use thrml_kernels::sigmoid_bernoulli_fused;

/// A conditional update for spin-valued random variables that performs a Gibbs sampling update
/// given one or more DiscreteEBMInteractions.
pub struct SpinGibbsConditional;

impl SpinGibbsConditional {
    pub fn new() -> Self {
        SpinGibbsConditional
    }
}

impl Default for SpinGibbsConditional {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractConditionalSampler for SpinGibbsConditional {
    type SamplerState = (); // Stateless sampler

    fn sample(
        &self,
        _key: RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        _sampler_state: Self::SamplerState,
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, Self::SamplerState) {
        // Compute gamma parameter using n_spin to split states
        let gamma = self.compute_parameters(
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction,
            output_spec,
            device,
        );

        // Generate uniform random values
        let n_nodes = output_spec.shape[0];
        let uniform: Tensor<WgpuBackend, 1> =
            Tensor::random([n_nodes], Distribution::Uniform(0.0, 1.0), device);

        // Use fused kernel when feature is enabled
        #[cfg(feature = "fused-kernels")]
        {
            return (sigmoid_bernoulli_fused(gamma, uniform), ());
        }

        // Default implementation: separate operations
        #[cfg(not(feature = "fused-kernels"))]
        {
            // Sample: P(S=1) = sigmoid(2*gamma)
            let probs = burn::tensor::activation::sigmoid(gamma * 2.0);
            // Sample Bernoulli: output 1 if uniform < probs, else 0
            (uniform.lower_equal(probs).float(), ())
        }
    }
}

impl SpinGibbsConditional {
    /// Compute the parameter γ of a spin-valued Bernoulli distribution given DiscreteEBMInteractions.
    ///
    /// γ = Σ_i s_1^i ... s_K^i * W^i[x_1^i, ..., x_M^i]
    ///
    /// where the sum is over all the DiscreteEBMInteractions.
    fn compute_parameters(
        &self,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_nodes = output_spec.shape[0];
        let mut gamma: Tensor<WgpuBackend, 1> = Tensor::zeros([n_nodes], device);

        for (interaction, active, states, &n_spin) in itertools::izip!(
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction
        ) {
            match interaction {
                InteractionData::Tensor(tensor) => {
                    // interaction: [n_nodes, n_interactions, weight_dim]
                    // active: [n_nodes, n_interactions]
                    // states: Vec of [n_nodes, n_interactions] tensors (one per tail block)
                    // n_spin: number of spin tail blocks (first n_spin are spin, rest are categorical)

                    // Split states into spin and categorical
                    let states_spin: Vec<Tensor<WgpuBackend, 2>> =
                        states.iter().take(n_spin).cloned().collect();
                    let states_cat: Vec<Tensor<WgpuBackend, 2>> =
                        states.iter().skip(n_spin).cloned().collect();

                    // Compute spin product from spin states only
                    let spin_prod = compute_spin_product_2d(&states_spin, device);

                    // Get interaction dimensions
                    let dims = tensor.dims();
                    let n_interactions = dims[1];

                    // Index weights by categorical neighbor states
                    let weights = if states_cat.is_empty() {
                        // No categorical neighbors - weights should be [n_nodes, n_interactions, 1]
                        // Squeeze to [n_nodes, n_interactions]
                        tensor.clone().squeeze_dim(2)
                    } else {
                        // Use batch_gather to index by categorical states
                        // For SpinGibbsConditional, categorical indexing gives us a scalar per interaction
                        let cat = &states_cat[0]; // First categorical neighbor

                        // Flatten weights and indices for gather
                        let flat_weights: Tensor<WgpuBackend, 2> = tensor
                            .clone()
                            .reshape([(n_nodes * n_interactions) as i32, dims[2] as i32]);
                        let flat_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> = cat
                            .clone()
                            .reshape([(n_nodes * n_interactions) as i32])
                            .int();

                        // Gather and reshape
                        let indices_2d: Tensor<WgpuBackend, 2, burn::tensor::Int> =
                            flat_indices.unsqueeze_dim(1);
                        let gathered: Tensor<WgpuBackend, 2> = flat_weights.gather(1, indices_2d);
                        let result: Tensor<WgpuBackend, 1> = gathered.squeeze_dim::<1>(1);
                        result.reshape([n_nodes as i32, n_interactions as i32])
                    };

                    // Compute contribution: weights * active * spin_prod, then sum over interaction dim
                    let contribution = weights * active.clone() * spin_prod;
                    let contribution_sum = contribution.sum_dim(1).squeeze_dim(1);

                    gamma = gamma + contribution_sum;
                }
                InteractionData::Linear { weights } => {
                    // Linear interaction from continuous coupling: c_i * x_i
                    // Multiply by neighbor states and sum
                    let state_prod = if states.is_empty() {
                        let dims = weights.dims();
                        Tensor::ones([dims[0], dims[1]], device)
                    } else {
                        // Product of all neighbor states (typically continuous)
                        let first = states[0].clone();
                        states.iter().skip(1).fold(first, |acc, s| acc * s.clone())
                    };

                    let contribution = weights.clone() * active.clone() * state_prod;
                    let contribution_sum = contribution.sum_dim(1).squeeze_dim(1);
                    gamma = gamma + contribution_sum;
                }
                InteractionData::Quadratic { .. } => {
                    // Quadratic interactions are for continuous variables, not spin
                    // Skip for SpinGibbsConditional
                }
            }
        }

        gamma
    }
}

/// Compute the element-wise product of spin states.
///
/// For spin variables: True/1.0 -> +1, False/0.0 -> -1
/// Returns the element-wise product across all state tensors.
fn compute_spin_product_2d(
    states: &[Tensor<WgpuBackend, 2>],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2> {
    if states.is_empty() {
        // Return 1.0 tensor
        return Tensor::ones([1, 1], device);
    }

    // Convert each state to ±1: (2 * s - 1)
    let converted: Vec<Tensor<WgpuBackend, 2>> =
        states.iter().map(|s| s.clone() * 2.0 - 1.0).collect();

    // Compute element-wise product
    let first = converted[0].clone();
    converted
        .iter()
        .skip(1)
        .fold(first, |acc, x| acc * x.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_spin_gibbs_basic() {
        use thrml_core::backend::{ensure_metal_backend, init_gpu_device};

        ensure_metal_backend();
        let device = init_gpu_device();

        let sampler = SpinGibbsConditional::new();

        // Create simple test case
        let output_spec = TensorSpec {
            shape: vec![4],
            dtype: burn::tensor::DType::Bool,
        };

        // Empty interactions -> gamma = 0 -> P(S=1) = 0.5
        let interactions: Vec<InteractionData> = Vec::new();
        let active_flags: Vec<Tensor<WgpuBackend, 2>> = Vec::new();
        let neighbor_states: Vec<Vec<Tensor<WgpuBackend, 2>>> = Vec::new();
        let n_spin_per_interaction: Vec<usize> = Vec::new();

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
}
