use crate::sampler::AbstractConditionalSampler;
use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::interaction::InteractionData;
use thrml_core::node::TensorSpec;

/// Base Bernoulli conditional sampler for spin-valued variables.
///
/// Note: This is a base implementation. For discrete EBMs, use SpinGibbsConditional
/// which properly handles spin products and categorical neighbor indexing.
pub struct BernoulliConditional;

impl AbstractConditionalSampler for BernoulliConditional {
    type SamplerState = (); // Stateless sampler

    fn sample(
        &self,
        _key: crate::rng::RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        _neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        _n_spin_per_interaction: &[usize],
        _sampler_state: Self::SamplerState,
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, Self::SamplerState) {
        // Compute gamma parameter (sum of interaction contributions)
        // This is a base implementation for bias-only terms (no neighbor dependencies)
        // For full spin/categorical neighbor handling, use SpinGibbsConditional
        let n_nodes = output_spec.shape[0];
        let mut gamma: Tensor<WgpuBackend, 1> = Tensor::zeros([n_nodes], device);

        for (interaction, active) in itertools::izip!(interactions, active_flags) {
            match interaction {
                InteractionData::Tensor(tensor) => {
                    // Sum over interaction and trailing dimensions
                    let interaction_sum: Tensor<WgpuBackend, 1> = tensor
                        .clone()
                        .sum_dim(2)
                        .squeeze_dim::<2>(2) // [n_nodes, n_interactions]
                        .sum_dim(1)
                        .squeeze_dim::<1>(1); // [n_nodes]

                    let active_sum: Tensor<WgpuBackend, 1> =
                        active.clone().sum_dim(1).squeeze_dim::<1>(1);

                    gamma = gamma + interaction_sum * active_sum;
                }
                InteractionData::Linear { weights } => {
                    // Linear interactions contribute c_i * x_i to the energy
                    // For bias-only Bernoulli, just sum the weights
                    let weight_sum: Tensor<WgpuBackend, 1> =
                        weights.clone().sum_dim(1).squeeze_dim::<1>(1);
                    let active_sum: Tensor<WgpuBackend, 1> =
                        active.clone().sum_dim(1).squeeze_dim::<1>(1);
                    gamma = gamma + weight_sum * active_sum;
                }
                InteractionData::Quadratic { .. } => {
                    // Quadratic interactions are for continuous variables, not spin
                    // Skip for Bernoulli sampler
                }
                InteractionData::Sphere { .. } => {
                    // Sphere interactions are for Langevin dynamics, not Gibbs sampling
                    // Skip for Bernoulli sampler
                }
            }
        }

        // Sample: P(S=1) = sigmoid(2*gamma)
        let probs = burn::tensor::activation::sigmoid(gamma * 2.0);

        // Bernoulli sampling
        let uniform_random: Tensor<WgpuBackend, 1> = Tensor::random(
            probs.dims(),
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            device,
        );

        (uniform_random.lower_equal(probs).float(), ())
    }
}
