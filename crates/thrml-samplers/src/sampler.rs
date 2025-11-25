use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::interaction::InteractionData;
use thrml_core::node::TensorSpec;

/// Trait for conditional samplers that update node states during block sampling.
///
/// A conditional sampler takes neighbor states and produces samples for the current block.
/// Samplers can be stateful (carrying state between iterations) or stateless.
///
/// For stateless samplers (most common), use `SamplerState = ()`.
pub trait AbstractConditionalSampler {
    /// The type of state this sampler carries between iterations.
    /// Use `()` for stateless samplers.
    type SamplerState: Clone + Default;

    /// Initialize the sampler state before sampling begins.
    ///
    /// Returns the initial sampler state that will be passed to the first `sample` call.
    fn init_state(&self) -> Self::SamplerState {
        Self::SamplerState::default()
    }

    /// Sample from the conditional distribution.
    ///
    /// # Arguments
    ///
    /// - `key`: RNG key for deterministic sampling
    /// - `interactions`: Sliced interaction data (Tensor, Linear, or Quadratic)
    /// - `active_flags`: Boolean flags indicating which interactions are active, shape [n_nodes, n_interactions]
    /// - `neighbor_states`: Neighbor state tensors for each interaction, shape [n_nodes, n_interactions, ...]
    /// - `n_spin_per_interaction`: Number of spin tail blocks for each interaction group
    /// - `sampler_state`: Current sampler state (will be updated)
    /// - `output_spec`: Expected output shape/dtype
    /// - `device`: GPU device
    ///
    /// # Returns
    ///
    /// Tuple of (sampled state, updated sampler state)
    #[allow(clippy::too_many_arguments)]
    fn sample(
        &self,
        key: crate::rng::RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        sampler_state: Self::SamplerState,
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, Self::SamplerState);
}

/// Type-erased wrapper for conditional samplers to allow heterogeneous collections.
///
/// Since samplers have different `SamplerState` types, we can't store them directly
/// in a `Vec<Box<dyn AbstractConditionalSampler>>`. This wrapper erases the state type.
pub trait DynConditionalSampler: Send + Sync {
    /// Sample without explicit state (state managed internally).
    #[allow(clippy::too_many_arguments)]
    fn sample_stateless(
        &self,
        key: crate::rng::RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;
}

/// Blanket implementation for stateless samplers (SamplerState = ())
impl<T: AbstractConditionalSampler<SamplerState = ()> + Send + Sync> DynConditionalSampler for T {
    fn sample_stateless(
        &self,
        key: crate::rng::RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let (result, _) = self.sample(
            key,
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction,
            (),
            output_spec,
            device,
        );
        result
    }
}
