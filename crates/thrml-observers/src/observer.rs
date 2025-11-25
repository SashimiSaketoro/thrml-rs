use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::blockspec::BlockSpec;

/// Interface for objects that inspect the sampling program while it is running.
///
/// A concrete Observer is called once per block-sampling iteration and can maintain an
/// arbitrary "carry" state across calls (e.g. running averages, histogram buffers, etc.).
pub trait AbstractObserver {
    /// The type of the carry state maintained by this observer.
    type ObserveCarry: Clone;

    /// Make an observation.
    ///
    /// This function is called at the end of a block-sampling iteration and can record information about the
    /// current state of the sampling program that might be useful for something later.
    ///
    /// # Arguments
    ///
    /// * `gibbs_spec` - The BlockGibbsSpec from the sampling program (contains block structure and mappings)
    /// * `state_free` - The current state of the free nodes involved in the sampling program.
    /// * `state_clamped` - The state of the clamped nodes involved in the sampling program.
    /// * `carry` - The "memory" available to this observer. This function should modify this to record
    ///   information about the sampling program.
    /// * `iteration` - How many iterations of block sampling have happened before this function was called.
    ///
    /// # Returns
    ///
    /// A tuple, where the first element is the updated carry, and the second is an optional observation
    /// that will be recorded by the sampler. If None, nothing is recorded for this iteration.
    fn observe(
        &self,
        gibbs_spec: &BlockSpec,
        state_free: &[Tensor<WgpuBackend, 1>],
        state_clamped: &[Tensor<WgpuBackend, 1>],
        carry: Self::ObserveCarry,
        iteration: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Self::ObserveCarry, Option<Vec<Tensor<WgpuBackend, 1>>>);

    /// Initialize the memory for the observer.
    fn init(&self, device: &burn::backend::wgpu::WgpuDevice) -> Self::ObserveCarry;
}
