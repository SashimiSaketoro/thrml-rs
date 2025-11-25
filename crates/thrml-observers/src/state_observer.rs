use crate::observer::AbstractObserver;
use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::blockspec::BlockSpec;
use thrml_core::state_tree::{block_state_to_global, from_global_state};

/// Observer which logs the raw state of some set of nodes.
///
/// This observer extracts the state of the specified blocks and returns them
/// to be recorded by the sampler.
pub struct StateObserver {
    pub blocks_to_sample: Vec<Block>,
}

impl StateObserver {
    pub fn new(blocks_to_sample: Vec<Block>) -> Self {
        StateObserver { blocks_to_sample }
    }
}

impl AbstractObserver for StateObserver {
    type ObserveCarry = ();

    fn observe(
        &self,
        gibbs_spec: &BlockSpec,
        state_free: &[Tensor<WgpuBackend, 1>],
        state_clamped: &[Tensor<WgpuBackend, 1>],
        _carry: (),
        _iteration: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> ((), Option<Vec<Tensor<WgpuBackend, 1>>>) {
        // Build global state from free and clamped states
        let combined_state: Vec<Tensor<WgpuBackend, 1>> = state_free
            .iter()
            .chain(state_clamped.iter())
            .cloned()
            .collect();
        let global_state = block_state_to_global(&combined_state, gibbs_spec);

        // Extract states for the blocks we want to sample
        let sampled_state =
            from_global_state(&global_state, gibbs_spec, &self.blocks_to_sample, device);

        ((), Some(sampled_state))
    }

    fn init(&self, _device: &burn::backend::wgpu::WgpuDevice) {}
}
