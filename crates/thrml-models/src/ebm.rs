use crate::factor::AbstractFactor;
/// Energy-Based Model (EBM) abstractions.
///
/// EBMs define energy functions that map states to scalar values.
/// The Boltzmann distribution P(x) ∝ exp(-E(x)) is defined by the energy function.
use burn::tensor::Tensor;
use indexmap::IndexMap;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::blockspec::BlockSpec;
use thrml_core::node::{NodeType, TensorSpec};
use thrml_core::state_tree::block_state_to_global;

/// Trait for objects that have a well-defined energy function.
///
/// An EBM maps a state to a scalar energy value.
pub trait AbstractEBM {
    /// Evaluate the energy function of the EBM given some state information.
    ///
    /// # Arguments
    ///
    /// * `state` - The state for which to evaluate the energy function
    /// * `blocks` - Specifies how the information in `state` is organized
    /// * `device` - The device for tensor operations
    ///
    /// # Returns
    ///
    /// A scalar representing the energy value associated with `state`
    fn energy(
        &self,
        state: &[Tensor<WgpuBackend, 1>],
        blocks: &[Block],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;
}

/// Batched energy computation trait (vmap-style).
///
/// This trait enables efficient batch processing of energy computations
/// without explicit loops, similar to JAX's vmap.
pub trait BatchedEBM {
    /// Compute energy for a batch of states.
    ///
    /// # Arguments
    /// * `states` - Batched states [batch_size, n_nodes]
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Energy for each state `[batch_size]`
    fn energy_batched(
        &self,
        states: &Tensor<WgpuBackend, 2>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;
}

/// Trait for factors that define an energy function.
///
/// This combines the `AbstractFactor` trait with the ability to compute energy.
pub trait EBMFactor: AbstractFactor {
    /// Evaluate the energy function of the factor.
    ///
    /// # Arguments
    ///
    /// * `global_state` - The state information to use to evaluate the energy function
    /// * `block_spec` - The BlockSpec used to generate `global_state`
    /// * `device` - The device for tensor operations
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;
}

/// Trait for EBMs that are made up of Factors.
///
/// The energy function is: E(x) = Σ_i E^i(x)
/// where the sum is over factors.
pub trait AbstractFactorizedEBM: AbstractEBM {
    /// Get the node shape/dtypes for this EBM.
    fn node_shape_dtypes(&self) -> &IndexMap<NodeType, TensorSpec>;

    /// Get the factors that make up this EBM.
    fn factors(&self, device: &burn::backend::wgpu::WgpuDevice) -> Vec<Box<dyn EBMFactor>>;
}

/// A concrete factorized EBM defined by a list of factors.
pub struct FactorizedEBM {
    /// The factors that define this EBM
    pub factor_list: Vec<Box<dyn EBMFactor>>,
    /// Node shape/dtype specifications
    pub node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
}

impl FactorizedEBM {
    pub fn new(
        factors: Vec<Box<dyn EBMFactor>>,
        node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
    ) -> Self {
        FactorizedEBM {
            factor_list: factors,
            node_shape_dtypes,
        }
    }
}

impl AbstractEBM for FactorizedEBM {
    fn energy(
        &self,
        state: &[Tensor<WgpuBackend, 1>],
        blocks: &[Block],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Build BlockSpec from blocks
        let block_spec = BlockSpec::new(blocks.to_vec(), self.node_shape_dtypes.clone())
            .expect("Failed to create BlockSpec");

        // Convert to global state
        let global_state = block_state_to_global(state, &block_spec);

        // Sum energy from all factors
        let mut total_energy: Tensor<WgpuBackend, 1> = Tensor::zeros([1], device);
        for factor in &self.factor_list {
            let factor_energy = factor.factor_energy(&global_state, &block_spec, device);
            total_energy = total_energy + factor_energy;
        }

        total_energy
    }
}

impl AbstractFactorizedEBM for FactorizedEBM {
    fn node_shape_dtypes(&self) -> &IndexMap<NodeType, TensorSpec> {
        &self.node_shape_dtypes
    }

    fn factors(&self, _device: &burn::backend::wgpu::WgpuDevice) -> Vec<Box<dyn EBMFactor>> {
        // This is tricky since we can't clone Box<dyn EBMFactor> easily
        // For now, return an empty vec - this method is primarily used by IsingEBM
        Vec::new()
    }
}
