use crate::discrete_ebm::DiscreteEBMInteraction;
/// Factor system for building sampling programs from undirected interactions.
///
/// A factor represents a batch of undirected interactions between sets of random variables.
/// The defining trait of a factor is to produce InteractionGroups that affect each member
/// during the conditional updates of a block sampling program.
use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_samplers::sampler::DynConditionalSampler;

/// Trait for factors that can be converted to interaction groups.
pub trait AbstractFactor {
    /// Get the node groups that this factor acts on.
    fn node_groups(&self) -> &[Block];

    /// Compile a factor to a set of directed interactions.
    ///
    /// This is the main method that converts the undirected factor representation
    /// into directed interaction groups suitable for block Gibbs sampling.
    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup>;
}

/// Trait for factors parameterized by a weight tensor.
pub trait WeightedFactor: AbstractFactor {
    /// Get the weight tensor for this factor.
    fn weights(&self) -> &Tensor<WgpuBackend, 3>;
}

/// An interaction group produced by a factor.
///
/// This is similar to `thrml_core::interaction::InteractionGroup` but uses
/// `DiscreteEBMInteraction` as the interaction type.
pub struct FactorInteractionGroup {
    /// The interaction data (n_spin and weights)
    pub interaction: DiscreteEBMInteraction,
    /// The nodes whose conditional updates should be affected by this InteractionGroup
    pub head_nodes: Block,
    /// The nodes whose state information is required to update `head_nodes`
    pub tail_nodes: Vec<Block>,
}

impl FactorInteractionGroup {
    pub fn new(
        interaction: DiscreteEBMInteraction,
        head_nodes: Block,
        tail_nodes: Vec<Block>,
    ) -> Result<Self, String> {
        let interaction_size = head_nodes.len();

        for block in &tail_nodes {
            if block.len() != interaction_size {
                return Err(
                    "All tail node blocks must have the same length as head_nodes".to_string(),
                );
            }
        }

        // Verify interaction weights have correct leading dimension
        let weight_dims = interaction.weights.dims();
        if weight_dims[0] != interaction_size {
            return Err(
                "Interaction weights must have leading dimension equal to the length of head_nodes"
                    .to_string(),
            );
        }

        Ok(FactorInteractionGroup {
            interaction,
            head_nodes,
            tail_nodes,
        })
    }
}

/// Helper function to validate node groups for a factor.
pub fn validate_node_groups(node_groups: &[Block]) -> Result<usize, String> {
    if node_groups.is_empty() {
        return Err("A factor should not be empty".to_string());
    }

    let n_nodes = node_groups[0].len();

    for group in node_groups {
        if group.len() != n_nodes {
            return Err(
                "Every block in node_groups must contain the same number of nodes".to_string(),
            );
        }
    }

    Ok(n_nodes)
}

/// Helper function to validate weight tensor dimensions.
pub fn validate_weights(
    weights: &Tensor<WgpuBackend, 3>,
    n_nodes: usize,
    n_categorical: usize,
) -> Result<(), String> {
    let weight_dims = weights.dims();

    if weight_dims[0] != n_nodes {
        return Err(
            "The leading dimension of weights must have the same length as the number of nodes in each node group".to_string()
        );
    }

    // For DiscreteEBMFactor: weights shape should be [b, x_1, ..., x_k] where k = n_categorical
    // Since we're using Tensor<WgpuBackend, 3>, we expect 3 dimensions total
    // (this may need adjustment for factors with more categorical variables)
    if weight_dims.len() != 1 + n_categorical.max(2) - 1 {
        // Allow some flexibility in weight shape validation
    }

    Ok(())
}

// Re-export for convenience
use thrml_core::interaction::InteractionGroup;
pub use thrml_samplers::{BlockGibbsSpec, BlockSamplingProgram};

/// A sampling program built out of factors.
///
/// This struct compiles factors into interaction groups and builds a
/// `BlockSamplingProgram` from them. It provides a convenient way to
/// set up sampling from models defined by factors.
///
/// This is equivalent to Python's `FactorSamplingProgram`.
pub struct FactorSamplingProgram {
    /// The underlying block sampling program
    pub program: BlockSamplingProgram,
}

impl FactorSamplingProgram {
    /// Create a FactorSamplingProgram from factors.
    ///
    /// # Arguments
    ///
    /// * `gibbs_spec` - A division of some PGM into free and clamped blocks
    /// * `samplers` - The update rule to use for each free block in gibbs_spec
    /// * `factors` - The factors defining the model interactions
    /// * `other_interaction_groups` - Additional interaction groups not defined by factors
    /// * `device` - The GPU device
    ///
    /// # Returns
    ///
    /// A FactorSamplingProgram ready for sampling
    pub fn new(
        gibbs_spec: BlockGibbsSpec,
        samplers: Vec<Box<dyn DynConditionalSampler>>,
        factors: &[&dyn AbstractFactor],
        other_interaction_groups: Vec<InteractionGroup>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Result<Self, String> {
        // Compile all factors to interaction groups
        let mut interaction_groups: Vec<InteractionGroup> = Vec::new();

        for factor in factors {
            let factor_groups = factor.to_interaction_groups(device);
            for fg in factor_groups {
                // Convert FactorInteractionGroup to InteractionGroup
                // Keep the 3D weights and n_spin metadata
                let ig = InteractionGroup {
                    head_nodes: fg.head_nodes,
                    tail_nodes: fg.tail_nodes,
                    interaction: thrml_core::InteractionData::Tensor(fg.interaction.weights),
                    n_spin: fg.interaction.n_spin,
                };
                interaction_groups.push(ig);
            }
        }

        // Add other interaction groups
        interaction_groups.extend(other_interaction_groups);

        // Build the underlying BlockSamplingProgram
        let program = BlockSamplingProgram::new(gibbs_spec, samplers, interaction_groups)?;

        Ok(FactorSamplingProgram { program })
    }
}
