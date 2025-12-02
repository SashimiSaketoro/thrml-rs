//! Factors for continuous-valued random variables.
//!
//! This module provides factors for Gaussian PGMs and other continuous variable models:
//!
//! - [`LinearFactor`]: Linear bias factor `w * x`
//! - [`QuadraticFactor`]: Quadratic self-interaction `w * x^2` (variance term)
//! - [`CouplingFactor`]: Pairwise coupling `w * x_i * x_j`
//!
//! These factors are used to construct Gaussian Markov Random Fields (GMRFs)
//! and mixed continuous-discrete EBMs.

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::interaction::{InteractionData, InteractionGroup};

use crate::factor::{AbstractFactor, FactorInteractionGroup};
use crate::DiscreteEBMInteraction;

/// A linear factor of the form `w * x`.
///
/// This represents a linear bias term in the energy function:
/// E(x) = -w^T * x
///
/// In a Gaussian model, this corresponds to the mean parameter.
pub struct LinearFactor {
    /// Linear weights with shape `[n_nodes]`
    weights: Tensor<WgpuBackend, 1>,
    /// Block of continuous nodes
    block: Block,
    /// Node groups (for AbstractFactor trait)
    node_groups: Vec<Block>,
}

impl LinearFactor {
    /// Create a new linear factor.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight tensor with shape `[n_nodes]`
    /// * `block` - Block of continuous nodes this factor applies to
    pub fn new(weights: Tensor<WgpuBackend, 1>, block: Block) -> Self {
        let node_groups = vec![block.clone()];
        Self {
            weights,
            block,
            node_groups,
        }
    }

    /// Create an InteractionGroup for this factor.
    ///
    /// Returns an InteractionGroup with Linear interaction data.
    pub fn to_interaction_group(&self) -> InteractionGroup {
        let n_nodes = self.block.len();

        // Reshape weights to [n_nodes, 1] for consistency with InteractionGroup format
        let weights_2d: Tensor<WgpuBackend, 2> = self.weights.clone().reshape([n_nodes as i32, 1]);

        InteractionGroup::with_data(
            InteractionData::Linear {
                weights: weights_2d,
            },
            self.block.clone(),
            vec![], // No tail nodes for bias term
            0,      // No spin tail blocks
        )
        .expect("LinearFactor should create valid InteractionGroup")
    }
}

impl AbstractFactor for LinearFactor {
    fn node_groups(&self) -> &[Block] {
        &self.node_groups
    }

    fn to_interaction_groups(
        &self,
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        // For compatibility with the AbstractFactor trait, we return FactorInteractionGroup
        // but this is meant for discrete EBMs. For continuous factors, use to_interaction_group().
        // Here we create a placeholder that will work with the factor system.
        let n_nodes = self.block.len();

        // Create a 3D tensor [n_nodes, 1, 1] for compatibility
        let weights_3d: Tensor<WgpuBackend, 3> =
            self.weights.clone().reshape([n_nodes as i32, 1, 1]);

        vec![FactorInteractionGroup::new(
            DiscreteEBMInteraction::new(0, weights_3d),
            self.block.clone(),
            vec![],
        )
        .expect("LinearFactor should create valid FactorInteractionGroup")]
    }
}

/// A quadratic factor of the form `w * x^2`.
///
/// This represents the diagonal of the inverse covariance (precision) matrix
/// in a Gaussian model:
/// E(x) = (1/2) * sum_i (1/sigma_i^2) * x_i^2
///
/// The `inverse_weights` represent 1/sigma^2 (precision/inverse variance).
pub struct QuadraticFactor {
    /// Inverse variance weights with shape `[n_nodes]`
    /// These are the diagonal of the precision matrix.
    inverse_weights: Tensor<WgpuBackend, 1>,
    /// Block of continuous nodes
    block: Block,
    /// Node groups (for AbstractFactor trait)
    node_groups: Vec<Block>,
}

impl QuadraticFactor {
    /// Create a new quadratic factor.
    ///
    /// # Arguments
    ///
    /// * `inverse_weights` - Inverse variance tensor with shape `[n_nodes]`
    /// * `block` - Block of continuous nodes this factor applies to
    pub fn new(inverse_weights: Tensor<WgpuBackend, 1>, block: Block) -> Self {
        let node_groups = vec![block.clone()];
        Self {
            inverse_weights,
            block,
            node_groups,
        }
    }

    /// Create an InteractionGroup for this factor.
    ///
    /// Returns an InteractionGroup with Quadratic interaction data.
    pub fn to_interaction_group(&self) -> InteractionGroup {
        let n_nodes = self.block.len();

        // Reshape inverse_weights to [n_nodes, 1] for consistency
        let inverse_weights_2d: Tensor<WgpuBackend, 2> =
            self.inverse_weights.clone().reshape([n_nodes as i32, 1]);

        InteractionGroup::with_data(
            InteractionData::Quadratic {
                inverse_weights: inverse_weights_2d,
            },
            self.block.clone(),
            vec![], // No tail nodes for self-interaction
            0,      // No spin tail blocks
        )
        .expect("QuadraticFactor should create valid InteractionGroup")
    }
}

impl AbstractFactor for QuadraticFactor {
    fn node_groups(&self) -> &[Block] {
        &self.node_groups
    }

    fn to_interaction_groups(
        &self,
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        // For compatibility with the AbstractFactor trait
        let n_nodes = self.block.len();

        // Create a 3D tensor [n_nodes, 1, 1] for compatibility
        let weights_3d: Tensor<WgpuBackend, 3> =
            self.inverse_weights.clone().reshape([n_nodes as i32, 1, 1]);

        vec![FactorInteractionGroup::new(
            DiscreteEBMInteraction::new(0, weights_3d),
            self.block.clone(),
            vec![],
        )
        .expect("QuadraticFactor should create valid FactorInteractionGroup")]
    }
}

/// A coupling factor of the form `w * x_i * x_j`.
///
/// This represents off-diagonal elements of the precision matrix
/// in a Gaussian model, creating pairwise correlations between variables.
///
/// E(x) = -sum_{i,j} w_{ij} * x_i * x_j
pub struct CouplingFactor {
    /// Coupling weights with shape `[n_edges]`
    weights: Tensor<WgpuBackend, 1>,
    /// Block of source nodes (x_i)
    block_i: Block,
    /// Block of target nodes (x_j)
    block_j: Block,
    /// Node groups (for AbstractFactor trait)
    node_groups: Vec<Block>,
}

impl CouplingFactor {
    /// Create a new coupling factor.
    ///
    /// # Arguments
    ///
    /// * `weights` - Coupling weights with shape `[n_edges]`
    /// * `block_i` - Block of source continuous nodes
    /// * `block_j` - Block of target continuous nodes
    ///
    /// Note: `block_i` and `block_j` must have the same length.
    pub fn new(
        weights: Tensor<WgpuBackend, 1>,
        block_i: Block,
        block_j: Block,
    ) -> Result<Self, String> {
        if block_i.len() != block_j.len() {
            return Err(format!(
                "block_i and block_j must have the same length: {} vs {}",
                block_i.len(),
                block_j.len()
            ));
        }
        let weights_len = weights.dims()[0];
        if weights_len != block_i.len() {
            return Err(format!(
                "weights length {} must match block length {}",
                weights_len,
                block_i.len()
            ));
        }
        let node_groups = vec![block_i.clone(), block_j.clone()];
        Ok(Self {
            weights,
            block_i,
            block_j,
            node_groups,
        })
    }

    /// Create InteractionGroups for this coupling factor.
    ///
    /// Returns two InteractionGroups:
    /// - One with block_i as head and block_j as tail
    /// - One with block_j as head and block_i as tail
    ///
    /// This creates symmetric interactions for undirected edges.
    pub fn to_interaction_groups(&self) -> Vec<InteractionGroup> {
        let n_edges = self.block_i.len();

        // Reshape weights to [n_edges, 1] for Linear interaction format
        let weights_2d: Tensor<WgpuBackend, 2> = self.weights.clone().reshape([n_edges as i32, 1]);

        // Create interaction group: block_i -> block_j
        let ig1 = InteractionGroup::with_data(
            InteractionData::Linear {
                weights: weights_2d.clone(),
            },
            self.block_i.clone(),
            vec![self.block_j.clone()],
            0, // No spin tail blocks
        )
        .expect("CouplingFactor should create valid InteractionGroup");

        // Create symmetric interaction: block_j -> block_i
        let ig2 = InteractionGroup::with_data(
            InteractionData::Linear {
                weights: weights_2d,
            },
            self.block_j.clone(),
            vec![self.block_i.clone()],
            0, // No spin tail blocks
        )
        .expect("CouplingFactor should create valid InteractionGroup");

        vec![ig1, ig2]
    }
}

impl AbstractFactor for CouplingFactor {
    fn node_groups(&self) -> &[Block] {
        &self.node_groups
    }

    fn to_interaction_groups(
        &self,
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        // For compatibility with the AbstractFactor trait
        let n_edges = self.block_i.len();

        // Create 3D tensors [n_edges, 1, 1] for compatibility
        let weights_3d: Tensor<WgpuBackend, 3> =
            self.weights.clone().reshape([n_edges as i32, 1, 1]);

        // Create both directions of the coupling
        vec![
            FactorInteractionGroup::new(
                DiscreteEBMInteraction::new(0, weights_3d.clone()),
                self.block_i.clone(),
                vec![self.block_j.clone()],
            )
            .expect("CouplingFactor should create valid FactorInteractionGroup"),
            FactorInteractionGroup::new(
                DiscreteEBMInteraction::new(0, weights_3d),
                self.block_j.clone(),
                vec![self.block_i.clone()],
            )
            .expect("CouplingFactor should create valid FactorInteractionGroup"),
        ]
    }
}

/// Create a Gaussian MRF factor system from a precision matrix.
///
/// This creates factors for a Gaussian distribution:
/// p(x) ∝ exp(-0.5 * x^T * A * x + b^T * x)
///
/// where A is the precision matrix and b is the bias vector.
///
/// # Arguments
///
/// * `precision_diag` - Diagonal elements of the precision matrix `[n_nodes]`
/// * `precision_off_diag` - Off-diagonal precision values `[n_edges]`
/// * `edge_i` - Source node indices for each edge `[n_edges]`
/// * `edge_j` - Target node indices for each edge `[n_edges]`
/// * `bias` - Bias vector (mean contribution) `[n_nodes]`
/// * `block` - Block of continuous nodes
///
/// # Returns
///
/// Tuple of (quadratic_factor, linear_factor, coupling_factors)
pub fn create_gaussian_factors(
    precision_diag: Tensor<WgpuBackend, 1>,
    precision_off_diag: Tensor<WgpuBackend, 1>,
    edge_blocks_i: Block,
    edge_blocks_j: Block,
    bias: Tensor<WgpuBackend, 1>,
    block: Block,
) -> Result<(QuadraticFactor, LinearFactor, CouplingFactor), String> {
    // Create quadratic factor for diagonal (variance terms)
    let quadratic = QuadraticFactor::new(precision_diag, block.clone());

    // Create linear factor for bias
    let linear = LinearFactor::new(bias, block);

    // Create coupling factor for off-diagonal elements
    let coupling = CouplingFactor::new(precision_off_diag, edge_blocks_i, edge_blocks_j)?;

    Ok((quadratic, linear, coupling))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_linear_factor() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};
        use thrml_core::node::{Node, NodeType};

        ensure_backend();
        let device = init_gpu_device();

        // Create continuous nodes
        let nodes: Vec<Node> = (0..5).map(|_| Node::new(NodeType::Continuous)).collect();
        let block = Block::new(nodes).unwrap();

        // Create linear factor
        let weights: Tensor<WgpuBackend, 1> =
            Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0], &device);
        let factor = LinearFactor::new(weights, block);

        // Convert to interaction group
        let ig = factor.to_interaction_group();
        assert_eq!(ig.head_nodes.len(), 5);
        assert!(ig.tail_nodes.is_empty());
        assert!(ig.interaction.is_linear());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_quadratic_factor() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};
        use thrml_core::node::{Node, NodeType};

        ensure_backend();
        let device = init_gpu_device();

        // Create continuous nodes
        let nodes: Vec<Node> = (0..5).map(|_| Node::new(NodeType::Continuous)).collect();
        let block = Block::new(nodes).unwrap();

        // Create quadratic factor (inverse variance)
        let inverse_weights: Tensor<WgpuBackend, 1> =
            Tensor::from_data([0.5f32, 1.0, 1.5, 2.0, 2.5], &device);
        let factor = QuadraticFactor::new(inverse_weights, block);

        // Convert to interaction group
        let ig = factor.to_interaction_group();
        assert_eq!(ig.head_nodes.len(), 5);
        assert!(ig.tail_nodes.is_empty());
        assert!(ig.interaction.is_quadratic());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_coupling_factor() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};
        use thrml_core::node::{Node, NodeType};

        ensure_backend();
        let device = init_gpu_device();

        // Create two blocks of continuous nodes (for edges)
        let nodes_i: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Continuous)).collect();
        let nodes_j: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Continuous)).collect();
        let block_i = Block::new(nodes_i).unwrap();
        let block_j = Block::new(nodes_j).unwrap();

        // Create coupling factor
        let weights: Tensor<WgpuBackend, 1> = Tensor::from_data([0.1f32, 0.2, 0.3], &device);
        let factor = CouplingFactor::new(weights, block_i, block_j).unwrap();

        // Convert to interaction groups (should get 2: i->j and j->i)
        let igs = factor.to_interaction_groups();
        assert_eq!(igs.len(), 2);

        // Check first group (i -> j)
        assert_eq!(igs[0].head_nodes.len(), 3);
        assert_eq!(igs[0].tail_nodes.len(), 1);
        assert!(igs[0].interaction.is_linear());

        // Check second group (j -> i)
        assert_eq!(igs[1].head_nodes.len(), 3);
        assert_eq!(igs[1].tail_nodes.len(), 1);
        assert!(igs[1].interaction.is_linear());
    }
}
