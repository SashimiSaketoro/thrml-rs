//! MomentAccumulatorObserver - Observer that accumulates moment statistics.
//!
//! This observer computes running sums of products of state variables:
//! Σ_i f(x_1^i) * f(x_2^i) * ... * f(x_N^i)
//!
//! where f is a transformation function applied to the state values.

use crate::observer::AbstractObserver;
use burn::tensor::Tensor;
use std::collections::HashMap;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::blockspec::BlockSpec;
use thrml_core::node::Node;
use thrml_core::state_tree::{block_state_to_global, from_global_state};

/// Moment specification: defines which nodes to compute moments for.
///
/// Structure: `Vec<Vec<Vec<Node>>>`
/// - Outer Vec: different moment types (e.g., first moments, second moments)
/// - Middle Vec: groups within a moment type (e.g., individual nodes for first moments)
/// - Inner Vec: nodes involved in each moment computation
pub type MomentSpec = Vec<Vec<Vec<Node>>>;

/// Observer that accumulates and updates provided moments.
///
/// This observer accumulates a running sum of products of state variables,
/// but does not scale by the number of samples (caller must divide by n_samples).
pub struct MomentAccumulatorObserver {
    /// Blocks to sample from (one per node type)
    pub blocks_to_sample: Vec<Block>,
    /// Flat list of all nodes in the moments (each occurring only once)
    pub flat_nodes_list: Vec<Node>,
    /// For each node type, indices into flat_nodes_list
    pub flat_to_type_slices_list: Vec<Vec<usize>>,
    /// For each moment type, 2D array of indices into flat_nodes_list
    pub flat_to_full_moment_slices: Vec<Vec<Vec<usize>>>,
    /// Transform function type (for spin: converts bool to ±1)
    pub transform_to_spin: bool,
}

impl MomentAccumulatorObserver {
    /// Create a new MomentAccumulatorObserver.
    ///
    /// # Arguments
    ///
    /// * `moment_spec` - A 3-depth sequence defining which moments to compute.
    ///   First level: different moment types (e.g., first vs second moments)
    ///   Second level: groups within a moment type
    ///   Third level: nodes involved in each moment
    /// * `transform_to_spin` - If true, transform bool values to ±1 before computing moments
    pub fn new(moment_spec: MomentSpec, transform_to_spin: bool) -> Self {
        let mut flat_nodes_list = Vec::new();
        let mut node_to_flat_idx: HashMap<Node, usize> = HashMap::new();
        let mut flat_to_full_moment_slices = Vec::new();
        let mut nodes_by_type: HashMap<String, Vec<Node>> = HashMap::new();
        let mut flat_to_type_slices: HashMap<String, Vec<usize>> = HashMap::new();

        for moment in &moment_spec {
            // moment = list of "rows" => each row is a list of nodes
            let mut moment_slice: Vec<Vec<usize>> = Vec::new();

            for nodes in moment {
                let mut row_slice: Vec<usize> = Vec::new();

                for node in nodes {
                    // Get or assign index
                    let idx = if let Some(&existing_idx) = node_to_flat_idx.get(node) {
                        existing_idx
                    } else {
                        let new_idx = flat_nodes_list.len();
                        node_to_flat_idx.insert(node.clone(), new_idx);
                        flat_nodes_list.push(node.clone());
                        new_idx
                    };

                    row_slice.push(idx);

                    // Track by node type
                    let type_key = format!("{:?}", node.node_type());
                    nodes_by_type
                        .entry(type_key.clone())
                        .or_default()
                        .push(node.clone());
                    flat_to_type_slices.entry(type_key).or_default().push(idx);
                }

                moment_slice.push(row_slice);
            }

            flat_to_full_moment_slices.push(moment_slice);
        }

        // Build blocks to sample and type slices
        let mut blocks_to_sample = Vec::new();
        let mut flat_to_type_slices_list = Vec::new();

        for (_, nodes) in nodes_by_type {
            if !nodes.is_empty() {
                if let Ok(block) = Block::new(nodes) {
                    blocks_to_sample.push(block);
                }
            }
        }

        for (_type_key, slices) in flat_to_type_slices {
            flat_to_type_slices_list.push(slices);
        }

        Self {
            blocks_to_sample,
            flat_nodes_list,
            flat_to_type_slices_list,
            flat_to_full_moment_slices,
            transform_to_spin,
        }
    }

    /// Create a moment spec for first and second moments of an Ising model.
    ///
    /// # Arguments
    ///
    /// * `first_moment_nodes` - Nodes for first moment computation
    /// * `second_moment_edges` - Node pairs for second moment computation
    pub fn ising_moment_spec(
        first_moment_nodes: &[Node],
        second_moment_edges: &[(Node, Node)],
    ) -> MomentSpec {
        let mut spec = Vec::new();

        // First moments: each node separately
        if !first_moment_nodes.is_empty() {
            let first_moments: Vec<Vec<Node>> =
                first_moment_nodes.iter().map(|n| vec![n.clone()]).collect();
            spec.push(first_moments);
        }

        // Second moments: pairs of nodes
        if !second_moment_edges.is_empty() {
            let second_moments: Vec<Vec<Node>> = second_moment_edges
                .iter()
                .map(|(n1, n2)| vec![n1.clone(), n2.clone()])
                .collect();
            spec.push(second_moments);
        }

        spec
    }
}

/// Carry type for MomentAccumulatorObserver: accumulated sums for each moment type
pub type MomentCarry = Vec<Tensor<WgpuBackend, 1>>;

impl AbstractObserver for MomentAccumulatorObserver {
    type ObserveCarry = MomentCarry;

    fn observe(
        &self,
        spec: &BlockSpec,
        state_free: &[Tensor<WgpuBackend, 1>],
        state_clamped: &[Tensor<WgpuBackend, 1>],
        carry: MomentCarry,
        _iteration: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (MomentCarry, Option<Vec<Tensor<WgpuBackend, 1>>>) {
        // Combine free and clamped states
        let combined_state: Vec<Tensor<WgpuBackend, 1>> = state_free
            .iter()
            .chain(state_clamped.iter())
            .cloned()
            .collect();

        // Convert to global state
        let global_state = block_state_to_global(&combined_state, spec);

        // Extract states for our blocks
        let sampled_state = from_global_state(&global_state, spec, &self.blocks_to_sample, device);

        // Apply transformation if needed (for spin: bool -> ±1)
        let transformed_state: Vec<Tensor<WgpuBackend, 1>> = if self.transform_to_spin {
            sampled_state
                .iter()
                .map(|s| s.clone() * 2.0 - 1.0)
                .collect()
        } else {
            sampled_state
        };

        // Build flat state tensor
        let n_flat = self.flat_nodes_list.len();
        let mut flat_data = vec![0.0f32; n_flat];

        for (type_idx, type_slice) in self.flat_to_type_slices_list.iter().enumerate() {
            if type_idx < transformed_state.len() {
                let state_data: Vec<f32> = transformed_state[type_idx]
                    .clone()
                    .into_data()
                    .to_vec()
                    .expect("read state data");

                for (i, &flat_idx) in type_slice.iter().enumerate() {
                    if i < state_data.len() && flat_idx < n_flat {
                        flat_data[flat_idx] = state_data[i];
                    }
                }
            }
        }

        // Accumulate moments
        let mut new_carry = carry;

        for (moment_idx, moment_slices) in self.flat_to_full_moment_slices.iter().enumerate() {
            // Compute product for each group in this moment type
            let mut updates = Vec::new();

            for group_slices in moment_slices {
                // Compute product of all values in this group
                let mut product = 1.0f32;
                for &flat_idx in group_slices {
                    if flat_idx < flat_data.len() {
                        product *= flat_data[flat_idx];
                    }
                }
                updates.push(product);
            }

            // Add to carry
            if moment_idx < new_carry.len() {
                let update_tensor: Tensor<WgpuBackend, 1> =
                    Tensor::from_data(updates.as_slice(), device);
                new_carry[moment_idx] = new_carry[moment_idx].clone() + update_tensor;
            }
        }

        // MomentAccumulatorObserver doesn't return observations, only updates carry
        (new_carry, None)
    }

    fn init(&self, device: &burn::backend::wgpu::WgpuDevice) -> MomentCarry {
        // Initialize carry with zeros for each moment type
        self.flat_to_full_moment_slices
            .iter()
            .map(|moment_slices| {
                let n_groups = moment_slices.len();
                Tensor::<WgpuBackend, 1>::zeros([n_groups], device)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrml_core::node::NodeType;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_moment_observer_creation() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};

        ensure_backend();
        let _device = init_gpu_device();

        // Create some test nodes
        let node1 = Node::new(NodeType::Spin);
        let node2 = Node::new(NodeType::Spin);

        // Create moment spec for first and second moments
        let spec = MomentAccumulatorObserver::ising_moment_spec(
            &[node1.clone(), node2.clone()],
            &[(node1, node2)],
        );

        let observer = MomentAccumulatorObserver::new(spec, true);

        assert_eq!(
            observer.flat_nodes_list.len(),
            2,
            "Should have 2 unique nodes"
        );
        assert_eq!(
            observer.flat_to_full_moment_slices.len(),
            2,
            "Should have 2 moment types"
        );
    }
}
