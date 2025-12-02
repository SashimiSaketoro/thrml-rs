use burn::tensor::Tensor;
use indexmap::IndexMap;
use std::collections::HashMap;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::blockspec::BlockSpec;
use thrml_core::interaction::InteractionGroup;
use thrml_core::node::{Node, NodeType, TensorSpec};
use thrml_core::state_tree::block_state_to_global;

/// A collection of blocks sampled at the same "time".
///
/// Blocks in a SuperBlock are sampled separately but without updating state
/// in between (same algorithmic time, different computation time).
pub type SuperBlock = Vec<Block>;

/// A BlockGibbsSpec is a type of BlockSpec which contains additional information
/// on free and clamped blocks.
pub struct BlockGibbsSpec {
    pub spec: BlockSpec,
    pub free_blocks: Vec<Block>,
    pub sampling_order: Vec<Vec<usize>>,
    pub clamped_blocks: Vec<Block>,
    pub superblocks: Vec<Vec<Block>>,
}

impl BlockGibbsSpec {
    pub fn new(
        free_super_blocks: Vec<SuperBlock>,
        clamped_blocks: Vec<Block>,
        node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
    ) -> Result<Self, String> {
        let mut free_blocks = Vec::new();
        let mut sampling_order = Vec::new();
        let mut superblocks = Vec::new();
        let mut i = 0;

        for super_block in free_super_blocks {
            superblocks.push(super_block.clone());
            let mut sampling_group = Vec::new();
            for block in &super_block {
                free_blocks.push(block.clone());
                sampling_group.push(i);
                i += 1;
            }
            sampling_order.push(sampling_group);
        }

        let all_blocks: Vec<Block> = free_blocks
            .iter()
            .chain(clamped_blocks.iter())
            .cloned()
            .collect();

        let spec = BlockSpec::new(all_blocks, node_shape_dtypes)?;

        Ok(Self {
            spec,
            free_blocks,
            sampling_order,
            clamped_blocks,
            superblocks,
        })
    }
}

use thrml_core::interaction::InteractionData;

/// A PGM block-sampling program.
pub struct BlockSamplingProgram {
    pub gibbs_spec: BlockGibbsSpec,
    pub samplers: Vec<Box<dyn crate::sampler::DynConditionalSampler>>,
    // Store indices as GPU tensors for direct gather operations
    pub per_block_interaction_global_inds: Vec<Vec<Vec<usize>>>,
    pub per_block_interaction_global_slices:
        Vec<Vec<Vec<Tensor<WgpuBackend, 2, burn::tensor::Int>>>>,
    // Sliced interactions stored as InteractionData
    // - Tensor: 3D tensor [n_nodes, n_interactions, tail_dim]
    // - Linear: 2D tensor [n_nodes, n_interactions]
    // - Quadratic: 2D tensor [n_nodes, n_interactions]
    pub per_block_interactions: Vec<Vec<InteractionData>>,
    pub per_block_interaction_active: Vec<Vec<Tensor<WgpuBackend, 2>>>,
    // Number of spin tail blocks per interaction, for each block
    pub per_block_n_spin: Vec<Vec<usize>>,
}

impl BlockSamplingProgram {
    pub fn new(
        gibbs_spec: BlockGibbsSpec,
        samplers: Vec<Box<dyn crate::sampler::DynConditionalSampler>>,
        interaction_groups: Vec<InteractionGroup>,
    ) -> Result<Self, String> {
        if samplers.len() != gibbs_spec.free_blocks.len() {
            return Err(format!(
                "Number of samplers ({}) must match number of free blocks ({})",
                samplers.len(),
                gibbs_spec.free_blocks.len()
            ));
        }

        // First, construct a map from every head node to each interaction it shows up in
        let mut head_node_map: HashMap<Node, Vec<(usize, usize)>> = HashMap::new();

        for (i, interaction_group) in interaction_groups.iter().enumerate() {
            for (j, node) in interaction_group.head_nodes.nodes().iter().enumerate() {
                head_node_map.entry(node.clone()).or_default().push((i, j));
            }
        }

        // Organize interaction information into block format
        let mut interaction_inds: Vec<Vec<Vec<Vec<usize>>>> = Vec::new();
        let mut max_n_interactions: Vec<Vec<usize>> = Vec::new();

        for block in &gibbs_spec.free_blocks {
            let mut this_block_interaction_info: Vec<Vec<Vec<usize>>> =
                vec![vec![Vec::new(); block.len()]; interaction_groups.len()];

            for (j, node) in block.nodes().iter().enumerate() {
                if let Some(this_node_interaction_info) = head_node_map.get(node) {
                    for (interaction_idx, position_in_interaction) in this_node_interaction_info {
                        this_block_interaction_info[*interaction_idx][j]
                            .push(*position_in_interaction);
                    }
                }
            }

            let this_max_n: Vec<usize> = this_block_interaction_info
                .iter()
                .map(|this_int| this_int.iter().map(|x| x.len()).max().unwrap_or(0))
                .collect();

            interaction_inds.push(this_block_interaction_info);
            max_n_interactions.push(this_max_n);
        }

        // Construct the block-arranged interactions and slicers for the global state
        // Sliced interactions are stored as InteractionData
        let mut per_block_interactions: Vec<Vec<InteractionData>> = Vec::new();
        let mut per_block_interaction_active: Vec<Vec<Tensor<WgpuBackend, 2>>> = Vec::new();
        let mut per_block_interaction_global_inds: Vec<Vec<Vec<usize>>> = Vec::new();
        let mut per_block_interaction_global_slices: Vec<
            Vec<Vec<Tensor<WgpuBackend, 2, burn::tensor::Int>>>,
        > = Vec::new();
        let mut per_block_n_spin: Vec<Vec<usize>> = Vec::new();

        let device = burn::backend::wgpu::WgpuDevice::default();

        for (block, block_interact_inds, block_n_interactions) in itertools::izip!(
            &gibbs_spec.free_blocks,
            &interaction_inds,
            &max_n_interactions
        ) {
            let mut this_block_interactions = Vec::new();
            let mut this_block_active = Vec::new();
            let mut this_block_global_inds = Vec::new();
            let mut this_block_global_slices = Vec::new();
            let mut this_block_n_spin = Vec::new();

            for (interaction_group, interact_inds, n_interactions) in itertools::izip!(
                interaction_groups.iter(),
                block_interact_inds.iter(),
                block_n_interactions.iter()
            ) {
                if *n_interactions > 0 {
                    let n_nodes = block.len();

                    // Build interaction_slices: (n_nodes, n_interactions) array of indices
                    let mut interaction_slices_data = vec![0i32; n_nodes * n_interactions];
                    let mut active_data = vec![false; n_nodes * n_interactions];

                    // Build global_inds and global_slices for each tail block
                    let mut global_inds = Vec::new();
                    let mut global_slices_data: Vec<Vec<i32>> = Vec::new();

                    for tail_block in &interaction_group.tail_nodes {
                        let (sd_ind, _) = gibbs_spec.spec.get_node_locations(tail_block)?;
                        global_inds.push(sd_ind);
                        global_slices_data.push(vec![0i32; n_nodes * n_interactions]);
                    }

                    // Fill in the slices and active flags
                    for (i, inds) in interact_inds.iter().enumerate() {
                        for (j, ind) in inds.iter().enumerate() {
                            interaction_slices_data[i * n_interactions + j] = *ind as i32;
                            active_data[i * n_interactions + j] = true;

                            // Fill global slices for each tail block
                            for (k, tail_block) in interaction_group.tail_nodes.iter().enumerate() {
                                let node = &tail_block.nodes()[*ind];
                                let (_, pos) = gibbs_spec
                                    .spec
                                    .node_global_location_map
                                    .get(node)
                                    .ok_or_else(|| {
                                    "Node not found in global location map".to_string()
                                })?;
                                global_slices_data[k][i * n_interactions + j] = *pos as i32;
                            }
                        }
                    }

                    // Create interaction_slices tensor (2D Int tensor)
                    // from_data infers shape from data, so we need to reshape after creation
                    let interaction_slices_1d: Tensor<WgpuBackend, 1, burn::tensor::Int> =
                        Tensor::from_data(interaction_slices_data.as_slice(), &device);
                    let interaction_slices =
                        interaction_slices_1d.reshape([n_nodes as i32, *n_interactions as i32]);

                    // Slice the interaction data using interaction_slices
                    // The Python code uses jnp.take(x, sl, axis=0) where sl is 2D (n_nodes, n_interactions)
                    // This extracts elements along axis 0 using the 2D index array
                    let flat_indices = interaction_slices
                        .clone()
                        .reshape([(n_nodes * n_interactions) as i32]);

                    let sliced_interaction: InteractionData = match &interaction_group.interaction {
                        InteractionData::Tensor(tensor) => {
                            // Standard 3D tensor [head_nodes, dim1, dim2]
                            let interaction_dims = tensor.dims();
                            let dim1 = interaction_dims[1];
                            let dim2 = interaction_dims[2];

                            // Gather from interaction tensor along first dimension
                            // This gives us [n_nodes * n_interactions, dim1, dim2] (3D tensor)
                            let gathered: Tensor<WgpuBackend, 3> =
                                tensor.clone().select(0, flat_indices);
                            // Reshape to [n_nodes, n_interactions, dim1*dim2] as a 3D tensor
                            // The sampler will use batch_gather_with_k to index by neighbor categories
                            let sliced: Tensor<WgpuBackend, 3> = gathered.reshape([
                                n_nodes as i32,
                                *n_interactions as i32,
                                (dim1 * dim2) as i32,
                            ]);
                            InteractionData::Tensor(sliced)
                        }
                        InteractionData::Linear { weights } => {
                            // 2D tensor [head_nodes, k]
                            let k = weights.dims()[1];
                            // Gather from weights tensor along first dimension
                            let gathered: Tensor<WgpuBackend, 2> =
                                weights.clone().select(0, flat_indices);
                            // Reshape to [n_nodes, n_interactions * k]
                            let sliced: Tensor<WgpuBackend, 2> =
                                gathered.reshape([n_nodes as i32, (*n_interactions * k) as i32]);
                            InteractionData::Linear { weights: sliced }
                        }
                        InteractionData::Quadratic { inverse_weights } => {
                            // 2D tensor [head_nodes, k]
                            let k = inverse_weights.dims()[1];
                            // Gather from inverse_weights tensor along first dimension
                            let gathered: Tensor<WgpuBackend, 2> =
                                inverse_weights.clone().select(0, flat_indices);
                            // Reshape to [n_nodes, n_interactions * k]
                            let sliced: Tensor<WgpuBackend, 2> =
                                gathered.reshape([n_nodes as i32, (*n_interactions * k) as i32]);
                            InteractionData::Quadratic {
                                inverse_weights: sliced,
                            }
                        }
                        InteractionData::Sphere {
                            ideal_radii,
                            similarity,
                            interaction_radius,
                        } => {
                            // Sphere interactions are for Langevin dynamics, not Gibbs sampling
                            // Pass through unchanged since they shouldn't be used in BlockSamplingProgram
                            InteractionData::Sphere {
                                ideal_radii: ideal_radii.clone(),
                                similarity: similarity.clone(),
                                interaction_radius: *interaction_radius,
                            }
                        }
                    };

                    // Create active tensor (2D Bool tensor converted to Float for compatibility)
                    let active_bool_1d: Tensor<WgpuBackend, 1, burn::tensor::Bool> =
                        Tensor::from_data(active_data.as_slice(), &device);
                    let active_bool =
                        active_bool_1d.reshape([n_nodes as i32, *n_interactions as i32]);
                    let active = active_bool.float();

                    // Create global slices tensors (2D Int tensors)
                    let global_slices: Vec<Tensor<WgpuBackend, 2, burn::tensor::Int>> =
                        global_slices_data
                            .iter()
                            .map(|data| {
                                let tensor_1d: Tensor<WgpuBackend, 1, burn::tensor::Int> =
                                    Tensor::from_data(data.as_slice(), &device);
                                tensor_1d.reshape([n_nodes as i32, *n_interactions as i32])
                            })
                            .collect();

                    // Store the sliced interaction data
                    this_block_interactions.push(sliced_interaction);
                    this_block_active.push(active);
                    this_block_global_inds.push(global_inds);
                    this_block_global_slices.push(global_slices);
                    this_block_n_spin.push(interaction_group.n_spin);
                }
            }

            per_block_interactions.push(this_block_interactions);
            per_block_interaction_active.push(this_block_active);
            per_block_interaction_global_inds.push(this_block_global_inds);
            per_block_interaction_global_slices.push(this_block_global_slices);
            per_block_n_spin.push(this_block_n_spin);
        }

        Ok(Self {
            gibbs_spec,
            samplers,
            per_block_interaction_global_inds,
            per_block_interaction_global_slices,
            per_block_interactions,
            per_block_interaction_active,
            per_block_n_spin,
        })
    }

    /// Sample a single block using the current state.
    ///
    /// This function is generic over StateLeaf, but in practice works with Tensor<WgpuBackend, 1>.
    /// The sampler interface currently expects Tensor types, so we constrain to that.
    pub fn sample_single_block(
        &self,
        block_idx: usize,
        key: crate::rng::RngKey,
        state_free: &[Tensor<WgpuBackend, 1>],
        clamp_state: &[Tensor<WgpuBackend, 1>],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Build global state (concatenate free + clamped)
        let combined_state: Vec<Tensor<WgpuBackend, 1>> = state_free
            .iter()
            .chain(clamp_state.iter())
            .cloned()
            .collect();
        let global_state = block_state_to_global(&combined_state, &self.gibbs_spec.spec);

        // Extract neighbor states using precomputed slices
        let mut all_interaction_states = Vec::new();
        if block_idx < self.per_block_interaction_global_slices.len() {
            for (interaction_global_inds, interaction_slices) in self
                .per_block_interaction_global_inds[block_idx]
                .iter()
                .zip(self.per_block_interaction_global_slices[block_idx].iter())
            {
                let mut this_interaction_states = Vec::new();
                for (ind, sl) in interaction_global_inds
                    .iter()
                    .zip(interaction_slices.iter())
                {
                    // Use the 2D slice tensor to gather from global state
                    // The slice tensor has shape [n_nodes, n_interactions]
                    // We need to flatten it and use it to gather from global_state[*ind]
                    let flat_slice = sl.clone().reshape([sl.dims()[0] * sl.dims()[1]]);
                    let gathered = global_state[*ind].clone().select(0, flat_slice);
                    // Reshape back to match expected shape [n_nodes, n_interactions]
                    let n_nodes = sl.dims()[0];
                    let n_interactions = sl.dims()[1];
                    let gathered_2d = gathered.reshape([n_nodes, n_interactions]);
                    this_interaction_states.push(gathered_2d);
                }
                all_interaction_states.push(this_interaction_states);
            }
        }

        // Get output spec for this block
        let this_block = &self.gibbs_spec.free_blocks[block_idx];
        let node_type = this_block.node_type();
        let output_spec = self
            .gibbs_spec
            .spec
            .node_shape_dtypes
            .get(node_type)
            .expect("Node type not found in node_shape_dtypes");

        // Resize spec to match block length
        let mut resized_spec = output_spec.clone();
        resized_spec.shape = vec![this_block.len()];

        // Call sampler with n_spin metadata for each interaction
        let sampler = &self.samplers[block_idx];
        sampler.sample_stateless(
            key,
            &self.per_block_interactions[block_idx],
            &self.per_block_interaction_active[block_idx],
            &all_interaction_states,
            &self.per_block_n_spin[block_idx],
            &resized_spec,
            device,
        )
    }
}
