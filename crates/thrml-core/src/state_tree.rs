use crate::backend::WgpuBackend;
use crate::blockspec::BlockSpec;
use burn::tensor::Tensor;

pub trait StateLeaf: Clone {
    fn stack(leaves: &[Self]) -> Self
    where
        Self: Sized;
    fn take_indices(&self, indices: Tensor<WgpuBackend, 1, burn::tensor::Int>) -> Self;
}

impl<const D: usize> StateLeaf for Tensor<WgpuBackend, D> {
    fn stack(leaves: &[Self]) -> Self {
        if leaves.is_empty() {
            panic!("Cannot stack empty leaves");
        }
        if leaves.len() == 1 {
            return leaves[0].clone();
        }
        // Use Burn's concat operation
        Tensor::cat(leaves.to_vec(), 0)
    }

    fn take_indices(&self, indices: Tensor<WgpuBackend, 1, burn::tensor::Int>) -> Self {
        // Burn's select operation for indexing
        // select takes (dim, indices) where indices is a 1D Int tensor
        // select takes ownership, so we clone first
        self.clone().select(0, indices)
    }
}

pub fn block_state_to_global<L: StateLeaf>(block_state: &[L], spec: &BlockSpec) -> Vec<L> {
    let mut global_state = Vec::new();
    for sd_indexes in &spec.block_to_global_slice_spec {
        if sd_indexes.is_empty() {
            continue; // Skip None equivalent
        }
        let collected: Vec<L> = sd_indexes.iter().map(|&i| block_state[i].clone()).collect();
        if collected.len() == 1 {
            global_state.push(collected[0].clone());
        } else {
            global_state.push(L::stack(&collected));
        }
    }
    global_state
}

pub fn from_global_state<L: StateLeaf>(
    global_state: &[L],
    spec_from: &BlockSpec,
    blocks_to_extract: &[crate::block::Block],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Vec<L> {
    let mut result = Vec::new();
    for block in blocks_to_extract {
        let (sd_ind, slices) = spec_from
            .get_node_locations(block)
            .expect("Failed to get node locations");
        // Convert slices to tensor indices (as Int tensor)
        // Use from_data with proper type annotation for Int tensor
        let indices: Tensor<WgpuBackend, 1, burn::tensor::Int> = Tensor::from_data(
            slices
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );
        let extracted = global_state[sd_ind].take_indices(indices);
        result.push(extracted);
    }
    result
}

pub fn make_empty_block_state(
    blocks: &[crate::block::Block],
    node_shape_dtypes: &indexmap::IndexMap<crate::node::NodeType, crate::node::TensorSpec>,
    batch_shape: Option<&[usize]>,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Vec<Tensor<WgpuBackend, 1>> {
    let mut state = Vec::new();
    for block in blocks {
        let _spec = node_shape_dtypes
            .get(block.node_type())
            .expect("Node type not found in node_shape_dtypes");
        let block_len = block.len();
        let shape: [usize; 1] = if let Some(batch) = batch_shape {
            let total = batch.iter().product::<usize>() * block_len;
            [total]
        } else {
            [block_len]
        };
        // Create zero tensor - dtype will be determined by usage context
        // For now, use f32 as default (will be cast as needed)
        let tensor = Tensor::zeros(shape, device);
        state.push(tensor);
    }
    state
}

/// Verify that a block state matches the expected structure.
///
/// This validates that:
/// - Number of states equals number of blocks
/// - Each state tensor has correct shape for its block
/// - State dtype is compatible with node type
///
/// # Arguments
///
/// * `blocks` - The blocks to validate against
/// * `states` - The state tensors to validate
/// * `node_shape_dtypes` - Expected shape/dtype per node type
/// * `block_axis` - Optional axis index where block length should appear
///
/// # Returns
///
/// `Ok(())` if valid, `Err(message)` if invalid
pub fn verify_block_state(
    blocks: &[crate::block::Block],
    states: &[Tensor<WgpuBackend, 1>],
    node_shape_dtypes: &indexmap::IndexMap<crate::node::NodeType, crate::node::TensorSpec>,
    block_axis: Option<usize>,
) -> Result<(), String> {
    // Check length match
    if blocks.len() != states.len() {
        return Err(format!(
            "Number of states ({}) not equal to number of blocks ({})",
            states.len(),
            blocks.len()
        ));
    }

    // Check each block-state pair
    for (block, state) in blocks.iter().zip(states.iter()) {
        // Get expected spec for this node type
        let expected_spec = node_shape_dtypes.get(block.node_type()).ok_or_else(|| {
            format!(
                "Node type {:?} not found in node_shape_dtypes",
                block.node_type()
            )
        })?;

        let state_dims = state.dims();
        let block_len = block.len();

        // Validate shape
        if let Some(axis) = block_axis {
            // Check that the specified axis has the block length
            if axis >= state_dims.len() {
                return Err(format!(
                    "block_axis {} is out of bounds for state with {} dimensions",
                    axis,
                    state_dims.len()
                ));
            }
            if state_dims[axis] != block_len {
                return Err(format!(
                    "State dimension at axis {} is {}, expected {} (block length)",
                    axis, state_dims[axis], block_len
                ));
            }
        } else {
            // Default: check that total size is compatible
            let total_size: usize = state_dims.iter().product();
            if !total_size.is_multiple_of(block_len) {
                return Err(format!(
                    "State size {} is not compatible with block length {}",
                    total_size, block_len
                ));
            }
        }

        // Validate dtype compatibility
        let state_dtype = state.dtype();
        match block.node_type() {
            crate::node::NodeType::Spin => {
                // Spin nodes should be Bool or compatible float
                if state_dtype != burn::tensor::DType::Bool
                    && state_dtype != burn::tensor::DType::F32
                    && state_dtype != burn::tensor::DType::F64
                {
                    return Err(format!(
                        "Spin block has incompatible dtype {:?}, expected Bool or Float",
                        state_dtype
                    ));
                }
            }
            crate::node::NodeType::Categorical { n_categories } => {
                // Categorical should be unsigned int or compatible
                // In practice, we use f32 for GPU compatibility
                let max_expected = *n_categories as usize;
                // Note: We can't easily validate value ranges on GPU tensors
                // Just check dtype is numeric
                if state_dtype != burn::tensor::DType::U8
                    && state_dtype != burn::tensor::DType::I32
                    && state_dtype != burn::tensor::DType::I64
                    && state_dtype != burn::tensor::DType::F32
                    && state_dtype != burn::tensor::DType::F64
                {
                    return Err(format!(
                        "Categorical block with {} categories has incompatible dtype {:?}",
                        max_expected, state_dtype
                    ));
                }
            }
            crate::node::NodeType::Continuous => {
                // Continuous nodes should be float
                if state_dtype != burn::tensor::DType::F32
                    && state_dtype != burn::tensor::DType::F64
                {
                    return Err(format!(
                        "Continuous block has incompatible dtype {:?}, expected F32 or F64",
                        state_dtype
                    ));
                }
            }
        }

        // Check against expected spec shape if provided
        if !expected_spec.shape.is_empty() {
            let expected_trailing_size: usize = expected_spec.shape.iter().product();
            let state_size: usize = state_dims.iter().product();
            if !state_size.is_multiple_of(expected_trailing_size) {
                return Err(format!(
                    "State shape {:?} not compatible with expected spec shape {:?}",
                    state_dims, expected_spec.shape
                ));
            }
        }
    }

    Ok(())
}
