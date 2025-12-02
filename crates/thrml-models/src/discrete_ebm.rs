//! Discrete EBM (Energy-Based Model) factors for spin and categorical variables.
//!
//! This module provides discrete factor implementations that define energy functions
//! for models with binary (spin) and categorical variables.
//!
//! ## Precision Routing
//!
//! When using `factor_energy_routed` with a [`ComputeBackend`], the energy computation
//! routes accumulation based on hardware capabilities:
//!
//! | Hardware | Backend | Precision | Path |
//! |----------|---------|-----------|------|
//! | H100/B200 (with `cuda` feature) | `GpuHpcF64` | f64 on GPU | `factor_energy_cuda_f64` |
//! | Apple Silicon, Consumer NVIDIA | `UnifiedHybrid` | f64 on CPU | `factor_energy_cpu_f64` |
//! | Any (bulk ops / non-precision) | `GpuOnly` | f32 on GPU | `factor_energy` |
//!
//! ## Fused Kernels
//!
//! When the `fused-kernels` feature is enabled, the batch_gather operations use
//! GPU-fused kernels that eliminate intermediate memory allocations.

use crate::ebm::EBMFactor;
use crate::factor::{AbstractFactor, FactorInteractionGroup};
use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::blockspec::BlockSpec;
use thrml_core::compute::{ComputeBackend, OpType};
use thrml_core::node::Node;
use thrml_core::state_tree::from_global_state;

// Note: When fused-kernels feature is enabled, batch_gather can use the GPU-fused
// kernel for improved performance. The current implementation provides the same
// functionality using standard Burn tensor operations.
#[cfg(feature = "fused-kernels")]
use thrml_kernels::batch_gather_fused;

/// An interaction that shows up when sampling from discrete-variable EBMs.
#[derive(Clone)]
pub struct DiscreteEBMInteraction {
    /// Number of spin states involved in the interaction
    pub n_spin: usize,
    /// Weight tensor associated with this interaction [batch, ..., dims]
    pub weights: Tensor<WgpuBackend, 3>,
}

impl DiscreteEBMInteraction {
    pub const fn new(n_spin: usize, weights: Tensor<WgpuBackend, 3>) -> Self {
        Self { n_spin, weights }
    }
}

/// Multiply spin values (convert bool to -1/+1)
pub fn spin_product(
    spin_vals: &[Tensor<WgpuBackend, 1, burn::tensor::Bool>],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 1> {
    if spin_vals.is_empty() {
        return Tensor::ones([1], device); // Return 1.0
    }
    // Convert bool to f32: True -> 1.0, False -> -1.0
    let converted: Vec<Tensor<WgpuBackend, 1>> = spin_vals
        .iter()
        .map(|v| {
            let float_tensor: Tensor<WgpuBackend, 1> = v.clone().float();
            float_tensor * 2.0 - 1.0
        })
        .collect();
    // Multiply all together - need to dereference for multiplication
    let first = converted[0].clone();
    converted
        .iter()
        .skip(1)
        .fold(first, |acc, x| acc * x.clone())
}

/// Index into weight tensor using categorical indices
///
/// This implements multi-dimensional advanced indexing using linear indexing.
/// The weights tensor has shape [batch, dim1, dim2, ..., dimN] where:
/// - The first dimension is the batch dimension
/// - The remaining dimensions correspond to categorical indices
///
/// The function computes flat indices using strides and then uses a single
/// `select` operation on the flattened tensor for efficiency.
///
/// When the `fused-kernels` feature is enabled, this can use a GPU-fused kernel
/// that eliminates intermediate tensor allocations.
///
/// # Arguments
///
/// * `weights` - 3D tensor with shape [batch, dim1, dim2, ...]
/// * `indices` - Array of 1D integer tensors, one for each trailing dimension
///
/// # Returns
///
/// 1D tensor with shape `[batch]` containing the gathered values
pub fn batch_gather(
    weights: &Tensor<WgpuBackend, 3>, // [batch, dim1, dim2, ...]
    indices: &[Tensor<WgpuBackend, 1, burn::tensor::Int>],
) -> Tensor<WgpuBackend, 1> {
    let n_indices = indices.len();
    if n_indices == 0 {
        // No indices means we want the entire batch flattened
        let dims = weights.dims();
        let batch_size = dims[0];
        let total_trailing: usize = dims[1..].iter().product();
        let total_size = batch_size * total_trailing;
        return weights.clone().reshape([total_size as i32]);
    }

    let dims = weights.dims();
    let batch_size = dims[0];
    let trailing_dims = &dims[1..];

    // Verify we have the right number of indices
    if indices.len() != trailing_dims.len() {
        panic!(
            "batch_gather: Expected {} index tensors (one per trailing dimension), got {}",
            trailing_dims.len(),
            indices.len()
        );
    }

    // Verify all indices have the same length (batch_size)
    for (i, idx) in indices.iter().enumerate() {
        if idx.dims()[0] != batch_size {
            panic!(
                "batch_gather: Index tensor {} has length {}, expected {} (batch_size)",
                i,
                idx.dims()[0],
                batch_size
            );
        }
    }

    // Compute strides for trailing dimensions
    let mut strides: Vec<usize> = Vec::new();
    let mut stride = 1;
    for &dim in trailing_dims.iter().rev() {
        strides.push(stride);
        stride *= dim;
    }
    strides.reverse();

    // Batch stride is the product of all trailing dimensions
    let batch_stride: usize = trailing_dims.iter().product();

    // Use fused kernel when feature is enabled
    #[cfg(feature = "fused-kernels")]
    {
        // Stack indices into a 2D tensor [batch_size, n_indices]
        let indices_stacked: Vec<Tensor<WgpuBackend, 2, burn::tensor::Int>> = indices
            .iter()
            .map(|idx| idx.clone().unsqueeze_dim::<2>(1))
            .collect();
        let indices_2d: Tensor<WgpuBackend, 2, burn::tensor::Int> = Tensor::cat(indices_stacked, 1);

        batch_gather_fused(weights.clone(), indices_2d, &strides, batch_stride)
    }

    // Default implementation: standard tensor operations
    #[cfg(not(feature = "fused-kernels"))]
    {
        let device = weights.device();

        // Compute linear indices: batch_idx * batch_stride + idx0 * stride0 + idx1 * stride1 + ...
        let batch_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> = Tensor::from_data(
            (0..batch_size)
                .map(|i| i as i32)
                .collect::<Vec<_>>()
                .as_slice(),
            &device,
        );

        let batch_stride_tensor =
            Tensor::from_data(vec![batch_stride as i32; batch_size].as_slice(), &device);
        let mut linear_idx = batch_indices * batch_stride_tensor;

        for (idx, &stride_val) in indices.iter().zip(strides.iter()) {
            let stride_tensor =
                Tensor::from_data(vec![stride_val as i32; batch_size].as_slice(), &device);
            linear_idx = linear_idx + idx.clone() * stride_tensor;
        }

        // Flatten weights to 1D for efficient indexing
        let total_size: usize = dims.iter().product();
        let weights_flat = weights.clone().reshape([total_size as i32]);

        // Select using linear indices (select along dimension 0 on the flattened tensor)
        weights_flat.select(0, linear_idx)
    }
}

/// Batch gather with an extra "k" dimension for interactions.
///
/// This is similar to `batch_gather` but handles an extra trailing dimension
/// in the weights tensor. Used for categorical Gibbs sampling.
///
/// # Arguments
///
/// * `weights` - Tensor with shape [batch, k, dim1, dim2, ...]
/// * `indices` - Array of 1D integer tensors, one for each trailing dimension after k
///
/// # Returns
///
/// Tensor with shape [batch, k] containing the gathered values
pub fn batch_gather_with_k(
    weights: &Tensor<WgpuBackend, 3>,
    indices: &[Tensor<WgpuBackend, 1, burn::tensor::Int>],
) -> Tensor<WgpuBackend, 2> {
    let dims = weights.dims();
    let batch_size = dims[0];
    let k = dims[1];

    if indices.is_empty() {
        // No categorical indices, just return the tensor as-is
        return weights.clone().reshape([batch_size as i32, k as i32]);
    }

    let _device = weights.device();

    // Expand indices to include k dimension
    // For each index tensor of shape [batch], expand to [batch * k]
    let expanded_indices: Vec<Tensor<WgpuBackend, 1, burn::tensor::Int>> = indices
        .iter()
        .map(|idx| {
            // Repeat each element k times: [a, b, c] -> [a, a, ..., b, b, ..., c, c, ...]
            let idx_expanded: Tensor<WgpuBackend, 2, burn::tensor::Int> =
                idx.clone().unsqueeze_dim::<2>(1).repeat_dim(1, k);
            // Flatten to [batch * k]
            idx_expanded.reshape([(batch_size * k) as i32])
        })
        .collect();

    // Reshape weights to [batch * k, dim1, dim2, ...]
    let trailing_dims = &dims[2..];
    let flattened_batch = batch_size * k;

    // Create a 3D tensor for batch_gather
    // Reshape to [batch * k, trailing_dims...]
    let weights_reshaped = if trailing_dims.is_empty() {
        weights.clone().reshape([flattened_batch as i32, 1, 1])
    } else if trailing_dims.len() == 1 {
        weights
            .clone()
            .reshape([flattened_batch as i32, trailing_dims[0] as i32, 1])
    } else {
        weights.clone().reshape([
            flattened_batch as i32,
            trailing_dims[0] as i32,
            trailing_dims[1] as i32,
        ])
    };

    // Use batch_gather on the reshaped tensor
    let gathered = batch_gather(&weights_reshaped, &expanded_indices);

    // Reshape result from [batch * k] to [batch, k]
    gathered.reshape([batch_size as i32, k as i32])
}

/// Separate spin vs categorical states
#[allow(clippy::type_complexity)]
pub fn split_states(
    states: &[Tensor<WgpuBackend, 2>],
    n_spin: usize,
) -> (
    Vec<Tensor<WgpuBackend, 2, burn::tensor::Bool>>,
    Vec<Tensor<WgpuBackend, 2, burn::tensor::Int>>,
) {
    let states_spin: Vec<Tensor<WgpuBackend, 2, burn::tensor::Bool>> = states[..n_spin]
        .iter()
        .map(|s| {
            // Convert to bool tensor
            s.clone().bool()
        })
        .collect();

    // For categorical, use Int type (i32) for indexing operations
    // Burn uses Int for all index tensors (gather, select, etc.)
    let states_cat: Vec<Tensor<WgpuBackend, 2, burn::tensor::Int>> = states[n_spin..]
        .iter()
        .map(|s| {
            // Convert to int tensor
            s.clone().int()
        })
        .collect();

    (states_spin, states_cat)
}

/// A factor that defines an energy function for discrete EBMs
pub struct DiscreteEBMFactor {
    pub spin_node_groups: Vec<Block>,
    pub categorical_node_groups: Vec<Block>,
    pub weights: Tensor<WgpuBackend, 3>,
}

impl DiscreteEBMFactor {
    pub fn new(
        spin_node_groups: Vec<Block>,
        categorical_node_groups: Vec<Block>,
        weights: Tensor<WgpuBackend, 3>,
    ) -> Result<Self, String> {
        // Validate that all node groups have the same length
        let n_nodes = if let Some(first) = spin_node_groups
            .first()
            .or_else(|| categorical_node_groups.first())
        {
            first.len()
        } else {
            return Err("At least one node group must be provided".to_string());
        };

        for group in spin_node_groups
            .iter()
            .chain(categorical_node_groups.iter())
        {
            if group.len() != n_nodes {
                return Err(
                    "Every block in node_groups must contain the same number of nodes".to_string(),
                );
            }
        }

        // Validate weights shape
        let weight_dims = weights.dims();
        if weight_dims[0] != n_nodes {
            return Err("The leading dimension of weights must have the same length as the number of nodes in each node group".to_string());
        }

        // Check that the effective dimensions (ignoring trailing 1s) match categorical_node_groups
        // This allows [n_nodes, 1, 1] for spin-only, [n_nodes, n_cats, 1] for 1 categorical, etc.
        let expected_dims = 1 + categorical_node_groups.len();

        // Count effective dimensions (non-trailing-1 dimensions)
        let mut effective_dims = weight_dims.len();
        for i in (1..weight_dims.len()).rev() {
            if weight_dims[i] == 1 {
                effective_dims -= 1;
            } else {
                break;
            }
        }

        // The effective dimensions should be at least 1 (for n_nodes)
        effective_dims = effective_dims.max(1);

        if effective_dims < expected_dims {
            return Err(format!(
                "The shape of the weight tensor must be [b, x_1, ..., x_k], where k is the length of categorical_node_groups ({}). Got effective dims: {}",
                categorical_node_groups.len(), effective_dims - 1
            ));
        }

        Ok(Self {
            spin_node_groups,
            categorical_node_groups,
            weights,
        })
    }

    /// Get all node groups (spin + categorical)
    pub fn node_groups(&self) -> Vec<Block> {
        let mut groups = self.spin_node_groups.clone();
        groups.extend(self.categorical_node_groups.clone());
        groups
    }
}

impl AbstractFactor for DiscreteEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // This is a bit awkward since we need to return a slice but have two vecs
        // For now, just return spin groups (this is primarily used for validation)
        &self.spin_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        let mut interaction_groups = Vec::new();

        let n_spin = self.spin_node_groups.len();
        let n_cat = self.categorical_node_groups.len();
        let n_total = n_spin + n_cat;

        // Handle the interaction groups with spin head nodes
        if n_spin > 0 {
            // Generate combinations: (head_index, [tail_indices])
            // Each spin group takes a turn being the head, others are tail
            let spin_inds: Vec<usize> = (0..n_spin).collect();
            let spin_combos: Vec<(usize, Vec<usize>)> = spin_inds
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let tail: Vec<usize> = spin_inds
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, &v)| v)
                        .collect();
                    (x, tail)
                })
                .collect();

            // Collect all head nodes and tail nodes across all combos
            let mut all_head_nodes: Vec<Node> = Vec::new();
            let mut all_tail_nodes: Vec<Vec<Node>> = vec![Vec::new(); n_total - 1];

            for (head_idx, tail_inds) in &spin_combos {
                // Add head nodes from this spin group
                all_head_nodes.extend(self.spin_node_groups[*head_idx].nodes().iter().cloned());

                // Add tail nodes from other spin groups
                for (i, &tail_ind) in tail_inds.iter().enumerate() {
                    all_tail_nodes[i]
                        .extend(self.spin_node_groups[tail_ind].nodes().iter().cloned());
                }

                // Add all categorical groups as tail nodes
                for (j, cat_group) in self.categorical_node_groups.iter().enumerate() {
                    all_tail_nodes[n_spin - 1 + j].extend(cat_group.nodes().iter().cloned());
                }
            }

            // Tile the weights: repeat n_spin times along the batch dimension
            let weight_dims = self.weights.dims();
            let batch_size = weight_dims[0];
            let _new_batch_size = batch_size * n_spin;

            // Create tiled weights by repeating the tensor
            let mut tiled_weights_vec = Vec::new();
            for _ in 0..n_spin {
                tiled_weights_vec.push(self.weights.clone());
            }
            let rep_weights = Tensor::cat(tiled_weights_vec, 0);

            // Create the interaction group
            let head_block = Block::new(all_head_nodes)
                .expect("Failed to create head block for spin interaction");
            let tail_blocks: Vec<Block> = all_tail_nodes
                .into_iter()
                .map(|nodes| Block::new(nodes).expect("Failed to create tail block"))
                .collect();

            let interaction = DiscreteEBMInteraction::new(n_spin - 1, rep_weights);

            if let Ok(group) = FactorInteractionGroup::new(interaction, head_block, tail_blocks) {
                interaction_groups.push(group);
            }
        }

        // Handle the interaction groups with categorical head nodes
        if n_cat > 0 {
            let cat_inds: Vec<usize> = (0..n_cat).collect();

            // Generate combinations for categorical variables
            let cat_combos: Vec<(usize, Vec<usize>)> = cat_inds
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let tail: Vec<usize> = cat_inds
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, &v)| v)
                        .collect();
                    (x, tail)
                })
                .collect();

            for (head_idx, tail_inds) in cat_combos {
                let head_nodes = self.categorical_node_groups[head_idx].clone();

                // Tail nodes: all spin groups + selected categorical groups
                let mut tail_blocks: Vec<Block> = self.spin_node_groups.clone();
                for &i in &tail_inds {
                    tail_blocks.push(self.categorical_node_groups[i].clone());
                }

                // Reorder weight axes: move the head category dimension to position 1
                // Original: [batch, cat0, cat1, ...]
                // We want: [batch, cat_head, cat_tail0, cat_tail1, ...]
                //
                // In Python: reind = (0, combo[0] + 1, *[x + 1 for x in combo[1]])
                //            weights_reind = jnp.moveaxis(self.weights, reind, list(range(len(reind))))
                //
                // For now, if there's only one categorical variable, no reordering needed
                let weights_reind = if n_cat == 1 {
                    self.weights.clone()
                } else {
                    // Implement axis permutation
                    // This requires permute/transpose operations
                    // For 3D tensors with categorical groups, we need to reorder
                    self.permute_weights_for_categorical(head_idx, &tail_inds, device)
                };

                let interaction = DiscreteEBMInteraction::new(n_spin, weights_reind);

                if let Ok(group) = FactorInteractionGroup::new(interaction, head_nodes, tail_blocks)
                {
                    interaction_groups.push(group);
                }
            }
        }

        interaction_groups
    }
}

impl DiscreteEBMFactor {
    /// Permute weights for categorical head node processing.
    ///
    /// This implements the equivalent of jnp.moveaxis to reorder the categorical
    /// dimensions so the head category is in the right position.
    fn permute_weights_for_categorical(
        &self,
        head_idx: usize,
        _tail_inds: &[usize],
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 3> {
        // Build the permutation indices
        // reind = (0, head_idx + 1, *[x + 1 for x in tail_inds])
        // This moves axis `head_idx + 1` to position 1

        let weight_dims = self.weights.dims();
        let n_dims = weight_dims.len();

        // For 3D tensor: dims are [batch, cat0, cat1]
        // If head_idx = 1, we want [batch, cat1, cat0] -> swap_dims(1, 2)

        if n_dims == 3 {
            // Only two categorical dimensions, simple swap if needed
            if head_idx == 0 {
                // Head is already at position 1, no change needed
                self.weights.clone()
            } else {
                // Swap dimensions 1 and 2
                self.weights.clone().swap_dims(1, 2)
            }
        } else {
            // For higher dimensional cases, we'd need more complex permutation
            // For now, just return the weights as-is (this is a simplification)
            self.weights.clone()
        }
    }
}

impl EBMFactor for DiscreteEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Get spin values from global state
        let spin_vals = from_global_state(global_state, block_spec, &self.spin_node_groups, device);

        // Get categorical values from global state
        let cat_vals = from_global_state(
            global_state,
            block_spec,
            &self.categorical_node_groups,
            device,
        );

        // Compute spin product
        let spin_vals_bool: Vec<Tensor<WgpuBackend, 1, burn::tensor::Bool>> =
            spin_vals.iter().map(|t| t.clone().bool()).collect();
        let spin_prod = spin_product(&spin_vals_bool, device);

        // Convert categorical values to Int for batch_gather
        let cat_vals_int: Vec<Tensor<WgpuBackend, 1, burn::tensor::Int>> =
            cat_vals.iter().map(|t| t.clone().int()).collect();

        // Index into weights using categorical values
        let weights = if cat_vals_int.is_empty() {
            // No categorical variables, weights are just the batch dimension
            let dims = self.weights.dims();
            self.weights.clone().reshape([dims[0] as i32])
        } else {
            batch_gather(&self.weights, &cat_vals_int)
        };

        // Energy = -sum(weights * spin_prod)
        let energy = -(weights * spin_prod).sum();

        // Return as 1D tensor with single element
        energy.unsqueeze_dim(0)
    }
}

impl DiscreteEBMFactor {
    /// Compute factor energy with precision routing.
    ///
    /// Routes computation based on [`ComputeBackend`] configuration:
    /// - **CUDA f64**: If `backend.uses_gpu_f64()` and CUDA feature enabled
    /// - **CPU f64**: If `backend.use_cpu(OpType::EnergyCompute, ...)` returns true
    /// - **GPU f32**: Default path
    pub fn factor_energy_routed(
        &self,
        backend: &ComputeBackend,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Estimate problem size for routing decision
        let n_elements = self.weights.dims().iter().product::<usize>();

        // Priority 1: CUDA f64 for HPC GPUs
        #[cfg(feature = "cuda")]
        if backend.uses_gpu_f64() {
            return self.factor_energy_cuda_f64(global_state, block_spec, device);
        }

        // Priority 2: CPU f64 for precision-sensitive computation
        if backend.use_cpu(OpType::EnergyCompute, Some(n_elements)) {
            return self.factor_energy_cpu_f64(global_state, block_spec, device);
        }

        // Priority 3: Standard GPU f32 path
        self.factor_energy(global_state, block_spec, device)
    }

    /// Compute factor energy using CPU f64 accumulation.
    #[allow(clippy::needless_range_loop)]
    fn factor_energy_cpu_f64(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Get spin values from global state
        let spin_vals = from_global_state(global_state, block_spec, &self.spin_node_groups, device);

        // Get categorical values from global state
        let cat_vals = from_global_state(
            global_state,
            block_spec,
            &self.categorical_node_groups,
            device,
        );

        // Extract data to CPU
        let spin_data: Vec<Vec<f32>> = spin_vals
            .iter()
            .map(|t| t.clone().into_data().to_vec().unwrap())
            .collect();

        let cat_data: Vec<Vec<f32>> = cat_vals
            .iter()
            .map(|t| t.clone().into_data().to_vec().unwrap())
            .collect();

        let weights_data: Vec<f32> = self.weights.clone().into_data().to_vec().unwrap();
        let weight_dims = self.weights.dims();

        // Compute spin product in f64
        let n_nodes = if !spin_data.is_empty() {
            spin_data[0].len()
        } else if !cat_data.is_empty() {
            cat_data[0].len()
        } else {
            weight_dims[0]
        };

        let spin_prod_f64: Vec<f64> = if spin_data.is_empty() {
            vec![1.0; n_nodes]
        } else {
            let mut prod = vec![1.0f64; n_nodes];
            for spin_vec in &spin_data {
                for (i, &s) in spin_vec.iter().enumerate() {
                    // Convert 0/1 to -1/+1
                    let spin_val = 2.0f64.mul_add(s as f64, -1.0);
                    prod[i] *= spin_val;
                }
            }
            prod
        };

        // Index into weights using categorical values and compute energy in f64
        let mut energy_f64: f64 = 0.0;

        if cat_data.is_empty() {
            // No categorical variables - simple weighted sum
            for i in 0..n_nodes {
                let w = weights_data.get(i).copied().unwrap_or(0.0) as f64;
                energy_f64 -= w * spin_prod_f64[i];
            }
        } else {
            // With categorical variables - need to index into weights
            let trailing_dims = &weight_dims[1..];
            let mut strides: Vec<usize> = Vec::with_capacity(trailing_dims.len());
            let mut stride = 1;
            for &dim in trailing_dims.iter().rev() {
                strides.push(stride);
                stride *= dim;
            }
            strides.reverse();
            let batch_stride: usize = trailing_dims.iter().product();

            for i in 0..n_nodes {
                // Compute linear index
                let mut linear_idx = i * batch_stride;
                for (cat_vec, &s) in cat_data.iter().zip(strides.iter()) {
                    let cat_idx = cat_vec.get(i).copied().unwrap_or(0.0) as usize;
                    linear_idx += cat_idx * s;
                }

                let w = weights_data.get(linear_idx).copied().unwrap_or(0.0) as f64;
                energy_f64 -= w * spin_prod_f64[i];
            }
        }

        // Convert back to f32 tensor
        Tensor::from_floats([energy_f64 as f32].as_slice(), device)
    }

    /// Compute factor energy using CUDA f64.
    #[cfg(feature = "cuda")]
    fn factor_energy_cuda_f64(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // For complex gather operations, fall back to CPU f64
        // A full CUDA implementation would require CUDA-native gather
        self.factor_energy_cpu_f64(global_state, block_spec, device)
    }
}

// ============================================================================
// Specialized Factor Types
// ============================================================================

/// A DiscreteEBMFactor that involves only spin variables.
///
/// This is a convenience wrapper around DiscreteEBMFactor with no categorical node groups.
pub struct SpinEBMFactor {
    inner: DiscreteEBMFactor,
}

impl SpinEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Tensor<WgpuBackend, 3>) -> Result<Self, String> {
        let inner = DiscreteEBMFactor::new(node_groups, vec![], weights)?;
        Ok(Self { inner })
    }

    pub const fn inner(&self) -> &DiscreteEBMFactor {
        &self.inner
    }
}

impl AbstractFactor for SpinEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // SpinEBMFactor only has spin node groups
        &self.inner.spin_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        self.inner.to_interaction_groups(device)
    }
}

impl EBMFactor for SpinEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner.factor_energy(global_state, block_spec, device)
    }
}

impl SpinEBMFactor {
    /// Compute factor energy with precision routing.
    pub fn factor_energy_routed(
        &self,
        backend: &ComputeBackend,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner
            .factor_energy_routed(backend, global_state, block_spec, device)
    }
}

/// A DiscreteEBMFactor that involves only categorical variables.
///
/// This is a convenience wrapper around DiscreteEBMFactor with no spin node groups.
pub struct CategoricalEBMFactor {
    inner: DiscreteEBMFactor,
}

impl CategoricalEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Tensor<WgpuBackend, 3>) -> Result<Self, String> {
        let inner = DiscreteEBMFactor::new(vec![], node_groups, weights)?;
        Ok(Self { inner })
    }

    pub const fn inner(&self) -> &DiscreteEBMFactor {
        &self.inner
    }
}

impl AbstractFactor for CategoricalEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // CategoricalEBMFactor only has categorical node groups
        &self.inner.categorical_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        self.inner.to_interaction_groups(device)
    }
}

impl EBMFactor for CategoricalEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner.factor_energy(global_state, block_spec, device)
    }
}

impl CategoricalEBMFactor {
    /// Compute factor energy with precision routing.
    pub fn factor_energy_routed(
        &self,
        backend: &ComputeBackend,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner
            .factor_energy_routed(backend, global_state, block_spec, device)
    }
}

/// A discrete factor with a square interaction weight tensor.
///
/// If a discrete factor is square (shape [b, x, x, ..., x]), the interaction groups
/// corresponding to different choices of the head node blocks can be merged for
/// improved runtime performance.
pub struct SquareDiscreteEBMFactor {
    inner: DiscreteEBMFactor,
}

impl SquareDiscreteEBMFactor {
    pub fn new(
        spin_node_groups: Vec<Block>,
        categorical_node_groups: Vec<Block>,
        weights: Tensor<WgpuBackend, 3>,
    ) -> Result<Self, String> {
        // Validate that weights are square (all non-batch dimensions equal)
        let weight_dims = weights.dims();
        if weight_dims.len() > 2 {
            let target_shape = weight_dims[1];
            for &dim in &weight_dims[1..] {
                if dim != target_shape {
                    return Err("Interaction tensor is not square".to_string());
                }
            }
        }

        let inner = DiscreteEBMFactor::new(spin_node_groups, categorical_node_groups, weights)?;
        Ok(Self { inner })
    }

    pub const fn inner(&self) -> &DiscreteEBMFactor {
        &self.inner
    }
}

impl AbstractFactor for SquareDiscreteEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // Return spin node groups (for validation purposes)
        &self.inner.spin_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        // Get base interaction groups
        let base_groups = self.inner.to_interaction_groups(device);

        // For square factors, merge groups with the same n_spin for efficiency
        // This reduces the number of interaction groups, leading to smaller XLA/GPU programs
        merge_interaction_groups(base_groups, device)
    }
}

/// Merge multiple interaction groups into fewer, larger groups for efficiency.
///
/// Groups with the same number of tail blocks and same n_spin can be merged
/// by concatenating their head nodes, tail nodes, and weight tensors.
fn merge_interaction_groups(
    groups: Vec<FactorInteractionGroup>,
    _device: &burn::backend::wgpu::WgpuDevice,
) -> Vec<FactorInteractionGroup> {
    if groups.is_empty() {
        return groups;
    }

    // Group by (n_tail_blocks, n_spin) for merging compatibility
    let mut merge_buckets: std::collections::HashMap<(usize, usize), Vec<&FactorInteractionGroup>> =
        std::collections::HashMap::new();

    for group in &groups {
        let key = (group.tail_nodes.len(), group.interaction.n_spin);
        merge_buckets.entry(key).or_default().push(group);
    }

    let mut merged = Vec::new();

    for ((n_tail, n_spin), bucket) in merge_buckets {
        if bucket.len() == 1 {
            // Only one group, no merging needed
            let g = bucket[0];
            merged.push(FactorInteractionGroup {
                interaction: DiscreteEBMInteraction::new(
                    g.interaction.n_spin,
                    g.interaction.weights.clone(),
                ),
                head_nodes: g.head_nodes.clone(),
                tail_nodes: g.tail_nodes.clone(),
            });
            continue;
        }

        // Merge multiple groups
        let mut all_head_nodes: Vec<Node> = Vec::new();
        let mut all_tail_nodes: Vec<Vec<Node>> = vec![Vec::new(); n_tail];
        let mut all_weights: Vec<Tensor<WgpuBackend, 3>> = Vec::new();

        for group in &bucket {
            // Collect head nodes
            all_head_nodes.extend(group.head_nodes.nodes().iter().cloned());

            // Collect tail nodes for each tail block
            for (i, tail_block) in group.tail_nodes.iter().enumerate() {
                all_tail_nodes[i].extend(tail_block.nodes().iter().cloned());
            }

            // Collect weights
            all_weights.push(group.interaction.weights.clone());
        }

        // Concatenate weights along batch dimension (axis 0)
        let merged_weights: Tensor<WgpuBackend, 3> = if all_weights.len() == 1 {
            all_weights[0].clone()
        } else {
            Tensor::cat(all_weights, 0)
        };

        // Create merged group
        let head_block = Block::new(all_head_nodes).expect("Failed to create merged head block");
        let tail_blocks: Vec<Block> = all_tail_nodes
            .into_iter()
            .map(|nodes| Block::new(nodes).expect("Failed to create merged tail block"))
            .collect();

        let merged_group = FactorInteractionGroup {
            interaction: DiscreteEBMInteraction::new(n_spin, merged_weights),
            head_nodes: head_block,
            tail_nodes: tail_blocks,
        };

        merged.push(merged_group);
    }

    merged
}

impl EBMFactor for SquareDiscreteEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner.factor_energy(global_state, block_spec, device)
    }
}

impl SquareDiscreteEBMFactor {
    /// Compute factor energy with precision routing.
    pub fn factor_energy_routed(
        &self,
        backend: &ComputeBackend,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner
            .factor_energy_routed(backend, global_state, block_spec, device)
    }
}

/// A DiscreteEBMFactor with only categorical variables and a square weight tensor.
pub struct SquareCategoricalEBMFactor {
    inner: SquareDiscreteEBMFactor,
}

impl SquareCategoricalEBMFactor {
    pub fn new(node_groups: Vec<Block>, weights: Tensor<WgpuBackend, 3>) -> Result<Self, String> {
        let inner = SquareDiscreteEBMFactor::new(vec![], node_groups, weights)?;
        Ok(Self { inner })
    }
}

impl AbstractFactor for SquareCategoricalEBMFactor {
    fn node_groups(&self) -> &[Block] {
        // SquareCategoricalEBMFactor only has categorical node groups
        &self.inner.inner.categorical_node_groups
    }

    fn to_interaction_groups(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<FactorInteractionGroup> {
        self.inner.to_interaction_groups(device)
    }
}

impl EBMFactor for SquareCategoricalEBMFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner.factor_energy(global_state, block_spec, device)
    }
}

impl SquareCategoricalEBMFactor {
    /// Compute factor energy with precision routing.
    pub fn factor_energy_routed(
        &self,
        backend: &ComputeBackend,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.inner
            .factor_energy_routed(backend, global_state, block_spec, device)
    }
}
