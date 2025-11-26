use crate::discrete_ebm::SpinEBMFactor;
use crate::ebm::AbstractEBM;
use crate::factor::AbstractFactor;
/// Ising Energy-Based Model implementation.
///
/// The Ising model defines an energy function:
/// E(s) = -β * (Σ_i b_i * s_i + Σ_(i,j) J_ij * s_i * s_j)
///
/// where s_i are spin variables, b_i are biases, J_ij are coupling weights,
/// and β is the inverse temperature.
use burn::tensor::{Distribution, Tensor};
use indexmap::IndexMap;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::interaction::InteractionGroup;
use thrml_core::node::{Node, NodeType, TensorSpec};
use thrml_samplers::rng::RngKey;
use thrml_samplers::{
    BlockGibbsSpec, BlockSamplingProgram, SamplingSchedule, SpinGibbsConditional,
};

/// An EBM with the Ising energy function.
///
/// E(s) = -β * (Σ_i b_i * s_i + Σ_(i,j) J_ij * s_i * s_j)
#[derive(Clone)]
pub struct IsingEBM {
    pub nodes: Vec<Node>,
    pub biases: Tensor<WgpuBackend, 1>,
    pub edges: Vec<Edge>,
    pub weights: Tensor<WgpuBackend, 1>,
    pub beta: Tensor<WgpuBackend, 1>,
    /// Cached beta value on CPU (avoids GPU sync)
    beta_cached: f32,
    /// Cached node shape/dtype map
    node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
}

impl IsingEBM {
    pub fn new(
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        biases: Tensor<WgpuBackend, 1>,
        weights: Tensor<WgpuBackend, 1>,
        beta: Tensor<WgpuBackend, 1>,
    ) -> Self {
        // All nodes should be spin type
        let node_type = if !nodes.is_empty() {
            nodes[0].node_type().clone()
        } else {
            NodeType::Spin
        };

        let mut node_shape_dtypes = IndexMap::new();
        node_shape_dtypes.insert(
            node_type,
            TensorSpec {
                shape: vec![],
                dtype: burn::tensor::DType::Bool,
            },
        );

        // Cache beta value on CPU to avoid GPU sync during training
        let beta_cached: f32 = beta
            .clone()
            .into_data()
            .to_vec()
            .map(|v: Vec<f32>| v.first().copied().unwrap_or(1.0))
            .unwrap_or(1.0);

        IsingEBM {
            nodes,
            biases,
            edges,
            weights,
            beta,
            beta_cached,
            node_shape_dtypes,
        }
    }

    /// Update weights and biases in-place (avoids full model recreation).
    ///
    /// This is much faster than calling `IsingEBM::new()` because it only
    /// updates the tensor fields without rebuilding the node/edge structures.
    pub fn update_weights(
        &mut self,
        new_weights: Tensor<WgpuBackend, 1>,
        new_biases: Tensor<WgpuBackend, 1>,
    ) {
        self.weights = new_weights;
        self.biases = new_biases;
    }

    /// Get the cached beta value (avoids GPU sync).
    #[inline]
    pub fn beta_value(&self) -> f32 {
        self.beta_cached
    }

    /// Get the factors that make up this Ising EBM.
    ///
    /// Returns two SpinEBMFactors:
    /// 1. Bias factor: for each node, energy contribution is b_i * s_i
    /// 2. Edge factor: for each edge (i,j), energy contribution is J_ij * s_i * s_j
    pub fn get_factors(&self, _device: &burn::backend::wgpu::WgpuDevice) -> Vec<SpinEBMFactor> {
        let mut factors = Vec::new();

        // Bias factor: SpinEBMFactor with single node group
        if !self.nodes.is_empty() {
            let bias_block = Block::new(self.nodes.clone()).expect("Failed to create bias block");

            // Weights for bias: beta * biases, shaped as [n_nodes, 1, 1] for 3D tensor
            let scaled_biases = self.beta.clone() * self.biases.clone();
            let n_nodes = self.nodes.len();
            let bias_weights: Tensor<WgpuBackend, 3> =
                scaled_biases.reshape([n_nodes as i32, 1, 1]);

            if let Ok(factor) = SpinEBMFactor::new(vec![bias_block], bias_weights) {
                factors.push(factor);
            }
        }

        // Edge factor: SpinEBMFactor with two node groups (edge endpoints)
        if !self.edges.is_empty() {
            let edge_nodes_0: Vec<Node> = self.edges.iter().map(|(n, _)| n.clone()).collect();
            let edge_nodes_1: Vec<Node> = self.edges.iter().map(|(_, n)| n.clone()).collect();

            let edge_block_0 = Block::new(edge_nodes_0).expect("Failed to create edge block 0");
            let edge_block_1 = Block::new(edge_nodes_1).expect("Failed to create edge block 1");

            // Weights for edges: beta * weights, shaped as [n_edges, 1, 1]
            let scaled_weights = self.beta.clone() * self.weights.clone();
            let n_edges = self.edges.len();
            let edge_weights: Tensor<WgpuBackend, 3> =
                scaled_weights.reshape([n_edges as i32, 1, 1]);

            if let Ok(factor) = SpinEBMFactor::new(vec![edge_block_0, edge_block_1], edge_weights) {
                factors.push(factor);
            }
        }

        factors
    }

    /// Get the node shape/dtype specification for this model.
    pub fn node_shape_dtypes(&self) -> &IndexMap<NodeType, TensorSpec> {
        &self.node_shape_dtypes
    }
}

impl AbstractEBM for IsingEBM {
    fn energy(
        &self,
        state: &[Tensor<WgpuBackend, 1>],
        _blocks: &[Block],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        // Simple direct energy computation for Ising model
        // E = -Σ_i β*b_i*s_i - Σ_(i,j) β*J_ij*s_i*s_j
        //
        // Assume first block contains all nodes in order
        let node_state = if !state.is_empty() {
            state[0].clone()
        } else {
            return Tensor::zeros([1], device);
        };

        // Convert bool state to spin state: s = 2*x - 1 where x ∈ {0, 1}
        let spin_state = node_state.clone() * 2.0 - 1.0;

        // Get beta value
        let beta_data: Vec<f32> = self.beta.clone().into_data().to_vec().expect("read beta");
        let beta = beta_data[0];

        // Bias contribution: Σ_i b_i * s_i
        let bias_data: Vec<f32> = self
            .biases
            .clone()
            .into_data()
            .to_vec()
            .expect("read biases");
        let spin_data: Vec<f32> = spin_state
            .clone()
            .into_data()
            .to_vec()
            .expect("read spin state");

        let bias_energy: f32 = bias_data
            .iter()
            .zip(spin_data.iter())
            .map(|(b, s)| b * s)
            .sum();

        // Edge contribution: Σ_(i,j) J_ij * s_i * s_j
        // Need to map nodes to indices
        let node_to_idx: std::collections::HashMap<_, _> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id(), i))
            .collect();

        let weight_data: Vec<f32> = self
            .weights
            .clone()
            .into_data()
            .to_vec()
            .expect("read weights");

        let edge_energy: f32 = self
            .edges
            .iter()
            .zip(weight_data.iter())
            .map(|((n1, n2), w)| {
                let idx1 = *node_to_idx.get(&n1.id()).unwrap_or(&0);
                let idx2 = *node_to_idx.get(&n2.id()).unwrap_or(&0);
                w * spin_data.get(idx1).unwrap_or(&0.0) * spin_data.get(idx2).unwrap_or(&0.0)
            })
            .sum();

        // Total energy = -beta * (bias_energy + edge_energy)
        let total = -beta * (bias_energy + edge_energy);

        Tensor::from_data(vec![total].as_slice(), device)
    }
}

use crate::ebm::BatchedEBM;
use std::collections::HashMap;

impl BatchedEBM for IsingEBM {
    fn energy_batched(
        &self,
        states: &Tensor<WgpuBackend, 2>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let [_batch_size, n_nodes] = states.dims();

        // Validate dimensions
        if n_nodes != self.nodes.len() {
            panic!(
                "State dimension {} does not match model nodes {}",
                n_nodes,
                self.nodes.len()
            );
        }

        // Convert {0,1} states to {-1,+1} spins
        let spins = states.clone() * 2.0 - 1.0; // [batch, nodes]

        // === Bias energy: sum_i(b_i * s_i) ===
        // biases: [nodes] -> [1, nodes] for broadcasting
        let biases_expanded = self.biases.clone().unsqueeze_dim::<2>(0);
        let bias_energy: Tensor<WgpuBackend, 1> = (spins.clone() * biases_expanded)
            .sum_dim(1)
            .squeeze_dim::<1>(1);

        // === Edge energy: sum_ij(J_ij * s_i * s_j) ===
        let edge_energy = self.compute_edge_energy_batched(&spins, device);

        // === Total energy: E = -β * (bias + edge) ===
        let beta_val = self.beta.clone().into_data().to_vec::<f32>().unwrap()[0];
        (bias_energy + edge_energy) * (-beta_val)
    }
}

impl IsingEBM {
    /// Compute edge energy for batch using tensor gather operations.
    fn compute_edge_energy_batched(
        &self,
        spins: &Tensor<WgpuBackend, 2>, // [batch, nodes]
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        if self.edges.is_empty() {
            let batch_size = spins.dims()[0];
            return Tensor::zeros([batch_size], device);
        }

        let [batch_size, _] = spins.dims();

        // Build node-to-index map (do this once, could be cached)
        let node_to_idx: HashMap<usize, i32> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id(), i as i32))
            .collect();

        // Build edge endpoint index tensors
        let edge_i: Vec<i32> = self
            .edges
            .iter()
            .map(|(n1, _)| *node_to_idx.get(&n1.id()).unwrap_or(&0))
            .collect();
        let edge_j: Vec<i32> = self
            .edges
            .iter()
            .map(|(_, n2)| *node_to_idx.get(&n2.id()).unwrap_or(&0))
            .collect();

        let idx_i: Tensor<WgpuBackend, 1, burn::tensor::Int> =
            Tensor::from_data(edge_i.as_slice(), device);
        let idx_j: Tensor<WgpuBackend, 1, burn::tensor::Int> =
            Tensor::from_data(edge_j.as_slice(), device);

        // Expand indices for batched gather: [n_edges] -> [batch, n_edges]
        let idx_i_2d: Tensor<WgpuBackend, 2, burn::tensor::Int> = idx_i
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, batch_size);
        let idx_j_2d: Tensor<WgpuBackend, 2, burn::tensor::Int> = idx_j
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, batch_size);

        // Gather spin values at edge endpoints
        // spins: [batch, nodes], idx: [batch, edges] -> s: [batch, edges]
        let s_i: Tensor<WgpuBackend, 2> = spins.clone().gather(1, idx_i_2d);
        let s_j: Tensor<WgpuBackend, 2> = spins.clone().gather(1, idx_j_2d);

        // weights: [edges] -> [1, edges]
        let weights_expanded = self.weights.clone().unsqueeze_dim::<2>(0);

        // Edge energy: sum over edges of (s_i * s_j * J_ij)
        (s_i * s_j * weights_expanded)
            .sum_dim(1)
            .squeeze_dim::<1>(1)
    }
}

/// SuperBlock type alias for Ising sampling (a block or list of blocks)
pub type SuperBlock = Block;

/// A sampling program specialized for Ising models.
///
/// Uses SpinGibbsConditional for all blocks.
pub struct IsingSamplingProgram {
    pub program: BlockSamplingProgram,
}

impl IsingSamplingProgram {
    /// Create a new Ising sampling program.
    ///
    /// # Arguments
    ///
    /// * `ebm` - The Ising EBM to sample from
    /// * `free_blocks` - List of blocks that are free to vary
    /// * `clamped_blocks` - List of blocks that are held fixed
    /// * `device` - The device for tensor operations
    pub fn new(
        ebm: &IsingEBM,
        free_blocks: Vec<Block>,
        clamped_blocks: Vec<Block>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Result<Self, String> {
        // Create superblocks (each free block is its own superblock for Ising)
        let superblocks: Vec<Vec<Block>> = free_blocks.iter().map(|b| vec![b.clone()]).collect();

        // Create BlockGibbsSpec
        let gibbs_spec =
            BlockGibbsSpec::new(superblocks, clamped_blocks, ebm.node_shape_dtypes.clone())?;

        // Create samplers (one SpinGibbsConditional per free block)
        let samplers: Vec<Box<dyn thrml_samplers::sampler::DynConditionalSampler>> = gibbs_spec
            .free_blocks
            .iter()
            .map(|_| {
                Box::new(SpinGibbsConditional::new())
                    as Box<dyn thrml_samplers::sampler::DynConditionalSampler>
            })
            .collect();

        // Get interaction groups from factors and convert to InteractionGroup
        let factors = ebm.get_factors(device);
        let mut interaction_groups: Vec<InteractionGroup> = Vec::new();
        for factor in factors {
            let factor_groups = factor.to_interaction_groups(device);
            for fg in factor_groups {
                // Convert FactorInteractionGroup to InteractionGroup
                // Keep the 3D weights and n_spin metadata
                if let Ok(ig) = InteractionGroup::new(
                    fg.interaction.weights,
                    fg.head_nodes,
                    fg.tail_nodes,
                    fg.interaction.n_spin,
                ) {
                    interaction_groups.push(ig);
                }
            }
        }

        // Create the sampling program
        let program = BlockSamplingProgram::new(gibbs_spec, samplers, interaction_groups)?;

        Ok(IsingSamplingProgram { program })
    }
}

/// Specification for training an Ising model using sampling-based gradients.
pub struct IsingTrainingSpec {
    pub ebm: IsingEBM,
    pub program_positive: IsingSamplingProgram,
    pub program_negative: IsingSamplingProgram,
    pub schedule_positive: SamplingSchedule,
    pub schedule_negative: SamplingSchedule,
}

impl IsingTrainingSpec {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ebm: IsingEBM,
        data_blocks: Vec<Block>,
        conditioning_blocks: Vec<Block>,
        positive_sampling_blocks: Vec<Block>,
        negative_sampling_blocks: Vec<Block>,
        schedule_positive: SamplingSchedule,
        schedule_negative: SamplingSchedule,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Result<Self, String> {
        // Positive phase: sample hidden given visible (clamped data + conditioning)
        let clamped_positive: Vec<Block> = data_blocks
            .iter()
            .chain(conditioning_blocks.iter())
            .cloned()
            .collect();
        let program_positive =
            IsingSamplingProgram::new(&ebm, positive_sampling_blocks, clamped_positive, device)?;

        // Negative phase: sample all (only condition on conditioning blocks)
        let program_negative =
            IsingSamplingProgram::new(&ebm, negative_sampling_blocks, conditioning_blocks, device)?;

        Ok(IsingTrainingSpec {
            ebm,
            program_positive,
            program_negative,
            schedule_positive,
            schedule_negative,
        })
    }
}

/// Edge type alias for moment estimation
pub type Edge = (Node, Node);

/// Estimate the first and second moments of an Ising model Boltzmann distribution via sampling.
///
/// # Arguments
///
/// * `key` - RNG key for reproducibility
/// * `first_moment_nodes` - Nodes for which to estimate first moments
/// * `second_moment_edges` - Edges for which to estimate second moments
/// * `program` - The BlockSamplingProgram to use for sampling
/// * `schedule` - Sampling schedule
/// * `init_state` - Initial state for the sampling chain
/// * `clamped_data` - Values for clamped nodes
/// * `device` - Device for tensor operations
///
/// # Returns
///
/// Tuple of (node_moments, edge_moments) - first and second moment estimates
#[allow(clippy::too_many_arguments)]
pub fn estimate_moments(
    key: RngKey,
    first_moment_nodes: &[Node],
    second_moment_edges: &[(Node, Node)],
    program: &BlockSamplingProgram,
    schedule: &SamplingSchedule,
    init_state: Vec<Tensor<WgpuBackend, 1>>,
    clamped_data: &[Tensor<WgpuBackend, 1>],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<(Tensor<WgpuBackend, 1>, Tensor<WgpuBackend, 1>), String> {
    use thrml_observers::moment::MomentAccumulatorObserver;
    use thrml_observers::observer::AbstractObserver;
    use thrml_samplers::sampling::sample_with_observation;

    // Build moment spec: first moments (individual nodes) + second moments (edges)
    let moment_spec =
        MomentAccumulatorObserver::ising_moment_spec(first_moment_nodes, second_moment_edges);

    // Create observer with spin transform (bool -> ±1)
    let observer = MomentAccumulatorObserver::new(moment_spec, true);
    let init_carry = observer.init(device);

    // Run sampling with observation
    let (final_carry, _observations) = sample_with_observation(
        key,
        program,
        schedule,
        init_state,
        clamped_data,
        init_carry,
        &observer,
        device,
    )?;

    // Divide accumulated sums by n_samples to get averages
    let n_samples = schedule.n_samples as f32;

    let node_moments = if !final_carry.is_empty() {
        final_carry[0].clone().div_scalar(n_samples)
    } else {
        Tensor::zeros([0], device)
    };

    let edge_moments = if final_carry.len() > 1 {
        final_carry[1].clone().div_scalar(n_samples)
    } else {
        Tensor::zeros([0], device)
    };

    Ok((node_moments, edge_moments))
}

/// Estimate the KL-divergence gradients of an Ising model.
///
/// Uses the standard two-term Monte Carlo estimator:
/// - Δb = -β(⟨sᵢ⟩₊ - ⟨sᵢ⟩₋)
/// - Δw = -β(⟨sᵢsⱼ⟩₊ - ⟨sᵢsⱼ⟩₋)
///
/// where ⟨·⟩₊ is the positive phase (data-clamped) and ⟨·⟩₋ is the negative phase.
///
/// # Arguments
///
/// * `key` - RNG key
/// * `training_spec` - IsingTrainingSpec containing the model and programs
/// * `bias_nodes` - Nodes for bias gradients
/// * `weight_edges` - Edges for weight gradients  
/// * `data` - Data values for positive phase [batch, nodes]
/// * `conditioning_values` - Values for conditioning nodes
/// * `init_state_positive` - Initial state for positive chain
/// * `init_state_negative` - Initial state for negative chain
/// * `device` - Device for tensor operations
///
/// # Returns
///
/// Tuple of (grad_weights, grad_biases)
#[allow(clippy::too_many_arguments)]
pub fn estimate_kl_grad(
    key: RngKey,
    training_spec: &IsingTrainingSpec,
    bias_nodes: &[Node],
    weight_edges: &[(Node, Node)],
    data: &[Tensor<WgpuBackend, 2>],
    conditioning_values: &[Tensor<WgpuBackend, 1>],
    init_state_positive: Vec<Tensor<WgpuBackend, 1>>,
    init_state_negative: Vec<Tensor<WgpuBackend, 1>>,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<(Tensor<WgpuBackend, 1>, Tensor<WgpuBackend, 1>), String> {
    let (key_pos, key_neg) = key.split_two();

    // Get cached beta value (no GPU sync!)
    let beta: f32 = training_spec.ebm.beta_value();

    // Positive phase: estimate moments with data clamped
    // Use random sample from batch for stochastic gradient
    let mut clamped_pos: Vec<Tensor<WgpuBackend, 1>> = Vec::new();
    for d in data {
        let dims = d.dims();
        let batch_size = dims[0];
        let n_cols = dims[1];
        // Random sample index from batch
        let sample_idx = (key_pos.0 as usize) % batch_size;
        let row: Tensor<WgpuBackend, 2> =
            d.clone().slice([sample_idx..sample_idx + 1, 0..n_cols]);
        let squeezed: Tensor<WgpuBackend, 1> = row.reshape([n_cols as i32]);
        clamped_pos.push(squeezed);
    }
    for c in conditioning_values {
        clamped_pos.push(c.clone());
    }

    let (moms_b_pos, moms_w_pos) = estimate_moments(
        key_pos,
        bias_nodes,
        weight_edges,
        &training_spec.program_positive.program,
        &training_spec.schedule_positive,
        init_state_positive,
        &clamped_pos,
        device,
    )?;

    // Negative phase: estimate moments with only conditioning clamped
    let clamped_neg: Vec<Tensor<WgpuBackend, 1>> = conditioning_values.to_vec();

    let (moms_b_neg, moms_w_neg) = estimate_moments(
        key_neg,
        bias_nodes,
        weight_edges,
        &training_spec.program_negative.program,
        &training_spec.schedule_negative,
        init_state_negative,
        &clamped_neg,
        device,
    )?;

    // Compute gradients: Δ = -β(positive - negative)
    let grad_b = (moms_b_pos - moms_b_neg).mul_scalar(-beta);
    let grad_w = (moms_w_pos - moms_w_neg).mul_scalar(-beta);

    Ok((grad_w, grad_b))
}

/// Initialize blocks according to the marginal bias (Hinton initialization).
///
/// Each binary unit i in a block is sampled independently as:
/// P(S_i = 1) = σ(β * h_i)
///
/// where h_i is the bias of unit i and β is the inverse temperature.
pub fn hinton_init(
    key: RngKey,
    model: &IsingEBM,
    blocks: &[Block],
    batch_shape: &[usize],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Vec<Tensor<WgpuBackend, 2>> {
    // Build node -> bias index map
    let node_map: std::collections::HashMap<Node, usize> = model
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (node.clone(), i))
        .collect();

    let mut data = Vec::new();
    let _keys = key.split(blocks.len());

    for block in blocks.iter() {
        let block_len = block.len();
        if block_len == 0 {
            // Empty block
            let batch_size = batch_shape.iter().product::<usize>();
            data.push(Tensor::<WgpuBackend, 2>::zeros([batch_size, 0], device));
            continue;
        }

        // Get bias indices for this block
        let block_indices: Vec<i32> = block
            .nodes()
            .iter()
            .map(|node| node_map.get(node).copied().unwrap_or(0) as i32)
            .collect();

        // Create index tensor
        let indices: Tensor<WgpuBackend, 1, burn::tensor::Int> =
            Tensor::from_data(block_indices.as_slice(), device);

        // Get biases for this block
        let block_biases = model.biases.clone().select(0, indices);

        // Compute probabilities: sigmoid(beta * biases)
        let beta_scalar = model
            .beta
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("read beta")[0];
        let probs = burn::tensor::activation::sigmoid(block_biases * beta_scalar);

        // Sample Bernoulli for each batch
        let batch_size: usize = batch_shape.iter().product();
        let probs_expanded = probs
            .clone()
            .unsqueeze_dim::<2>(0)
            .repeat_dim(0, batch_size);

        // Generate uniform random values
        let uniform: Tensor<WgpuBackend, 2> = Tensor::random(
            [batch_size, block_len],
            Distribution::Uniform(0.0, 1.0),
            device,
        );

        // Sample: output 1 if uniform < probs, else 0
        let samples = uniform.lower_equal(probs_expanded).float();

        data.push(samples);
    }

    data
}
