//! Test utilities for thrml-models
//!
//! Port of Python tests/utils.py

use burn::tensor::{Int, Tensor};
use itertools::Itertools;
use std::collections::HashMap;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::node::NodeType;
use thrml_models::ebm::AbstractEBM;
use thrml_samplers::program::BlockSamplingProgram;
use thrml_samplers::rng::RngKey;
use thrml_samplers::sampling::sample_states;
use thrml_samplers::schedule::SamplingSchedule;

/// Generate all binary states for num_binary variables.
/// Returns shape [2^num_binary, num_binary]
pub fn generate_all_states_binary(
    num_binary: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2> {
    if num_binary == 0 {
        return Tensor::<WgpuBackend, 2>::zeros([1, 0], device);
    }

    let n_states = 1usize << num_binary; // 2^num_binary
    let mut data = Vec::with_capacity(n_states * num_binary);

    for state_idx in 0..n_states {
        for bit_idx in 0..num_binary {
            // Check if bit is set (MSB first to match Python)
            let bit_pos = num_binary - 1 - bit_idx;
            let bit = ((state_idx >> bit_pos) & 1) as f32;
            data.push(bit);
        }
    }

    // Create a 1D tensor first, then reshape to 2D
    let flat_tensor: Tensor<WgpuBackend, 1> = Tensor::from_data(data.as_slice(), device);
    flat_tensor.reshape([n_states, num_binary])
}

/// Generate all categorical states for num_categorical variables with n_categories.
/// Returns shape [n_categories^num_categorical, num_categorical]
pub fn generate_all_states_categorical(
    num_categorical: usize,
    n_categories: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2, Int> {
    if num_categorical == 0 {
        return Tensor::<WgpuBackend, 2, Int>::zeros([1, 0], device);
    }

    let n_states = n_categories.pow(num_categorical as u32);
    let mut data = Vec::with_capacity(n_states * num_categorical);

    // Generate all combinations using itertools
    for combo in (0..num_categorical)
        .map(|_| 0..n_categories)
        .multi_cartesian_product()
    {
        for &val in &combo {
            data.push(val as i32);
        }
    }

    // Create a 1D tensor first, then reshape to 2D
    let flat_tensor: Tensor<WgpuBackend, 1, Int> = Tensor::from_data(data.as_slice(), device);
    flat_tensor.reshape([n_states, num_categorical])
}

/// Generate all states for mixed binary/categorical system.
/// Returns (binary_states, categorical_states)
pub fn generate_all_states_bin_cat(
    num_binary: usize,
    num_categorical: usize,
    n_categories: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> (Tensor<WgpuBackend, 2>, Tensor<WgpuBackend, 2, Int>) {
    let bin_states = generate_all_states_binary(num_binary, device);
    let cat_states = generate_all_states_categorical(num_categorical, n_categories, device);

    let bin_dims = bin_states.dims();
    let cat_dims = cat_states.dims();

    let n_bin_states = bin_dims[0];
    let n_cat_states = cat_dims[0];
    let total_states = n_bin_states * n_cat_states;

    if total_states == 0
        || (n_bin_states == 1 && bin_dims[1] == 0) && (n_cat_states == 1 && cat_dims[1] == 0)
    {
        return (bin_states, cat_states);
    }

    // Expand and combine
    let mut bin_expanded_data = Vec::new();
    let mut cat_expanded_data = Vec::new();

    let bin_data: Vec<f32> = bin_states
        .clone()
        .into_data()
        .to_vec()
        .expect("read bin data");
    let cat_data: Vec<i32> = cat_states
        .clone()
        .into_data()
        .to_vec()
        .expect("read cat data");

    for i in 0..n_bin_states {
        for j in 0..n_cat_states {
            // Add binary state
            for k in 0..bin_dims[1] {
                bin_expanded_data.push(bin_data[i * bin_dims[1] + k]);
            }
            // Add categorical state
            for k in 0..cat_dims[1] {
                cat_expanded_data.push(cat_data[j * cat_dims[1] + k]);
            }
        }
    }

    let bin_result = Tensor::<WgpuBackend, 2>::from_data(bin_expanded_data.as_slice(), device)
        .reshape([total_states as i32, bin_dims[1] as i32]);
    let cat_result = Tensor::<WgpuBackend, 2, Int>::from_data(cat_expanded_data.as_slice(), device)
        .reshape([total_states as i32, cat_dims[1] as i32]);

    (bin_result, cat_result)
}

/// Count sample frequencies against all possible states.
/// Returns normalized counts.
pub fn count_samples(
    all_states_bin: &Tensor<WgpuBackend, 2>,
    all_states_cat: &Tensor<WgpuBackend, 2, Int>,
    samples_bin: &Tensor<WgpuBackend, 2>,
    samples_cat: &Tensor<WgpuBackend, 2, Int>,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 1> {
    let all_bin_data: Vec<f32> = all_states_bin
        .clone()
        .into_data()
        .to_vec()
        .expect("read all bin");
    let all_cat_data: Vec<i32> = all_states_cat
        .clone()
        .into_data()
        .to_vec()
        .expect("read all cat");
    let samples_bin_data: Vec<f32> = samples_bin
        .clone()
        .into_data()
        .to_vec()
        .expect("read samples bin");
    let samples_cat_data: Vec<i32> = samples_cat
        .clone()
        .into_data()
        .to_vec()
        .expect("read samples cat");

    let all_bin_dims = all_states_bin.dims();
    let all_cat_dims = all_states_cat.dims();
    let samples_bin_dims = samples_bin.dims();
    let samples_cat_dims = samples_cat.dims();

    let n_all_states = all_bin_dims[0];
    let n_bin = all_bin_dims[1];
    let n_cat = all_cat_dims[1];
    let n_samples = samples_bin_dims[0];

    // Build lookup table
    let mut state_to_idx: HashMap<(Vec<u8>, Vec<i32>), usize> = HashMap::new();

    for i in 0..n_all_states {
        let bin_state: Vec<u8> = (0..n_bin)
            .map(|k| (all_bin_data[i * n_bin + k] > 0.5) as u8)
            .collect();
        let cat_state: Vec<i32> = (0..n_cat).map(|k| all_cat_data[i * n_cat + k]).collect();
        state_to_idx.insert((bin_state, cat_state), i);
    }

    // Count samples
    let mut counts = vec![0usize; n_all_states];

    for s in 0..n_samples {
        let bin_sample: Vec<u8> = (0..samples_bin_dims[1])
            .map(|k| (samples_bin_data[s * samples_bin_dims[1] + k] > 0.5) as u8)
            .collect();
        let cat_sample: Vec<i32> = (0..samples_cat_dims[1])
            .map(|k| samples_cat_data[s * samples_cat_dims[1] + k])
            .collect();

        if let Some(&idx) = state_to_idx.get(&(bin_sample, cat_sample)) {
            counts[idx] += 1;
        }
    }

    // Normalize
    let total = n_samples as f32;
    let normalized: Vec<f32> = counts.iter().map(|&c| c as f32 / total).collect();

    Tensor::<WgpuBackend, 1>::from_data(normalized.as_slice(), device)
}

/// Sample from a model and compare empirical distribution to exact Boltzmann.
/// Returns (empirical_dist, exact_dist)
#[allow(dead_code)]
pub fn sample_and_compare_distribution<E: AbstractEBM>(
    key: RngKey,
    ebm: &E,
    program: &BlockSamplingProgram,
    clamp_vals: &[Tensor<WgpuBackend, 1>],
    schedule: &SamplingSchedule,
    n_cats: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<(Tensor<WgpuBackend, 1>, Tensor<WgpuBackend, 1>), String> {
    // Count binary and categorical nodes
    let mut all_binary_nodes = Vec::new();
    let mut all_cat_nodes = Vec::new();

    for block in &program.gibbs_spec.free_blocks {
        for node in block.nodes() {
            match node.node_type() {
                NodeType::Spin => all_binary_nodes.push(node.clone()),
                NodeType::Categorical { .. } => all_cat_nodes.push(node.clone()),
                NodeType::Continuous => {
                    panic!("Continuous nodes not supported in discrete EBM tests")
                }
                NodeType::Spherical { .. } => {
                    panic!("Spherical nodes not supported in discrete EBM tests")
                }
            }
        }
    }

    // Create initial state
    let mut init_free = Vec::new();
    for block in &program.gibbs_spec.free_blocks {
        let n = block.len();
        init_free.push(Tensor::<WgpuBackend, 1>::zeros([n], device));
    }

    // Observe blocks
    let mut observe_blocks = Vec::new();
    if !all_binary_nodes.is_empty() {
        observe_blocks.push(Block::new(all_binary_nodes.clone())?);
    }
    if !all_cat_nodes.is_empty() {
        observe_blocks.push(Block::new(all_cat_nodes.clone())?);
    }

    // Sample
    let all_block_samples = sample_states(
        key,
        program,
        schedule,
        init_free,
        clamp_vals,
        &observe_blocks,
        device,
    )?;

    // Collect samples
    let samples_bin = if !all_binary_nodes.is_empty() && !all_block_samples.is_empty() {
        all_block_samples[0].clone()
    } else {
        Tensor::<WgpuBackend, 2>::zeros([schedule.n_samples, 0], device)
    };

    let samples_cat = if !all_cat_nodes.is_empty()
        && all_block_samples.len() > (!all_binary_nodes.is_empty() as usize)
    {
        all_block_samples[1].clone().int()
    } else {
        Tensor::<WgpuBackend, 2, Int>::zeros([schedule.n_samples, 0], device)
    };

    // Generate all possible states
    let (all_bin_states, all_cat_states) =
        generate_all_states_bin_cat(all_binary_nodes.len(), all_cat_nodes.len(), n_cats, device);

    // Count samples
    let empirical_dist = count_samples(
        &all_bin_states,
        &all_cat_states,
        &samples_bin,
        &samples_cat,
        device,
    );

    // Compute exact distribution via energies
    // TODO: Implement vmap-style batched energy computation
    let n_states = all_bin_states.dims()[0];
    let mut energies = Vec::with_capacity(n_states);

    // For now, compute energies one at a time (inefficient but correct)
    for i in 0..n_states {
        let bin_dims = all_bin_states.dims();
        let cat_dims = all_cat_states.dims();

        // Extract single state
        let bin_slice: Tensor<WgpuBackend, 2> =
            all_bin_states.clone().slice([i..i + 1, 0..bin_dims[1]]);
        let bin_state: Tensor<WgpuBackend, 1> = bin_slice.reshape([bin_dims[1] as i32]);

        let cat_slice: Tensor<WgpuBackend, 2, Int> =
            all_cat_states.clone().slice([i..i + 1, 0..cat_dims[1]]);
        let cat_state: Tensor<WgpuBackend, 1, Int> = cat_slice.reshape([cat_dims[1] as i32]);

        // Combine with clamped values - bin_state is already float, cat_state needs conversion
        let mut full_state: Vec<Tensor<WgpuBackend, 1>> = vec![bin_state, cat_state.float()];
        for c in clamp_vals {
            full_state.push(c.clone());
        }

        // Build blocks for energy computation
        let mut energy_blocks = observe_blocks.clone();
        for clamped_block in &program.gibbs_spec.clamped_blocks {
            energy_blocks.push(clamped_block.clone());
        }

        let energy = ebm.energy(&full_state, &energy_blocks, device);
        let energy_val: Vec<f32> = energy.into_data().to_vec().expect("read energy");
        energies.push(energy_val[0]);
    }

    // Convert to probabilities via softmax
    let max_neg_energy = energies.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let unnormalized: Vec<f32> = energies
        .iter()
        .map(|e| (-e - max_neg_energy).exp())
        .collect();
    let sum: f32 = unnormalized.iter().sum();
    let exact_probs: Vec<f32> = unnormalized.iter().map(|p| p / sum).collect();

    let exact_dist = Tensor::<WgpuBackend, 1>::from_data(exact_probs.as_slice(), device);

    Ok((empirical_dist, exact_dist))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_generate_all_states_binary() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};

        ensure_backend();
        let device = init_gpu_device();

        let states = generate_all_states_binary(3, &device);
        let dims = states.dims();

        assert_eq!(dims[0], 8, "Should have 2^3 = 8 states");
        assert_eq!(dims[1], 3, "Should have 3 binary variables");

        // First state should be [0,0,0], last should be [1,1,1]
        let data: Vec<f32> = states.into_data().to_vec().expect("read data");
        assert_eq!(&data[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&data[21..24], &[1.0, 1.0, 1.0]);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_generate_all_states_categorical() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};

        ensure_backend();
        let device = init_gpu_device();

        let states = generate_all_states_categorical(2, 3, &device);
        let dims = states.dims();

        assert_eq!(dims[0], 9, "Should have 3^2 = 9 states");
        assert_eq!(dims[1], 2, "Should have 2 categorical variables");
    }
}
