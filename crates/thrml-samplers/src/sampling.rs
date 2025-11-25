use crate::program::BlockSamplingProgram;
use crate::rng::RngKey;
use crate::schedule::SamplingSchedule;
/// High-level sampling functions for block Gibbs sampling.
///
/// This module provides the main user-facing APIs for sampling from probabilistic
/// graphical models using block Gibbs sampling.
use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_observers::observer::AbstractObserver;

/// Perform one iteration of sampling, visiting every block.
///
/// This function implements a single step of block Gibbs sampling, where each
/// free block is sampled in the order specified by the program's sampling_order.
///
/// # Arguments
///
/// * `key` - RNG key for this iteration
/// * `state_free` - Current states of free blocks (will be updated)
/// * `clamp_state` - States of clamped blocks (remain fixed)
/// * `program` - The block sampling program
///
/// # Returns
///
/// Updated free-block state list
pub fn sample_blocks(
    key: RngKey,
    state_free: &mut [Tensor<WgpuBackend, 1>],
    clamp_state: &[Tensor<WgpuBackend, 1>],
    program: &BlockSamplingProgram,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<(), String> {
    // Split key for each free block
    let n_blocks = program.gibbs_spec.free_blocks.len();
    let keys = key.split(n_blocks);

    // Iterate through sampling groups (superblocks)
    for sampling_group in &program.gibbs_spec.sampling_order {
        // Sample all blocks in this group (they can be sampled in parallel conceptually,
        // but we do them sequentially for now)
        let mut state_updates = Vec::new();
        for &block_idx in sampling_group {
            // Sample this block
            let samples = program.sample_single_block(
                block_idx,
                keys[block_idx],
                state_free,
                clamp_state,
                device,
            );
            state_updates.push((block_idx, samples));
        }

        // Update state_free with the new samples
        for (block_idx, new_state) in state_updates {
            state_free[block_idx] = new_state;
        }
    }

    Ok(())
}

/// Run multiple iterations of block sampling.
///
/// This function performs `n_iters` steps of block Gibbs sampling, accumulating
/// the state over iterations.
///
/// # Arguments
///
/// * `key` - RNG key
/// * `program` - The block sampling program
/// * `init_chain_state` - Initial state of free blocks
/// * `state_clamp` - Clamped block states
/// * `n_iters` - Number of iterations to run
///
/// # Returns
///
/// Final state after n_iters iterations
pub fn run_blocks(
    key: RngKey,
    program: &BlockSamplingProgram,
    mut init_chain_state: Vec<Tensor<WgpuBackend, 1>>,
    state_clamp: &[Tensor<WgpuBackend, 1>],
    n_iters: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<Vec<Tensor<WgpuBackend, 1>>, String> {
    if n_iters == 0 {
        return Ok(init_chain_state);
    }

    let keys = key.split(n_iters);

    for iter_key in keys {
        sample_blocks(
            iter_key,
            &mut init_chain_state,
            state_clamp,
            program,
            device,
        )?;
    }

    Ok(init_chain_state)
}

/// Sample states according to a schedule.
///
/// This is the main user-facing function for sampling. It runs warmup iterations,
/// then collects samples according to the schedule.
///
/// Convenience wrapper that builds a StateObserver and calls sample_with_observation.
///
/// # Arguments
///
/// * `key` - RNG key
/// * `program` - The block sampling program
/// * `schedule` - Sampling schedule (warmup, n_samples, steps_per_sample)
/// * `init_state_free` - Initial state of free blocks
/// * `state_clamp` - Clamped block states
/// * `nodes_to_sample` - Blocks to collect samples from
///
/// # Returns
///
/// Collected samples as a list of tensors, each with shape `[n_samples, ...]`
/// (one tensor per block in nodes_to_sample)
pub fn sample_states(
    key: RngKey,
    program: &BlockSamplingProgram,
    schedule: &SamplingSchedule,
    init_state_free: Vec<Tensor<WgpuBackend, 1>>,
    state_clamp: &[Tensor<WgpuBackend, 1>],
    nodes_to_sample: &[Block],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<Vec<Tensor<WgpuBackend, 2>>, String> {
    use thrml_observers::state_observer::StateObserver;

    // Create StateObserver for the blocks we want to sample
    let observer = StateObserver::new(nodes_to_sample.to_vec());
    observer.init(device);

    // Use sample_with_observation
    let (_final_carry, samples) = sample_with_observation(
        key,
        program,
        schedule,
        init_state_free,
        state_clamp,
        (),
        &observer,
        device,
    )?;

    Ok(samples)
}

/// Run the full chain and call an Observer after every recorded sample.
///
/// This function runs warmup iterations, then collects samples according to the schedule,
/// calling the observer after each sample to record observations.
///
/// # Arguments
///
/// * `key` - RNG key
/// * `program` - The sampling program
/// * `schedule` - Warm-up length, number of samples, number of steps between samples
/// * `init_chain_state` - Initial free-block state
/// * `state_clamp` - Clamped-block state
/// * `observation_carry_init` - Initial carry handed to the observer
/// * `f_observe` - Observer instance
///
/// # Returns
///
/// Tuple `(final_observer_carry, samples)` where `samples` is a list of tensors,
/// each with shape `\[n_samples, ...\]` (one tensor per block being observed)
#[allow(clippy::too_many_arguments)]
pub fn sample_with_observation<O: AbstractObserver>(
    key: RngKey,
    program: &BlockSamplingProgram,
    schedule: &SamplingSchedule,
    init_chain_state: Vec<Tensor<WgpuBackend, 1>>,
    state_clamp: &[Tensor<WgpuBackend, 1>],
    observation_carry_init: O::ObserveCarry,
    f_observe: &O,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<(O::ObserveCarry, Vec<Tensor<WgpuBackend, 2>>), String> {
    // Run warmup
    let (warmup_key, sample_key) = key.split_two();
    let warmup_state = run_blocks(
        warmup_key,
        program,
        init_chain_state,
        state_clamp,
        schedule.n_warmup,
        device,
    )?;

    // Call observer on warmup state
    let (mem, warmup_observation) = f_observe.observe(
        &program.gibbs_spec.spec,
        &warmup_state,
        state_clamp,
        observation_carry_init,
        0,
        device,
    );

    // Handle observers that don't return observations (like MomentAccumulatorObserver)
    // They only accumulate in carry, not in observations
    let warmup_obs = match warmup_observation {
        Some(obs) => obs,
        None => {
            // No observations returned - run the sampling loop but only track carry
            if schedule.n_samples <= 1 {
                return Ok((mem, Vec::new()));
            }

            let mut current_state = warmup_state;
            let mut current_carry = mem;

            let keys = sample_key.split(schedule.n_samples - 1);
            for (i, iter_key) in keys.iter().enumerate() {
                current_state = run_blocks(
                    *iter_key,
                    program,
                    current_state,
                    state_clamp,
                    schedule.steps_per_sample,
                    device,
                )?;

                let (new_carry, _) = f_observe.observe(
                    &program.gibbs_spec.spec,
                    &current_state,
                    state_clamp,
                    current_carry.clone(),
                    i + 1,
                    device,
                );
                current_carry = new_carry;
            }

            return Ok((current_carry, Vec::new()));
        }
    };

    // Handle case where n_samples <= 1
    if schedule.n_samples <= 1 {
        // Prepend dimension to warmup observation (equivalent to Python's [None])
        // Each tensor in warmup_obs has shape [n], we add dimension to get [1, n]
        let warmup_with_dim: Vec<Tensor<WgpuBackend, 2>> = warmup_obs
            .into_iter()
            .map(|t| t.unsqueeze_dim::<2>(0)) // Add dimension: [n] -> [1, n]
            .collect();
        return Ok((mem, warmup_with_dim));
    }

    // Collect remaining samples
    let mut current_state = warmup_state;
    let mut current_carry = mem;
    let mut all_observations = Vec::new();

    let keys = sample_key.split(schedule.n_samples - 1);
    for (i, iter_key) in keys.iter().enumerate() {
        // Run steps_per_sample iterations
        current_state = run_blocks(
            *iter_key,
            program,
            current_state,
            state_clamp,
            schedule.steps_per_sample,
            device,
        )?;

        // Call observer
        let (new_carry, observation) = f_observe.observe(
            &program.gibbs_spec.spec,
            &current_state,
            state_clamp,
            current_carry.clone(),
            i + 1,
            device,
        );
        current_carry = new_carry;

        if let Some(obs) = observation {
            all_observations.push(obs);
        }
    }

    // Prepend warmup observation to results
    // In Python: jnp.concatenate([_warmup[None], _rest], axis=0)
    // We need to stack along dimension 0
    // Each observation is Vec<Tensor<WgpuBackend, 1>> (one tensor per block)
    // We want to stack them to get Vec<Tensor<WgpuBackend, 2>> (one tensor per block, shape [n_samples, n_nodes])
    let n_blocks = warmup_obs.len();
    let mut final_observations = Vec::with_capacity(n_blocks);

    for block_idx in 0..n_blocks {
        // Start with warmup observation for this block
        let warmup_tensor = warmup_obs[block_idx].clone();
        let warmup_with_dim = warmup_tensor.unsqueeze_dim::<2>(0); // [n] -> [1, n]
        let mut to_stack = vec![warmup_with_dim];

        // Add all other observations for this block
        for obs_list in &all_observations {
            if block_idx < obs_list.len() {
                let obs_tensor = obs_list[block_idx].clone();
                to_stack.push(obs_tensor.unsqueeze_dim::<2>(0)); // [n] -> [1, n]
            }
        }

        // Concatenate along dimension 0: [1, n] + [1, n] + ... -> [n_samples, n]
        let stacked = Tensor::cat(to_stack, 0);
        final_observations.push(stacked);
    }

    Ok((current_carry, final_observations))
}
