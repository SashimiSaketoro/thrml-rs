//! Gaussian-Bernoulli mixed EBM example.
//!
//! This example demonstrates sampling from a mixed model with both
//! continuous (Gaussian) and discrete (spin) variables.
//!
//! The energy function is:
//! E(x,s) = E_G(x) + E_GB(x,s) + E_B(s)
//!
//! where:
//! - E_G(x) is the Gaussian energy (quadratic + linear)
//! - E_GB(x,s) is the coupling between continuous and spin variables
//! - E_B(s) is the spin energy (biases + pairwise)
//!
//! This is a port of Part 2 of Python's `01_all_of_thrml.ipynb`.

use burn::tensor::{Distribution, Tensor};
use indexmap::IndexMap;
use thrml_core::backend::{ensure_backend, init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::interaction::{InteractionData, InteractionGroup};
use thrml_core::node::{Node, NodeType, TensorSpec};
use thrml_samplers::{
    BlockGibbsSpec, BlockSamplingProgram, GaussianSampler, RngKey, SamplingSchedule,
    SpinGibbsConditional,
};

fn main() {
    println!("=== Gaussian-Bernoulli Mixed EBM Example ===\n");

    // Initialize GPU
    ensure_backend();
    let device = init_gpu_device();

    // Parameters
    let n_continuous = 16; // 4x4 grid of continuous nodes
    let n_spin = 9; // 3x3 grid of spin nodes
    let seed = 12345u64;
    let n_samples = 500;

    println!("Continuous nodes: {}", n_continuous);
    println!("Spin nodes: {}", n_spin);

    // Create nodes
    let continuous_nodes: Vec<Node> = (0..n_continuous)
        .map(|_| Node::new(NodeType::Continuous))
        .collect();
    let spin_nodes: Vec<Node> = (0..n_spin).map(|_| Node::new(NodeType::Spin)).collect();

    // Create blocks
    // Color 0: first half of continuous, first half of spin
    // Color 1: second half of continuous, second half of spin
    let cont_half = n_continuous / 2;
    let spin_half = n_spin / 2;

    let cont_block0 = Block::new(continuous_nodes[..cont_half].to_vec()).expect("cont block 0");
    let cont_block1 = Block::new(continuous_nodes[cont_half..].to_vec()).expect("cont block 1");
    let spin_block0 = Block::new(spin_nodes[..spin_half].to_vec()).expect("spin block 0");
    let spin_block1 = Block::new(spin_nodes[spin_half..].to_vec()).expect("spin block 1");

    println!("\nBlocks:");
    println!("  Continuous block 0: {} nodes", cont_block0.len());
    println!("  Continuous block 1: {} nodes", cont_block1.len());
    println!("  Spin block 0: {} nodes", spin_block0.len());
    println!("  Spin block 1: {} nodes", spin_block1.len());

    // Create RNG key
    let mut key = RngKey::new(seed);

    // === Create factors ===

    println!("\nCreating factors...");

    // 1. Quadratic factor for continuous nodes (diagonal of precision)
    let precision_diag: Tensor<WgpuBackend, 1> =
        Tensor::random([n_continuous], Distribution::Uniform(2.0, 3.0), &device);
    let inverse_precision: Tensor<WgpuBackend, 1> = precision_diag.recip();

    // 2. Linear factor for continuous nodes (bias)
    let (_k, k2) = key.split_two();
    key = k2;
    let cont_bias: Tensor<WgpuBackend, 1> =
        Tensor::random([n_continuous], Distribution::Normal(0.0, 1.0), &device);

    // 3. Continuous-continuous coupling (some edges)
    let n_cc_edges = n_continuous / 2; // Approximate number of edges
    let cc_weights: Tensor<WgpuBackend, 1> =
        Tensor::random([n_cc_edges], Distribution::Uniform(-0.1, 0.1), &device);

    // Create edge blocks for continuous coupling
    let cc_edge_i: Vec<Node> = continuous_nodes.iter().take(n_cc_edges).cloned().collect();
    let cc_edge_j: Vec<Node> = continuous_nodes
        .iter()
        .skip(1)
        .take(n_cc_edges)
        .cloned()
        .collect();
    let cc_block_i = Block::new(cc_edge_i).expect("cc edge block i");
    let cc_block_j = Block::new(cc_edge_j).expect("cc edge block j");

    // 4. Spin biases
    let spin_biases: Tensor<WgpuBackend, 1> =
        Tensor::random([n_spin], Distribution::Normal(0.0, 0.5), &device);

    // 5. Spin-spin coupling (some edges)
    let n_ss_edges = n_spin / 2;
    let ss_weights: Tensor<WgpuBackend, 1> =
        Tensor::random([n_ss_edges], Distribution::Normal(0.0, 0.3), &device);

    let ss_edge_i: Vec<Node> = spin_nodes.iter().take(n_ss_edges).cloned().collect();
    let ss_edge_j: Vec<Node> = spin_nodes
        .iter()
        .skip(1)
        .take(n_ss_edges)
        .cloned()
        .collect();
    let ss_block_i = Block::new(ss_edge_i).expect("ss edge block i");
    let ss_block_j = Block::new(ss_edge_j).expect("ss edge block j");

    // 6. Spin-continuous coupling
    let n_sc_edges = n_spin.min(n_continuous);
    let sc_weights: Tensor<WgpuBackend, 1> =
        Tensor::random([n_sc_edges], Distribution::Normal(0.0, 0.5), &device);

    let sc_edge_spin: Vec<Node> = spin_nodes.iter().take(n_sc_edges).cloned().collect();
    let sc_edge_cont: Vec<Node> = continuous_nodes.iter().take(n_sc_edges).cloned().collect();
    let sc_block_spin = Block::new(sc_edge_spin).expect("sc edge block spin");
    let sc_block_cont = Block::new(sc_edge_cont).expect("sc edge block cont");

    // === Create interaction groups ===

    let mut interaction_groups: Vec<InteractionGroup> = Vec::new();

    // Continuous node interactions
    let cont_all_block = Block::new(continuous_nodes).expect("cont all block");
    let spin_all_block = Block::new(spin_nodes).expect("spin all block");

    // Quadratic (variance) for continuous
    let inv_prec_2d: Tensor<WgpuBackend, 2> = inverse_precision.reshape([n_continuous as i32, 1]);
    interaction_groups.push(
        InteractionGroup::with_data(
            InteractionData::Quadratic {
                inverse_weights: inv_prec_2d,
            },
            cont_all_block.clone(),
            vec![],
            0,
        )
        .expect("quadratic ig"),
    );

    // Linear (bias) for continuous
    let cont_bias_2d: Tensor<WgpuBackend, 2> = cont_bias.reshape([n_continuous as i32, 1]);
    interaction_groups.push(
        InteractionGroup::with_data(
            InteractionData::Linear {
                weights: cont_bias_2d,
            },
            cont_all_block,
            vec![],
            0,
        )
        .expect("linear ig"),
    );

    // Continuous-continuous coupling
    let cc_weights_2d: Tensor<WgpuBackend, 2> = cc_weights.reshape([n_cc_edges as i32, 1]);
    interaction_groups.push(
        InteractionGroup::with_data(
            InteractionData::Linear {
                weights: cc_weights_2d.clone(),
            },
            cc_block_i.clone(),
            vec![cc_block_j.clone()],
            0,
        )
        .expect("cc coupling ig 1"),
    );
    interaction_groups.push(
        InteractionGroup::with_data(
            InteractionData::Linear {
                weights: cc_weights_2d,
            },
            cc_block_j,
            vec![cc_block_i],
            0,
        )
        .expect("cc coupling ig 2"),
    );

    // Spin biases - as a 3D tensor [n_spin, 1, 1]
    let spin_bias_3d: Tensor<WgpuBackend, 3> = spin_biases.reshape([n_spin as i32, 1, 1]);
    interaction_groups.push(
        InteractionGroup::new(spin_bias_3d, spin_all_block, vec![], 0).expect("spin bias ig"),
    );

    // Spin-spin coupling - as 3D tensor [n_edges, 1, 1]
    let ss_weights_3d: Tensor<WgpuBackend, 3> = ss_weights.reshape([n_ss_edges as i32, 1, 1]);
    interaction_groups.push(
        InteractionGroup::new(
            ss_weights_3d.clone(),
            ss_block_i.clone(),
            vec![ss_block_j.clone()],
            1,
        )
        .expect("ss coupling ig 1"),
    );
    interaction_groups.push(
        InteractionGroup::new(ss_weights_3d, ss_block_j, vec![ss_block_i], 1)
            .expect("ss coupling ig 2"),
    );

    // Spin-continuous coupling - using Linear interaction
    // When spin is head: contributes to spin's field based on continuous neighbor
    // When continuous is head: contributes to continuous mean based on spin neighbor
    let sc_weights_2d: Tensor<WgpuBackend, 2> = sc_weights.clone().reshape([n_sc_edges as i32, 1]);
    let sc_weights_3d: Tensor<WgpuBackend, 3> = sc_weights.reshape([n_sc_edges as i32, 1, 1]);

    // Spin <- Continuous coupling (affects spin sampling)
    interaction_groups.push(
        InteractionGroup::new(
            sc_weights_3d,
            sc_block_spin.clone(),
            vec![sc_block_cont.clone()],
            0, // continuous tail, not spin
        )
        .expect("sc coupling ig spin"),
    );

    // Continuous <- Spin coupling (affects continuous sampling)
    interaction_groups.push(
        InteractionGroup::with_data(
            InteractionData::Linear {
                weights: sc_weights_2d,
            },
            sc_block_cont,
            vec![sc_block_spin],
            0, // spin tail but for linear interaction it doesn't matter
        )
        .expect("sc coupling ig cont"),
    );

    println!("Created {} interaction groups", interaction_groups.len());

    // === Create sampling program ===

    // Free super blocks:
    // Step 0: continuous block 0 + spin block 0
    // Step 1: continuous block 1 + spin block 1
    let free_super_blocks = vec![
        vec![cont_block0, spin_block0],
        vec![cont_block1, spin_block1],
    ];

    let mut node_shape_dtypes = IndexMap::new();
    node_shape_dtypes.insert(NodeType::Continuous, TensorSpec::for_continuous());
    node_shape_dtypes.insert(NodeType::Spin, TensorSpec::for_spin());

    let gibbs_spec = BlockGibbsSpec::new(free_super_blocks, vec![], node_shape_dtypes)
        .expect("create gibbs spec");

    println!(
        "\nGibbs spec: {} free blocks in {} sampling steps",
        gibbs_spec.free_blocks.len(),
        gibbs_spec.sampling_order.len()
    );

    // Create samplers (one per free block)
    // Order: cont_block0, spin_block0, cont_block1, spin_block1
    let samplers: Vec<Box<dyn thrml_samplers::DynConditionalSampler>> = vec![
        Box::new(GaussianSampler::new()),
        Box::new(SpinGibbsConditional::new()),
        Box::new(GaussianSampler::new()),
        Box::new(SpinGibbsConditional::new()),
    ];

    println!("Creating BlockSamplingProgram...");
    let program = BlockSamplingProgram::new(gibbs_spec, samplers, interaction_groups)
        .expect("create sampling program");

    println!("Program created successfully!");

    // === Run sampling ===

    let schedule = SamplingSchedule::new(50, n_samples, 3);
    println!(
        "\nSampling: {} warmup, {} samples, {} steps/sample",
        schedule.n_warmup, schedule.n_samples, schedule.steps_per_sample
    );

    // Initialize state
    let init_cont0: Tensor<WgpuBackend, 1> =
        Tensor::random([cont_half], Distribution::Normal(0.0, 0.1), &device);

    // Initialize spin as random 0/1 values
    let uniform0: Tensor<WgpuBackend, 1> =
        Tensor::random([spin_half], Distribution::Uniform(0.0, 1.0), &device);
    let threshold0: Tensor<WgpuBackend, 1> = Tensor::full([spin_half], 0.5, &device);
    let init_spin0: Tensor<WgpuBackend, 1> = uniform0.lower_equal(threshold0).float();

    let init_cont1: Tensor<WgpuBackend, 1> = Tensor::random(
        [n_continuous - cont_half],
        Distribution::Normal(0.0, 0.1),
        &device,
    );

    let uniform1: Tensor<WgpuBackend, 1> = Tensor::random(
        [n_spin - spin_half],
        Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let threshold1: Tensor<WgpuBackend, 1> = Tensor::full([n_spin - spin_half], 0.5, &device);
    let init_spin1: Tensor<WgpuBackend, 1> = uniform1.lower_equal(threshold1).float();

    let mut state_free = vec![init_cont0, init_spin0, init_cont1, init_spin1];

    // Track statistics
    let mut cont_means = vec![0.0f32; n_continuous];
    let mut spin_means = vec![0.0f32; n_spin];

    println!("\nRunning sampling...");

    // Warmup
    for _ in 0..schedule.n_warmup {
        for step_indices in &program.gibbs_spec.sampling_order {
            for &block_idx in step_indices {
                let (step_key, new_key) = key.split_two();
                key = new_key;
                let new_state =
                    program.sample_single_block(block_idx, step_key, &state_free, &[], &device);
                state_free[block_idx] = new_state;
            }
        }
    }

    // Collect samples
    for _sample_idx in 0..schedule.n_samples {
        // Take steps between samples
        for _ in 0..schedule.steps_per_sample {
            for step_indices in &program.gibbs_spec.sampling_order {
                for &block_idx in step_indices {
                    let (step_key, new_key) = key.split_two();
                    key = new_key;
                    let new_state =
                        program.sample_single_block(block_idx, step_key, &state_free, &[], &device);
                    state_free[block_idx] = new_state;
                }
            }
        }

        // Accumulate statistics
        let cont0: Vec<f32> = state_free[0]
            .clone()
            .into_data()
            .to_vec()
            .expect("read cont0");
        let cont1: Vec<f32> = state_free[2]
            .clone()
            .into_data()
            .to_vec()
            .expect("read cont1");
        let spin0: Vec<f32> = state_free[1]
            .clone()
            .into_data()
            .to_vec()
            .expect("read spin0");
        let spin1: Vec<f32> = state_free[3]
            .clone()
            .into_data()
            .to_vec()
            .expect("read spin1");

        for (i, &v) in cont0.iter().enumerate() {
            cont_means[i] += v;
        }
        for (i, &v) in cont1.iter().enumerate() {
            cont_means[cont_half + i] += v;
        }
        for (i, &v) in spin0.iter().enumerate() {
            spin_means[i] += v;
        }
        for (i, &v) in spin1.iter().enumerate() {
            spin_means[spin_half + i] += v;
        }
    }

    // Compute averages
    let n_samples_f = n_samples as f32;
    for m in &mut cont_means {
        *m /= n_samples_f;
    }
    for m in &mut spin_means {
        *m /= n_samples_f;
    }

    // Print results
    println!("\n=== Results ===");
    println!("\nContinuous variable means:");
    for (i, &m) in cont_means.iter().enumerate() {
        println!("  x_{}: {:.4}", i, m);
    }

    println!("\nSpin variable means (should be near 0.5 for unbiased):");
    for (i, &m) in spin_means.iter().enumerate() {
        let spin_val = 2.0f32.mul_add(m, -1.0); // Convert 0/1 to -1/+1
        println!("  s_{}: {:.4} (as ±1: {:.4})", i, m, spin_val);
    }

    // Compute overall statistics
    let cont_mean: f32 = cont_means.iter().sum::<f32>() / n_continuous as f32;
    let spin_mean: f32 = spin_means.iter().sum::<f32>() / n_spin as f32;

    println!("\n=== Summary ===");
    println!("Average continuous value: {:.4}", cont_mean);
    println!(
        "Average spin probability (0->1): {:.4} (±1 magnetization: {:.4})",
        spin_mean,
        2.0f32.mul_add(spin_mean, -1.0)
    );
    println!("\n✓ Mixed Gaussian-Bernoulli sampling completed successfully!");
}
