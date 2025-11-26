//! Gaussian PGM sampling example.
//!
//! This example demonstrates continuous variable sampling using the GaussianSampler.
//! It creates a 5x5 grid of continuous nodes, defines a Gaussian distribution via
//! inverse covariance matrix, and verifies the sampled covariances match theory.
//!
//! This is a port of Part 1 of Python's `01_all_of_thrml.ipynb`.

use burn::tensor::{Distribution, Tensor};
use indexmap::IndexMap;
use std::collections::HashMap;
use thrml_core::backend::{ensure_backend, init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::interaction::{InteractionData, InteractionGroup};
use thrml_core::node::{Node, NodeType, TensorSpec};
use thrml_samplers::{
    BlockGibbsSpec, BlockSamplingProgram, GaussianSampler, RngKey, SamplingSchedule,
};

/// Generate a 2D grid of continuous nodes with bipartite coloring.
///
/// Returns:
/// - `all_nodes`: All nodes in the grid
/// - `color0`: Nodes in first color group
/// - `color1`: Nodes in second color group  
/// - `edges`: Pairs of adjacent node indices
fn generate_grid_graph(
    rows: usize,
    cols: usize,
) -> (Vec<Node>, Vec<Node>, Vec<Node>, Vec<(usize, usize)>) {
    // Create nodes in row-major order
    let mut all_nodes = Vec::new();
    let mut node_coords: HashMap<(usize, usize), usize> = HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let node = Node::new(NodeType::Continuous);
            node_coords.insert((r, c), all_nodes.len());
            all_nodes.push(node);
        }
    }

    // Bipartite coloring: checkerboard pattern
    let mut color0 = Vec::new();
    let mut color1 = Vec::new();

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            if (r + c) % 2 == 0 {
                color0.push(all_nodes[idx].clone());
            } else {
                color1.push(all_nodes[idx].clone());
            }
        }
    }

    // Generate edges (only horizontal and vertical neighbors)
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let idx1 = r * cols + c;
            // Right neighbor
            if c + 1 < cols {
                let idx2 = r * cols + (c + 1);
                edges.push((idx1, idx2));
            }
            // Down neighbor
            if r + 1 < rows {
                let idx2 = (r + 1) * cols + c;
                edges.push((idx1, idx2));
            }
        }
    }

    (all_nodes, color0, color1, edges)
}

fn main() {
    println!("=== Gaussian PGM Example ===\n");

    // Initialize GPU
    ensure_backend();
    let device = init_gpu_device();

    // Parameters
    let rows = 5;
    let cols = 5;
    let n_batches = 100;
    let n_samples = 1000;
    let seed = 4242u64;

    // Generate grid graph
    let (all_nodes, color0, color1, edges) = generate_grid_graph(rows, cols);
    let n_nodes = all_nodes.len();
    let n_edges = edges.len();

    println!(
        "Grid: {}x{} = {} nodes, {} edges",
        rows, cols, n_nodes, n_edges
    );
    println!(
        "Color 0: {} nodes, Color 1: {} nodes\n",
        color0.len(),
        color1.len()
    );

    // Create RNG key
    let mut key = RngKey::new(seed);

    // Generate random inverse covariance matrix
    // Diagonal: uniform [1, 2]
    let diag: Tensor<WgpuBackend, 1> =
        Tensor::random([n_nodes], Distribution::Uniform(1.0, 2.0), &device);
    let diag_data: Vec<f32> = diag.clone().into_data().to_vec().expect("read diag");

    // Off-diagonal: uniform [-0.25, 0.25] (ensures PSD)
    let off_diag: Tensor<WgpuBackend, 1> =
        Tensor::random([n_edges], Distribution::Uniform(-0.25, 0.25), &device);
    let off_diag_data: Vec<f32> = off_diag
        .clone()
        .into_data()
        .to_vec()
        .expect("read off_diag");

    // Construct full precision matrix for verification
    let mut inv_cov_mat = vec![vec![0.0f32; n_nodes]; n_nodes];
    for i in 0..n_nodes {
        inv_cov_mat[i][i] = diag_data[i];
    }
    for (idx, &(i, j)) in edges.iter().enumerate() {
        inv_cov_mat[i][j] = off_diag_data[idx];
        inv_cov_mat[j][i] = off_diag_data[idx];
    }

    // Generate random mean vector
    let (_key1, key2) = key.split_two();
    key = key2;
    let mean_vec: Tensor<WgpuBackend, 1> =
        Tensor::random([n_nodes], Distribution::Normal(0.0, 1.0), &device);
    let mean_data: Vec<f32> = mean_vec.clone().into_data().to_vec().expect("read mean");

    // Compute bias: b = -A * mu
    let mut bias_data = vec![0.0f32; n_nodes];
    for i in 0..n_nodes {
        for j in 0..n_nodes {
            bias_data[i] -= inv_cov_mat[i][j] * mean_data[j];
        }
    }
    let bias: Tensor<WgpuBackend, 1> = Tensor::from_data(bias_data.as_slice(), &device);

    println!("Setting up factors...");

    // Create blocks
    let block_all = Block::new(all_nodes.clone()).expect("create block");
    let block0 = Block::new(color0.clone()).expect("create block 0");
    let block1 = Block::new(color1.clone()).expect("create block 1");

    // Create edge blocks for coupling factor
    let edge_nodes_i: Vec<Node> = edges.iter().map(|(i, _)| all_nodes[*i].clone()).collect();
    let edge_nodes_j: Vec<Node> = edges.iter().map(|(_, j)| all_nodes[*j].clone()).collect();
    let edge_block_i = Block::new(edge_nodes_i).expect("create edge block i");
    let edge_block_j = Block::new(edge_nodes_j).expect("create edge block j");

    // Create interaction groups
    let mut interaction_groups = Vec::new();

    // 1. Quadratic factor (diagonal of precision matrix) - for variance
    // We need 1/A_ii (inverse variance)
    let inverse_diag: Tensor<WgpuBackend, 1> = diag.clone().recip();
    let inverse_diag_2d: Tensor<WgpuBackend, 2> = inverse_diag.reshape([n_nodes as i32, 1]);

    let quadratic_ig = InteractionGroup::with_data(
        InteractionData::Quadratic {
            inverse_weights: inverse_diag_2d,
        },
        block_all.clone(),
        vec![],
        0,
    )
    .expect("create quadratic interaction group");
    interaction_groups.push(quadratic_ig);

    // 2. Linear factor (bias term) - for mean contribution
    let bias_2d: Tensor<WgpuBackend, 2> = bias.reshape([n_nodes as i32, 1]);

    let linear_ig = InteractionGroup::with_data(
        InteractionData::Linear { weights: bias_2d },
        block_all.clone(),
        vec![],
        0,
    )
    .expect("create linear interaction group");
    interaction_groups.push(linear_ig);

    // 3. Coupling factors (off-diagonal precision) - i -> j and j -> i
    let off_diag_2d: Tensor<WgpuBackend, 2> = off_diag.clone().reshape([n_edges as i32, 1]);

    // i -> j coupling
    let coupling_ig1 = InteractionGroup::with_data(
        InteractionData::Linear {
            weights: off_diag_2d.clone(),
        },
        edge_block_i.clone(),
        vec![edge_block_j.clone()],
        0,
    )
    .expect("create coupling interaction group 1");
    interaction_groups.push(coupling_ig1);

    // j -> i coupling
    let coupling_ig2 = InteractionGroup::with_data(
        InteractionData::Linear {
            weights: off_diag_2d,
        },
        edge_block_j.clone(),
        vec![edge_block_i.clone()],
        0,
    )
    .expect("create coupling interaction group 2");
    interaction_groups.push(coupling_ig2);

    println!("Created {} interaction groups", interaction_groups.len());

    // Create block spec
    let free_super_blocks = vec![vec![block0.clone()], vec![block1.clone()]];
    let clamped_blocks = vec![];

    let mut node_shape_dtypes = IndexMap::new();
    node_shape_dtypes.insert(NodeType::Continuous, TensorSpec::for_continuous());

    let gibbs_spec = BlockGibbsSpec::new(free_super_blocks, clamped_blocks, node_shape_dtypes)
        .expect("create gibbs spec");

    // Create samplers (one per free block)
    let samplers: Vec<Box<dyn thrml_samplers::DynConditionalSampler>> = vec![
        Box::new(GaussianSampler::new()),
        Box::new(GaussianSampler::new()),
    ];

    // Create sampling program
    println!("Creating BlockSamplingProgram...");
    let program = BlockSamplingProgram::new(gibbs_spec, samplers, interaction_groups)
        .expect("create sampling program");

    println!(
        "Program created with {} free blocks\n",
        program.gibbs_spec.free_blocks.len()
    );

    // Sampling schedule
    let schedule = SamplingSchedule::new(0, n_samples, 5);

    println!("Sampling {} batches x {} samples...", n_batches, n_samples);

    // Initialize state
    let n0 = color0.len();
    let n1 = color1.len();

    // Accumulate moments
    let mut first_moments = vec![0.0f32; n_edges];
    let mut second_moments = vec![0.0f32; n_edges];
    let mut product_moments = vec![0.0f32; n_edges];

    for batch in 0..n_batches {
        if batch % 20 == 0 {
            println!("  Batch {}/{}", batch, n_batches);
        }

        // Initialize with small random values
        let (_key_init, key_new) = key.split_two();
        key = key_new;

        let init0: Tensor<WgpuBackend, 1> =
            Tensor::random([n0], Distribution::Normal(0.0, 0.1), &device);
        let init1: Tensor<WgpuBackend, 1> =
            Tensor::random([n1], Distribution::Normal(0.0, 0.1), &device);

        let mut state_free = vec![init0, init1];

        // Run sampling
        let (_key_sample, key_next) = key.split_two();
        key = key_next;

        for _ in 0..schedule.n_warmup {
            // Warmup iterations
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
                        let new_state = program.sample_single_block(
                            block_idx,
                            step_key,
                            &state_free,
                            &[],
                            &device,
                        );
                        state_free[block_idx] = new_state;
                    }
                }
            }

            // Reconstruct global state and accumulate moments for edges
            // Get state values from each block
            let state0: Vec<f32> = state_free[0]
                .clone()
                .into_data()
                .to_vec()
                .expect("read state0");
            let state1: Vec<f32> = state_free[1]
                .clone()
                .into_data()
                .to_vec()
                .expect("read state1");

            // Map node index to value using the block spec
            // all_nodes is in the same order as the original grid creation
            let mut state_data = vec![0.0f32; n_nodes];

            // Color 0 nodes map to indices in block 0
            for (local_idx, node) in color0.iter().enumerate() {
                // Find original index in all_nodes
                if let Some(global_idx) = all_nodes.iter().position(|n| n.id() == node.id()) {
                    state_data[global_idx] = state0[local_idx];
                }
            }
            // Color 1 nodes map to indices in block 1
            for (local_idx, node) in color1.iter().enumerate() {
                if let Some(global_idx) = all_nodes.iter().position(|n| n.id() == node.id()) {
                    state_data[global_idx] = state1[local_idx];
                }
            }

            for (edge_idx, &(i, j)) in edges.iter().enumerate() {
                first_moments[edge_idx] += state_data[i];
                second_moments[edge_idx] += state_data[j];
                product_moments[edge_idx] += state_data[i] * state_data[j];
            }
        }
    }

    // Compute averages
    let total_samples = (n_batches * n_samples) as f32;
    for edge_idx in 0..n_edges {
        first_moments[edge_idx] /= total_samples;
        second_moments[edge_idx] /= total_samples;
        product_moments[edge_idx] /= total_samples;
    }

    // Compute sampled covariances
    let mut sampled_covs = vec![0.0f32; n_edges];
    for edge_idx in 0..n_edges {
        sampled_covs[edge_idx] =
            product_moments[edge_idx] - first_moments[edge_idx] * second_moments[edge_idx];
    }

    println!("\nComputing theoretical covariances...");

    // Compute theoretical covariances by inverting precision matrix
    // Using simple Gaussian elimination (for small matrices)
    let cov_mat = invert_matrix(&inv_cov_mat);

    let mut real_covs = vec![0.0f32; n_edges];
    for (edge_idx, &(i, j)) in edges.iter().enumerate() {
        real_covs[edge_idx] = cov_mat[i][j];
    }

    // Compute max relative error
    let max_real = real_covs.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
    let mut max_error = 0.0f32;
    for edge_idx in 0..n_edges {
        let error = (sampled_covs[edge_idx] - real_covs[edge_idx]).abs();
        if error > max_error {
            max_error = error;
        }
    }
    let relative_error = max_error / max_real;

    println!("\n=== Results ===");
    println!("Max absolute error: {:.6}", max_error);
    println!("Max |real covariance|: {:.6}", max_real);
    println!("Relative error: {:.4}%", relative_error * 100.0);

    if relative_error < 0.05 {
        println!("\n✓ SUCCESS: Relative error < 5%");
    } else {
        println!("\n✗ WARNING: Relative error >= 5% (may need more samples)");
    }

    // Print a few sample covariances
    println!("\nSample covariance comparisons:");
    for i in 0..5.min(n_edges) {
        let (node_i, node_j) = edges[i];
        println!(
            "  Edge ({}, {}): sampled={:.4}, theoretical={:.4}, diff={:.4}",
            node_i,
            node_j,
            sampled_covs[i],
            real_covs[i],
            (sampled_covs[i] - real_covs[i]).abs()
        );
    }
}

/// Simple matrix inversion using Gaussian elimination with partial pivoting.
fn invert_matrix(mat: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = mat.len();
    let mut a: Vec<Vec<f64>> = mat
        .iter()
        .map(|row| row.iter().map(|&x| x as f64).collect())
        .collect();
    let mut inv = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }

    // Forward elimination
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[max_row][col].abs() {
                max_row = row;
            }
        }
        a.swap(col, max_row);
        inv.swap(col, max_row);

        let pivot = a[col][col];
        if pivot.abs() < 1e-10 {
            panic!("Matrix is singular");
        }

        // Scale pivot row
        for j in 0..n {
            a[col][j] /= pivot;
            inv[col][j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = a[row][col];
                for j in 0..n {
                    a[row][j] -= factor * a[col][j];
                    inv[row][j] -= factor * inv[col][j];
                }
            }
        }
    }

    inv.iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect()
}
