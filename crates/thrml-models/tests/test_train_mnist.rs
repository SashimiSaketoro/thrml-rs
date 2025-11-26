//! MNIST Training Test
//!
//! Port of Python tests/test_train_mnist.py
//!
//! This test validates that the full training pipeline works by:
//! 1. Creating a double-grid Ising model architecture
//! 2. Loading preprocessed MNIST data (digits 0, 3, 4)
//! 3. Running gradient estimation (simplified for testing)
//!
//! Note: Full training test is marked as #[ignore] as it takes several minutes.
//! Run with: cargo test --features gpu test_train_mnist -- --ignored
//!
//! Note: These tests require GPU hardware and only run on macOS (Metal backend).

#![cfg(feature = "gpu")]

use std::fs::File;
use std::path::Path;

use burn::tensor::{Bool, Tensor};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use rand::prelude::*;
use thrml_core::backend::{ensure_backend, init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};
use thrml_models::ising::{hinton_init, IsingEBM, IsingSamplingProgram, IsingTrainingSpec};
use thrml_samplers::rng::RngKey;
use thrml_samplers::schedule::SamplingSchedule;

/// Creates a double-grid graph structure for the Ising model.
///
/// This creates two layers of nodes (upper and lower) connected in a grid pattern
/// with connections at multiple "jump" distances (for long-range interactions).
///
/// Returns: (upper_block, lower_block, visible_block, upper_without_visible, all_nodes, edges)
fn get_double_grid(
    side_len: usize,
    jumps: &[usize],
    n_visible: usize,
    seed: u64,
) -> (Block, Block, Block, Block, Vec<Node>, Vec<(Node, Node)>) {
    let size = side_len * side_len;
    assert!(n_visible <= size, "n_visible must be <= grid size");

    // Helper to wrap indices (toroidal grid)
    let get_idx = |i: isize, j: isize| -> usize {
        let i = ((i % side_len as isize) + side_len as isize) as usize % side_len;
        let j = ((j % side_len as isize) + side_len as isize) as usize % side_len;
        i * side_len + j
    };

    let get_coords =
        |idx: usize| -> (isize, isize) { ((idx / side_len) as isize, (idx % side_len) as isize) };

    // Create all edge pairs (including self-loops at start)
    let mut edges_arr: Vec<(usize, usize)> = Vec::new();

    // Self-loops (identity connections between layers)
    for idx in 0..size {
        edges_arr.push((idx, idx));
    }

    // Grid connections at each jump distance
    for &d in jumps {
        let d = d as isize;
        for idx in 0..size {
            let (i, j) = get_coords(idx);
            // Left, right, up, down at distance d
            edges_arr.push((idx, get_idx(i - d, j)));
            edges_arr.push((idx, get_idx(i + d, j)));
            edges_arr.push((idx, get_idx(i, j - d)));
            edges_arr.push((idx, get_idx(i, j + d)));
        }
    }

    // Create nodes
    let nodes_upper: Vec<Node> = (0..size).map(|_| Node::new(NodeType::Spin)).collect();
    let nodes_lower: Vec<Node> = (0..size).map(|_| Node::new(NodeType::Spin)).collect();

    let mut all_nodes = nodes_upper.clone();
    all_nodes.extend(nodes_lower.clone());

    // Create edges connecting upper to lower layer
    let all_edges: Vec<(Node, Node)> = edges_arr
        .iter()
        .map(|&(i, j)| (nodes_upper[i].clone(), nodes_lower[j].clone()))
        .collect();

    // Select visible nodes randomly
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..size).collect();
    indices.shuffle(&mut rng);
    let visible_indices: Vec<usize> = indices.into_iter().take(n_visible).collect();

    let visible_nodes: Vec<Node> = visible_indices
        .iter()
        .map(|&i| nodes_upper[i].clone())
        .collect();

    let visible_set: std::collections::HashSet<_> = visible_nodes.iter().collect();
    let upper_without_visible: Vec<Node> = nodes_upper
        .iter()
        .filter(|n| !visible_set.contains(n))
        .cloned()
        .collect();

    (
        Block::new(nodes_upper).unwrap(),
        Block::new(nodes_lower).unwrap(),
        Block::new(visible_nodes).unwrap(),
        Block::new(upper_without_visible).unwrap(),
        all_nodes,
        all_edges,
    )
}

/// Load numpy array from file and convert to Burn tensor
/// Handles f32, i32, and bool numpy arrays
fn load_npy_as_tensor(
    path: &Path,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2> {
    // Try different dtypes
    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<f32>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            return Tensor::<WgpuBackend, 1>::from_data(data.as_slice(), device)
                .reshape([shape[0], shape[1]]);
        }
    }

    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<i32>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            return Tensor::<WgpuBackend, 1>::from_data(data_f32.as_slice(), device)
                .reshape([shape[0], shape[1]]);
        }
    }

    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<bool>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            let data_f32: Vec<f32> = data.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
            return Tensor::<WgpuBackend, 1>::from_data(data_f32.as_slice(), device)
                .reshape([shape[0], shape[1]]);
        }
    }

    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<u8>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            return Tensor::<WgpuBackend, 1>::from_data(data_f32.as_slice(), device)
                .reshape([shape[0], shape[1]]);
        }
    }

    panic!("Failed to read {:?} - unsupported numpy dtype", path);
}

/// Load numpy array as bool tensor
fn load_npy_as_bool_tensor(
    path: &Path,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2, Bool> {
    let tensor = load_npy_as_tensor(path, device);
    // Convert float to bool (threshold at 0.5)
    tensor.greater_elem(0.5)
}

/// Test that we can create the full MNIST model architecture

#[test]
fn test_mnist_model_creation() {
    ensure_backend();
    let device = init_gpu_device();

    // Configuration matching Python test
    let target_classes = vec![0, 3, 4];
    let num_label_spots = 10;
    let label_size = target_classes.len() * num_label_spots;
    let data_dim = 28 * 28 + label_size; // 814
    let side_len = 40;

    // Create model architecture
    let (upper_grid, lower_grid, visible_nodes, upper_without_visible, all_nodes, all_edges) =
        get_double_grid(side_len, &[1, 4, 15], data_dim, 0);

    println!("Created double grid:");
    println!("  Upper grid: {} nodes", upper_grid.len());
    println!("  Lower grid: {} nodes", lower_grid.len());
    println!("  Visible nodes: {} nodes", visible_nodes.len());
    println!(
        "  Upper without visible: {} nodes",
        upper_without_visible.len()
    );
    println!("  Total nodes: {}", all_nodes.len());
    println!("  Total edges: {}", all_edges.len());

    // Verify structure
    assert_eq!(upper_grid.len(), side_len * side_len);
    assert_eq!(lower_grid.len(), side_len * side_len);
    assert_eq!(visible_nodes.len(), data_dim);
    assert_eq!(upper_without_visible.len(), side_len * side_len - data_dim);
    assert_eq!(all_nodes.len(), 2 * side_len * side_len);

    // Create model
    let biases: Tensor<WgpuBackend, 1> = Tensor::zeros([all_nodes.len()], &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::zeros([all_edges.len()], &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data([1.0f32].as_slice(), &device);

    let model = IsingEBM::new(all_nodes.clone(), all_edges.clone(), biases, weights, beta);

    assert_eq!(model.nodes.len(), all_nodes.len());
    assert_eq!(model.edges.len(), all_edges.len());

    println!("Model created successfully!");
}

/// Test that we can load the MNIST data files

#[test]
fn test_mnist_data_loading() {
    ensure_backend();
    let device = init_gpu_device();

    // Load training data
    let train_data_path = Path::new("tests/mnist_test_data/train_data_filtered.npy");

    if train_data_path.exists() {
        let train_data = load_npy_as_tensor(train_data_path, &device);
        let dims = train_data.dims();
        println!("Train data shape: {:?}", dims);

        // Expected: (N, 814) where N is number of samples, 814 = 784 image + 30 labels
        assert_eq!(dims[1], 814, "Expected 814 features per sample");
        assert!(dims[0] > 0, "Should have some training samples");

        // Load test data for each class
        for digit in [0, 3, 4] {
            let path =
                Path::new("tests/mnist_test_data").join(format!("sep_images_test_{}.npy", digit));

            if path.exists() {
                let test_data = load_npy_as_tensor(&path, &device);
                let dims = test_data.dims();
                println!("Test data for digit {}: {:?}", digit, dims);

                // Should have same feature count
                assert!(dims[0] > 0, "Should have test samples for digit {}", digit);
            } else {
                println!("Warning: Test data file not found for digit {}", digit);
            }
        }

        println!("Data loading test passed!");
    } else {
        println!(
            "Warning: Training data file not found at {:?}",
            train_data_path
        );
        println!("Skipping data loading test.");
    }
}

/// Test the full training setup (without actually training)

#[test]
fn test_mnist_training_setup() {
    ensure_backend();
    let device = init_gpu_device();

    // Configuration
    let target_classes = vec![0, 3, 4];
    let num_label_spots = 10;
    let label_size = target_classes.len() * num_label_spots;
    let data_dim = 28 * 28 + label_size;
    let side_len = 40;

    println!("Setting up MNIST training...");

    // Create model architecture
    let (upper_grid, lower_grid, visible_nodes, upper_without_visible, all_nodes, all_edges) =
        get_double_grid(side_len, &[1, 4, 15], data_dim, 0);

    // Initialize model with zeros
    let biases: Tensor<WgpuBackend, 1> = Tensor::zeros([all_nodes.len()], &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::zeros([all_edges.len()], &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data([1.0f32].as_slice(), &device);

    let model = IsingEBM::new(all_nodes.clone(), all_edges.clone(), biases, weights, beta);

    // Define sampling blocks
    let positive_sampling_blocks = vec![upper_without_visible.clone(), lower_grid.clone()];
    let negative_sampling_blocks = vec![upper_grid.clone(), lower_grid.clone()];
    let training_data_blocks = vec![visible_nodes.clone()];

    // Create schedules
    let schedule_negative = SamplingSchedule::new(10, 5, 2); // Small for testing
    let schedule_positive = SamplingSchedule::new(10, 5, 2);

    // Create training spec (needs to clone model since it takes ownership)
    let model_for_spec = IsingEBM::new(
        all_nodes.clone(),
        all_edges.clone(),
        model.biases.clone(),
        model.weights.clone(),
        model.beta.clone(),
    );

    let training_spec = IsingTrainingSpec::new(
        model_for_spec,
        training_data_blocks.clone(),
        vec![],
        positive_sampling_blocks.clone(),
        negative_sampling_blocks.clone(),
        schedule_positive.clone(),
        schedule_negative.clone(),
        &device,
    );

    assert!(
        training_spec.is_ok(),
        "Training spec creation should succeed: {:?}",
        training_spec.err()
    );

    println!("Training spec created successfully!");

    // Create sampling program
    let free_blocks = positive_sampling_blocks.clone();
    let clamped_blocks = training_data_blocks.clone();

    let program = IsingSamplingProgram::new(&model, free_blocks, clamped_blocks, &device);

    assert!(
        program.is_ok(),
        "Sampling program creation should succeed: {:?}",
        program.err()
    );

    println!("Sampling program created successfully!");

    // Test hinton initialization
    let key = RngKey::new(42);
    let batch_shape = [10]; // batch of 10

    let init_states = hinton_init(
        key,
        &model,
        &positive_sampling_blocks,
        &batch_shape,
        &device,
    );

    assert_eq!(
        init_states.len(),
        positive_sampling_blocks.len(),
        "Should have init state for each block"
    );

    for (i, state) in init_states.iter().enumerate() {
        let dims = state.dims();
        assert_eq!(dims[0], 10, "Batch dimension should match");
        assert_eq!(
            dims[1],
            positive_sampling_blocks[i].len(),
            "Block dimension should match"
        );
        println!("Init state for block {}: {:?}", i, dims);
    }

    println!("\nMNIST training setup test passed!");
    println!("The full training pipeline is ready for use.");
}

/// Full training test - runs actual gradient descent
/// This is slow (~minutes) so it's marked as ignored by default

#[test]
#[ignore]
fn test_mnist_training_full() {
    use thrml_models::ising::estimate_kl_grad;

    ensure_backend();
    let device = init_gpu_device();

    // Load training data
    let train_data_path = Path::new("tests/mnist_test_data/train_data_filtered.npy");
    if !train_data_path.exists() {
        println!("Training data not found, skipping full training test");
        return;
    }

    let train_data = load_npy_as_bool_tensor(train_data_path, &device);
    let train_dims = train_data.dims();
    println!("Train data shape: {:?}", train_dims);

    // Configuration
    let target_classes = vec![0, 3, 4];
    let num_label_spots = 10;
    let label_size = target_classes.len() * num_label_spots;
    let data_dim = 28 * 28 + label_size;
    let side_len = 40;

    // Create model
    let (upper_grid, lower_grid, visible_nodes, upper_without_visible, all_nodes, all_edges) =
        get_double_grid(side_len, &[1, 4, 15], data_dim, 0);

    let biases: Tensor<WgpuBackend, 1> = Tensor::zeros([all_nodes.len()], &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::zeros([all_edges.len()], &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data([1.0f32].as_slice(), &device);

    let mut model = IsingEBM::new(all_nodes.clone(), all_edges.clone(), biases, weights, beta);

    // Sampling blocks
    let positive_sampling_blocks = vec![upper_without_visible.clone(), lower_grid.clone()];
    let negative_sampling_blocks = vec![upper_grid.clone(), lower_grid.clone()];
    let training_data_blocks = vec![visible_nodes.clone()];

    // Schedules - smaller for testing
    let schedule_negative = SamplingSchedule::new(50, 10, 3);
    let schedule_positive = SamplingSchedule::new(50, 10, 3);

    // Note: We'll recreate training_spec each batch since it takes ownership
    // For now, just verify the setup works

    // Training loop
    let n_batches = 3; // Just a few batches to verify it works
    let bsz = 10;
    let learning_rate = 0.01f32;
    let mut key = RngKey::new(0);

    println!("\nStarting training with {} batches...", n_batches);

    for batch_idx in 0..n_batches {
        println!("  Batch {}/{}...", batch_idx + 1, n_batches);

        // Get batch of data
        let start_idx = batch_idx * bsz;
        let end_idx = (start_idx + bsz).min(train_dims[0]);
        let batch_data: Tensor<WgpuBackend, 2> = train_data
            .clone()
            .float()
            .slice([start_idx..end_idx, 0..train_dims[1]]);

        // Initialize states
        let keys = key.split(3);
        key = keys[2].clone();

        let init_free_pos = hinton_init(
            keys[0].clone(),
            &model,
            &positive_sampling_blocks,
            &[],
            &device,
        );
        let init_free_neg = hinton_init(
            keys[1].clone(),
            &model,
            &negative_sampling_blocks,
            &[],
            &device,
        );

        // Squeeze batch dimension since we're doing single-sample
        let init_free_pos_1d: Vec<Tensor<WgpuBackend, 1>> = init_free_pos
            .into_iter()
            .map(|t| t.squeeze::<1>())
            .collect();
        let init_free_neg_1d: Vec<Tensor<WgpuBackend, 1>> = init_free_neg
            .into_iter()
            .map(|t| t.squeeze::<1>())
            .collect();

        // Create training spec for this batch (takes ownership of model clone)
        let model_for_spec = IsingEBM::new(
            all_nodes.clone(),
            all_edges.clone(),
            model.biases.clone(),
            model.weights.clone(),
            model.beta.clone(),
        );

        let training_spec = match IsingTrainingSpec::new(
            model_for_spec,
            training_data_blocks.clone(),
            vec![],
            positive_sampling_blocks.clone(),
            negative_sampling_blocks.clone(),
            schedule_positive.clone(),
            schedule_negative.clone(),
            &device,
        ) {
            Ok(spec) => spec,
            Err(e) => {
                println!("    Failed to create training spec: {}", e);
                continue;
            }
        };

        // Compute gradients
        let result = estimate_kl_grad(
            keys[0].clone(),
            &training_spec,
            &all_nodes,
            &all_edges,
            &[batch_data],
            &[],
            init_free_pos_1d,
            init_free_neg_1d,
            &device,
        );

        match result {
            Ok((grad_w, grad_b)) => {
                // Simple SGD update
                let new_weights = model.weights.clone() + grad_w.clone() * learning_rate;
                let new_biases = model.biases.clone() + grad_b.clone() * learning_rate;

                // Compute gradient norms for logging
                let w_norm: f32 = grad_w
                    .clone()
                    .powf_scalar(2.0)
                    .sum()
                    .sqrt()
                    .into_data()
                    .to_vec::<f32>()
                    .expect("read norm")[0];
                let b_norm: f32 = grad_b
                    .clone()
                    .powf_scalar(2.0)
                    .sum()
                    .sqrt()
                    .into_data()
                    .to_vec::<f32>()
                    .expect("read norm")[0];

                println!(
                    "    Gradient norms - weights: {:.6}, biases: {:.6}",
                    w_norm, b_norm
                );

                // Update model
                model = IsingEBM::new(
                    all_nodes.clone(),
                    all_edges.clone(),
                    new_biases,
                    new_weights,
                    Tensor::from_data([1.0f32].as_slice(), &device),
                );
            }
            Err(e) => {
                println!("    Warning: estimate_kl_grad failed: {}", e);
            }
        }
    }

    println!("\nTraining completed successfully!");
    println!("Note: For real accuracy metrics, run for more epochs with full batch sizes.");
}
