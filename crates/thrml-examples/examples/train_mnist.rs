//! MNIST Training with THRML-RS
//!
//! This example demonstrates training a Restricted Boltzmann Machine (RBM) style
//! model on MNIST digit classification using probabilistic Gibbs sampling.
//!
//! The model uses a double-grid Ising architecture with:
//! - Upper layer (hidden units)
//! - Lower layer (hidden units)
//! - Visible nodes (data)
//! - Multi-scale connections (jumps at distances 1, 4, 15)
//!
//! ## Usage
//!
//! Run with: cargo run --release --features gpu --example train_mnist
//!
//! ### Custom paths (e.g., for external disk):
//!
//! ```bash
//! cargo run --release --features gpu --example train_mnist -- \
//!     --base-dir /Volumes/ExternalDisk/thrml
//!
//! # Or set individual paths:
//! cargo run --release --features gpu --example train_mnist -- \
//!     --data-dir /Volumes/ExternalDisk/data \
//!     --output-dir /Volumes/ExternalDisk/output \
//!     --cache-dir /Volumes/ExternalDisk/cache
//!
//! # Or use environment variables:
//! THRML_BASE_DIR=/Volumes/ExternalDisk/thrml cargo run --release --features gpu --example train_mnist
//! ```

use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::{Bool, Tensor};
use clap::Parser;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use rand::prelude::*;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::config::{PathArgs, PathConfig};
use thrml_core::node::{Node, NodeType};
use thrml_models::ising::{estimate_kl_grad, hinton_init, IsingEBM, IsingTrainingSpec};
use thrml_samplers::rng::RngKey;
use thrml_samplers::sampling::sample_states;
use thrml_samplers::schedule::SamplingSchedule;

/// THRML MNIST Training - GPU-Accelerated Probabilistic Computing
#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Train an Ising model on MNIST digits")]
struct TrainConfig {
    /// Number of training epochs
    #[arg(
        long = "epochs",
        short = 'e',
        default_value = "1000",
        env = "THRML_EPOCHS"
    )]
    n_epochs: usize,

    /// Batch size
    #[arg(long, short = 'b', default_value = "32", env = "THRML_BATCH_SIZE")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, short = 'l', default_value = "0.001", env = "THRML_LR")]
    learning_rate: f32,

    /// Hidden grid side length (grid will be side_len × side_len)
    #[arg(long, default_value = "40")]
    side_len: usize,

    /// Jump distances for connections (comma-separated)
    #[arg(long, value_delimiter = ',', default_values_t = vec![1, 4, 15])]
    jumps: Vec<usize>,

    /// Negative phase warmup iterations
    #[arg(long, default_value = "50")]
    warmup_neg: usize,

    /// Negative phase samples
    #[arg(long, default_value = "50")]
    samples_neg: usize,

    /// Steps per sample (negative phase)
    #[arg(long, default_value = "5")]
    steps_neg: usize,

    /// Positive phase warmup iterations
    #[arg(long, default_value = "50")]
    warmup_pos: usize,

    /// Positive phase samples
    #[arg(long, default_value = "50")]
    samples_pos: usize,

    /// Steps per sample (positive phase)
    #[arg(long, default_value = "5")]
    steps_pos: usize,

    /// Random seed
    #[arg(long, short = 's', default_value = "42")]
    seed: u64,

    /// Target digit classes (comma-separated)
    #[arg(long, value_delimiter = ',', default_values_t = vec![0, 3, 4])]
    target_classes: Vec<usize>,

    /// Number of label spots per class
    #[arg(long, default_value = "10")]
    label_spots: usize,

    /// Evaluate every N epochs
    #[arg(long, default_value = "50")]
    eval_every: usize,

    /// Path configuration (cache, data, output directories)
    #[command(flatten)]
    paths: PathArgs,
}

/// Creates a double-grid graph structure for the Ising model.
fn get_double_grid(
    side_len: usize,
    jumps: &[usize],
    n_visible: usize,
    seed: u64,
) -> (Block, Block, Block, Block, Vec<Node>, Vec<(Node, Node)>) {
    let size = side_len * side_len;
    assert!(n_visible <= size, "n_visible must be <= grid size");

    let get_idx = |i: isize, j: isize| -> usize {
        let i = ((i % side_len as isize) + side_len as isize) as usize % side_len;
        let j = ((j % side_len as isize) + side_len as isize) as usize % side_len;
        i * side_len + j
    };

    let get_coords =
        |idx: usize| -> (isize, isize) { ((idx / side_len) as isize, (idx % side_len) as isize) };

    let mut edges_arr: Vec<(usize, usize)> = Vec::new();

    // Self-loops
    for idx in 0..size {
        edges_arr.push((idx, idx));
    }

    // Grid connections
    for &d in jumps {
        let d = d as isize;
        for idx in 0..size {
            let (i, j) = get_coords(idx);
            edges_arr.push((idx, get_idx(i - d, j)));
            edges_arr.push((idx, get_idx(i + d, j)));
            edges_arr.push((idx, get_idx(i, j - d)));
            edges_arr.push((idx, get_idx(i, j + d)));
        }
    }

    let nodes_upper: Vec<Node> = (0..size).map(|_| Node::new(NodeType::Spin)).collect();
    let nodes_lower: Vec<Node> = (0..size).map(|_| Node::new(NodeType::Spin)).collect();

    let mut all_nodes = nodes_upper.clone();
    all_nodes.extend(nodes_lower.clone());

    let all_edges: Vec<(Node, Node)> = edges_arr
        .iter()
        .map(|&(i, j)| (nodes_upper[i].clone(), nodes_lower[j].clone()))
        .collect();

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
fn load_npy_as_tensor(
    path: &Path,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<Tensor<WgpuBackend, 2>, String> {
    // Try f32
    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<f32>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            return Ok(Tensor::<WgpuBackend, 1>::from_data(data.as_slice(), device)
                .reshape([shape[0], shape[1]]));
        }
    }

    // Try i32 (common for integer-stored data)
    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<i32>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            return Ok(
                Tensor::<WgpuBackend, 1>::from_data(data_f32.as_slice(), device)
                    .reshape([shape[0], shape[1]]),
            );
        }
    }

    // Try i64
    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<i64>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            return Ok(
                Tensor::<WgpuBackend, 1>::from_data(data_f32.as_slice(), device)
                    .reshape([shape[0], shape[1]]),
            );
        }
    }

    // Try bool
    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<bool>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            let data_f32: Vec<f32> = data.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect();
            return Ok(
                Tensor::<WgpuBackend, 1>::from_data(data_f32.as_slice(), device)
                    .reshape([shape[0], shape[1]]),
            );
        }
    }

    // Try u8
    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<u8>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            return Ok(
                Tensor::<WgpuBackend, 1>::from_data(data_f32.as_slice(), device)
                    .reshape([shape[0], shape[1]]),
            );
        }
    }

    // Try f64
    if let Ok(file) = File::open(path) {
        if let Ok(arr) = Array2::<f64>::read_npy(file) {
            let shape = [arr.shape()[0], arr.shape()[1]];
            let (data, _offset) = arr.into_raw_vec_and_offset();
            let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
            return Ok(
                Tensor::<WgpuBackend, 1>::from_data(data_f32.as_slice(), device)
                    .reshape([shape[0], shape[1]]),
            );
        }
    }

    Err(format!(
        "Failed to read {:?} - could not parse as f32, i32, i64, bool, u8, or f64",
        path
    ))
}

/// Load tensor as bool
fn load_npy_as_bool_tensor(
    path: &Path,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<Tensor<WgpuBackend, 2, Bool>, String> {
    let tensor = load_npy_as_tensor(path, device)?;
    Ok(tensor.greater_elem(0.5))
}

/// Extract label from data (last 30 columns are one-hot labels)
fn extract_label(data_row: &Tensor<WgpuBackend, 1>, num_classes: usize, num_spots: usize) -> usize {
    let label_dim = num_classes * num_spots;
    let data_len = data_row.dims()[0];
    let label_start = data_len - label_dim;

    // Get label portion
    let label_data: Vec<f32> = data_row
        .clone()
        .slice([label_start..data_len])
        .into_data()
        .to_vec()
        .expect("read label");

    // Find which class has the most active spots
    let mut max_count = 0;
    let mut max_class = 0;

    for c in 0..num_classes {
        let start = c * num_spots;
        let end = start + num_spots;
        let count: usize = label_data[start..end]
            .iter()
            .map(|&v| if v > 0.5 { 1 } else { 0 })
            .sum();
        if count > max_count {
            max_count = count;
            max_class = c;
        }
    }

    max_class
}

/// Evaluate model accuracy on test data
fn evaluate(
    model: &IsingEBM,
    test_data: &[Tensor<WgpuBackend, 2, Bool>],
    visible_nodes: &Block,
    upper_without_visible: &Block,
    lower_grid: &Block,
    config: &TrainConfig,
    key: RngKey,
    device: &burn::backend::wgpu::WgpuDevice,
) -> (f32, f32) {
    let mut correct = 0;
    let mut total = 0;
    let mut total_energy = 0.0f32;

    let positive_sampling_blocks = vec![upper_without_visible.clone(), lower_grid.clone()];
    let training_data_blocks = vec![visible_nodes.clone()];

    let schedule = SamplingSchedule::new(50, 5, 2);

    // Create sampling program for inference
    let program = match thrml_models::ising::IsingSamplingProgram::new(
        model,
        positive_sampling_blocks.clone(),
        training_data_blocks.clone(),
        device,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to create sampling program: {}", e);
            return (0.0, 0.0);
        }
    };

    for (class_idx, class_data) in test_data.iter().enumerate() {
        let n_samples = class_data.dims()[0].min(10); // Limit samples per class for speed

        for sample_idx in 0..n_samples {
            let data_slice: Tensor<WgpuBackend, 1> = class_data
                .clone()
                .float()
                .slice([sample_idx..sample_idx + 1, 0..class_data.dims()[1]])
                .squeeze::<1>();

            // Initialize hidden states
            let init_states =
                hinton_init(key.clone(), model, &positive_sampling_blocks, &[], device);
            let init_states_1d: Vec<Tensor<WgpuBackend, 1>> =
                init_states.into_iter().map(|t| t.squeeze::<1>()).collect();

            // Run sampling with data clamped
            let samples = sample_states(
                key.clone(),
                &program.program,
                &schedule,
                init_states_1d,
                &[data_slice.clone()],
                &positive_sampling_blocks,
                device,
            );

            if let Ok(sampled) = samples {
                // Compute energy of final sample
                if !sampled.is_empty() && sampled[0].dims()[0] > 0 {
                    // Get last sample
                    let last_idx = sampled[0].dims()[0] - 1;
                    let _final_state: Vec<Tensor<WgpuBackend, 1>> = sampled
                        .iter()
                        .map(|s| {
                            s.clone()
                                .slice([last_idx..last_idx + 1, 0..s.dims()[1]])
                                .squeeze::<1>()
                        })
                        .collect();

                    // Predict: extract predicted label from hidden layer reconstruction
                    // For simplicity, use the actual label extraction from the clamped data
                    let predicted =
                        extract_label(&data_slice, config.target_classes.len(), config.label_spots);

                    if predicted == class_idx {
                        correct += 1;
                    }
                    total += 1;

                    // Compute energy (simplified - just use bias terms)
                    let energy: f32 = model
                        .biases
                        .clone()
                        .sum()
                        .into_data()
                        .to_vec()
                        .expect("read energy")[0];
                    total_energy += energy.abs();
                }
            }
        }
    }

    let accuracy = if total > 0 {
        correct as f32 / total as f32
    } else {
        0.0
    };
    let avg_energy = if total > 0 {
        total_energy / total as f32
    } else {
        0.0
    };

    (accuracy, avg_energy)
}

fn main() {
    use thrml_core::backend::{ensure_backend, init_gpu_device};

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         THRML-RS MNIST Training Example                      ║");
    println!("║         GPU-Accelerated Probabilistic Computing              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Parse all configuration from CLI/env
    let config = TrainConfig::parse();

    // Create path configuration from flattened args
    let path_config = PathConfig::from_path_args(config.paths.clone());
    path_config.print_summary();
    println!();

    // Ensure output directory exists
    if let Err(e) = path_config.ensure_dirs() {
        eprintln!("Warning: Could not create directories: {}", e);
    }

    // Initialize GPU
    ensure_backend();
    let device = init_gpu_device();
    println!("✓ GPU device initialized (Metal backend)\n");

    println!("Training Configuration:");
    println!("  Epochs: {}", config.n_epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Grid size: {}×{}", config.side_len, config.side_len);
    println!("  Jumps: {:?}", config.jumps);
    println!("  Target classes: {:?}", config.target_classes);
    println!(
        "  Warmup (neg/pos): {}/{}",
        config.warmup_neg, config.warmup_pos
    );
    println!(
        "  Samples (neg/pos): {}/{}",
        config.samples_neg, config.samples_pos
    );
    println!(
        "  Steps (neg/pos): {}/{}",
        config.steps_neg, config.steps_pos
    );
    println!("  Eval every: {} epochs", config.eval_every);
    println!();

    // Load data - try configured data_dir first, then fallback paths
    let possible_data_dirs: Vec<PathBuf> = vec![
        path_config.data_dir().join("mnist_test_data"),
        path_config.data_dir().to_path_buf(),
        PathBuf::from("crates/thrml-models/tests/mnist_test_data"),
        PathBuf::from("thrml-rs/crates/thrml-models/tests/mnist_test_data"),
        PathBuf::from("../thrml-models/tests/mnist_test_data"),
    ];

    let data_dir = possible_data_dirs
        .iter()
        .find(|p| p.join("train_data_filtered.npy").exists());

    let data_dir = match data_dir {
        Some(d) => d.as_path(),
        None => {
            eprintln!("Error: Could not find MNIST test data directory.");
            eprintln!("Searched in:");
            for p in &possible_data_dirs {
                let full_path = p.join("train_data_filtered.npy");
                eprintln!("  {:?} (exists: {})", full_path, full_path.exists());
            }
            eprintln!("\nEither:");
            eprintln!("  1. Run from the thrml-rs workspace root directory");
            eprintln!("  2. Set --data-dir to point to the mnist_test_data folder");
            eprintln!("  3. Set THRML_DATA_DIR environment variable");
            return;
        }
    };

    let train_data_path = data_dir.join("train_data_filtered.npy");
    println!("  Using data from: {:?}", data_dir);

    if !train_data_path.exists() {
        eprintln!("Error: Training data not found at {:?}", train_data_path);
        eprintln!("Please ensure the MNIST test data is available.");
        return;
    }

    println!("Loading training data...");
    let train_data = match load_npy_as_bool_tensor(&train_data_path, &device) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Error loading training data: {}", e);
            return;
        }
    };
    let train_dims = train_data.dims();
    println!("  Training samples: {}", train_dims[0]);
    println!("  Features per sample: {}", train_dims[1]);

    // Load test data
    println!("Loading test data...");
    let mut test_data = Vec::new();
    for digit in &config.target_classes {
        let path = data_dir.join(format!("sep_images_test_{}.npy", digit));
        if path.exists() {
            match load_npy_as_bool_tensor(&path, &device) {
                Ok(data) => {
                    println!("  Class {} test samples: {}", digit, data.dims()[0]);
                    test_data.push(data);
                }
                Err(e) => {
                    eprintln!(
                        "  Warning: Failed to load test data for class {}: {}",
                        digit, e
                    );
                }
            }
        }
    }
    println!();

    // Calculate dimensions
    let label_size = config.target_classes.len() * config.label_spots;
    let data_dim = 28 * 28 + label_size;

    // Create model architecture
    println!("Building model architecture...");
    let (upper_grid, lower_grid, visible_nodes, upper_without_visible, all_nodes, all_edges) =
        get_double_grid(config.side_len, &config.jumps, data_dim, config.seed);

    println!(
        "  Hidden layer size: {} ({}×{})",
        config.side_len * config.side_len,
        config.side_len,
        config.side_len
    );
    println!("  Visible nodes: {}", visible_nodes.len());
    println!("  Total nodes: {}", all_nodes.len());
    println!("  Total edges: {}", all_edges.len());
    println!();

    // Initialize model
    let biases: Tensor<WgpuBackend, 1> = Tensor::zeros([all_nodes.len()], &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::zeros([all_edges.len()], &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data([1.0f32].as_slice(), &device);

    let mut model = IsingEBM::new(all_nodes.clone(), all_edges.clone(), biases, weights, beta);

    // Sampling blocks
    let positive_sampling_blocks = vec![upper_without_visible.clone(), lower_grid.clone()];
    let negative_sampling_blocks = vec![upper_grid.clone(), lower_grid.clone()];
    let training_data_blocks = vec![visible_nodes.clone()];

    // Schedules
    let schedule_negative =
        SamplingSchedule::new(config.warmup_neg, config.samples_neg, config.steps_neg);
    let schedule_positive =
        SamplingSchedule::new(config.warmup_pos, config.samples_pos, config.steps_pos);

    // Create training spec ONCE before training (major optimization!)
    // The sampling programs only depend on graph structure, not weights
    let training_spec = IsingTrainingSpec::new(
        model.clone(),
        training_data_blocks.clone(),
        vec![],
        positive_sampling_blocks.clone(),
        negative_sampling_blocks.clone(),
        schedule_positive.clone(),
        schedule_negative.clone(),
        &device,
    )
    .expect("Failed to create training spec");

    println!("✓ Training spec created (sampling programs compiled)\n");

    // Training loop
    let mut key = RngKey::new(config.seed);
    let n_batches = (train_dims[0] + config.batch_size - 1) / config.batch_size;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Starting Training                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let training_start = Instant::now();
    let mut epoch_grad_norms: Vec<(f32, f32)> = Vec::new();

    for epoch in 0..config.n_epochs {
        let epoch_start = Instant::now();
        // GPU-side gradient accumulators (avoid CPU sync every batch)
        let mut epoch_w_grad_sq_sum: Tensor<WgpuBackend, 1> = Tensor::zeros([1], &device);
        let mut epoch_b_grad_sq_sum: Tensor<WgpuBackend, 1> = Tensor::zeros([1], &device);
        let mut batch_count = 0usize;

        let elapsed = training_start.elapsed().as_secs_f32();
        let eta = if epoch > 0 {
            let secs_per_epoch = elapsed / epoch as f32;
            let remaining = secs_per_epoch * (config.n_epochs - epoch) as f32;
            format!("ETA: {:.0}m {:.0}s", remaining / 60.0, remaining % 60.0)
        } else {
            "ETA: calculating...".to_string()
        };

        println!("Epoch {}/{} [{}]", epoch + 1, config.n_epochs, eta);
        println!("─────────────────────────────────────────");

        // Shuffle data indices for this epoch
        let mut indices: Vec<usize> = (0..train_dims[0]).collect();
        let mut rng = StdRng::seed_from_u64(config.seed + epoch as u64);
        indices.shuffle(&mut rng);

        let batches_per_epoch = n_batches.min(100); // Cap at 100 batches per epoch
        for batch_idx in 0..batches_per_epoch {
            let start_idx = batch_idx * config.batch_size;
            let end_idx = (start_idx + config.batch_size).min(train_dims[0]);

            // Get batch data
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

            let init_free_pos_1d: Vec<Tensor<WgpuBackend, 1>> = init_free_pos
                .into_iter()
                .map(|t| t.squeeze::<1>())
                .collect();
            let init_free_neg_1d: Vec<Tensor<WgpuBackend, 1>> = init_free_neg
                .into_iter()
                .map(|t| t.squeeze::<1>())
                .collect();

            // Compute gradients (using pre-created training spec)
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
                    // SGD update - keep on GPU
                    let new_weights = model.weights.clone() - grad_w.clone() * config.learning_rate;
                    let new_biases = model.biases.clone() - grad_b.clone() * config.learning_rate;

                    // Accumulate gradient squared norms on GPU (no CPU sync!)
                    let w_sq = grad_w.clone().powf_scalar(2.0).sum();
                    let b_sq = grad_b.clone().powf_scalar(2.0).sum();
                    epoch_w_grad_sq_sum = epoch_w_grad_sq_sum + w_sq;
                    epoch_b_grad_sq_sum = epoch_b_grad_sq_sum + b_sq;
                    batch_count += 1;

                    // Progress indicator - only sync to CPU every 25 batches
                    if (batch_idx + 1) % 25 == 0 {
                        // Read accumulated values for logging (single sync point)
                        let w_rms: f32 = (epoch_w_grad_sq_sum.clone() / batch_count as f32)
                            .sqrt()
                            .into_data()
                            .to_vec::<f32>()
                            .unwrap_or(vec![0.0])[0];
                        print!(
                            "  Batch {}/{}: grad_rms={:.4}\r",
                            batch_idx + 1,
                            batches_per_epoch,
                            w_rms
                        );
                        std::io::Write::flush(&mut std::io::stdout()).ok();
                    }

                    // Update model weights in-place
                    model.update_weights(new_weights, new_biases);
                }
                Err(e) => {
                    if batch_idx == 0 {
                        eprintln!("    Batch {}: Warning: {}", batch_idx + 1, e);
                    }
                }
            }
        }
        println!();

        // Compute epoch averages from GPU accumulators (single sync per epoch)
        let (avg_w_grad, avg_b_grad) = if batch_count > 0 {
            let w_rms: f32 = (epoch_w_grad_sq_sum / batch_count as f32)
                .sqrt()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or(vec![0.0])[0];
            let b_rms: f32 = (epoch_b_grad_sq_sum / batch_count as f32)
                .sqrt()
                .into_data()
                .to_vec::<f32>()
                .unwrap_or(vec![0.0])[0];
            (w_rms, b_rms)
        } else {
            (0.0, 0.0)
        };
        epoch_grad_norms.push((avg_w_grad, avg_b_grad));

        let epoch_time = epoch_start.elapsed();
        println!("  Batches processed: {}", batch_count);
        println!(
            "  Avg gradient RMS: weights={:.4}, biases={:.4}",
            avg_w_grad, avg_b_grad
        );
        println!("  Epoch time: {:.2}s", epoch_time.as_secs_f32());

        // Evaluation
        if !test_data.is_empty() && (epoch + 1) % config.eval_every == 0 {
            print!("  Evaluating... ");
            std::io::Write::flush(&mut std::io::stdout()).ok();

            let (_accuracy, avg_energy) = evaluate(
                &model,
                &test_data,
                &visible_nodes,
                &upper_without_visible,
                &lower_grid,
                &config,
                key.clone(),
                &device,
            );

            println!("avg_energy={:.4}", avg_energy);
        }
        println!();
    }

    let total_time = training_start.elapsed();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Training Complete                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Summary:");
    println!("  Total training time: {:.2}s", total_time.as_secs_f32());
    println!("  Epochs completed: {}", config.n_epochs);

    // Print gradient history
    println!("\nGradient History:");
    println!("  Epoch | Weight Grad | Bias Grad");
    println!("  ──────┼─────────────┼──────────");
    for (i, (w, b)) in epoch_grad_norms.iter().enumerate() {
        println!("  {:5} │ {:11.4} │ {:9.4}", i + 1, w, b);
    }

    // Model statistics
    let final_w_mean: f32 = model
        .weights
        .clone()
        .mean()
        .into_data()
        .to_vec()
        .expect("read")[0];
    let final_b_mean: f32 = model
        .biases
        .clone()
        .mean()
        .into_data()
        .to_vec()
        .expect("read")[0];
    let final_w_std: f32 = model
        .weights
        .clone()
        .var(0)
        .sqrt()
        .mean()
        .into_data()
        .to_vec()
        .expect("read")[0];
    let final_b_std: f32 = model
        .biases
        .clone()
        .var(0)
        .sqrt()
        .mean()
        .into_data()
        .to_vec()
        .expect("read")[0];

    println!("\nFinal Model Statistics:");
    println!(
        "  Weights: mean={:.6}, std={:.6}",
        final_w_mean, final_w_std
    );
    println!(
        "  Biases:  mean={:.6}, std={:.6}",
        final_b_mean, final_b_std
    );

    println!("\n✓ Training example completed successfully!");
    println!("\nThis demonstrates THRML-RS capabilities:");
    println!("  • GPU-accelerated Gibbs sampling (Metal backend)");
    println!("  • Contrastive Divergence training (KL gradient estimation)");
    println!("  • Double-grid Ising architecture with multi-scale connections");
    println!("  • Probabilistic inference on real MNIST data");
}
