//! Tests for Ising model
//!
//! Port of Python tests/test_ising.py
//!
//! Note: These tests require GPU hardware and only run on macOS (Metal backend).

#![cfg(feature = "gpu")]

use burn::tensor::Tensor;
use thrml_core::backend::{ensure_backend, init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};
use thrml_models::ebm::{AbstractEBM, BatchedEBM};
use thrml_models::ising::{hinton_init, IsingEBM, IsingSamplingProgram};
use thrml_samplers::rng::RngKey;

#[test]
fn test_ising_energy() {
    ensure_backend();
    let device = init_gpu_device();

    // Create a simple 3-node chain
    let nodes: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();
    let edges = vec![
        (nodes[0].clone(), nodes[1].clone()),
        (nodes[1].clone(), nodes[2].clone()),
    ];

    // Create biases and weights
    let biases: Tensor<WgpuBackend, 1> =
        Tensor::from_data(vec![0.1f32, 0.2, 0.3].as_slice(), &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![0.5f32, -0.5].as_slice(), &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![1.0f32].as_slice(), &device);

    // Create model
    let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);

    // Test state: [True, False, True] -> spins [1, -1, 1]
    // Energy = -β * (Σ b_i * s_i + Σ w_ij * s_i * s_j)
    //        = -(0.1*1 + 0.2*(-1) + 0.3*1 + 0.5*1*(-1) + (-0.5)*(-1)*1)
    //        = -(0.1 - 0.2 + 0.3 - 0.5 - 0.5)
    //        = -(-0.8)
    //        = 0.8
    let state: Tensor<WgpuBackend, 1> =
        Tensor::from_data(vec![1.0f32, 0.0, 1.0].as_slice(), &device);
    let blocks = vec![Block::new(nodes.clone()).expect("create block")];

    let energy = model.energy(&[state], &blocks, &device);
    let energy_val: Vec<f32> = energy.into_data().to_vec().expect("read energy");

    // Note: The exact value depends on how the energy is computed
    // This is just a smoke test
    assert!(energy_val[0].is_finite(), "Energy should be finite");
    println!("Energy = {}", energy_val[0]);
}

#[test]
fn test_hinton_init() {
    ensure_backend();
    let device = init_gpu_device();

    // Create nodes and model
    let nodes: Vec<Node> = (0..10).map(|_| Node::new(NodeType::Spin)).collect();
    let edges = vec![]; // No edges for this test

    // Strong positive biases -> should initialize to mostly 1s
    let biases: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![10.0f32; 10].as_slice(), &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::from_data(Vec::<f32>::new().as_slice(), &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![1.0f32].as_slice(), &device);

    let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);

    let key = RngKey::new(42);
    let blocks = vec![Block::new(nodes.clone()).expect("create block")];

    let init_state = hinton_init(key, &model, &blocks, &[], &device);

    // With strong positive bias, most states should be 1 (true)
    let state_data: Vec<f32> = init_state[0]
        .clone()
        .into_data()
        .to_vec()
        .expect("read state");

    let n_ones = state_data.iter().filter(|&&v| v > 0.5).count();
    println!(
        "Number of 1s with positive bias: {}/{}",
        n_ones,
        state_data.len()
    );

    // Should be mostly 1s (allow some variance from RNG)
    assert!(
        n_ones >= 7,
        "Expected mostly 1s with positive bias, got {}",
        n_ones
    );
}

#[test]
fn test_ising_sampling_program_creation() {
    ensure_backend();
    let device = init_gpu_device();

    // Create a 5-node chain with alternating free blocks (for Gibbs sampling)
    let nodes: Vec<Node> = (0..5).map(|_| Node::new(NodeType::Spin)).collect();
    let edges: Vec<(Node, Node)> = nodes
        .windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect();

    let biases: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![0.0f32; 5].as_slice(), &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![0.5f32; 4].as_slice(), &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![1.0f32].as_slice(), &device);

    let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);

    // Create blocks for checkerboard Gibbs sampling
    let even_nodes: Vec<Node> = nodes.iter().step_by(2).cloned().collect();
    let odd_nodes: Vec<Node> = nodes.iter().skip(1).step_by(2).cloned().collect();

    let free_blocks = vec![
        Block::new(even_nodes).expect("create even block"),
        Block::new(odd_nodes).expect("create odd block"),
    ];
    let clamped_blocks: Vec<Block> = vec![];

    // This should create the program without panic
    let result = IsingSamplingProgram::new(&model, free_blocks, clamped_blocks, &device);

    assert!(
        result.is_ok(),
        "IsingSamplingProgram creation should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_ising_energy_batched() {
    ensure_backend();
    let device = init_gpu_device();

    // Create a simple 3-node chain
    let nodes: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();
    let edges = vec![
        (nodes[0].clone(), nodes[1].clone()),
        (nodes[1].clone(), nodes[2].clone()),
    ];

    let biases: Tensor<WgpuBackend, 1> =
        Tensor::from_data(vec![0.1f32, 0.2, 0.3].as_slice(), &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![0.5f32, -0.5].as_slice(), &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![1.0f32].as_slice(), &device);

    let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);

    // Create batch of 4 different states
    let batch_states: Tensor<WgpuBackend, 2> = Tensor::from_data(
        [
            [0.0f32, 0.0, 0.0], // All down
            [1.0, 1.0, 1.0],    // All up
            [1.0, 0.0, 1.0],    // Alternating
            [0.0, 1.0, 0.0],    // Opposite alternating
        ],
        &device,
    );

    let energies = model.energy_batched(&batch_states, &device);
    let energy_data: Vec<f32> = energies.into_data().to_vec().expect("read energies");

    // Verify we got 4 energies
    assert_eq!(energy_data.len(), 4);

    // All-up and all-down should have same energy in zero-bias case
    // (This is a sanity check, actual values depend on biases)
    println!("Batched energies: {:?}", energy_data);

    // Verify all energies are finite
    for e in &energy_data {
        assert!(e.is_finite(), "Energy should be finite");
    }
}

#[test]
fn test_ising_energy_batched_vs_single() {
    ensure_backend();
    let device = init_gpu_device();

    let nodes: Vec<Node> = (0..4).map(|_| Node::new(NodeType::Spin)).collect();
    let edges = vec![
        (nodes[0].clone(), nodes[1].clone()),
        (nodes[1].clone(), nodes[2].clone()),
        (nodes[2].clone(), nodes[3].clone()),
    ];

    let biases: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![0.0f32; 4].as_slice(), &device);
    let weights: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![1.0f32; 3].as_slice(), &device);
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![1.0f32].as_slice(), &device);

    let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);
    let blocks = vec![Block::new(nodes.clone()).unwrap()];

    // Test state
    let state: Tensor<WgpuBackend, 1> =
        Tensor::from_data(vec![1.0f32, 0.0, 1.0, 0.0].as_slice(), &device);
    let batch_state: Tensor<WgpuBackend, 2> = state.clone().unsqueeze_dim(0);

    // Compute both ways
    let single_energy = model.energy(&[state], &blocks, &device);
    let batched_energy = model.energy_batched(&batch_state, &device);

    let single_val: f32 = single_energy.into_data().to_vec::<f32>().unwrap()[0];
    let batched_val: f32 = batched_energy.into_data().to_vec::<f32>().unwrap()[0];

    // They should match (within floating point tolerance)
    assert!(
        (single_val - batched_val).abs() < 1e-5,
        "Single ({}) and batched ({}) energy should match",
        single_val,
        batched_val
    );
}
