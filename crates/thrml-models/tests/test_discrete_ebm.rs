//! Tests for DiscreteEBMFactor and related types
//!
//! Port of Python tests/test_discrete_ebm.py
//!
//! Note: These tests require GPU hardware and only run on macOS (Metal backend).

#![cfg(feature = "gpu")]

use burn::tensor::Tensor;
use thrml_core::backend::{ensure_backend, init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};
use thrml_models::discrete_ebm::{
    batch_gather, spin_product, split_states, CategoricalEBMFactor, DiscreteEBMFactor,
    SpinEBMFactor, SquareDiscreteEBMFactor,
};
use thrml_models::factor::AbstractFactor;

#[test]
fn test_discrete_ebm_factor_creation() {
    ensure_backend();
    let device = init_gpu_device();

    let fac_size = 4;

    let spin_nodes: Vec<Node> = (0..fac_size).map(|_| Node::new(NodeType::Spin)).collect();
    let cat_nodes: Vec<Node> = (0..fac_size)
        .map(|_| Node::new(NodeType::Categorical { n_categories: 3 }))
        .collect();

    let spin_block = Block::new(spin_nodes).unwrap();
    let cat_block = Block::new(cat_nodes).unwrap();

    // Weights shape: [n_nodes, n_categories]
    let weights: Tensor<WgpuBackend, 3> = Tensor::zeros([fac_size, 3, 1], &device);

    let factor = DiscreteEBMFactor::new(vec![spin_block], vec![cat_block], weights);
    assert!(factor.is_ok(), "DiscreteEBMFactor creation should succeed");
}

#[test]
fn test_discrete_ebm_factor_wrong_leading_dim() {
    ensure_backend();
    let device = init_gpu_device();

    let fac_size = 4;

    let spin_nodes: Vec<Node> = (0..fac_size).map(|_| Node::new(NodeType::Spin)).collect();
    let cat_nodes: Vec<Node> = (0..fac_size)
        .map(|_| Node::new(NodeType::Categorical { n_categories: 3 }))
        .collect();

    let spin_block = Block::new(spin_nodes).unwrap();
    let cat_block = Block::new(cat_nodes).unwrap();

    // Wrong leading dimension (5 instead of 4)
    let weights: Tensor<WgpuBackend, 3> = Tensor::zeros([fac_size + 1, 3, 1], &device);

    let factor = DiscreteEBMFactor::new(vec![spin_block], vec![cat_block], weights);
    assert!(factor.is_err(), "Should fail with wrong leading dimension");

    let err = factor.err().unwrap();
    assert!(
        err.contains("leading dimension"),
        "Error should mention leading dimension, got: {}",
        err
    );
}

#[test]
fn test_spin_ebm_factor() {
    ensure_backend();
    let device = init_gpu_device();

    let n_nodes = 3;
    let nodes1: Vec<Node> = (0..n_nodes).map(|_| Node::new(NodeType::Spin)).collect();
    let nodes2: Vec<Node> = (0..n_nodes).map(|_| Node::new(NodeType::Spin)).collect();

    let block1 = Block::new(nodes1).unwrap();
    let block2 = Block::new(nodes2).unwrap();

    // Weights for spin-only factor: [n_nodes, 1, 1] for no categorical dims
    let weights: Tensor<WgpuBackend, 3> = Tensor::zeros([n_nodes, 1, 1], &device);

    let factor = SpinEBMFactor::new(vec![block1, block2], weights);
    assert!(factor.is_ok(), "SpinEBMFactor creation should succeed");
}

#[test]
fn test_categorical_ebm_factor() {
    ensure_backend();
    let device = init_gpu_device();

    let n_cats = 3usize;
    let n_nodes = 5;
    let nodes: Vec<Node> = (0..n_nodes)
        .map(|_| {
            Node::new(NodeType::Categorical {
                n_categories: n_cats as u8,
            })
        })
        .collect();

    let block = Block::new(nodes).unwrap();

    // Weights shape for biases: [n_nodes, n_categories, 1]
    let biases: Tensor<WgpuBackend, 3> = Tensor::zeros([n_nodes, n_cats, 1], &device);

    let factor = CategoricalEBMFactor::new(vec![block], biases);
    assert!(
        factor.is_ok(),
        "CategoricalEBMFactor creation should succeed"
    );
}

#[test]
fn test_square_discrete_ebm_factor() {
    ensure_backend();
    let device = init_gpu_device();

    let block_len = 4;
    let n_cat_blocks = 2; // Use 2 categorical blocks (fits in 3D tensor)
    let n_categories = 5u8;

    let cat_nodes: Vec<Vec<Node>> = (0..n_cat_blocks)
        .map(|_| {
            (0..block_len)
                .map(|_| Node::new(NodeType::Categorical { n_categories }))
                .collect()
        })
        .collect();

    let blocks: Vec<Block> = cat_nodes
        .into_iter()
        .map(|nodes| Block::new(nodes).unwrap())
        .collect();

    // Square weights: [n_nodes, 5, 5] for 2 categorical blocks
    let weights: Tensor<WgpuBackend, 3> = Tensor::zeros(
        [block_len, n_categories as usize, n_categories as usize],
        &device,
    );

    let factor = SquareDiscreteEBMFactor::new(vec![], blocks, weights);
    assert!(
        factor.is_ok(),
        "SquareDiscreteEBMFactor creation should succeed: {:?}",
        factor.err()
    );
}

#[test]
fn test_square_factor_not_square_fails() {
    ensure_backend();
    let device = init_gpu_device();

    let block_len = 4;

    let cat_nodes: Vec<Vec<Node>> = (0..3)
        .map(|_| {
            (0..block_len)
                .map(|_| Node::new(NodeType::Categorical { n_categories: 5 }))
                .collect()
        })
        .collect();

    let blocks: Vec<Block> = cat_nodes
        .into_iter()
        .map(|nodes| Block::new(nodes).unwrap())
        .collect();

    // Non-square weights: [n_nodes, 5, 3]
    let weights: Tensor<WgpuBackend, 3> = Tensor::zeros([block_len, 5, 3], &device);

    let factor = SquareDiscreteEBMFactor::new(vec![], blocks, weights);
    assert!(factor.is_err(), "Should fail with non-square weights");

    let err = factor.err().unwrap();
    assert!(
        err.contains("square"),
        "Error should mention square, got: {}",
        err
    );
}

#[test]
fn test_to_interaction_groups() {
    ensure_backend();
    let device = init_gpu_device();

    let n_cats = 3;
    let chain_len = 4;

    let spin_nodes1: Vec<Node> = (0..chain_len).map(|_| Node::new(NodeType::Spin)).collect();
    let cat_nodes1: Vec<Node> = (0..chain_len)
        .map(|_| {
            Node::new(NodeType::Categorical {
                n_categories: n_cats,
            })
        })
        .collect();
    let spin_nodes2: Vec<Node> = (0..chain_len).map(|_| Node::new(NodeType::Spin)).collect();
    let cat_nodes2: Vec<Node> = (0..chain_len)
        .map(|_| {
            Node::new(NodeType::Categorical {
                n_categories: n_cats,
            })
        })
        .collect();

    let block1 = Block::new(spin_nodes1).unwrap();
    let block2 = Block::new(cat_nodes1).unwrap();
    let block3 = Block::new(spin_nodes2).unwrap();
    let block4 = Block::new(cat_nodes2).unwrap();

    let weights: Tensor<WgpuBackend, 3> =
        Tensor::zeros([chain_len, n_cats as usize, n_cats as usize], &device);

    let factor =
        DiscreteEBMFactor::new(vec![block1, block3], vec![block2, block4], weights).unwrap();

    let groups = factor.to_interaction_groups(&device);

    // Should produce multiple interaction groups (one for each possible head)
    assert!(
        groups.len() >= 2,
        "Should produce at least 2 interaction groups, got {}",
        groups.len()
    );
}

#[test]
fn test_spin_product() {
    ensure_backend();
    let device = init_gpu_device();

    // Test with some spin values (bool tensors)
    let spin1: Tensor<WgpuBackend, 1, burn::tensor::Bool> =
        Tensor::from_data([true, false, true].as_slice(), &device);
    let spin2: Tensor<WgpuBackend, 1, burn::tensor::Bool> =
        Tensor::from_data([true, true, false].as_slice(), &device);

    let product = spin_product(&[spin1, spin2], &device);
    let data: Vec<f32> = product.into_data().to_vec().expect("read product");

    // true*true = 1*1 = 1
    // false*true = (-1)*1 = -1
    // true*false = 1*(-1) = -1
    assert_eq!(data.len(), 3);
    assert!(
        (data[0] - 1.0).abs() < 1e-5,
        "Expected 1.0, got {}",
        data[0]
    );
    assert!(
        (data[1] - (-1.0)).abs() < 1e-5,
        "Expected -1.0, got {}",
        data[1]
    );
    assert!(
        (data[2] - (-1.0)).abs() < 1e-5,
        "Expected -1.0, got {}",
        data[2]
    );
}

#[test]
fn test_spin_product_empty() {
    ensure_backend();
    let device = init_gpu_device();

    let product = spin_product(&[], &device);
    let data: Vec<f32> = product.into_data().to_vec().expect("read product");

    // Empty product should return 1.0
    assert_eq!(data.len(), 1);
    assert!((data[0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_batch_gather() {
    ensure_backend();
    let device = init_gpu_device();

    // Create a simple 3D tensor [2, 3, 3] (batch=2, 3x3 matrix per batch)
    let data: Vec<f32> = vec![
        // batch 0
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // batch 1
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
    ];
    let weights: Tensor<WgpuBackend, 3> =
        Tensor::<WgpuBackend, 1>::from_data(data.as_slice(), &device).reshape([2, 3, 3]);

    // Index [0, 1] for first dimension, [2, 0] for second
    let idx0: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        Tensor::from_data([0i32, 1].as_slice(), &device);
    let idx1: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        Tensor::from_data([2i32, 0].as_slice(), &device);

    let result = batch_gather(&weights, &[idx0, idx1]);
    let data: Vec<f32> = result.into_data().to_vec().expect("read result");

    // Batch 0, index [0, 2] -> row 0, col 2 -> 3.0
    // Batch 1, index [1, 0] -> row 1, col 0 -> 40.0
    assert_eq!(data.len(), 2);
    assert!(
        (data[0] - 3.0).abs() < 1e-5,
        "Expected 3.0, got {}",
        data[0]
    );
    assert!(
        (data[1] - 40.0).abs() < 1e-5,
        "Expected 40.0, got {}",
        data[1]
    );
}

#[test]
fn test_split_states() {
    ensure_backend();
    let device = init_gpu_device();

    // Create some state tensors
    let states: Vec<Tensor<WgpuBackend, 2>> = vec![
        Tensor::ones([3, 2], &device),  // First is spin
        Tensor::zeros([3, 2], &device), // Second is spin
        Tensor::ones([3, 2], &device),  // Third is categorical
    ];

    let n_spin = 2;
    let (spin_states, cat_states) = split_states(&states, n_spin);

    assert_eq!(spin_states.len(), 2, "Should have 2 spin states");
    assert_eq!(cat_states.len(), 1, "Should have 1 categorical state");
}
