//! Tests for InteractionGroup
//!
//! Port of Python tests/test_interaction.py

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::interaction::InteractionGroup;
use thrml_core::node::{Node, NodeType};

#[cfg(feature = "gpu")]
#[test]
fn test_interaction_group_good() {
    use thrml_core::backend::{ensure_metal_backend, init_gpu_device};

    ensure_metal_backend();
    let device = init_gpu_device();

    let block_size = 4;
    let head_nodes: Vec<Node> = (0..block_size).map(|_| Node::new(NodeType::Spin)).collect();
    let tail_nodes1: Vec<Node> = (0..block_size).map(|_| Node::new(NodeType::Spin)).collect();
    let tail_nodes2: Vec<Node> = (0..block_size).map(|_| Node::new(NodeType::Spin)).collect();

    let head = Block::new(head_nodes).unwrap();
    let tail1 = Block::new(tail_nodes1).unwrap();
    let tail2 = Block::new(tail_nodes2).unwrap();

    // 3D tensor: [n_nodes, dim1, dim2]
    let interaction: Tensor<WgpuBackend, 3> = Tensor::zeros([block_size, 2, 2], &device);

    let result = InteractionGroup::new(interaction, head, vec![tail1, tail2], 2);
    assert!(
        result.is_ok(),
        "Should create InteractionGroup successfully"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_interaction_group_bad_tail_size() {
    use thrml_core::backend::{ensure_metal_backend, init_gpu_device};

    ensure_metal_backend();
    let device = init_gpu_device();

    let block_size = 4;
    let head_nodes: Vec<Node> = (0..block_size).map(|_| Node::new(NodeType::Spin)).collect();
    // Tail has wrong size
    let tail_nodes: Vec<Node> = (0..block_size + 1)
        .map(|_| Node::new(NodeType::Spin))
        .collect();

    let head = Block::new(head_nodes).unwrap();
    let tail = Block::new(tail_nodes).unwrap();

    let interaction: Tensor<WgpuBackend, 3> = Tensor::zeros([block_size, 2, 2], &device);

    let result = InteractionGroup::new(interaction, head, vec![tail], 1);
    assert!(result.is_err(), "Should fail with mismatched tail size");

    let err = result.err().unwrap();
    assert!(
        err.contains("tail node blocks"),
        "Error should mention tail node blocks, got: {}",
        err
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_interaction_group_bad_interaction_size() {
    use thrml_core::backend::{ensure_metal_backend, init_gpu_device};

    ensure_metal_backend();
    let device = init_gpu_device();

    let block_size = 4;
    let head_nodes: Vec<Node> = (0..block_size).map(|_| Node::new(NodeType::Spin)).collect();
    let tail_nodes: Vec<Node> = (0..block_size).map(|_| Node::new(NodeType::Spin)).collect();

    let head = Block::new(head_nodes).unwrap();
    let tail = Block::new(tail_nodes).unwrap();

    // Wrong leading dimension in interaction tensor
    let interaction: Tensor<WgpuBackend, 3> = Tensor::zeros([block_size + 1, 2, 2], &device);

    let result = InteractionGroup::new(interaction, head, vec![tail], 1);
    assert!(
        result.is_err(),
        "Should fail with mismatched interaction dimension"
    );

    let err = result.err().unwrap();
    assert!(
        err.contains("leading dimension"),
        "Error should mention leading dimension, got: {}",
        err
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_interaction_group_n_spin_validation() {
    use thrml_core::backend::{ensure_metal_backend, init_gpu_device};

    ensure_metal_backend();
    let device = init_gpu_device();

    let block_size = 4;
    let head_nodes: Vec<Node> = (0..block_size).map(|_| Node::new(NodeType::Spin)).collect();
    let tail_nodes: Vec<Node> = (0..block_size).map(|_| Node::new(NodeType::Spin)).collect();

    let head = Block::new(head_nodes).unwrap();
    let tail = Block::new(tail_nodes).unwrap();

    let interaction: Tensor<WgpuBackend, 3> = Tensor::zeros([block_size, 2, 2], &device);

    // n_spin = 5 but only 1 tail block
    let result = InteractionGroup::new(interaction, head, vec![tail], 5);
    assert!(result.is_err(), "Should fail when n_spin > num tail blocks");

    let err = result.err().unwrap();
    assert!(
        err.contains("n_spin"),
        "Error should mention n_spin, got: {}",
        err
    );
}
