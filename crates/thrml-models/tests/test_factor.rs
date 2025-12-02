//! Tests for the Factor system
//!
//! Port of Python tests/test_factor.py

use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};
use thrml_models::factor::validate_node_groups;

#[test]
fn test_validate_node_groups_good() {
    let n_nodes = 5;
    let blocks: Vec<Block> = (0..3)
        .map(|_| {
            let nodes: Vec<Node> = (0..n_nodes).map(|_| Node::new(NodeType::Spin)).collect();
            Block::new(nodes).unwrap()
        })
        .collect();

    let result = validate_node_groups(&blocks);
    assert!(result.is_ok(), "Should validate good node groups");
    assert_eq!(result.unwrap(), n_nodes, "Should return correct node count");
}

#[test]
fn test_validate_node_groups_empty_fails() {
    let blocks: Vec<Block> = vec![];
    let result = validate_node_groups(&blocks);
    assert!(result.is_err(), "Should fail with empty node groups");
    assert!(
        result.unwrap_err().contains("empty"),
        "Error should mention empty"
    );
}

#[test]
fn test_validate_node_groups_ragged_fails() {
    let nodes1: Vec<Node> = (0..5).map(|_| Node::new(NodeType::Spin)).collect();
    let nodes2: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();

    let block1 = Block::new(nodes1).unwrap();
    let block2 = Block::new(nodes2).unwrap();

    let result = validate_node_groups(&[block1, block2]);
    assert!(result.is_err(), "Should fail with ragged node groups");
    assert!(
        result.unwrap_err().contains("same number"),
        "Error should mention same number"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_spin_factor_to_interactions() {
    use burn::tensor::Tensor;
    use thrml_core::backend::{ensure_backend, init_gpu_device, WgpuBackend};
    use thrml_models::discrete_ebm::SpinEBMFactor;
    use thrml_models::factor::AbstractFactor;

    ensure_backend();
    let device = init_gpu_device();

    let n_nodes = 4;

    // Create three blocks of spin nodes
    let block1 = Block::new(
        (0..n_nodes)
            .map(|_| Node::new(NodeType::Spin))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let block2 = Block::new(
        (0..n_nodes)
            .map(|_| Node::new(NodeType::Spin))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let block3 = Block::new(
        (0..n_nodes)
            .map(|_| Node::new(NodeType::Spin))
            .collect::<Vec<_>>(),
    )
    .unwrap();

    // SpinEBMFactor needs 3D weights: [n_nodes, 1, 1]
    let weights: Tensor<WgpuBackend, 3> = Tensor::ones([n_nodes, 1, 1], &device);

    let factor = SpinEBMFactor::new(vec![block1, block2, block3], weights);
    assert!(
        factor.is_ok(),
        "SpinEBMFactor creation should succeed: {:?}",
        factor.err()
    );
    let factor = factor.unwrap();

    let interactions = factor.to_interaction_groups(&device);

    // For 3 blocks, we should get 1 merged interaction group
    // (since SpinEBMFactor uses SquareDiscreteEBMFactor which merges)
    assert!(
        !interactions.is_empty(),
        "Should produce interaction groups"
    );

    // Check that head and tail nodes are properly set
    let group = &interactions[0];
    assert!(
        !group.head_nodes.is_empty(),
        "Head nodes should not be empty"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_factor_interaction_group_structure() {
    use burn::tensor::Tensor;
    use thrml_core::backend::{ensure_backend, init_gpu_device, WgpuBackend};
    use thrml_models::discrete_ebm::DiscreteEBMFactor;
    use thrml_models::factor::AbstractFactor;

    ensure_backend();
    let device = init_gpu_device();

    let n_nodes = 3;

    let spin_nodes: Vec<Node> = (0..n_nodes).map(|_| Node::new(NodeType::Spin)).collect();
    let cat_nodes: Vec<Node> = (0..n_nodes)
        .map(|_| Node::new(NodeType::Categorical { n_categories: 4 }))
        .collect();

    let spin_block = Block::new(spin_nodes).unwrap();
    let cat_block = Block::new(cat_nodes).unwrap();

    let weights: Tensor<WgpuBackend, 3> = Tensor::ones([n_nodes, 4, 1], &device);

    let factor = DiscreteEBMFactor::new(vec![spin_block], vec![cat_block], weights).unwrap();

    let interactions = factor.to_interaction_groups(&device);

    for group in &interactions {
        // Head nodes and tail nodes should have same length
        for tail in &group.tail_nodes {
            assert_eq!(
                group.head_nodes.len(),
                tail.len(),
                "Head and tail blocks should have same length"
            );
        }

        // Interaction weights should have head_nodes.len() as first dimension
        let weight_dims = group.interaction.weights.dims();
        assert_eq!(
            weight_dims[0],
            group.head_nodes.len(),
            "Weights first dim should match head nodes"
        );
    }
}
