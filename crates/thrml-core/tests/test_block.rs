//! Tests for Block and BlockSpec
//!
//! Port of Python tests/test_block_management.py

use indexmap::IndexMap;
use thrml_core::block::Block;
use thrml_core::blockspec::BlockSpec;
use thrml_core::node::{Node, NodeType, TensorSpec};

/// Test that blocks enforce same node type
#[test]
fn test_block_same_type() {
    let spin_nodes: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();
    let block = Block::new(spin_nodes);
    assert!(block.is_ok(), "Should create block with same types");
}

#[test]
fn test_block_mixed_types_fails() {
    let spin = Node::new(NodeType::Spin);
    let cat = Node::new(NodeType::Categorical { n_categories: 3 });
    let mixed = vec![spin, cat];
    let block = Block::new(mixed);
    assert!(block.is_err(), "Should fail with mixed node types");
    assert!(
        block.unwrap_err().contains("same type"),
        "Error should mention type mismatch"
    );
}

#[test]
fn test_block_empty_fails() {
    let empty: Vec<Node> = vec![];
    let block = Block::new(empty);
    assert!(block.is_err(), "Should fail with empty nodes");
}

#[test]
fn test_block_addition() {
    let nodes1: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();
    let nodes2: Vec<Node> = (0..2).map(|_| Node::new(NodeType::Spin)).collect();

    let block1 = Block::new(nodes1).unwrap();
    let block2 = Block::new(nodes2).unwrap();

    let combined = (block1 + block2).unwrap();
    assert_eq!(combined.len(), 5, "Combined block should have 5 nodes");
}

#[test]
fn test_block_addition_different_types_fails() {
    let spin_nodes: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();
    let cat_nodes: Vec<Node> = (0..2)
        .map(|_| Node::new(NodeType::Categorical { n_categories: 3 }))
        .collect();

    let block1 = Block::new(spin_nodes).unwrap();
    let block2 = Block::new(cat_nodes).unwrap();

    let result = block1 + block2;
    assert!(result.is_err(), "Should fail adding different types");
}

/// Test BlockSpec creation and node location lookups
#[test]
fn test_blockspec_creation() {
    let nodes1: Vec<Node> = (0..5).map(|_| Node::new(NodeType::Spin)).collect();
    let nodes2: Vec<Node> = (0..3)
        .map(|_| Node::new(NodeType::Categorical { n_categories: 4 }))
        .collect();

    let blocks = vec![Block::new(nodes1).unwrap(), Block::new(nodes2).unwrap()];

    let mut node_shape_dtypes = IndexMap::new();
    node_shape_dtypes.insert(NodeType::Spin, TensorSpec::for_spin());
    node_shape_dtypes.insert(
        NodeType::Categorical { n_categories: 4 },
        TensorSpec::for_categorical(4),
    );

    let spec = BlockSpec::new(blocks, node_shape_dtypes);
    assert!(spec.is_ok(), "Should create BlockSpec successfully");

    let spec = spec.unwrap();
    assert_eq!(spec.blocks.len(), 2, "Should have 2 blocks");
}

#[test]
fn test_blockspec_duplicate_node_fails() {
    let nodes: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();
    let block1 = Block::new(nodes.clone()).unwrap();
    let block2 = Block::new(nodes).unwrap(); // Same nodes!

    let blocks = vec![block1, block2];

    let mut node_shape_dtypes = IndexMap::new();
    node_shape_dtypes.insert(NodeType::Spin, TensorSpec::for_spin());

    let spec = BlockSpec::new(blocks, node_shape_dtypes);
    assert!(spec.is_err(), "Should fail with duplicate nodes");

    let err = spec.err().unwrap();
    assert!(
        err.contains("twice"),
        "Error should mention duplicate, got: {}",
        err
    );
}

#[test]
fn test_blockspec_get_node_locations() {
    let nodes: Vec<Node> = (0..5).map(|_| Node::new(NodeType::Spin)).collect();
    let block = Block::new(nodes).unwrap();

    let mut node_shape_dtypes = IndexMap::new();
    node_shape_dtypes.insert(NodeType::Spin, TensorSpec::for_spin());

    let spec = BlockSpec::new(vec![block.clone()], node_shape_dtypes).unwrap();

    let (sd_idx, locations) = spec.get_node_locations(&block).unwrap();
    assert_eq!(sd_idx, 0, "First (only) SD should be at index 0");
    assert_eq!(locations.len(), 5, "Should have 5 locations");

    // Locations should be sequential for single block
    for (i, &loc) in locations.iter().enumerate() {
        assert_eq!(loc, i, "Location {} should be {}", i, loc);
    }
}

#[test]
fn test_blockspec_multiple_blocks_same_type() {
    // Two blocks of the same type
    let nodes1: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();
    let nodes2: Vec<Node> = (0..4).map(|_| Node::new(NodeType::Spin)).collect();

    let block1 = Block::new(nodes1).unwrap();
    let block2 = Block::new(nodes2).unwrap();

    let mut node_shape_dtypes = IndexMap::new();
    node_shape_dtypes.insert(NodeType::Spin, TensorSpec::for_spin());

    let spec = BlockSpec::new(vec![block1.clone(), block2.clone()], node_shape_dtypes).unwrap();

    // First block should have indices 0,1,2
    let (sd_idx1, locs1) = spec.get_node_locations(&block1).unwrap();
    assert_eq!(sd_idx1, 0);
    assert_eq!(locs1, vec![0, 1, 2]);

    // Second block should have indices 3,4,5,6
    let (sd_idx2, locs2) = spec.get_node_locations(&block2).unwrap();
    assert_eq!(sd_idx2, 0); // Same type, same global array
    assert_eq!(locs2, vec![3, 4, 5, 6]);
}
