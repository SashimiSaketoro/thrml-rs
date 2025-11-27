//! Graph construction utilities for probabilistic graphical models.
//!
//! This module provides utilities for constructing common graph topologies
//! used in PGMs, particularly for block Gibbs sampling.
//!
//! ## 2D Lattice Graphs
//!
//! The [`make_lattice_graph`] function creates a square lattice with:
//! - Optional torus (periodic boundary) topology
//! - Beyond-nearest-neighbor connections via jump patterns
//! - Two-color (checkerboard) blocking for efficient parallel Gibbs updates
//!
//! ```rust,ignore
//! use thrml_models::graph_utils::make_lattice_graph;
//!
//! // Create an 8x8 lattice with periodic boundaries
//! let (nodes, sidecar, upper_block, lower_block) = make_lattice_graph(8, true);
//!
//! // Use blocks for two-color Gibbs sampling
//! let free_blocks = vec![upper_block, lower_block];
//! ```

use crate::graph_ebm::GraphSidecar;
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};

/// Jump patterns for lattice connections.
///
/// Each (di, dj) defines a connection from cell (i, j) to (i+di, j+dj).
/// These create beyond-nearest-neighbor interactions for richer graph structure.
const JUMP_PATTERNS: [(i32, i32); 4] = [
    (1, 0),  // Vertical neighbor
    (2, 1),  // Knight's move variant
    (3, 2),  // Extended diagonal
    (1, 4),  // Extended horizontal
];

/// Create a 2D square lattice graph with two-color blocking for block Gibbs sampling.
///
/// This function creates a lattice graph suitable for efficient block Gibbs sampling
/// using the two-color (checkerboard) decomposition. Nodes in each color can be
/// updated in parallel since they share no edges within the same color.
///
/// # Arguments
///
/// * `side_len` - Side length of the square lattice. Will be rounded up to the
///   nearest even number for proper two-coloring.
/// * `torus` - If true, use periodic boundary conditions (wrap edges).
///   If false, edges that would go out of bounds are omitted.
///
/// # Returns
///
/// A tuple containing:
/// * `nodes` - All SpinNodes in the lattice, indexed row-major as `i * side_len + j`
/// * `sidecar` - GraphSidecar containing all edges with unit weights
/// * `upper_block` - Block containing "white" checkerboard squares (even parity)
/// * `lower_block` - Block containing "black" checkerboard squares (odd parity)
///
/// # Example
///
/// ```rust,ignore
/// use thrml_models::graph_utils::make_lattice_graph;
/// use thrml_models::ising::{IsingEBM, IsingSamplingProgram};
///
/// // Create 6x6 lattice with periodic boundaries
/// let (nodes, sidecar, upper, lower) = make_lattice_graph(6, true);
///
/// // Use for Ising model with checkerboard Gibbs sampling
/// let free_blocks = vec![upper, lower];
/// ```
///
/// # Panics
///
/// Panics if `side_len` is 0.
pub fn make_lattice_graph(side_len: usize, torus: bool) -> (Vec<Node>, GraphSidecar, Block, Block) {
    assert!(side_len > 0, "side_len must be positive");

    // Round up to even for proper two-coloring
    let side_len = (side_len + 1) / 2 * 2;
    let size = side_len * side_len;

    // Helper: convert (i, j) grid coordinates to linear index
    // Returns None for out-of-bounds when torus=false
    let get_idx = |i: i32, j: i32| -> Option<usize> {
        let side = side_len as i32;
        if torus {
            // Wrap around with modulo (handle negative values)
            let i_wrapped = ((i % side) + side) % side;
            let j_wrapped = ((j % side) + side) % side;
            Some((i_wrapped * side + j_wrapped) as usize)
        } else {
            // Boundary check
            if i >= 0 && i < side && j >= 0 && j < side {
                Some((i * side + j) as usize)
            } else {
                None
            }
        }
    };

    // Create all nodes
    let nodes: Vec<Node> = (0..size).map(|_| Node::new(NodeType::Spin)).collect();

    // Build edge set (using set to avoid duplicates from bidirectional patterns)
    let mut edge_set = std::collections::HashSet::new();

    for idx in 0..size {
        let i = (idx / side_len) as i32;
        let j = (idx % side_len) as i32;

        // Add edges for each jump pattern (positive and negative directions)
        for &(di, dj) in &JUMP_PATTERNS {
            // Positive direction
            if let Some(target_idx) = get_idx(i + di, j + dj) {
                let edge = if idx < target_idx {
                    (idx, target_idx)
                } else {
                    (target_idx, idx)
                };
                edge_set.insert(edge);
            }

            // Negative direction
            if let Some(target_idx) = get_idx(i - di, j - dj) {
                let edge = if idx < target_idx {
                    (idx, target_idx)
                } else {
                    (target_idx, idx)
                };
                edge_set.insert(edge);
            }
        }
    }

    // Build GraphSidecar with unit weights
    let mut sidecar = GraphSidecar::new(size);
    for (src, dst) in &edge_set {
        sidecar.add_edge(*src, *dst, 1.0);
    }

    // Two-color blocking: checkerboard pattern
    // A cell at (i, j) with linear index = i * side_len + j
    // Parity = (i + j) % 2
    // Even parity (0) -> upper block ("white" squares)
    // Odd parity (1) -> lower block ("black" squares)
    let mut upper_nodes = Vec::new();
    let mut lower_nodes = Vec::new();

    for idx in 0..size {
        let i = idx / side_len;
        let j = idx % side_len;
        let parity = (i + j) % 2;

        if parity == 0 {
            upper_nodes.push(nodes[idx].clone());
        } else {
            lower_nodes.push(nodes[idx].clone());
        }
    }

    let upper_block = Block::new(upper_nodes).expect("upper block should be non-empty");
    let lower_block = Block::new(lower_nodes).expect("lower block should be non-empty");

    (nodes, sidecar, upper_block, lower_block)
}

/// Create a simple nearest-neighbor 2D lattice (4-connected grid).
///
/// This is a simpler version of [`make_lattice_graph`] that only connects
/// each cell to its 4 immediate neighbors (up, down, left, right).
///
/// # Arguments
///
/// * `side_len` - Side length of the square lattice
/// * `torus` - If true, use periodic boundary conditions
///
/// # Returns
///
/// Same tuple as [`make_lattice_graph`].
pub fn make_nearest_neighbor_lattice(
    side_len: usize,
    torus: bool,
) -> (Vec<Node>, GraphSidecar, Block, Block) {
    assert!(side_len > 0, "side_len must be positive");

    let side_len = (side_len + 1) / 2 * 2;
    let size = side_len * side_len;

    let get_idx = |i: i32, j: i32| -> Option<usize> {
        let side = side_len as i32;
        if torus {
            let i_wrapped = ((i % side) + side) % side;
            let j_wrapped = ((j % side) + side) % side;
            Some((i_wrapped * side + j_wrapped) as usize)
        } else {
            if i >= 0 && i < side && j >= 0 && j < side {
                Some((i * side + j) as usize)
            } else {
                None
            }
        }
    };

    let nodes: Vec<Node> = (0..size).map(|_| Node::new(NodeType::Spin)).collect();

    // Only 4-connected: up, down, left, right
    let nn_jumps: [(i32, i32); 2] = [(1, 0), (0, 1)];
    let mut edge_set = std::collections::HashSet::new();

    for idx in 0..size {
        let i = (idx / side_len) as i32;
        let j = (idx % side_len) as i32;

        for &(di, dj) in &nn_jumps {
            if let Some(target_idx) = get_idx(i + di, j + dj) {
                let edge = if idx < target_idx {
                    (idx, target_idx)
                } else {
                    (target_idx, idx)
                };
                edge_set.insert(edge);
            }
        }
    }

    let mut sidecar = GraphSidecar::new(size);
    for (src, dst) in &edge_set {
        sidecar.add_edge(*src, *dst, 1.0);
    }

    let mut upper_nodes = Vec::new();
    let mut lower_nodes = Vec::new();

    for idx in 0..size {
        let i = idx / side_len;
        let j = idx % side_len;
        if (i + j) % 2 == 0 {
            upper_nodes.push(nodes[idx].clone());
        } else {
            lower_nodes.push(nodes[idx].clone());
        }
    }

    let upper_block = Block::new(upper_nodes).expect("upper block should be non-empty");
    let lower_block = Block::new(lower_nodes).expect("lower block should be non-empty");

    (nodes, sidecar, upper_block, lower_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_size() {
        // Test that lattice has correct number of nodes
        let (nodes, sidecar, upper, lower) = make_lattice_graph(4, false);

        assert_eq!(nodes.len(), 16, "4x4 lattice should have 16 nodes");
        assert_eq!(sidecar.n_nodes, 16);

        // Blocks should partition all nodes
        assert_eq!(upper.len() + lower.len(), 16);
        assert_eq!(upper.len(), 8, "Half should be in upper block");
        assert_eq!(lower.len(), 8, "Half should be in lower block");
    }

    #[test]
    fn test_odd_side_rounds_up() {
        // Odd side_len should round up to even
        let (nodes, _, _, _) = make_lattice_graph(5, false);

        // 5 rounds up to 6, so 6x6 = 36 nodes
        assert_eq!(nodes.len(), 36);
    }

    #[test]
    fn test_torus_vs_non_torus_edges() {
        // Torus should have more edges due to wrapping
        let (_, sidecar_torus, _, _) = make_lattice_graph(4, true);
        let (_, sidecar_open, _, _) = make_lattice_graph(4, false);

        assert!(
            sidecar_torus.n_edges() >= sidecar_open.n_edges(),
            "Torus should have at least as many edges as open boundary"
        );
    }

    #[test]
    fn test_blocks_are_disjoint() {
        let (nodes, _, upper, lower) = make_lattice_graph(6, true);

        // Collect node IDs from each block
        let upper_ids: std::collections::HashSet<_> =
            upper.nodes().iter().map(|n| n.id()).collect();
        let lower_ids: std::collections::HashSet<_> =
            lower.nodes().iter().map(|n| n.id()).collect();

        // Check disjoint
        let intersection: Vec<_> = upper_ids.intersection(&lower_ids).collect();
        assert!(intersection.is_empty(), "Blocks should be disjoint");

        // Check complete coverage
        let all_ids: std::collections::HashSet<_> = nodes.iter().map(|n| n.id()).collect();
        let block_ids: std::collections::HashSet<_> =
            upper_ids.union(&lower_ids).copied().collect();
        assert_eq!(all_ids, block_ids, "Blocks should cover all nodes");
    }

    #[test]
    fn test_nearest_neighbor_lattice() {
        let (nodes, sidecar, upper, lower) = make_nearest_neighbor_lattice(4, false);

        assert_eq!(nodes.len(), 16);
        assert_eq!(upper.len() + lower.len(), 16);

        // 4x4 grid with open boundaries should have:
        // Horizontal edges: 4 rows * 3 edges = 12
        // Vertical edges: 3 rows * 4 edges = 12
        // Total: 24 edges
        assert_eq!(sidecar.n_edges(), 24);
    }

    #[test]
    fn test_nearest_neighbor_torus() {
        let (_, sidecar, _, _) = make_nearest_neighbor_lattice(4, true);

        // 4x4 torus should have:
        // Horizontal edges: 4 rows * 4 edges = 16
        // Vertical edges: 4 rows * 4 edges = 16
        // Total: 32 edges
        assert_eq!(sidecar.n_edges(), 32);
    }

    #[test]
    #[should_panic(expected = "side_len must be positive")]
    fn test_zero_side_len_panics() {
        make_lattice_graph(0, false);
    }
}
