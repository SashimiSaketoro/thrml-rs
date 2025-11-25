//! # thrml-examples utilities
//!
//! Shared utilities for THRML examples, including graph generation,
//! coloring algorithms, and visualization helpers.

use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};

/// A graph with THRML nodes as node weights.
pub type ThrmlGraph = UnGraph<Node, ()>;

/// Result of graph coloring - maps each node to a color index.
pub type Coloring = HashMap<NodeIndex, usize>;

/// Generate a 2D lattice/grid graph with the given dimensions.
///
/// Similar to NetworkX's `grid_graph(dim=(rows, cols))`.
///
/// # Arguments
///
/// * `rows` - Number of rows in the grid
/// * `cols` - Number of columns in the grid
/// * `node_type` - The type of node to create (Spin or Categorical)
///
/// # Returns
///
/// A tuple of (graph, node_grid) where `node_grid[row][col]` gives the NodeIndex.
pub fn generate_lattice_graph(
    rows: usize,
    cols: usize,
    node_type: NodeType,
) -> (ThrmlGraph, Vec<Vec<NodeIndex>>) {
    let mut graph = UnGraph::new_undirected();
    let mut node_grid: Vec<Vec<NodeIndex>> = Vec::with_capacity(rows);

    // Create nodes
    for _row in 0..rows {
        let mut row_nodes = Vec::with_capacity(cols);
        for _col in 0..cols {
            let node = Node::new(node_type.clone());
            let idx = graph.add_node(node);
            row_nodes.push(idx);
        }
        node_grid.push(row_nodes);
    }

    // Create edges (4-connectivity: up, down, left, right)
    for row in 0..rows {
        for col in 0..cols {
            let current = node_grid[row][col];

            // Right neighbor
            if col + 1 < cols {
                graph.add_edge(current, node_grid[row][col + 1], ());
            }

            // Down neighbor
            if row + 1 < rows {
                graph.add_edge(current, node_grid[row + 1][col], ());
            }
        }
    }

    (graph, node_grid)
}

/// Generate a 1D chain graph.
///
/// # Arguments
///
/// * `length` - Number of nodes in the chain
/// * `node_type` - The type of node to create
///
/// # Returns
///
/// A tuple of (graph, nodes) where nodes is a vector of NodeIndex in order.
pub fn generate_chain_graph(length: usize, node_type: NodeType) -> (ThrmlGraph, Vec<NodeIndex>) {
    let mut graph = UnGraph::new_undirected();
    let mut nodes = Vec::with_capacity(length);

    // Create nodes
    for _ in 0..length {
        let node = Node::new(node_type.clone());
        let idx = graph.add_node(node);
        nodes.push(idx);
    }

    // Create edges
    for i in 0..length.saturating_sub(1) {
        graph.add_edge(nodes[i], nodes[i + 1], ());
    }

    (graph, nodes)
}

/// Perform greedy graph coloring.
///
/// Uses a simple greedy algorithm that assigns the smallest available color
/// to each node. This is not optimal but is fast and works well for regular lattices.
///
/// # Arguments
///
/// * `graph` - The graph to color
///
/// # Returns
///
/// A Coloring mapping each NodeIndex to a color (0-indexed).
pub fn greedy_coloring(graph: &ThrmlGraph) -> Coloring {
    let mut coloring: Coloring = HashMap::new();

    for node_idx in graph.node_indices() {
        // Find colors used by neighbors
        let mut neighbor_colors: Vec<usize> = Vec::new();
        for edge in graph.edges(node_idx) {
            let neighbor = if edge.source() == node_idx {
                edge.target()
            } else {
                edge.source()
            };

            if let Some(&color) = coloring.get(&neighbor) {
                neighbor_colors.push(color);
            }
        }

        // Find smallest color not used by neighbors
        neighbor_colors.sort_unstable();
        let mut color = 0;
        for &nc in &neighbor_colors {
            if nc == color {
                color += 1;
            } else if nc > color {
                break;
            }
        }

        coloring.insert(node_idx, color);
    }

    coloring
}

/// Compute bipartite coloring for a bipartite graph (like a grid).
///
/// For a 2D grid, this produces a checkerboard pattern with 2 colors.
///
/// # Arguments
///
/// * `node_grid` - 2D array of NodeIndex from generate_lattice_graph
///
/// # Returns
///
/// A Coloring with exactly 2 colors (0 and 1).
pub fn bipartite_coloring(node_grid: &[Vec<NodeIndex>]) -> Coloring {
    let mut coloring = HashMap::new();

    for (row, row_nodes) in node_grid.iter().enumerate() {
        for (col, &node_idx) in row_nodes.iter().enumerate() {
            // Checkerboard pattern
            let color = (row + col) % 2;
            coloring.insert(node_idx, color);
        }
    }

    coloring
}

/// Convert a graph coloring into THRML Blocks.
///
/// Groups nodes by their color and creates a Block for each color.
///
/// # Arguments
///
/// * `graph` - The graph containing the nodes
/// * `coloring` - The coloring to use
///
/// # Returns
///
/// A vector of Blocks, one per color, in color order (0, 1, 2, ...).
pub fn coloring_to_blocks(graph: &ThrmlGraph, coloring: &Coloring) -> Vec<Block> {
    // Find number of colors
    let n_colors = coloring.values().max().map(|&c| c + 1).unwrap_or(0);

    // Group nodes by color
    let mut color_groups: Vec<Vec<Node>> = vec![Vec::new(); n_colors];

    for (&node_idx, &color) in coloring {
        if let Some(node) = graph.node_weight(node_idx) {
            color_groups[color].push(node.clone());
        }
    }

    // Create blocks
    color_groups
        .into_iter()
        .filter(|nodes| !nodes.is_empty())
        .map(|nodes| Block::new(nodes).expect("Failed to create block"))
        .collect()
}

/// Get all nodes from a graph as a vector.
pub fn graph_nodes(graph: &ThrmlGraph) -> Vec<Node> {
    graph.node_weights().cloned().collect()
}

/// Get all edges from a graph as pairs of Nodes.
pub fn graph_edges(graph: &ThrmlGraph) -> Vec<(Node, Node)> {
    graph
        .edge_indices()
        .filter_map(|edge_idx| {
            let (a, b) = graph.edge_endpoints(edge_idx)?;
            let node_a = graph.node_weight(a)?.clone();
            let node_b = graph.node_weight(b)?.clone();
            Some((node_a, node_b))
        })
        .collect()
}

/// Create output directory if it doesn't exist.
pub fn ensure_output_dir() -> std::io::Result<std::path::PathBuf> {
    let output_dir = std::path::PathBuf::from("output");
    std::fs::create_dir_all(&output_dir)?;
    Ok(output_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_chain_graph() {
        let (graph, nodes) = generate_chain_graph(5, NodeType::Spin);
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 4);
        assert_eq!(nodes.len(), 5);
    }

    #[test]
    fn test_generate_lattice_graph() {
        let (graph, node_grid) = generate_lattice_graph(3, 4, NodeType::Spin);
        assert_eq!(graph.node_count(), 12);
        // Edges: 3*3 horizontal + 2*4 vertical = 9 + 8 = 17
        assert_eq!(graph.edge_count(), 17);
        assert_eq!(node_grid.len(), 3);
        assert_eq!(node_grid[0].len(), 4);
    }

    #[test]
    fn test_greedy_coloring() {
        let (graph, _) = generate_chain_graph(5, NodeType::Spin);
        let coloring = greedy_coloring(&graph);
        assert_eq!(coloring.len(), 5);

        // Chain should be 2-colorable
        let n_colors = *coloring.values().max().unwrap() + 1;
        assert_eq!(n_colors, 2);
    }

    #[test]
    fn test_bipartite_coloring() {
        let (graph, node_grid) = generate_lattice_graph(3, 3, NodeType::Spin);
        let coloring = bipartite_coloring(&node_grid);

        // Should have exactly 2 colors
        let colors: std::collections::HashSet<_> = coloring.values().collect();
        assert_eq!(colors.len(), 2);

        // Adjacent nodes should have different colors
        for edge in graph.edge_indices() {
            let (a, b) = graph.edge_endpoints(edge).unwrap();
            assert_ne!(coloring[&a], coloring[&b]);
        }
    }

    #[test]
    fn test_coloring_to_blocks() {
        let (graph, node_grid) = generate_lattice_graph(2, 2, NodeType::Spin);
        let coloring = bipartite_coloring(&node_grid);
        let blocks = coloring_to_blocks(&graph, &coloring);

        assert_eq!(blocks.len(), 2);
        // 2x2 grid has 2 nodes of each color
        assert_eq!(blocks[0].len(), 2);
        assert_eq!(blocks[1].len(), 2);
    }
}
