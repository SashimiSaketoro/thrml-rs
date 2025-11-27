//! Graph-based energy models.
//!
//! This module provides energy-based models that use graph connectivity
//! to define interactions between nodes.
//!
//! - [`GraphEdge`]: An edge connecting two nodes with a weight
//! - [`GraphSidecar`]: Graph structure with edges and optional node attributes
//! - [`SpringEBM`]: Spring-like forces between connected nodes

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::distance::pairwise_distances_sq;

/// An edge in a graph connecting two nodes.
#[derive(Clone, Debug)]
pub struct GraphEdge {
    /// Source node index
    pub src: usize,
    /// Destination node index
    pub dst: usize,
    /// Edge weight
    pub weight: f32,
}

impl GraphEdge {
    /// Create a new edge.
    pub fn new(src: usize, dst: usize, weight: f32) -> Self {
        Self { src, dst, weight }
    }
}

/// Graph structure (sidecar) with edges and optional node attributes.
///
/// This can represent:
/// - Hypergraphs from BLT pipelines
/// - Knowledge graphs
/// - Dependency graphs
/// - Any connectivity structure
#[derive(Clone, Debug, Default)]
pub struct GraphSidecar {
    /// Edges in the graph
    pub edges: Vec<GraphEdge>,
    /// Number of nodes
    pub n_nodes: usize,
    /// Optional node weights/scores
    pub node_weights: Option<Vec<f32>>,
    /// Optional edge types (for heterogeneous graphs)
    pub edge_types: Option<Vec<u8>>,
}

impl GraphSidecar {
    /// Create a new graph sidecar.
    pub fn new(n_nodes: usize) -> Self {
        Self {
            edges: Vec::new(),
            n_nodes,
            node_weights: None,
            edge_types: None,
        }
    }
    
    /// Add an edge to the graph.
    pub fn add_edge(&mut self, src: usize, dst: usize, weight: f32) {
        assert!(src < self.n_nodes && dst < self.n_nodes, "Node index out of bounds");
        self.edges.push(GraphEdge::new(src, dst, weight));
    }
    
    /// Add a bidirectional edge.
    pub fn add_edge_bidir(&mut self, a: usize, b: usize, weight: f32) {
        self.add_edge(a, b, weight);
        self.add_edge(b, a, weight);
    }
    
    /// Set node weights.
    pub fn with_node_weights(mut self, weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), self.n_nodes);
        self.node_weights = Some(weights);
        self
    }
    
    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
    
    /// Check if the graph is empty.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
    
    /// Get adjacency list representation.
    pub fn to_adjacency_list(&self) -> Vec<Vec<(usize, f32)>> {
        let mut adj = vec![Vec::new(); self.n_nodes];
        for edge in &self.edges {
            adj[edge.src].push((edge.dst, edge.weight));
        }
        adj
    }
    
    /// Convert to dense adjacency matrix.
    ///
    /// Returns [N, N] tensor.
    pub fn to_dense_adjacency(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let n = self.n_nodes;
        let mut data = vec![0.0f32; n * n];
        
        for edge in &self.edges {
            data[edge.src * n + edge.dst] = edge.weight;
        }
        
        let tensor_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(data.as_slice(), device);
        tensor_1d.reshape([n as i32, n as i32])
    }
    
    /// Convert to symmetric dense adjacency matrix.
    ///
    /// Adds both directions for each edge.
    pub fn to_symmetric_adjacency(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let n = self.n_nodes;
        let mut data = vec![0.0f32; n * n];
        
        for edge in &self.edges {
            data[edge.src * n + edge.dst] = edge.weight;
            data[edge.dst * n + edge.src] = edge.weight;
        }
        
        let tensor_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(data.as_slice(), device);
        tensor_1d.reshape([n as i32, n as i32])
    }
    
    /// Get node weights as tensor.
    pub fn node_weights_tensor(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Option<Tensor<WgpuBackend, 1>> {
        self.node_weights.as_ref().map(|w| {
            Tensor::from_data(w.as_slice(), device)
        })
    }
}

/// Spring-based energy model for graph connectivity.
///
/// Connected nodes experience spring-like attraction:
/// ```text
/// E_spring[i] = k * sum_j adj[i,j] * ||pos[i] - pos[j]||^2
/// ```
///
/// This encourages connected nodes to cluster together.
#[derive(Clone)]
pub struct SpringEBM {
    /// Adjacency matrix [N, N]
    pub adjacency: Tensor<WgpuBackend, 2>,
    /// Spring constant
    pub spring_constant: f32,
}

impl SpringEBM {
    /// Create from adjacency matrix.
    pub fn new(adjacency: Tensor<WgpuBackend, 2>, spring_constant: f32) -> Self {
        Self {
            adjacency,
            spring_constant,
        }
    }
    
    /// Create from graph sidecar.
    pub fn from_graph(
        graph: &GraphSidecar,
        spring_constant: f32,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let adjacency = graph.to_symmetric_adjacency(device);
        Self::new(adjacency, spring_constant)
    }
    
    /// Computes spring energy per node.
    ///
    /// `E_spring[i] = k * sum_j adj[i,j] * ||pos[i] - pos[j]||^2`
    ///
    /// # Arguments
    ///
    /// * `positions` - Node positions \[N, D\]
    ///
    /// # Returns
    ///
    /// Energy per node \[N\].
    pub fn energy(&self, positions: &Tensor<WgpuBackend, 2>) -> Tensor<WgpuBackend, 1> {
        let n = positions.dims()[0];
        let dist_sq = pairwise_distances_sq(positions);
        
        // Weighted distances: adj_ij * dist_sq_ij
        let weighted_dist = self.adjacency.clone() * dist_sq;
        
        // Sum over connections for each node
        let energy_2d = weighted_dist.sum_dim(1);
        let energy: Tensor<WgpuBackend, 1> = energy_2d.reshape([n as i32]);
        
        energy.mul_scalar(self.spring_constant)
    }
    
    /// Total energy across all nodes.
    pub fn total_energy(&self, positions: &Tensor<WgpuBackend, 2>) -> f32 {
        let energy = self.energy(positions);
        let energy_data: Vec<f32> = energy.into_data().to_vec().expect("energy to vec");
        energy_data.iter().sum()
    }
    
    /// Computes spring force per node in Cartesian coordinates.
    ///
    /// `F_spring[i] = 2k * sum_j adj[i,j] * (pos[j] - pos[i])`
    ///
    /// # Arguments
    ///
    /// * `positions` - Node positions \[N, D\]
    ///
    /// # Returns
    ///
    /// Force per node \[N, D\].
    pub fn force(&self, positions: &Tensor<WgpuBackend, 2>) -> Tensor<WgpuBackend, 2> {
        // For each node i: sum_j adj[i,j] * (pos[j] - pos[i])
        // = adj @ positions - positions * row_sum(adj)
        
        let weighted_pos = self.adjacency.clone().matmul(positions.clone());
        let row_sums = self.adjacency.clone().sum_dim(1); // [N, 1]
        let scaled_pos = positions.clone() * row_sums;
        
        // Force = 2k * (weighted_pos - scaled_pos)
        (weighted_pos - scaled_pos).mul_scalar(2.0 * self.spring_constant)
    }
    
    /// Compute energy gradient (negative force).
    pub fn gradient(&self, positions: &Tensor<WgpuBackend, 2>) -> Tensor<WgpuBackend, 2> {
        -self.force(positions)
    }
}

/// Weighted node bias energy.
///
/// `E_bias[i] = -weight[i] * f(pos[i])`
///
/// Where f is a function of position (e.g., distance from origin).
#[derive(Clone)]
pub struct NodeBiasEBM {
    /// Node weights \[N\].
    pub weights: Tensor<WgpuBackend, 1>,
    /// Bias strength.
    pub strength: f32,
}

impl NodeBiasEBM {
    /// Create from weights.
    pub fn new(weights: Tensor<WgpuBackend, 1>, strength: f32) -> Self {
        Self { weights, strength }
    }
    
    /// Computes bias energy per node based on distance from origin.
    ///
    /// `E_bias[i] = -strength * weight[i] * log(||pos[i]|| / max_dist)`
    ///
    /// Higher weight nodes are biased toward smaller distances (toward origin).
    pub fn energy_radial(
        &self,
        positions: &Tensor<WgpuBackend, 2>,
        max_dist: f32,
    ) -> Tensor<WgpuBackend, 1> {
        let n = positions.dims()[0];
        
        // Compute distances from origin
        let dist = positions.clone().powf_scalar(2.0).sum_dim(1).sqrt(); // [N, 1]
        let dist_1d: Tensor<WgpuBackend, 1> = dist.reshape([n as i32]);
        
        // log(dist / max_dist) is negative for dist < max_dist
        let log_ratio = (dist_1d / max_dist).log();
        
        // E = -strength * weight * log_ratio
        // Negate so high weight + small dist = low energy
        -self.weights.clone() * log_ratio * self.strength
    }
    
    /// Computes radial force from bias.
    ///
    /// `F_bias[i] = strength * weight[i] / ||pos[i]||`
    ///
    /// Points toward origin for positive weights.
    pub fn force_radial(
        &self,
        positions: &Tensor<WgpuBackend, 2>,
    ) -> Tensor<WgpuBackend, 1> {
        let n = positions.dims()[0];
        
        let dist = positions.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        let dist_1d: Tensor<WgpuBackend, 1> = dist.reshape([n as i32]);
        let dist_safe = dist_1d.clamp(1e-8, f32::MAX);
        
        self.weights.clone() * self.strength / dist_safe
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_graph_sidecar_adjacency() {
        let device = init_gpu_device();
        
        let mut graph = GraphSidecar::new(4);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 0.5);
        graph.add_edge(2, 3, 0.8);
        
        let adj = graph.to_symmetric_adjacency(&device);
        let adj_data: Vec<f32> = adj.into_data().to_vec().expect("adj to vec");
        
        // Check edge 0-1 is symmetric
        assert!((adj_data[0 * 4 + 1] - 1.0).abs() < 1e-6);
        assert!((adj_data[1 * 4 + 0] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_spring_ebm_energy() {
        let device = init_gpu_device();
        
        let mut graph = GraphSidecar::new(2);
        graph.add_edge_bidir(0, 1, 1.0);
        
        let ebm = SpringEBM::from_graph(&graph, 1.0, &device);
        
        // Nodes at same position - zero energy
        let pos_same_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(
            [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0].as_slice(),
            &device,
        );
        let pos_same = pos_same_1d.reshape([2, 3]);
        let e_same = ebm.total_energy(&pos_same);
        assert!(e_same.abs() < 1e-5, "Same position should have ~0 energy");
        
        // Nodes at different positions - positive energy
        let pos_diff_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(
            [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0].as_slice(),
            &device,
        );
        let pos_diff = pos_diff_1d.reshape([2, 3]);
        let e_diff = ebm.total_energy(&pos_diff);
        assert!(e_diff > 0.0, "Different positions should have positive energy");
    }
    
    #[test]
    fn test_spring_force_attracts() {
        let device = init_gpu_device();
        
        let mut graph = GraphSidecar::new(2);
        graph.add_edge_bidir(0, 1, 1.0);
        
        let ebm = SpringEBM::from_graph(&graph, 1.0, &device);
        
        // Node 0 at origin, node 1 at (1, 0, 0)
        let positions_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(
            [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0].as_slice(),
            &device,
        );
        let positions = positions_1d.reshape([2, 3]);
        
        let force = ebm.force(&positions);
        let force_data: Vec<f32> = force.into_data().to_vec().expect("force to vec");
        
        // Force on node 0 should point toward node 1 (positive x)
        assert!(force_data[0] > 0.0, "Force should attract node 0 toward node 1");
        
        // Force on node 1 should point toward node 0 (negative x)
        assert!(force_data[3] < 0.0, "Force should attract node 1 toward node 0");
    }
}

