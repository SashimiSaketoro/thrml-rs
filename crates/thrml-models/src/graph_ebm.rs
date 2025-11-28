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
        assert!(
            src < self.n_nodes && dst < self.n_nodes,
            "Node index out of bounds"
        );
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
        self.node_weights
            .as_ref()
            .map(|w| Tensor::from_data(w.as_slice(), device))
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
    pub fn force_radial(&self, positions: &Tensor<WgpuBackend, 2>) -> Tensor<WgpuBackend, 1> {
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
        let pos_same_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data([0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0].as_slice(), &device);
        let pos_same = pos_same_1d.reshape([2, 3]);
        let e_same = ebm.total_energy(&pos_same);
        assert!(e_same.abs() < 1e-5, "Same position should have ~0 energy");

        // Nodes at different positions - positive energy
        let pos_diff_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data([0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0].as_slice(), &device);
        let pos_diff = pos_diff_1d.reshape([2, 3]);
        let e_diff = ebm.total_energy(&pos_diff);
        assert!(
            e_diff > 0.0,
            "Different positions should have positive energy"
        );
    }

    #[test]
    fn test_spring_force_attracts() {
        let device = init_gpu_device();

        let mut graph = GraphSidecar::new(2);
        graph.add_edge_bidir(0, 1, 1.0);

        let ebm = SpringEBM::from_graph(&graph, 1.0, &device);

        // Node 0 at origin, node 1 at (1, 0, 0)
        let positions_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data([0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0].as_slice(), &device);
        let positions = positions_1d.reshape([2, 3]);

        let force = ebm.force(&positions);
        let force_data: Vec<f32> = force.into_data().to_vec().expect("force to vec");

        // Force on node 0 should point toward node 1 (positive x)
        assert!(
            force_data[0] > 0.0,
            "Force should attract node 0 toward node 1"
        );

        // Force on node 1 should point toward node 0 (negative x)
        assert!(
            force_data[3] < 0.0,
            "Force should attract node 1 toward node 0"
        );
    }

    // ============================================
    // Property-Based Random Tests
    // ============================================

    /// Helper: create a random graph with n nodes and approximately edge_density * n^2 edges
    fn create_random_graph(n: usize, edge_density: f32) -> GraphSidecar {
        let mut graph = GraphSidecar::new(n);
        let mut rng_state = 12345u64; // Simple LCG for deterministic "randomness"

        for i in 0..n {
            for j in (i + 1)..n {
                // Simple LCG: next = (a * current + c) mod m
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let rand_val = (rng_state >> 33) as f32 / (u32::MAX >> 1) as f32;

                if rand_val < edge_density {
                    // Random weight between 0.1 and 2.0
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let weight = 0.1 + 1.9 * (rng_state >> 33) as f32 / (u32::MAX >> 1) as f32;
                    graph.add_edge_bidir(i, j, weight);
                }
            }
        }
        graph
    }

    #[test]
    fn test_spring_energy_non_negative_random() {
        let device = init_gpu_device();
        let n = 20;
        let d = 3;

        // Create random graph
        let graph = create_random_graph(n, 0.3);
        let ebm = SpringEBM::from_graph(&graph, 1.0, &device);

        // Test with multiple random position configurations
        for seed in [42, 123, 456, 789, 1000] {
            let positions: Tensor<WgpuBackend, 2> =
                Tensor::random([n, d], Distribution::Normal(0.0, 10.0), &device);

            let energy = ebm.total_energy(&positions);
            assert!(
                energy >= -1e-5,
                "Spring energy should be non-negative, got {} (seed={})",
                energy,
                seed
            );
        }
    }

    #[test]
    fn test_spring_energy_zero_at_coincident() {
        let device = init_gpu_device();
        let n = 10;
        let d = 3;

        // Create a connected graph
        let graph = create_random_graph(n, 0.5);
        let ebm = SpringEBM::from_graph(&graph, 1.0, &device);

        // All nodes at the same random position
        let single_point: Tensor<WgpuBackend, 1> =
            Tensor::random([d], Distribution::Normal(0.0, 5.0), &device);
        let single_point_data: Vec<f32> = single_point.clone().into_data().to_vec().expect("vec");

        // Repeat for all nodes
        let mut all_positions = Vec::new();
        for _ in 0..n {
            all_positions.extend(single_point_data.iter().cloned());
        }

        let positions_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data(all_positions.as_slice(), &device);
        let positions = positions_1d.reshape([n as i32, d as i32]);

        let energy = ebm.total_energy(&positions);
        assert!(
            energy.abs() < 1e-3, // Looser tolerance for GPU floating-point precision
            "Energy should be ~0 when all connected nodes are coincident, got {}",
            energy
        );
    }

    #[test]
    fn test_spring_energy_scales_with_constant() {
        let device = init_gpu_device();
        let n = 15;
        let d = 3;

        let graph = create_random_graph(n, 0.4);

        // Create two EBMs with different spring constants
        let ebm_k1 = SpringEBM::from_graph(&graph, 1.0, &device);
        let ebm_k2 = SpringEBM::from_graph(&graph, 2.0, &device);
        let ebm_k5 = SpringEBM::from_graph(&graph, 5.0, &device);

        // Random positions
        let positions: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 5.0), &device);

        let e1 = ebm_k1.total_energy(&positions);
        let e2 = ebm_k2.total_energy(&positions);
        let e5 = ebm_k5.total_energy(&positions);

        // Energy should scale linearly with spring constant (use relative tolerance)
        let rel_tol = 1e-5;
        let rel_err_2 = (e2 - 2.0 * e1).abs() / e1.abs().max(1e-10);
        let rel_err_5 = (e5 - 5.0 * e1).abs() / e1.abs().max(1e-10);

        assert!(
            rel_err_2 < rel_tol,
            "E(k=2) should equal 2*E(k=1): {} vs {}, rel_err={}",
            e2,
            2.0 * e1,
            rel_err_2
        );
        assert!(
            rel_err_5 < rel_tol,
            "E(k=5) should equal 5*E(k=1): {} vs {}, rel_err={}",
            e5,
            5.0 * e1,
            rel_err_5
        );
    }

    #[test]
    fn test_spring_force_is_negative_gradient() {
        let device = init_gpu_device();
        let n = 8;
        let d = 3;

        let graph = create_random_graph(n, 0.5);
        let ebm = SpringEBM::from_graph(&graph, 1.0, &device);

        // Use deterministic positions for reproducible tests across different hardware
        // Values chosen to cover a range of positive/negative coordinates
        let pos_data: Vec<f32> = vec![
            1.2, -0.5, 2.1, // node 0
            -1.8, 0.3, -0.7, // node 1
            0.9, 1.5, -2.3, // node 2
            -0.4, -1.2, 0.8, // node 3
            2.5, 0.1, -1.1, // node 4
            -1.3, 2.0, 0.5, // node 5
            0.7, -1.8, 1.9, // node 6
            -2.1, 0.6, -0.3, // node 7
        ];
        let pos_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(pos_data.as_slice(), &device);
        let positions: Tensor<WgpuBackend, 2> = pos_1d.reshape([n as i32, d as i32]);

        // Compute force
        let force = ebm.force(&positions);
        let force_data: Vec<f32> = force.into_data().to_vec().expect("force vec");

        // Compute numerical gradient: -dE/dx using finite differences
        // Note: total_energy sums per-node energies, which double-counts symmetric edges.
        // The force formula accounts for this, so we expect force = -0.5 * d(total_energy)/dx
        let eps = 1e-3;
        for i in 0..n {
            for j in 0..d {
                let idx = i * d + j;

                // E(x + eps)
                let mut pos_plus = pos_data.clone();
                pos_plus[idx] += eps;
                let pos_plus_1d: Tensor<WgpuBackend, 1> =
                    Tensor::from_data(pos_plus.as_slice(), &device);
                let pos_plus_2d = pos_plus_1d.reshape([n as i32, d as i32]);
                let e_plus = ebm.total_energy(&pos_plus_2d);

                // E(x - eps)
                let mut pos_minus = pos_data.clone();
                pos_minus[idx] -= eps;
                let pos_minus_1d: Tensor<WgpuBackend, 1> =
                    Tensor::from_data(pos_minus.as_slice(), &device);
                let pos_minus_2d = pos_minus_1d.reshape([n as i32, d as i32]);
                let e_minus = ebm.total_energy(&pos_minus_2d);

                // Numerical gradient: dE_total/dx = (E+ - E-) / (2*eps)
                // Due to symmetric adjacency, total_energy double-counts each edge contribution.
                // The force formula is F = -d(E_node)/dx, so force = -0.5 * d(E_total)/dx
                let numerical_grad = (e_plus - e_minus) / (2.0 * eps);
                let expected_force = -0.5 * numerical_grad;

                let actual_force = force_data[idx];
                let diff = (actual_force - expected_force).abs();
                let rel_diff = diff / (expected_force.abs().max(0.01));

                assert!(
                    rel_diff < 0.10, // 10% relative tolerance (accounts for numerical precision)
                    "Force mismatch at [{},{}]: actual={}, expected={}, rel_diff={}",
                    i,
                    j,
                    actual_force,
                    expected_force,
                    rel_diff
                );
            }
        }
    }

    #[test]
    fn test_node_bias_energy_properties() {
        let device = init_gpu_device();
        let n = 10;
        let d = 3;

        // Create weights with varying prominence
        let weights_data: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) / n as f32).collect();
        let weights: Tensor<WgpuBackend, 1> = Tensor::from_data(weights_data.as_slice(), &device);
        let bias_ebm = NodeBiasEBM::new(weights.clone(), 1.0);

        // Test 1: Energy scales with weight
        // All nodes at same distance from origin
        let uniform_dist = 10.0f32;
        let mut pos_data = Vec::with_capacity(n * d);
        for _ in 0..n {
            pos_data.extend_from_slice(&[uniform_dist, 0.0, 0.0]);
        }
        let positions_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(pos_data.as_slice(), &device);
        let positions_uniform = positions_1d.reshape([n as i32, d as i32]);

        let energy = bias_ebm.energy_radial(&positions_uniform, 100.0);
        let energy_data: Vec<f32> = energy.into_data().to_vec().expect("energy vec");

        // Higher weight nodes should have higher magnitude energy at same distance
        // E = -weight * log(dist/max_dist), at dist=10, max_dist=100: log(0.1) ≈ -2.3
        // So E ≈ weight * 2.3 (positive, proportional to weight)
        for i in 1..n {
            assert!(
                energy_data[i] > energy_data[i - 1],
                "Higher weight should give higher energy at same distance: E[{}]={} vs E[{}]={}",
                i,
                energy_data[i],
                i - 1,
                energy_data[i - 1]
            );
        }

        // Test 2: Force points inward (toward origin) for positive weights
        let positions: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Uniform(5.0, 20.0), &device);
        let force = bias_ebm.force_radial(&positions);
        let force_data: Vec<f32> = force.into_data().to_vec().expect("force vec");

        // All forces should be positive (pointing toward origin reduces distance)
        for (i, &f) in force_data.iter().enumerate() {
            assert!(
                f > 0.0,
                "Force should be positive (toward origin) for node {}: got {}",
                i,
                f
            );
        }

        // Test 3: Force magnitude scales with weight at same distance
        // Use uniform positions so only weight varies
        let force_uniform = bias_ebm.force_radial(&positions_uniform);
        let force_uniform_data: Vec<f32> = force_uniform
            .into_data()
            .to_vec()
            .expect("force uniform vec");

        for i in 1..n {
            assert!(
                force_uniform_data[i] > force_uniform_data[i - 1],
                "Higher weight should give stronger force at same distance: F[{}]={} vs F[{}]={}",
                i,
                force_uniform_data[i],
                i - 1,
                force_uniform_data[i - 1]
            );
        }
    }

    #[test]
    fn test_large_random_graph() {
        let device = init_gpu_device();
        let n = 100;
        let d = 3;

        // Create a large graph with moderate connectivity
        let graph = create_random_graph(n, 0.1); // ~500 edges
        let ebm = SpringEBM::from_graph(&graph, 1.0, &device);

        println!(
            "Large graph test: {} nodes, {} edges",
            graph.n_nodes,
            graph.n_edges()
        );

        // Random positions
        let positions: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 10.0), &device);

        // Verify energy computation completes and is valid
        let energy = ebm.total_energy(&positions);
        assert!(energy.is_finite(), "Energy should be finite");
        assert!(energy >= 0.0, "Energy should be non-negative");

        // Verify force computation completes and is valid
        let force = ebm.force(&positions);
        let force_data: Vec<f32> = force.into_data().to_vec().expect("force vec");
        assert_eq!(force_data.len(), n * d);
        assert!(
            force_data.iter().all(|&f| f.is_finite()),
            "All forces should be finite"
        );

        // Verify gradient computation
        let gradient = ebm.gradient(&positions);
        let grad_data: Vec<f32> = gradient.into_data().to_vec().expect("grad vec");
        assert_eq!(grad_data.len(), n * d);

        println!("Large graph energy: {}", energy);
    }
}
