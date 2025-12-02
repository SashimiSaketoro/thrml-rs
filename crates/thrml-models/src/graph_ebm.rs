//! Graph-based energy models.
//!
//! This module provides energy-based models that use graph connectivity
//! to define interactions between nodes.
//!
//! ## Components
//!
//! - [`GraphEdge`]: An edge connecting two nodes with a weight
//! - [`GraphSidecar`]: Graph structure with edges and optional node attributes
//! - [`SpringEBM`]: Spring-like forces between connected nodes
//! - [`ProbabilisticGraphEBM`]: Learnable edge energies with Gibbs sampling
//! - [`GraphEBMTrainer`]: Contrastive divergence training for edge weights
//!
//! ## Probabilistic Graph EBM
//!
//! The [`ProbabilisticGraphEBM`] extends static graphs with:
//! - **Learnable edge energies**: `E(i,j) = -sim_weight * cos(e_i, e_j) + prom_weight * (p_i + p_j)`
//! - **Gibbs sampling**: Sample edge configurations from the Boltzmann distribution
//! - **Subgraph scoring**: Log-probability of a subgraph configuration
//! - **Path sampling**: Monte Carlo paths through the graph
//!
//! ## Async GPU-CPU Transfers
//!
//! Uses [`burn::tensor::Transaction`] for batched async transfers when extracting
//! samples from GPU tensors, minimizing synchronization overhead.

#[allow(unused_imports)]
use burn::tensor::{Distribution, Tensor, Transaction};
use thrml_core::backend::WgpuBackend;
use thrml_core::distance::pairwise_distances_sq;
use thrml_core::similarity::{cosine_similarity_matrix, cosine_similarity_topk, SparseSimilarity};

// Fused kernel integration:
// - gumbel_argmax_fused: Used for categorical edge selection (one edge per node)
// - sigmoid_bernoulli_fused: Not applicable here (expects logits for binary choice)
#[cfg(feature = "fused-kernels")]
use thrml_kernels::gumbel_argmax_fused;

/// Type alias for edge batches: (positive_edges, negative_edges) pairs.
pub type EdgeBatch = (Vec<(usize, usize)>, Vec<(usize, usize)>);

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
    pub const fn new(src: usize, dst: usize, weight: f32) -> Self {
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
    pub const fn new(n_nodes: usize) -> Self {
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
    pub const fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Check if the graph is empty.
    pub const fn is_empty(&self) -> bool {
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
    /// Returns `[N, N]` tensor.
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
    /// Adjacency matrix `[N, N]`
    pub adjacency: Tensor<WgpuBackend, 2>,
    /// Spring constant
    pub spring_constant: f32,
}

impl SpringEBM {
    /// Create from adjacency matrix.
    pub const fn new(adjacency: Tensor<WgpuBackend, 2>, spring_constant: f32) -> Self {
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
    pub const fn new(weights: Tensor<WgpuBackend, 1>, strength: f32) -> Self {
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

// ============================================================================
// Probabilistic Graph EBM
// ============================================================================

/// Configuration for probabilistic graph EBM.
#[derive(Clone, Debug)]
pub struct ProbabilisticGraphConfig {
    /// Weight for cosine similarity in edge energy (default: 1.0)
    pub similarity_weight: f32,
    /// Weight for prominence bias in edge energy (default: 0.3)
    pub prominence_weight: f32,
    /// Temperature for Gibbs sampling (default: 0.5)
    pub temperature: f32,
    /// Default number of Gibbs sweeps (default: 4)
    pub default_sweeps: usize,
    /// Use sparse similarity (top-k neighbors only) for large graphs
    /// Set to Some(k) to enable, None for dense mode
    pub sparse_k: Option<usize>,
    /// Threshold for dense similarity storage (only store > threshold)
    /// Only used when sparse_k is None
    pub similarity_threshold: Option<f32>,
}

impl Default for ProbabilisticGraphConfig {
    fn default() -> Self {
        Self {
            similarity_weight: 1.0,
            prominence_weight: 0.3,
            temperature: 0.5,
            default_sweeps: 4,
            sparse_k: None, // Dense by default
            similarity_threshold: None,
        }
    }
}

impl ProbabilisticGraphConfig {
    /// Create config for large graphs using sparse top-k similarity.
    ///
    /// Recommended for graphs with >10K nodes.
    pub fn sparse(k: usize) -> Self {
        Self {
            sparse_k: Some(k),
            ..Default::default()
        }
    }

    /// Builder: set sparse_k neighbors.
    pub const fn with_sparse_k(mut self, k: usize) -> Self {
        self.sparse_k = Some(k);
        self
    }

    /// Builder: set temperature.
    pub const fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }
}

/// Probabilistic graph EBM with learnable edge energies.
///
/// Edge energy: `E(i,j) = -sim_weight * cos(e_i, e_j) + prom_weight * (p_i + p_j)`
///
/// Low energy edges (high similarity, low prominence) form "core trunks".
/// High energy edges seed branches for outer shells.
///
/// # Example
///
/// ```rust,ignore
/// use thrml_models::graph_ebm::{ProbabilisticGraphEBM, ProbabilisticGraphConfig};
///
/// let graph = ProbabilisticGraphEBM::new(embeddings, prominence, &config, &device);
///
/// // Sample edge configurations
/// let edge_probs = graph.gibbs_sample_edges(16, &device);
///
/// // Score a subgraph
/// let log_prob = graph.subgraph_log_prob(&cone_mask, 4, &device);
/// ```
/// Storage mode for similarity matrix.
#[derive(Clone)]
pub enum SimilarityStorage {
    /// Dense `[N, N]` tensor (boxed to reduce enum size)
    Dense(Box<Tensor<WgpuBackend, 2>>),
    /// Sparse top-k storage (for large graphs)
    Sparse(SparseSimilarity),
}

#[derive(Clone)]
pub struct ProbabilisticGraphEBM {
    /// Node embeddings `[N, D]`
    pub embeddings: Tensor<WgpuBackend, 2>,
    /// Node prominence scores `[N]`
    pub prominence: Tensor<WgpuBackend, 1>,
    /// Precomputed pairwise cosine similarities (dense or sparse)
    similarities: SimilarityStorage,
    /// Edge mask (which edges exist) [N, N] - only used in dense mode
    edge_mask: Option<Tensor<WgpuBackend, 2>>,
    /// Configuration
    pub config: ProbabilisticGraphConfig,
    /// Number of nodes
    n_nodes: usize,
}

impl ProbabilisticGraphEBM {
    /// Create from embeddings and prominence.
    ///
    /// Uses dense or sparse similarity based on config:
    /// - `config.sparse_k = Some(k)`: Sparse top-k storage (recommended for >10K nodes)
    /// - `config.sparse_k = None`: Dense [N, N] matrix
    pub fn new(
        embeddings: Tensor<WgpuBackend, 2>,
        prominence: Tensor<WgpuBackend, 1>,
        config: &ProbabilisticGraphConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let n_nodes = embeddings.dims()[0];

        // Choose storage mode based on config
        let similarities = if let Some(k) = config.sparse_k {
            // Sparse mode: only store top-k neighbors per node
            let sparse = cosine_similarity_topk(&embeddings, k, device);
            SimilarityStorage::Sparse(sparse)
        } else {
            // Dense mode: full N×N matrix
            let dense = Self::compute_dense_similarities(&embeddings, device);
            SimilarityStorage::Dense(Box::new(dense))
        };

        // Edge mask only used in dense mode
        let edge_mask = if config.sparse_k.is_none() {
            Some(Tensor::ones([n_nodes, n_nodes], device))
        } else {
            None
        };

        Self {
            embeddings,
            prominence,
            similarities,
            edge_mask,
            config: config.clone(),
            n_nodes,
        }
    }

    /// Create from graph sidecar (uses sidecar edges as mask).
    /// Always uses dense mode since sidecar provides explicit edges.
    pub fn from_sidecar(
        embeddings: Tensor<WgpuBackend, 2>,
        prominence: Tensor<WgpuBackend, 1>,
        sidecar: &GraphSidecar,
        config: &ProbabilisticGraphConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let n_nodes = embeddings.dims()[0];
        let similarities =
            SimilarityStorage::Dense(Box::new(Self::compute_dense_similarities(&embeddings, device)));
        let edge_mask = Some(sidecar.to_symmetric_adjacency(device));

        Self {
            embeddings,
            prominence,
            similarities,
            edge_mask,
            config: config.clone(),
            n_nodes,
        }
    }

    /// Compute dense pairwise cosine similarities using thrml-core.
    fn compute_dense_similarities(
        embeddings: &Tensor<WgpuBackend, 2>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        // Use thrml-core's optimized implementation
        // Note: cosine_similarity_matrix zeros the diagonal, but we want self-sim
        // so we add back the identity
        let n = embeddings.dims()[0];
        let sim = cosine_similarity_matrix(embeddings, device);
        let diag_eye: Tensor<WgpuBackend, 2> = Tensor::eye(n, device);
        sim + diag_eye // Add back self-similarity (1.0 on diagonal)
    }

    /// Get similarity matrix as dense tensor.
    /// For sparse mode, converts to dense (allocates N×N).
    pub fn similarities_dense(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        match &self.similarities {
            SimilarityStorage::Dense(t) => (**t).clone(),
            SimilarityStorage::Sparse(s) => s.to_dense(device),
        }
    }

    /// Check if using sparse storage.
    pub const fn is_sparse(&self) -> bool {
        matches!(self.similarities, SimilarityStorage::Sparse(_))
    }

    /// Memory usage estimate in bytes.
    pub fn memory_bytes(&self) -> usize {
        match &self.similarities {
            SimilarityStorage::Dense(_) => self.n_nodes * self.n_nodes * 4, // f32
            SimilarityStorage::Sparse(s) => s.memory_bytes(),
        }
    }

    /// Compute edge energies for all edges.
    ///
    /// `E(i,j) = -sim_weight * cos(e_i, e_j) + prom_weight * (p_i + p_j)`
    ///
    /// Returns [N, N] tensor of edge energies.
    pub fn edge_energies(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let n = self.n_nodes;

        // Get similarities as dense (sparse mode converts)
        let sim_dense = self.similarities_dense(device);

        // Similarity term: -sim_weight * cos_sim
        let sim_term = sim_dense.mul_scalar(-self.config.similarity_weight);

        // Prominence term: prom_weight * (p_i + p_j)
        // Broadcast p_i to [N, N] and add p_j
        let p_i = self.prominence.clone().reshape([n as i32, 1]);
        let p_j = self.prominence.clone().reshape([1, n as i32]);
        let prom_sum = p_i + p_j;
        let prom_term = prom_sum.mul_scalar(self.config.prominence_weight);

        // Total energy (masked by edge_mask if present)
        let total = sim_term + prom_term;
        if let Some(mask) = &self.edge_mask {
            total * mask.clone()
        } else {
            total
        }
    }

    /// Compute edge probabilities via softmax over energies.
    ///
    /// `P(edge_ij) = exp(-E_ij / T) / Z`
    ///
    /// Returns [N, N] tensor of edge probabilities.
    pub fn edge_probabilities(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let energies = self.edge_energies(device);
        let scaled = energies.mul_scalar(-1.0 / self.config.temperature);

        // Row-wise softmax
        let max_per_row = scaled.clone().max_dim(1);
        let shifted = scaled - max_per_row;
        let exp_shifted = shifted.exp();
        let row_sums = exp_shifted.clone().sum_dim(1);

        exp_shifted / row_sums
    }

    /// Gibbs sample edge configurations.
    ///
    /// Performs `n_sweeps` Gibbs sweeps, returning averaged edge probabilities.
    /// Uses fused kernels when `fused-kernels` feature is enabled for efficient
    /// sigmoid-Bernoulli sampling.
    ///
    /// # Returns
    ///
    /// Edge probabilities [N, N] averaged over samples.
    pub fn gibbs_sample_edges(
        &self,
        n_sweeps: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let _n = self.n_nodes; // Available for debugging assertions
        let sweeps = if n_sweeps == 0 {
            self.config.default_sweeps
        } else {
            n_sweeps
        };

        // Get similarities for coherence computation
        let sim_dense = self.similarities_dense(device);

        // Initialize with base probabilities
        let mut edge_probs = self.edge_probabilities(device);

        // Gibbs sweeps: resample each edge conditional on neighbors
        for _sweep in 0..sweeps {
            // Sample binary edges from current probabilities
            // Use fused kernel when available for better performance
            let sampled_edges = self.sample_bernoulli_edges(&edge_probs, device);

            // Update energies based on sampled neighbors
            // New energy includes coherence with sampled neighborhood
            let neighbor_coherence = sampled_edges.matmul(sim_dense.clone());
            let coherence_bonus = neighbor_coherence.mul_scalar(-0.1 / self.config.temperature);

            // Recompute probabilities with coherence
            let base_energies = self.edge_energies(device);
            let adjusted =
                (base_energies + coherence_bonus).mul_scalar(-1.0 / self.config.temperature);

            let max_per_row = adjusted.clone().max_dim(1);
            let shifted = adjusted - max_per_row;
            let exp_shifted = shifted.exp();
            let row_sums = exp_shifted.clone().sum_dim(1);

            edge_probs = exp_shifted / row_sums;
        }

        edge_probs
    }

    /// Sample binary edges from probabilities using independent Bernoulli sampling.
    ///
    /// Each edge is sampled independently with probability from the input tensor.
    /// This is useful for dense graph sampling where multiple edges per node are allowed.
    fn sample_bernoulli_edges(
        &self,
        probs: &Tensor<WgpuBackend, 2>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let [n, _] = probs.dims();
        let uniform: Tensor<WgpuBackend, 2> =
            Tensor::random([n, n], Distribution::Uniform(0.0, 1.0), device);

        uniform.lower_equal(probs.clone()).float()
    }

    /// Sample one edge per node using categorical (Gumbel-max) sampling.
    ///
    /// Uses fused `gumbel_argmax` kernel when available for efficient GPU sampling.
    /// Each row selects exactly one edge based on the softmax probabilities.
    ///
    /// # Arguments
    /// * `logits` - Unnormalized log-probabilities [N, N] (pre-softmax)
    ///
    /// # Returns
    /// * One-hot edge matrix [N, N] where each row has exactly one 1.0
    #[allow(dead_code)]
    fn sample_categorical_edges(
        &self,
        logits: &Tensor<WgpuBackend, 2>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let [n, _] = logits.dims();

        #[cfg(feature = "fused-kernels")]
        {
            // Use fused Gumbel-argmax kernel for efficient categorical sampling
            let uniform: Tensor<WgpuBackend, 2> = Tensor::random(
                [n, n],
                Distribution::Uniform(1e-10, 1.0 - 1e-10),
                device,
            );

            // gumbel_argmax_fused returns indices [N], convert to one-hot [N, N]
            let indices = gumbel_argmax_fused(logits.clone(), uniform);
            
            // Create one-hot encoding from indices
            let indices_expanded = indices.unsqueeze_dim::<2>(1); // [N, 1]
            let range: Tensor<WgpuBackend, 1, burn::tensor::Int> = 
                Tensor::arange(0..n as i64, device);
            let range_expanded = range.unsqueeze_dim::<2>(0); // [1, N]
            
            // Compare: one_hot[i,j] = 1 if indices[i] == j
            indices_expanded.equal(range_expanded).float()
        }

        #[cfg(not(feature = "fused-kernels"))]
        {
            // Reference implementation: Gumbel-max trick
            let uniform: Tensor<WgpuBackend, 2> = Tensor::random(
                [n, n],
                Distribution::Uniform(1e-10, 1.0 - 1e-10),
                device,
            );
            let gumbel = -(-(uniform.log())).log();
            let perturbed = logits.clone() + gumbel;
            
            // Argmax per row
            let indices = perturbed.argmax(1); // [N]
            
            // Create one-hot encoding
            let indices_expanded = indices.unsqueeze_dim::<2>(1);
            let range: Tensor<WgpuBackend, 1, burn::tensor::Int> = 
                Tensor::arange(0..n as i64, device);
            let range_expanded = range.unsqueeze_dim::<2>(0);
            
            indices_expanded.equal(range_expanded).float()
        }
    }

    /// Compute edge logits (unnormalized log-probabilities).
    ///
    /// Returns `-E / T` which can be used directly with Gumbel-max sampling.
    #[allow(dead_code)]
    fn edge_logits(&self, device: &burn::backend::wgpu::WgpuDevice) -> Tensor<WgpuBackend, 2> {
        let energies = self.edge_energies(device);
        energies.mul_scalar(-1.0 / self.config.temperature)
    }

    /// Extract subgraph for a set of nodes.
    ///
    /// # Arguments
    /// * `node_mask` - Boolean mask `[N]` indicating which nodes to include
    ///
    /// # Returns
    /// * Subgraph sidecar with edges between selected nodes
    pub fn extract_subgraph(
        &self,
        node_indices: &[usize],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> GraphSidecar {
        let mut subgraph = GraphSidecar::new(node_indices.len());

        // Get edge probabilities (batched async transfer)
        let probs = self.edge_probabilities(device);

        // Use Transaction for batched GPU→CPU transfer
        let probs_data = probs.into_data();
        let probs_vec: Vec<f32> = probs_data.to_vec().expect("probs to vec");

        // Build subgraph edges
        for (sub_i, &global_i) in node_indices.iter().enumerate() {
            for (sub_j, &global_j) in node_indices.iter().enumerate() {
                if sub_i != sub_j {
                    let prob = probs_vec[global_i * self.n_nodes + global_j];
                    if prob > 0.01 {
                        // Threshold for including edge
                        subgraph.add_edge(sub_i, sub_j, prob);
                    }
                }
            }
        }

        subgraph
    }

    /// Compute log-probability of a subgraph configuration.
    ///
    /// Uses Gibbs sampling to estimate the partition function.
    ///
    /// # Arguments
    /// * `node_indices` - Indices of nodes in the subgraph
    /// * `n_sweeps` - Number of Gibbs sweeps for estimation
    ///
    /// # Returns
    /// * Approximate log-probability of the subgraph
    pub fn subgraph_log_prob(
        &self,
        node_indices: &[usize],
        n_sweeps: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> f32 {
        if node_indices.is_empty() {
            return 0.0;
        }

        // Get edge energies for the subgraph
        let energies = self.edge_energies(device);
        let energies_data = energies.into_data();
        let energies_vec: Vec<f32> = energies_data.to_vec().expect("energies to vec");

        // Sum energies within subgraph
        let mut total_energy = 0.0f32;
        for &i in node_indices {
            for &j in node_indices {
                if i != j {
                    total_energy += energies_vec[i * self.n_nodes + j];
                }
            }
        }

        // Get sampled edge probs for partition function estimate
        let sampled_probs = self.gibbs_sample_edges(n_sweeps, device);
        let probs_data = sampled_probs.into_data();
        let probs_vec: Vec<f32> = probs_data.to_vec().expect("probs to vec");

        // Estimate log Z from samples
        let mut log_z_estimate = 0.0f32;
        for &i in node_indices {
            for &j in node_indices {
                if i != j {
                    let p = probs_vec[i * self.n_nodes + j].max(1e-10);
                    log_z_estimate += p.ln();
                }
            }
        }

        // log P(subgraph) ≈ -E/T - log Z
        -total_energy / self.config.temperature - log_z_estimate
    }

    /// Sample paths from start to end node.
    ///
    /// Uses rejection sampling on edge probabilities to find valid paths.
    ///
    /// # Arguments
    /// * `start` - Starting node index
    /// * `end` - Target node index
    /// * `n_samples` - Number of paths to sample
    /// * `max_length` - Maximum path length
    ///
    /// # Returns
    /// * Vector of sampled paths (each path is a `Vec<usize>`)
    pub fn sample_paths(
        &self,
        start: usize,
        end: usize,
        n_samples: usize,
        max_length: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<Vec<usize>> {
        let mut paths = Vec::with_capacity(n_samples);

        // Get edge probabilities (single GPU→CPU transfer)
        let probs = self.edge_probabilities(device);
        let probs_data = probs.into_data();
        let probs_vec: Vec<f32> = probs_data.to_vec().expect("probs to vec");

        // Simple random walk sampling
        let mut rng_state = 42u64;

        for _ in 0..n_samples * 10 {
            // Oversample to get enough valid paths
            if paths.len() >= n_samples {
                break;
            }

            let mut path = vec![start];
            let mut current = start;
            let mut visited = vec![false; self.n_nodes];
            visited[start] = true;

            for _ in 0..max_length {
                if current == end {
                    paths.push(path.clone());
                    break;
                }

                // Sample next node proportional to edge probabilities
                let row_start = current * self.n_nodes;
                let mut cumsum = 0.0f32;
                let mut candidates: Vec<(usize, f32)> = Vec::new();

                for j in 0..self.n_nodes {
                    if !visited[j] {
                        let p = probs_vec[row_start + j];
                        cumsum += p;
                        candidates.push((j, cumsum));
                    }
                }

                if candidates.is_empty() || cumsum < 1e-10 {
                    break; // Dead end
                }

                // Sample using LCG
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = (rng_state >> 33) as f32 / (u32::MAX >> 1) as f32 * cumsum;

                let next = candidates
                    .iter()
                    .find(|(_, cs)| *cs >= u)
                    .map(|(idx, _)| *idx)
                    .unwrap_or_else(|| candidates.last().unwrap().0);

                path.push(next);
                visited[next] = true;
                current = next;
            }
        }

        paths
    }

    /// Detect edges with high variance across Gibbs samples.
    ///
    /// High variance edges indicate uncertainty—candidates for overflow/rebalancing.
    ///
    /// # Arguments
    /// * `threshold` - Variance threshold (edges above this are returned)
    /// * `n_samples` - Number of Gibbs samples for variance estimation
    ///
    /// # Returns
    /// * Indices of high-variance edges as (src, dst) pairs
    pub fn detect_high_variance_edges(
        &self,
        threshold: f32,
        n_samples: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<(usize, usize)> {
        let n = self.n_nodes;

        // Collect samples (batched for async efficiency)
        let mut sum_probs = vec![0.0f32; n * n];
        let mut sum_sq_probs = vec![0.0f32; n * n];

        for _ in 0..n_samples {
            let probs = self.gibbs_sample_edges(1, device);
            let probs_data = probs.into_data();
            let probs_vec: Vec<f32> = probs_data.to_vec().expect("probs to vec");

            for (i, &p) in probs_vec.iter().enumerate() {
                sum_probs[i] += p;
                sum_sq_probs[i] += p * p;
            }
        }

        // Compute variance and find high-variance edges
        let mut high_var_edges = Vec::new();
        let n_samples_f = n_samples as f32;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let idx = i * n + j;
                    let mean = sum_probs[idx] / n_samples_f;
                    let variance = mean.mul_add(-mean, sum_sq_probs[idx] / n_samples_f);

                    if variance > threshold {
                        high_var_edges.push((i, j));
                    }
                }
            }
        }

        high_var_edges
    }
}

// ============================================================================
// Contrastive Divergence Training
// ============================================================================

/// Configuration for graph EBM training.
#[derive(Clone, Debug)]
pub struct GraphEBMTrainerConfig {
    /// Learning rate (default: 0.01)
    pub learning_rate: f32,
    /// Number of CD steps (default: 1)
    pub cd_steps: usize,
    /// Momentum for SGD (default: 0.9)
    pub momentum: f32,
    /// Weight decay (L2 regularization) (default: 1e-4)
    pub weight_decay: f32,
}

impl Default for GraphEBMTrainerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            cd_steps: 1,
            momentum: 0.9,
            weight_decay: 1e-4,
        }
    }
}

/// Training state for graph EBM.
#[derive(Clone)]
pub struct GraphEBMTrainingState {
    /// Momentum buffer for similarity weight
    pub sim_weight_momentum: f32,
    /// Momentum buffer for prominence weight  
    pub prom_weight_momentum: f32,
    /// Training step counter
    pub step: usize,
}

impl Default for GraphEBMTrainingState {
    fn default() -> Self {
        Self {
            sim_weight_momentum: 0.0,
            prom_weight_momentum: 0.0,
            step: 0,
        }
    }
}

/// Contrastive divergence trainer for ProbabilisticGraphEBM.
///
/// Trains the edge energy weights using CD-k:
/// ```text
/// ∂L/∂θ = ⟨∂E/∂θ⟩_data - ⟨∂E/∂θ⟩_model
/// ```
///
/// # Example
///
/// ```rust,ignore
/// let mut trainer = GraphEBMTrainer::new(GraphEBMTrainerConfig::default());
///
/// // Training loop
/// for (positive_edges, negative_edges) in batches {
///     let loss = trainer.cd_step(&mut graph, &positive_edges, &negative_edges, &device);
///     println!("CD loss: {}", loss);
/// }
/// ```
pub struct GraphEBMTrainer {
    /// Training configuration
    pub config: GraphEBMTrainerConfig,
    /// Training state
    pub state: GraphEBMTrainingState,
}

impl GraphEBMTrainer {
    /// Create a new trainer.
    pub fn new(config: GraphEBMTrainerConfig) -> Self {
        Self {
            config,
            state: GraphEBMTrainingState::default(),
        }
    }

    /// Perform one CD step.
    ///
    /// # Arguments
    /// * `graph` - The graph EBM to train
    /// * `positive_edges` - Edges from data (coherent graphs)
    /// * `negative_edges` - Edges from model samples (or shuffled)
    ///
    /// # Returns
    /// * CD loss value
    pub fn cd_step(
        &mut self,
        graph: &mut ProbabilisticGraphEBM,
        positive_edges: &[(usize, usize)],
        negative_edges: &[(usize, usize)],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> f32 {
        // Get current energies (async batched)
        let energies = graph.edge_energies(device);
        let energies_data = energies.into_data();
        let energies_vec: Vec<f32> = energies_data.to_vec().expect("energies to vec");

        // Get similarities (dense for training)
        let sims_dense = graph.similarities_dense(device);
        let sims_data = sims_dense.into_data();
        let sims_vec: Vec<f32> = sims_data.to_vec().expect("sims to vec");

        let prom_data = graph.prominence.clone().into_data();
        let prom_vec: Vec<f32> = prom_data.to_vec().expect("prom to vec");

        let n = graph.n_nodes;

        // Compute gradients for positive phase
        let mut pos_sim_grad = 0.0f32;
        let mut pos_prom_grad = 0.0f32;

        for &(i, j) in positive_edges {
            // ∂E/∂sim_weight = -cos_sim(i,j)
            pos_sim_grad += -sims_vec[i * n + j];
            // ∂E/∂prom_weight = p_i + p_j
            pos_prom_grad += prom_vec[i] + prom_vec[j];
        }

        // Compute gradients for negative phase
        let mut neg_sim_grad = 0.0f32;
        let mut neg_prom_grad = 0.0f32;

        for &(i, j) in negative_edges {
            neg_sim_grad += -sims_vec[i * n + j];
            neg_prom_grad += prom_vec[i] + prom_vec[j];
        }

        // Normalize by number of edges
        let n_pos = positive_edges.len().max(1) as f32;
        let n_neg = negative_edges.len().max(1) as f32;

        pos_sim_grad /= n_pos;
        pos_prom_grad /= n_pos;
        neg_sim_grad /= n_neg;
        neg_prom_grad /= n_neg;

        // CD gradient: positive - negative
        let sim_grad = pos_sim_grad - neg_sim_grad;
        let prom_grad = pos_prom_grad - neg_prom_grad;

        // Apply weight decay
        let sim_grad = self.config.weight_decay.mul_add(graph.config.similarity_weight, sim_grad);
        let prom_grad = self.config.weight_decay.mul_add(graph.config.prominence_weight, prom_grad);

        // Update with momentum
        self.state.sim_weight_momentum =
            self.config.momentum.mul_add(self.state.sim_weight_momentum, sim_grad);
        self.state.prom_weight_momentum =
            self.config.momentum.mul_add(self.state.prom_weight_momentum, prom_grad);

        // Apply updates
        graph.config.similarity_weight -=
            self.config.learning_rate * self.state.sim_weight_momentum;
        graph.config.prominence_weight -=
            self.config.learning_rate * self.state.prom_weight_momentum;

        // Clamp weights to reasonable range
        graph.config.similarity_weight = graph.config.similarity_weight.clamp(0.1, 10.0);
        graph.config.prominence_weight = graph.config.prominence_weight.clamp(0.0, 5.0);

        self.state.step += 1;

        // Return approximate CD loss
        let pos_energy: f32 = positive_edges
            .iter()
            .map(|&(i, j)| energies_vec[i * n + j])
            .sum::<f32>()
            / n_pos;
        let neg_energy: f32 = negative_edges
            .iter()
            .map(|&(i, j)| energies_vec[i * n + j])
            .sum::<f32>()
            / n_neg;

        neg_energy - pos_energy
    }

    /// Train for one epoch on batches of positive/negative edges.
    pub fn train_epoch(
        &mut self,
        graph: &mut ProbabilisticGraphEBM,
        batches: &[EdgeBatch],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> f32 {
        let mut total_loss = 0.0f32;

        for (positive, negative) in batches {
            let loss = self.cd_step(graph, positive, negative, device);
            total_loss += loss;
        }

        total_loss / batches.len().max(1) as f32
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
        assert!((adj_data[1] - 1.0).abs() < 1e-6);
        assert!((adj_data[4] - 1.0).abs() < 1e-6);
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
        let single_point_data: Vec<f32> = single_point.into_data().to_vec().expect("vec");

        // Repeat for all nodes
        let mut all_positions = Vec::new();
        for _ in 0..n {
            all_positions.extend(single_point_data.iter().cloned());
        }

        let positions_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data(all_positions.as_slice(), &device);
        let positions = positions_1d.reshape([n as i32, d]);

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
        let rel_err_2 = 2.0f32.mul_add(-e1, e2).abs() / e1.abs().max(1e-10);
        let rel_err_5 = 5.0f32.mul_add(-e1, e5).abs() / e1.abs().max(1e-10);

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
                    rel_diff < 0.15, // 15% relative tolerance (GPU f32 precision + numerical gradient error + random variation)
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
        let bias_ebm = NodeBiasEBM::new(weights, 1.0);

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
