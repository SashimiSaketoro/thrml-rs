//! Similarity computation utilities.
//!
//! This module provides GPU-accelerated similarity computations:
//!
//! - [`cosine_similarity_matrix`]: Cosine similarity between embedding vectors
//! - [`SparseSimilarity`]: Memory-efficient sparse similarity storage
//! - [`cosine_similarity_topk`]: Top-k neighbor similarity
//! - [`cosine_similarity_threshold`]: Threshold-based sparse similarity

use crate::backend::WgpuBackend;
use burn::tensor::Tensor;

/// Compute cosine similarity matrix on GPU.
///
/// Given embeddings [N, D], computes [N, N] similarity matrix where:
/// ```text
/// sim[i,j] = dot(emb[i], emb[j]) / (||emb[i]|| * ||emb[j]||)
/// ```
///
/// Self-similarity (diagonal) is set to 0 to avoid self-interaction.
///
/// # Arguments
/// * `embeddings` - Embedding matrix [N, D]
/// * `device` - GPU device
///
/// # Returns
/// Similarity matrix [N, N] with diagonal zeroed
///
/// # Example
/// ```rust,ignore
/// let embeddings: Tensor<WgpuBackend, 2> = /* [N, D] */;
/// let sim = cosine_similarity_matrix(&embeddings, &device);
/// ```
pub fn cosine_similarity_matrix(
    embeddings: &Tensor<WgpuBackend, 2>,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2> {
    let n = embeddings.dims()[0];

    // Compute L2 norms per row: ||emb[i]||
    // sum_dim(1) on [N, D] returns [N, 1] in Burn 0.19
    let norms = embeddings
        .clone()
        .powf_scalar(2.0)
        .sum_dim(1) // [N, 1]
        .sqrt()
        .clamp(1e-8, f32::MAX);

    // Normalize: emb_norm[i] = emb[i] / ||emb[i]||
    // norms is already [N, 1] - broadcasts correctly against [N, D]
    let normalized = embeddings.clone() / norms;

    // Compute similarity via matmul: sim = normalized @ normalized^T
    let similarity = normalized.clone().matmul(normalized.transpose());

    // Zero out diagonal (self-similarity)
    let eye: Tensor<WgpuBackend, 2> = Tensor::eye(n, device);
    similarity.clone() - similarity * eye
}

/// Sparse similarity representation using top-k neighbors.
///
/// For large datasets, storing the full N×N similarity matrix is infeasible.
/// This struct stores only the top-k most similar neighbors for each point.
#[derive(Clone, Debug)]
pub struct SparseSimilarity {
    /// Indices of top-k neighbors for each point [N, K]
    pub indices: Vec<Vec<usize>>,
    /// Similarity values for top-k neighbors [N, K]
    pub values: Vec<Vec<f32>>,
    /// Number of points
    pub n_points: usize,
    /// Maximum number of neighbors per point
    pub k: usize,
}

impl SparseSimilarity {
    /// Create empty sparse similarity.
    pub fn new(n_points: usize, k: usize) -> Self {
        Self {
            indices: vec![Vec::with_capacity(k); n_points],
            values: vec![Vec::with_capacity(k); n_points],
            n_points,
            k,
        }
    }

    /// Convert to dense tensor.
    ///
    /// Returns [N, N] tensor where entry [i, j] = similarity if j is in top-k
    /// neighbors of i, else 0.
    pub fn to_dense(&self, device: &burn::backend::wgpu::WgpuDevice) -> Tensor<WgpuBackend, 2> {
        let n = self.n_points;
        let mut data = vec![0.0f32; n * n];

        for (i, (indices, values)) in self.indices.iter().zip(self.values.iter()).enumerate() {
            for (&j, &v) in indices.iter().zip(values.iter()) {
                data[i * n + j] = v;
            }
        }

        // Create 1D tensor and reshape to 2D
        let tensor_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(data.as_slice(), device);
        tensor_1d.reshape([n as i32, n as i32])
    }

    /// Get neighbors for a specific point.
    pub fn neighbors(&self, idx: usize) -> (&[usize], &[f32]) {
        (&self.indices[idx], &self.values[idx])
    }

    /// Total number of stored entries.
    pub fn nnz(&self) -> usize {
        self.indices.iter().map(|v| v.len()).sum()
    }

    /// Memory usage in bytes (approximate).
    pub fn memory_bytes(&self) -> usize {
        // Each entry: 8 bytes (usize) + 4 bytes (f32) = 12 bytes
        self.nnz() * 12
    }
}

/// Compute sparse top-k cosine similarity.
///
/// For each point, finds the k most similar other points.
/// This is more memory-efficient than computing the full N×N matrix
/// for large datasets.
///
/// # Arguments
/// * `embeddings` - Embedding matrix [N, D]
/// * `k` - Number of neighbors to keep per point
/// * `device` - GPU device
///
/// # Returns
/// SparseSimilarity with top-k neighbors per point
///
/// # Performance Note
/// This currently computes the full similarity matrix on GPU then extracts
/// top-k on CPU. For very large N (>100K), consider batch processing.
pub fn cosine_similarity_topk(
    embeddings: &Tensor<WgpuBackend, 2>,
    k: usize,
    device: &burn::backend::wgpu::WgpuDevice,
) -> SparseSimilarity {
    let n = embeddings.dims()[0];
    let k = k.min(n - 1); // Can't have more neighbors than n-1

    // Compute full similarity matrix
    let sim = cosine_similarity_matrix(embeddings, device);
    let sim_data: Vec<f32> = sim.into_data().to_vec().expect("sim to vec");

    // Extract top-k for each row
    let mut sparse = SparseSimilarity::new(n, k);

    for i in 0..n {
        // Get row i similarities (excluding self)
        let mut row_sims: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, sim_data[i * n + j]))
            .collect();

        // Sort by similarity descending
        row_sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        for (j, v) in row_sims.into_iter().take(k) {
            sparse.indices[i].push(j);
            sparse.values[i].push(v);
        }
    }

    sparse
}

/// Compute sparse similarity with threshold cutoff.
///
/// Only stores similarities above the given threshold.
/// Useful when only high-similarity pairs matter.
///
/// # Arguments
/// * `embeddings` - Embedding matrix [N, D]
/// * `threshold` - Minimum similarity to include
/// * `device` - GPU device
///
/// # Returns
/// SparseSimilarity with only entries above threshold
pub fn cosine_similarity_threshold(
    embeddings: &Tensor<WgpuBackend, 2>,
    threshold: f32,
    device: &burn::backend::wgpu::WgpuDevice,
) -> SparseSimilarity {
    let n = embeddings.dims()[0];

    // Compute full similarity matrix
    let sim = cosine_similarity_matrix(embeddings, device);
    let sim_data: Vec<f32> = sim.into_data().to_vec().expect("sim to vec");

    // Count max neighbors (for allocation)
    let mut sparse = SparseSimilarity::new(n, n);
    sparse.k = 0;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let s = sim_data[i * n + j];
                if s >= threshold {
                    sparse.indices[i].push(j);
                    sparse.values[i].push(s);
                }
            }
        }
        sparse.k = sparse.k.max(sparse.indices[i].len());
    }

    sparse
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::init_gpu_device;
    use burn::tensor::Distribution;

    #[test]
    fn test_cosine_similarity_diagonal_zero() {
        let device = init_gpu_device();
        let n = 5;
        let d = 10;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);

        let sim = cosine_similarity_matrix(&embeddings, &device);
        let sim_data: Vec<f32> = sim.into_data().to_vec().expect("sim to vec");

        // Check diagonal is zero
        for i in 0..n {
            let diag_val = sim_data[i * n + i];
            assert!(
                diag_val.abs() < 1e-6,
                "Diagonal should be zero, got {}",
                diag_val
            );
        }
    }

    #[test]
    fn test_cosine_similarity_symmetric() {
        let device = init_gpu_device();
        let n = 5;
        let d = 10;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);

        let sim = cosine_similarity_matrix(&embeddings, &device);
        let sim_data: Vec<f32> = sim.into_data().to_vec().expect("sim to vec");

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                let s_ij = sim_data[i * n + j];
                let s_ji = sim_data[j * n + i];
                assert!((s_ij - s_ji).abs() < 1e-5, "Similarity should be symmetric");
            }
        }
    }

    #[test]
    fn test_sparse_similarity_topk() {
        let device = init_gpu_device();
        let n = 10;
        let d = 5;
        let k = 3;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);

        let sparse = cosine_similarity_topk(&embeddings, k, &device);

        // Each point should have exactly k neighbors
        for i in 0..n {
            assert_eq!(sparse.indices[i].len(), k, "Should have k neighbors");
            assert_eq!(sparse.values[i].len(), k, "Should have k values");
        }
    }
}
