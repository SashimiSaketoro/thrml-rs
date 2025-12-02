//! CubeCL kernels for cosine similarity computation.

use cubecl::prelude::*;

/// Single query cosine similarity kernel.
///
/// Computes cosine similarity between one query and multiple vectors.
///
/// # Arguments
/// * `query` - Query vector `[dim]`
/// * `vectors` - Matrix of vectors `[n_vectors, dim]` (flattened)
/// * `output` - Output similarities `[n_vectors]`
/// * `dim` - Dimension of vectors
#[cube(launch)]
pub fn cosine_similarity_single_kernel<F: Float>(
    query: &Tensor<F>,
    vectors: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] dim: u32,
) {
    // Epsilon for numerical stability (hardcoded to avoid Hash requirement)
    let epsilon = F::new(1e-8);
    let vec_idx = ABSOLUTE_POS;
    let n_vectors = output.len();

    if vec_idx >= n_vectors {
        terminate!();
    }

    // Compute query norm (could be precomputed but keeping simple)
    let mut query_norm_sq = F::new(0.0);
    for i in 0..dim {
        let q = query[i];
        query_norm_sq += q * q;
    }
    let query_norm = F::sqrt(query_norm_sq) + epsilon;

    // Compute vector norm and dot product
    let vec_start = vec_idx * dim;
    let mut vec_norm_sq = F::new(0.0);
    let mut dot = F::new(0.0);

    for i in 0..dim {
        let q = query[i];
        let v = vectors[vec_start + i];
        dot += q * v;
        vec_norm_sq += v * v;
    }

    let vec_norm = F::sqrt(vec_norm_sq) + epsilon;
    output[vec_idx] = dot / (query_norm * vec_norm);
}

/// Batched cosine similarity kernel.
///
/// Computes cosine similarity between multiple queries and multiple vectors.
/// Output is `[n_queries, n_vectors]`.
///
/// # Arguments
/// * `queries` - Query matrix `[n_queries, dim]` (flattened)
/// * `vectors` - Vector matrix `[n_vectors, dim]` (flattened)
/// * `output` - Output similarities `[n_queries, n_vectors]` (flattened)
/// * `n_queries` - Number of queries
/// * `n_vectors` - Number of vectors
/// * `dim` - Dimension of vectors
#[cube(launch)]
pub fn cosine_similarity_batched_kernel<F: Float>(
    queries: &Tensor<F>,
    vectors: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] n_queries: u32,
    #[comptime] n_vectors: u32,
    #[comptime] dim: u32,
) {
    // Epsilon for numerical stability (hardcoded to avoid Hash requirement)
    let epsilon = F::new(1e-8);
    let idx = ABSOLUTE_POS;
    let total = n_queries * n_vectors;

    if idx >= total {
        terminate!();
    }

    let query_idx = idx / n_vectors;
    let vec_idx = idx % n_vectors;

    let query_start = query_idx * dim;
    let vec_start = vec_idx * dim;

    // Compute norms and dot product
    let mut query_norm_sq = F::new(0.0);
    let mut vec_norm_sq = F::new(0.0);
    let mut dot = F::new(0.0);

    for i in 0..dim {
        let q = queries[query_start + i];
        let v = vectors[vec_start + i];
        dot += q * v;
        query_norm_sq += q * q;
        vec_norm_sq += v * v;
    }

    let query_norm = F::sqrt(query_norm_sq) + epsilon;
    let vec_norm = F::sqrt(vec_norm_sq) + epsilon;

    output[idx] = dot / (query_norm * vec_norm);
}

/// Pre-normalized cosine similarity kernel.
///
/// For when vectors are already L2 normalized, just computes dot products.
///
/// # Arguments
/// * `query` - Query vector `[dim]` (will be normalized)
/// * `normalized_vectors` - Pre-normalized vectors `[n_vectors, dim]` (flattened)
/// * `output` - Output similarities `[n_vectors]`
/// * `dim` - Dimension of vectors
#[cube(launch)]
pub fn cosine_similarity_prenorm_kernel<F: Float>(
    query: &Tensor<F>,
    normalized_vectors: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] dim: u32,
) {
    // Epsilon for numerical stability (hardcoded to avoid Hash requirement)
    let epsilon = F::new(1e-8);
    let vec_idx = ABSOLUTE_POS;
    let n_vectors = output.len();

    if vec_idx >= n_vectors {
        terminate!();
    }

    // Compute query norm
    let mut query_norm_sq = F::new(0.0);
    for i in 0..dim {
        let q = query[i];
        query_norm_sq += q * q;
    }
    let query_norm = F::sqrt(query_norm_sq) + epsilon;

    // Just dot product (vectors already normalized)
    let vec_start = vec_idx * dim;
    let mut dot = F::new(0.0);

    for i in 0..dim {
        dot += (query[i] / query_norm) * normalized_vectors[vec_start + i];
    }

    output[vec_idx] = dot;
}
