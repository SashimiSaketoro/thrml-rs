//! Re-export of similarity utilities from thrml-core.
//!
//! For sphere optimization, use these similarity functions:
//!
//! - [`cosine_similarity_matrix`]: Full N×N similarity matrix
//! - [`cosine_similarity_topk`]: Sparse top-k neighbors
//! - [`SparseSimilarity`]: Sparse representation for large datasets
//!
//! These are re-exported from `thrml_core::similarity` for convenience.

// Re-export from thrml-core
pub use thrml_core::similarity::{
    cosine_similarity_matrix,
    cosine_similarity_threshold,
    cosine_similarity_topk,
    SparseSimilarity,
};

// Re-export distance functions for convenience
pub use thrml_core::distance::{
    gaussian_kernel,
    laplacian_kernel,
    pairwise_distances,
    pairwise_distances_sq,
};
