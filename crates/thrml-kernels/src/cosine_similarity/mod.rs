//! Cosine similarity fused kernel.

mod forward;
mod kernel;

pub use forward::{
    cosine_similarity_fused,
    cosine_similarity_fused_batched,
    cosine_similarity_prenormalized,
};
