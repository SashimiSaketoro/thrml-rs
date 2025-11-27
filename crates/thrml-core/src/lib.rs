//! # thrml-core
//!
//! Core types and GPU backend for the THRML probabilistic computing library.
//!
//! This crate provides foundational types for building probabilistic graphical models:
//!
//! - [`Node`]: Represents a random variable in the graph
//! - [`NodeType`]: Spin (binary ±1), Categorical, Continuous, or Spherical variables
//! - [`Block`]: A collection of nodes of the same type
//! - [`BlockSpec`]: Specification for mapping between local and global state
//! - [`InteractionGroup`]: Defines interactions between node groups
//! - [`InteractionData`]: Tensor, Linear, Quadratic, or Sphere interaction parameters
//!
//! ## Hybrid Compute Backend
//!
//! The [`compute`] module provides CPU/GPU routing for precision-sensitive operations,
//! optimized for Apple Silicon's unified memory architecture:
//!
//! - [`ComputeBackend`]: Backend selection (CPU, GPU, Hybrid, Adaptive)
//! - [`OpType`]: Operation classification for routing decisions
//! - [`PrecisionMode`]: Precision mode selection (GpuFast, CpuPrecise, Adaptive)
//! - [`HybridConfig`]: Combined backend + precision configuration
//!
//! ```rust,ignore
//! use thrml_core::{ComputeBackend, OpType};
//!
//! // Auto-detect Apple Silicon unified memory
//! let backend = ComputeBackend::apple_silicon();
//!
//! // Route precision-sensitive ops to CPU f64
//! if backend.use_cpu(OpType::IsingSampling, None) {
//!     // Use CPU path with f64 precision
//! } else {
//!     // Use GPU path with f32 performance
//! }
//! ```
//!
//! ## GPU Utilities (requires `gpu` feature)
//!
//! - [`distance`]: Pairwise distance and kernel computations
//! - [`similarity`]: Cosine similarity and sparse representations
//! - [`spherical`]: Spherical coordinate utilities
//!
//! ## GPU Backend
//!
//! The `gpu` feature provides GPU acceleration via WGPU:
//!
//! ```rust,ignore
//! use thrml_core::backend::{init_gpu_device, ensure_backend};
//!
//! ensure_backend();
//! let device = init_gpu_device();
//! ```

#![recursion_limit = "256"] // Required for burn-wgpu

pub mod backend;
pub mod block;
pub mod blockspec;
pub mod compute;
pub mod config;
pub mod node;

// GPU-dependent modules (require gpu feature)
#[cfg(feature = "gpu")]
pub mod distance;
#[cfg(feature = "gpu")]
pub mod interaction;
#[cfg(feature = "gpu")]
pub mod similarity;
#[cfg(feature = "gpu")]
pub mod spherical;
#[cfg(feature = "gpu")]
pub mod state_tree;

pub use backend::*;
pub use block::*;
pub use blockspec::*;
pub use compute::*;
pub use config::*;
pub use node::*;

#[cfg(feature = "gpu")]
pub use distance::*;
#[cfg(feature = "gpu")]
pub use interaction::*;
#[cfg(feature = "gpu")]
pub use similarity::*;
#[cfg(feature = "gpu")]
pub use spherical::*;
#[cfg(feature = "gpu")]
pub use state_tree::*;
