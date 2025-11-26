//! # thrml-core
//!
//! Core types and GPU backend for the THRML probabilistic computing library.
//!
//! This crate provides foundational types for building probabilistic graphical models:
//!
//! - [`Node`]: Represents a random variable in the graph
//! - [`NodeType`]: Spin (binary ±1), Categorical, or Continuous variables
//! - [`Block`]: A collection of nodes of the same type
//! - [`BlockSpec`]: Specification for mapping between local and global state
//! - [`InteractionGroup`]: Defines interactions between node groups
//! - [`InteractionData`]: Tensor, Linear, or Quadratic interaction parameters
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
pub mod config;
pub mod interaction;
pub mod node;
pub mod state_tree;

pub use backend::*;
pub use block::*;
pub use blockspec::*;
pub use config::*;
pub use interaction::*;
pub use node::*;
pub use state_tree::*;
