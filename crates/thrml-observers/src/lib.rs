//! # thrml-observers
//!
//! Observation utilities for the THRML probabilistic computing library.

// Clippy pedantic allows - verified needed 2024-12-01
// (minimal crate - only 36 pedantic warnings, mostly acceptable)
#![allow(clippy::must_use_candidate)] // getters
//!
//! Observers allow you to collect data during sampling without modifying the sampling algorithm.
//!
//! ## StateObserver
//!
//! Collect state samples for specific blocks:
//!
//! ```rust,ignore
//! use thrml_observers::{StateObserver, AbstractObserver};
//!
//! let observer = StateObserver::new(blocks_to_observe);
//! let (carry, samples) = observer.observe(spec, free_state, clamped_state, carry, iter, &device);
//! ```
//!
//! ## MomentAccumulatorObserver
//!
//! Compute running moment statistics:
//!
//! ```rust,ignore
//! use thrml_observers::moment::{MomentAccumulatorObserver, MomentSpec};
//!
//! let spec = MomentAccumulatorObserver::ising_moment_spec(&nodes, &edges);
//! let observer = MomentAccumulatorObserver::new(spec, true); // spin transform
//! ```

#![recursion_limit = "256"] // Required for burn-wgpu

pub mod moment;
pub mod observer;
pub mod state_observer;

pub use moment::*;
pub use observer::*;
pub use state_observer::*;
