//! # thrml-viz
//!
//! Interactive 3D visualization for THRML sphere embeddings and ROOTS hierarchy.
//!
//! This crate provides a GPU-accelerated visualizer using `eframe` and `egui` with
//! custom `wgpu` rendering for point clouds and tree structures.
//!
//! ## Features
//!
//! - **3D Point Cloud Rendering**: Visualize sphere-optimized embeddings with prominence coloring
//! - **ROOTS Hierarchy View**: See the H-ROOTS tree structure as connected nodes
//! - **Dual View Mode**: Compare raw embeddings (PCA) vs optimized sphere positions
//! - **Live Monitoring**: Watch ingestion progress in real-time via IPC
//! - **Interactive Camera**: Orbit, zoom, and pan controls
//!
//! ## Usage
//!
//! ### Standalone Viewing
//!
//! ```bash
//! cargo run --example visualize -- --input output.safetensors
//! ```
//!
//! ### Live Monitoring
//!
//! ```bash
//! # Terminal 1: Start visualizer
//! cargo run --example visualize -- --monitor my-session
//!
//! # Terminal 2: Run ingestion with viz notifications
//! blt-burn ingest --features viz --viz-session my-session input.txt
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐                    ┌─────────────────┐
//! │   INGESTION     │                    │   VISUALIZER    │
//! │                 │    fire-and-forget │                 │
//! │  write tensors ─┼───────────────────→│  receive notify │
//! │  (never waits)  │    (unix socket)   │  lazy mmap load │
//! └─────────────────┘                    └─────────────────┘
//! ```

#![allow(clippy::too_many_arguments)]

// Monitor module is always available (just needs interprocess + serde)
pub mod monitor;

// These modules require the full feature set
#[cfg(feature = "full")]
pub mod app;
#[cfg(feature = "full")]
pub mod camera;
#[cfg(feature = "full")]
pub mod renderer;
#[cfg(feature = "full")]
pub mod tree_export;
#[cfg(feature = "full")]
pub mod widgets;

// Always export notification types
pub use monitor::{CheckpointNotify, VizNotifier, SOCKET_PREFIX};

// Full visualizer exports
#[cfg(feature = "full")]
pub use app::{run, VizApp, VizConfig};
#[cfg(feature = "full")]
pub use camera::OrbitalCamera;
#[cfg(feature = "full")]
pub use monitor::{IpcMonitor, MonitoredData};
#[cfg(feature = "full")]
pub use renderer::SphereRenderer;
#[cfg(feature = "full")]
pub use tree_export::{ExportNode, TreeVizData, VizNode};
