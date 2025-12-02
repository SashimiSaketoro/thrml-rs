//! Live monitoring infrastructure for ingestion visualization.
//!
//! This module provides zero-overhead IPC for monitoring ingestion progress
//! and lazy loading of checkpoint data.

// IPC listener (requires full features for threading)
#[cfg(feature = "full")]
mod ipc;
// Lazy data loader (requires safetensors, ndarray)
#[cfg(feature = "full")]
mod loader;
// Notification types (always available - minimal deps)
mod notify;

#[cfg(feature = "full")]
pub use ipc::IpcMonitor;
#[cfg(feature = "full")]
pub use loader::MonitoredData;
// Always export notification types
pub use notify::{CheckpointNotify, VizNotifier, SOCKET_PREFIX};
