//! Notification protocol for live ingestion monitoring.
//!
//! The protocol is simple JSON over Unix sockets:
//! - Each message is a JSON object followed by newline
//! - Messages are fire-and-forget (no ACK required)
//! - Visualizer polls for latest, ingestion never blocks

use interprocess::local_socket::{prelude::*, GenericFilePath, Stream};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Socket path prefix for IPC.
/// Full path: `/tmp/thrml-viz-{session_id}.sock`
pub const SOCKET_PREFIX: &str = "/tmp/thrml-viz-";

/// Checkpoint notification sent by ingestion process.
///
/// This is a lightweight ~100 byte JSON message that tells the visualizer
/// a new checkpoint is available for viewing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointNotify {
    /// Path to the SafeTensors file (embeddings + prominence)
    pub safetensors_path: PathBuf,

    /// Path to NPZ file with sphere coordinates (if optimization done)
    pub npz_path: Option<PathBuf>,

    /// Current batch/step number
    pub step: usize,

    /// Total expected steps (if known)
    pub total_steps: Option<usize>,

    /// Number of points/patches processed so far
    pub n_points: usize,

    /// Timestamp (unix milliseconds)
    pub timestamp: u64,

    /// Optional status message
    pub message: Option<String>,
}

impl CheckpointNotify {
    /// Create a new checkpoint notification.
    pub fn new(safetensors_path: PathBuf, step: usize, n_points: usize) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            safetensors_path,
            npz_path: None,
            step,
            total_steps: None,
            n_points,
            timestamp,
            message: None,
        }
    }

    /// Set the NPZ path (sphere coordinates).
    pub fn with_npz(mut self, path: PathBuf) -> Self {
        self.npz_path = Some(path);
        self
    }

    /// Set the total expected steps.
    pub fn with_total_steps(mut self, total: usize) -> Self {
        self.total_steps = Some(total);
        self
    }

    /// Set an optional status message.
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Calculate progress as a fraction (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        match self.total_steps {
            Some(total) if total > 0 => self.step as f32 / total as f32,
            _ => 0.0,
        }
    }
}

/// Fire-and-forget notification sender for ingestion processes.
///
/// This is designed to have zero overhead on the ingestion process:
/// - Non-blocking socket operations
/// - Errors are silently ignored (visualizer may not be running)
/// - Small message size (~100 bytes)
///
/// # Usage
///
/// ```ignore
/// let mut notifier = VizNotifier::try_connect("my-session");
///
/// // During ingestion loop
/// for batch in batches {
///     process_batch(&batch);
///     
///     // Fire-and-forget notification (~1µs)
///     notifier.notify(&CheckpointNotify::new(
///         output_path.clone(),
///         batch_idx,
///         total_points,
///     ));
/// }
/// ```
pub struct VizNotifier {
    socket: Option<Stream>,
    session_id: String,
}

impl VizNotifier {
    /// Attempt to connect to a visualizer session.
    ///
    /// If no visualizer is running, this returns a VizNotifier that
    /// silently drops all notifications.
    pub fn try_connect(session_id: &str) -> Self {
        let path = format!("{}{}.sock", SOCKET_PREFIX, session_id);
        let name = path.to_fs_name::<GenericFilePath>();
        let socket = name.ok().and_then(|n| Stream::connect(n).ok());

        if socket.is_some() {
            log::info!("Connected to visualizer session: {}", session_id);
        } else {
            log::debug!(
                "No visualizer running for session: {} (notifications will be dropped)",
                session_id
            );
        }

        Self {
            socket,
            session_id: session_id.to_string(),
        }
    }

    /// Send a checkpoint notification.
    ///
    /// This is fire-and-forget:
    /// - Never blocks the caller
    /// - Errors are silently ignored
    /// - If the visualizer is slow/gone, messages are dropped
    pub fn notify(&mut self, checkpoint: &CheckpointNotify) {
        if let Some(ref mut sock) = self.socket {
            // Serialize to JSON (typically ~100 bytes)
            if let Ok(mut msg) = serde_json::to_vec(checkpoint) {
                msg.push(b'\n'); // Newline delimiter

                // Non-blocking write, ignore errors
                if sock.write_all(&msg).is_err() {
                    // Connection lost, clear socket
                    log::debug!("Lost connection to visualizer, disabling notifications");
                    self.socket = None;
                }
            }
        }
    }

    /// Check if connected to a visualizer.
    pub fn is_connected(&self) -> bool {
        self.socket.is_some()
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

impl Default for VizNotifier {
    fn default() -> Self {
        Self {
            socket: None,
            session_id: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_notify_serialization() {
        let notify = CheckpointNotify::new(PathBuf::from("/tmp/test.safetensors"), 10, 1000)
            .with_total_steps(100)
            .with_message("Processing batch 10");

        let json = serde_json::to_string(&notify).unwrap();
        assert!(json.contains("safetensors_path"));
        assert!(json.contains("step"));

        // Should be small (~200 bytes)
        assert!(json.len() < 500);

        // Round-trip
        let parsed: CheckpointNotify = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.step, 10);
        assert_eq!(parsed.n_points, 1000);
    }

    #[test]
    fn test_progress_calculation() {
        let notify = CheckpointNotify::new(PathBuf::from("/tmp/test.safetensors"), 50, 1000)
            .with_total_steps(100);

        assert!((notify.progress() - 0.5).abs() < 0.001);
    }
}
