//! IPC listener for receiving checkpoint notifications.
//!
//! Runs a background thread that listens on a Unix socket for incoming
//! notifications from ingestion processes.

use super::notify::{CheckpointNotify, SOCKET_PREFIX};
use interprocess::local_socket::{prelude::*, GenericFilePath, Listener, ListenerOptions, Stream};
use std::io::{BufRead, BufReader};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

/// IPC monitor that listens for checkpoint notifications.
///
/// Spawns a background thread that listens on a Unix socket and
/// forwards notifications to the main thread via a channel.
///
/// # Usage
///
/// ```ignore
/// let monitor = IpcMonitor::start("my-session")?;
///
/// // In render loop
/// loop {
///     if let Some(notify) = monitor.poll() {
///         // New checkpoint available
///         reload_data(&notify);
///     }
///     render();
/// }
/// ```
pub struct IpcMonitor {
    receiver: Receiver<CheckpointNotify>,
    _listener_handle: JoinHandle<()>,
    session_id: String,
    socket_path: String,
}

impl IpcMonitor {
    /// Start listening for notifications on the given session.
    ///
    /// Creates a Unix socket at `/tmp/thrml-viz-{session_id}.sock`.
    pub fn start(session_id: &str) -> std::io::Result<Self> {
        let socket_path = format!("{}{}.sock", SOCKET_PREFIX, session_id);

        // Remove stale socket if exists
        let _ = std::fs::remove_file(&socket_path);

        // Create listener using new API
        let name = socket_path.clone().to_fs_name::<GenericFilePath>().map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, e)
        })?;
        let opts = ListenerOptions::new().name(name);
        let listener = opts.create_sync()?;
        log::info!("IPC monitor listening on: {}", socket_path);

        // Channel for forwarding notifications
        let (tx, rx) = mpsc::channel();

        // Spawn listener thread
        let handle = thread::spawn(move || {
            listener_thread(listener, tx);
        });

        Ok(Self {
            receiver: rx,
            _listener_handle: handle,
            session_id: session_id.to_string(),
            socket_path,
        })
    }

    /// Poll for the latest checkpoint notification (non-blocking).
    ///
    /// Returns the most recent notification if any are available,
    /// discarding older ones.
    pub fn poll(&self) -> Option<CheckpointNotify> {
        // Drain channel, return most recent
        let mut latest = None;
        while let Ok(notify) = self.receiver.try_recv() {
            latest = Some(notify);
        }
        latest
    }

    /// Check if there are pending notifications.
    pub fn has_pending(&self) -> bool {
        // This is a bit of a hack - we peek by trying to receive
        // Unfortunately mpsc doesn't have a proper peek
        false // Conservative: always check poll()
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the socket path.
    pub fn socket_path(&self) -> &str {
        &self.socket_path
    }
}

impl Drop for IpcMonitor {
    fn drop(&mut self) {
        // Clean up socket file
        let _ = std::fs::remove_file(&self.socket_path);
        log::debug!("Removed socket: {}", self.socket_path);
    }
}

/// Background thread that accepts connections and parses notifications.
fn listener_thread(listener: Listener, tx: Sender<CheckpointNotify>) {
    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let tx = tx.clone();
                // Handle connection in a separate thread to not block new connections
                thread::spawn(move || {
                    handle_connection(stream, tx);
                });
            }
            Err(e) => {
                log::warn!("Failed to accept connection: {}", e);
            }
        }
    }
}

/// Handle a single connection, reading newline-delimited JSON messages.
fn handle_connection(
    stream: Stream,
    tx: Sender<CheckpointNotify>,
) {
    let reader = BufReader::new(stream);

    for line in reader.lines() {
        match line {
            Ok(json) => {
                if json.is_empty() {
                    continue;
                }

                match serde_json::from_str::<CheckpointNotify>(&json) {
                    Ok(notify) => {
                        log::debug!(
                            "Received checkpoint: step={}, n_points={}",
                            notify.step,
                            notify.n_points
                        );
                        // Non-blocking send - if channel is full, older messages are already there
                        let _ = tx.send(notify);
                    }
                    Err(e) => {
                        log::warn!("Failed to parse notification: {}", e);
                    }
                }
            }
            Err(e) => {
                log::debug!("Connection closed: {}", e);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::VizNotifier;
    use std::path::PathBuf;
    use std::time::Duration;

    #[test]
    fn test_ipc_roundtrip() {
        let session_id = format!("test-{}", std::process::id());

        // Start monitor
        let monitor = IpcMonitor::start(&session_id).expect("Failed to start monitor");

        // Give it a moment to bind
        thread::sleep(Duration::from_millis(50));

        // Connect notifier
        let mut notifier = VizNotifier::try_connect(&session_id);
        assert!(notifier.is_connected());

        // Send notification
        let checkpoint = CheckpointNotify::new(PathBuf::from("/tmp/test.safetensors"), 42, 1000);
        notifier.notify(&checkpoint);

        // Give it time to arrive
        thread::sleep(Duration::from_millis(100));

        // Should receive it
        let received = monitor.poll();
        assert!(received.is_some());
        assert_eq!(received.unwrap().step, 42);
    }
}
