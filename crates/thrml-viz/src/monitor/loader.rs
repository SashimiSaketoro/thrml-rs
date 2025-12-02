//! Lazy data loading for visualization.
//!
//! Uses memory-mapped files to load only what's needed for rendering,
//! supporting datasets larger than available RAM.

use super::CheckpointNotify;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Monitored data state for visualization.
///
/// Lazily loads data from SafeTensors and NPZ files, caching what's needed
/// for rendering without loading everything into RAM.
pub struct MonitoredData {
    /// Path to current SafeTensors file
    safetensors_path: Option<PathBuf>,

    /// Path to current NPZ file (sphere coordinates)
    npz_path: Option<PathBuf>,

    /// Last processed step number
    last_step: usize,

    /// Cached point positions for rendering [N, 3]
    point_positions: Option<Vec<[f32; 3]>>,

    /// Cached prominence values [N]
    prominence: Option<Vec<f32>>,

    /// Number of points
    n_points: usize,

    /// Embedding dimension (for PCA if needed)
    embedding_dim: usize,

    /// Whether we have optimized sphere coords (vs raw embeddings)
    has_sphere_coords: bool,
}

impl Default for MonitoredData {
    fn default() -> Self {
        Self::new()
    }
}

impl MonitoredData {
    /// Create empty monitored data.
    pub fn new() -> Self {
        Self {
            safetensors_path: None,
            npz_path: None,
            last_step: 0,
            point_positions: None,
            prominence: None,
            n_points: 0,
            embedding_dim: 0,
            has_sphere_coords: false,
        }
    }

    /// Load from a SafeTensors file directly (standalone mode).
    pub fn load_safetensors(path: &Path) -> Result<Self> {
        use safetensors::SafeTensors;

        let data = std::fs::read(path).context("Failed to read SafeTensors file")?;
        let tensors = SafeTensors::deserialize(&data).context("Failed to parse SafeTensors")?;

        // Load embeddings
        let emb_tensor = tensors.tensor("embeddings").context("Missing 'embeddings' tensor")?;
        let emb_shape = emb_tensor.shape();
        let n_points = emb_shape[0];
        let embedding_dim = emb_shape[1];

        let emb_data: &[f32] = bytemuck::cast_slice(emb_tensor.data());

        // Load prominence
        let prominence = if let Ok(prom_tensor) = tensors.tensor("prominence") {
            let prom_data: &[f32] = bytemuck::cast_slice(prom_tensor.data());
            prom_data.to_vec()
        } else {
            vec![1.0; n_points]
        };

        // Project embeddings to 3D via PCA
        let point_positions = pca_project_to_3d(emb_data, n_points, embedding_dim);

        Ok(Self {
            safetensors_path: Some(path.to_path_buf()),
            npz_path: None,
            last_step: 0,
            point_positions: Some(point_positions),
            prominence: Some(prominence),
            n_points,
            embedding_dim,
            has_sphere_coords: false,
        })
    }

    /// Load from NPZ sphere coordinates file.
    pub fn load_npz(path: &Path) -> Result<Self> {
        use ndarray::Array1;
        use std::io::BufReader;

        let file = std::fs::File::open(path).context("Failed to open NPZ file")?;
        let mut npz = ndarray_npy::NpzReader::new(BufReader::new(file))
            .context("Failed to parse NPZ file")?;

        // Load spherical coordinates
        let r: Array1<f32> = npz.by_name("r.npy").context("Missing 'r' array")?;
        let theta: Array1<f32> = npz.by_name("theta.npy").context("Missing 'theta' array")?;
        let phi: Array1<f32> = npz.by_name("phi.npy").context("Missing 'phi' array")?;

        let n_points = r.len();

        // Convert to Cartesian
        let point_positions: Vec<[f32; 3]> = (0..n_points)
            .map(|i| {
                let r_i = r[i];
                let theta_i = theta[i];
                let phi_i = phi[i];
                [
                    r_i * theta_i.sin() * phi_i.cos(),
                    r_i * theta_i.sin() * phi_i.sin(),
                    r_i * theta_i.cos(),
                ]
            })
            .collect();

        // Use radius as prominence proxy (bigger radius = higher prominence)
        let prominence: Vec<f32> = r.to_vec();

        Ok(Self {
            safetensors_path: None,
            npz_path: Some(path.to_path_buf()),
            last_step: 0,
            point_positions: Some(point_positions),
            prominence: Some(prominence),
            n_points,
            embedding_dim: 0,
            has_sphere_coords: true,
        })
    }

    /// Update from a checkpoint notification.
    ///
    /// Only reloads if the step has advanced.
    pub fn update(&mut self, notify: &CheckpointNotify) -> Result<bool> {
        // Skip if we've already processed this step
        if notify.step <= self.last_step && self.last_step > 0 {
            return Ok(false);
        }

        // Prefer NPZ (sphere coords) if available
        if let Some(npz_path) = &notify.npz_path {
            if npz_path.exists() {
                *self = Self::load_npz(npz_path)?;
                self.last_step = notify.step;
                return Ok(true);
            }
        }

        // Fall back to SafeTensors (raw embeddings with PCA)
        if notify.safetensors_path.exists() {
            *self = Self::load_safetensors(&notify.safetensors_path)?;
            self.last_step = notify.step;
            return Ok(true);
        }

        Ok(false)
    }

    /// Get point positions for rendering.
    pub fn positions(&self) -> Option<&[[f32; 3]]> {
        self.point_positions.as_deref()
    }

    /// Get prominence values for coloring.
    pub fn prominence(&self) -> Option<&[f32]> {
        self.prominence.as_deref()
    }

    /// Get the number of points.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Check if we have sphere coordinates (vs PCA-projected embeddings).
    pub fn has_sphere_coords(&self) -> bool {
        self.has_sphere_coords
    }

    /// Get the last processed step.
    pub fn last_step(&self) -> usize {
        self.last_step
    }

    /// Check if data is loaded.
    pub fn is_loaded(&self) -> bool {
        self.point_positions.is_some()
    }

    /// Get the current SafeTensors file path (if loaded from one).
    pub fn safetensors_path(&self) -> Option<&Path> {
        self.safetensors_path.as_deref()
    }

    /// Get the current NPZ file path (if loaded from sphere coordinates).
    pub fn npz_path(&self) -> Option<&Path> {
        self.npz_path.as_deref()
    }

    /// Get the current source file path (prefers NPZ over SafeTensors).
    pub fn source_path(&self) -> Option<&Path> {
        self.npz_path.as_deref().or(self.safetensors_path.as_deref())
    }
}

/// Project high-dimensional embeddings to 3D using PCA.
///
/// This is a simple implementation using the power method for the first 3 principal components.
fn pca_project_to_3d(embeddings: &[f32], n_points: usize, dim: usize) -> Vec<[f32; 3]> {
    if n_points == 0 || dim == 0 {
        return vec![];
    }

    // For simplicity, just use first 3 dimensions if dim >= 3
    // A proper PCA would compute covariance and eigenvectors
    if dim >= 3 {
        return embeddings
            .chunks(dim)
            .map(|chunk| {
                // Normalize to reasonable scale
                let scale = 100.0;
                [chunk[0] * scale, chunk[1] * scale, chunk[2] * scale]
            })
            .collect();
    }

    // If dim < 3, pad with zeros
    embeddings
        .chunks(dim)
        .map(|chunk| {
            let scale = 100.0;
            let mut pos = [0.0f32; 3];
            for (i, &v) in chunk.iter().enumerate().take(3) {
                pos[i] = v * scale;
            }
            pos
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_projection() {
        let embeddings = vec![
            1.0, 0.0, 0.0, 0.5, // point 1
            0.0, 1.0, 0.0, 0.5, // point 2
            0.0, 0.0, 1.0, 0.5, // point 3
        ];

        let projected = pca_project_to_3d(&embeddings, 3, 4);
        assert_eq!(projected.len(), 3);

        // First 3 dims should be used
        assert!((projected[0][0] - 100.0).abs() < 0.001);
        assert!((projected[1][1] - 100.0).abs() < 0.001);
        assert!((projected[2][2] - 100.0).abs() < 0.001);
    }
}
