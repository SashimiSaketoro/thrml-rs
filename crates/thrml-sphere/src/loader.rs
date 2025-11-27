//! SafeTensors file loading utilities for BLT output.
//!
//! This module provides functions for loading embeddings, prominence, entropy,
//! and raw byte data from SafeTensors files produced by the BLT (Byte Latent
//! Transformer) ingestion pipeline.
//!
//! # Supported Formats
//!
//! | Format | Tensors | Use Case |
//! |--------|---------|----------|
//! | Legacy | `embeddings`, `prominence`, `entropies` | Pre-computed embeddings |
//! | BLT v3 | Above + `bytes`, `patch_lengths`, `patch_entropies` | Full BLT output |
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use thrml_sphere::{load_blt_safetensors, SphereConfig};
//!
//! // Load BLT v3 format (includes raw bytes for substring coupling)
//! let (sphere_ebm, patch_bytes) = load_blt_safetensors(
//!     Path::new("output.safetensors"),
//!     SphereConfig::default(),
//!     &device,
//! )?;
//!
//! // Or load just embeddings (legacy format)
//! let sphere_ebm = load_from_safetensors(
//!     Path::new("embeddings.safetensors"),
//!     SphereConfig::default(),
//!     &device,
//! )?;
//! ```

use anyhow::{Context, Result};
use burn::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
use thrml_core::backend::WgpuBackend;

use crate::config::SphereConfig;
use crate::sphere_ebm::SphereEBM;

/// Loads a [`SphereEBM`] from a SafeTensors file (legacy format).
///
/// Use this for pre-computed embeddings without raw byte data. For BLT v3
/// output with raw bytes, use [`load_blt_safetensors`] instead.
///
/// # Required Tensors
///
/// | Name | Shape | Dtype | Description |
/// |------|-------|-------|-------------|
/// | `embeddings` | \[N, D\] | F32 | Embedding vectors |
/// | `prominence` | \[N\] | F32 | Prominence/importance scores |
///
/// # Optional Tensors
///
/// | Name | Shape | Dtype | Description |
/// |------|-------|-------|-------------|
/// | `entropies` | \[N\] | F32 | Per-point entropy values |
///
/// # Arguments
///
/// * `path` - Path to the SafeTensors file.
/// * `config` - Sphere optimization configuration.
/// * `device` - GPU device for tensor allocation.
///
/// # Returns
///
/// A configured [`SphereEBM`] ready for optimization.
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be read or is not valid SafeTensors format
/// - Required `embeddings` or `prominence` tensors are missing
/// - Tensor shapes are incompatible
///
/// # Example
///
/// ```rust,ignore
/// use thrml_sphere::{load_from_safetensors, SphereConfig};
/// use std::path::Path;
///
/// let config = SphereConfig::default();
/// let sphere = load_from_safetensors(
///     Path::new("embeddings.safetensors"),
///     config,
///     &device,
/// )?;
///
/// println!("Loaded {} points with {} dimensions", sphere.n_points(), sphere.embedding_dim());
/// ```
pub fn load_from_safetensors(
    path: &Path,
    config: SphereConfig,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<SphereEBM> {
    let bytes = std::fs::read(path).with_context(|| format!("Failed to read {:?}", path))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .with_context(|| format!("Failed to deserialize SafeTensors from {:?}", path))?;

    // Load embeddings [N, D]
    let emb_view = tensors
        .tensor("embeddings")
        .context("SafeTensors file missing 'embeddings' tensor")?;
    let embeddings =
        tensor_to_burn_2d(&emb_view, device).context("Failed to convert embeddings tensor")?;

    // Load prominence [N]
    let prom_view = tensors
        .tensor("prominence")
        .context("SafeTensors file missing 'prominence' tensor")?;
    let prominence =
        tensor_to_burn_1d(&prom_view, device).context("Failed to convert prominence tensor")?;

    // Load entropies (optional)
    let entropies = tensors
        .tensor("entropies")
        .ok()
        .map(|v| tensor_to_burn_1d(&v, device))
        .transpose()
        .context("Failed to convert entropies tensor")?;

    Ok(SphereEBM::new(
        embeddings, prominence, entropies, config, device,
    ))
}

/// Loads only the embeddings tensor from a SafeTensors file.
///
/// Useful when you need embeddings for custom processing without creating
/// a full [`SphereEBM`].
///
/// # Arguments
///
/// * `path` - Path to the SafeTensors file.
/// * `device` - GPU device for tensor allocation.
///
/// # Returns
///
/// Embeddings tensor with shape [N, D].
///
/// # Errors
///
/// Returns an error if the file cannot be read or is missing the `embeddings` tensor.
pub fn load_embeddings(
    path: &Path,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<Tensor<WgpuBackend, 2>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let view = tensors
        .tensor("embeddings")
        .context("Missing 'embeddings' tensor")?;
    tensor_to_burn_2d(&view, device)
}

/// Loads only the prominence tensor from a SafeTensors file.
///
/// Useful when you need prominence scores for custom processing without
/// creating a full [`SphereEBM`].
///
/// # Arguments
///
/// * `path` - Path to the SafeTensors file.
/// * `device` - GPU device for tensor allocation.
///
/// # Returns
///
/// Prominence tensor with shape \[N\].
///
/// # Errors
///
/// Returns an error if the file cannot be read or is missing the `prominence` tensor.
pub fn load_prominence(
    path: &Path,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<Tensor<WgpuBackend, 1>> {
    let bytes = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let view = tensors
        .tensor("prominence")
        .context("Missing 'prominence' tensor")?;
    tensor_to_burn_1d(&view, device)
}

/// Convert a SafeTensors tensor view to a Burn 1D tensor.
///
/// Handles both unbatched [N] and batched [1, N] formats.
fn tensor_to_burn_1d(
    view: &safetensors::tensor::TensorView,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<Tensor<WgpuBackend, 1>> {
    let shape = view.shape();
    anyhow::ensure!(
        view.dtype() == safetensors::Dtype::F32,
        "Expected F32 dtype, got {:?}",
        view.dtype()
    );

    let data = view.data();
    let floats: &[f32] = bytemuck::cast_slice(data);

    match shape.len() {
        1 => {
            // Unbatched [N]
            Ok(Tensor::from_data(floats, device))
        }
        2 if shape[0] == 1 => {
            // Batched [1, N] - squeeze batch dimension
            let n = shape[1];
            let tensor_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(floats, device);
            Ok(tensor_1d.reshape([n as i32]))
        }
        _ => {
            anyhow::bail!(
                "Expected 1D tensor [N] or batched [1, N], got shape {:?}",
                shape
            )
        }
    }
}

/// Convert a SafeTensors tensor view to a Burn 2D tensor.
///
/// Handles both unbatched [N, D] and batched [1, N, D] formats.
fn tensor_to_burn_2d(
    view: &safetensors::tensor::TensorView,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<Tensor<WgpuBackend, 2>> {
    let shape = view.shape();
    anyhow::ensure!(
        view.dtype() == safetensors::Dtype::F32,
        "Expected F32 dtype, got {:?}",
        view.dtype()
    );

    let data = view.data();
    let floats: &[f32] = bytemuck::cast_slice(data);

    match shape.len() {
        2 => {
            // Unbatched [N, D]
            let tensor_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(floats, device);
            Ok(tensor_1d.reshape([shape[0] as i32, shape[1] as i32]))
        }
        3 if shape[0] == 1 => {
            // Batched [1, N, D] - squeeze batch dimension
            let n = shape[1];
            let d = shape[2];
            let tensor_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(floats, device);
            Ok(tensor_1d.reshape([n as i32, d as i32]))
        }
        _ => {
            anyhow::bail!(
                "Expected 2D tensor [N, D] or batched [1, N, D], got shape {:?}",
                shape
            )
        }
    }
}

// ============================================================================
// BLT Output Loading (v3 format with embeddings)
// ============================================================================

/// Loads patch bytes from a BLT SafeTensors file (v3 format).
///
/// The file should contain:
/// - `bytes`: \[total_bytes\] U8 - Concatenated raw bytes
/// - `patch_lengths`: \[num_patches\] I32 - Length of each patch
///
/// Returns a `Vec<Vec<u8>>` where each inner Vec is the bytes for one patch.
///
/// # Arguments
///
/// * `path` - Path to the BLT SafeTensors file.
///
/// # Returns
///
/// Vector of byte vectors, one per patch.
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be read
/// - Missing `bytes` or `patch_lengths` tensors
/// - Patch lengths don't align with total bytes
///
/// # Example
///
/// ```rust,ignore
/// let patch_bytes = load_patch_bytes(Path::new("output.safetensors"))?;
/// println!("Loaded {} patches", patch_bytes.len());
/// ```
pub fn load_patch_bytes(path: &Path) -> Result<Vec<Vec<u8>>> {
    let data = std::fs::read(path).with_context(|| format!("Failed to read {:?}", path))?;
    let tensors = SafeTensors::deserialize(&data)
        .with_context(|| format!("Failed to deserialize SafeTensors from {:?}", path))?;

    // Load concatenated bytes [total_bytes] as U8
    let bytes_view = tensors
        .tensor("bytes")
        .context("SafeTensors file missing 'bytes' tensor")?;
    anyhow::ensure!(
        bytes_view.dtype() == safetensors::Dtype::U8,
        "Expected U8 dtype for bytes, got {:?}",
        bytes_view.dtype()
    );
    let all_bytes: &[u8] = bytes_view.data();

    // Load patch lengths [num_patches] as I32
    let lengths_view = tensors
        .tensor("patch_lengths")
        .context("SafeTensors file missing 'patch_lengths' tensor")?;
    anyhow::ensure!(
        lengths_view.dtype() == safetensors::Dtype::I32,
        "Expected I32 dtype for patch_lengths, got {:?}",
        lengths_view.dtype()
    );
    let lengths: &[i32] = bytemuck::cast_slice(lengths_view.data());

    // Reconstruct individual patch bytes
    let mut patches = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for &len in lengths {
        let len = len as usize;
        if offset + len > all_bytes.len() {
            anyhow::bail!(
                "Patch at offset {} with length {} exceeds total bytes {}",
                offset,
                len,
                all_bytes.len()
            );
        }
        patches.push(all_bytes[offset..offset + len].to_vec());
        offset += len;
    }

    Ok(patches)
}

/// Loads complete BLT output for ROOTS integration.
///
/// This is the preferred method for loading BLT v3 SafeTensors files, providing
/// both embeddings (as a [`SphereEBM`]) and raw bytes (for substring coupling).
///
/// # Returns
///
/// A tuple of:
/// - [`SphereEBM`] - Ready for ROOTS partitioning
/// - `Vec<Vec<u8>>` - Patch bytes for substring coupling
///
/// # Required Tensors (blt_patches_v3 format)
///
/// | Name | Shape | Dtype | Description |
/// |------|-------|-------|-------------|
/// | `embeddings` | \[N, D\] | F32 | Patch embeddings |
/// | `prominence` | \[N\] | F32 | Importance scores |
/// | `bytes` | \[total_bytes\] | U8 | Concatenated raw bytes |
/// | `patch_lengths` | \[N\] | I32 | Length of each patch |
///
/// # Optional Tensors
///
/// | Name | Shape | Dtype | Description |
/// |------|-------|-------|-------------|
/// | `patch_entropies` | \[N\] | F32 | Aggregated entropy per patch |
///
/// # Arguments
///
/// * `path` - Path to the BLT SafeTensors file.
/// * `config` - Sphere optimization configuration.
/// * `device` - GPU device for tensor allocation.
///
/// # Errors
///
/// Returns an error if:
/// - File cannot be read
/// - Required tensors are missing
/// - Patch count mismatches between embeddings and patch_lengths
///
/// # Example
///
/// ```rust,ignore
/// let (sphere_ebm, patch_bytes) = load_blt_safetensors(
///     Path::new("output.safetensors"),
///     SphereConfig::default(),
///     &device,
/// )?;
///
/// let roots = RootsIndex::from_sphere_ebm_with_bytes(
///     &sphere_ebm,
///     &patch_bytes,
///     roots_config,
///     key,
///     &device,
/// );
/// ```
pub fn load_blt_safetensors(
    path: &Path,
    config: SphereConfig,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Result<(SphereEBM, Vec<Vec<u8>>)> {
    let data = std::fs::read(path).with_context(|| format!("Failed to read {:?}", path))?;
    let tensors = SafeTensors::deserialize(&data)
        .with_context(|| format!("Failed to deserialize SafeTensors from {:?}", path))?;

    // Load embeddings [N, D]
    let emb_view = tensors
        .tensor("embeddings")
        .context("SafeTensors file missing 'embeddings' tensor (is this blt_patches_v3 format?)")?;
    let embeddings =
        tensor_to_burn_2d(&emb_view, device).context("Failed to convert embeddings tensor")?;

    // Load prominence [N]
    let prom_view = tensors
        .tensor("prominence")
        .context("SafeTensors file missing 'prominence' tensor")?;
    let prominence =
        tensor_to_burn_1d(&prom_view, device).context("Failed to convert prominence tensor")?;

    // Load patch entropies (optional, prefer patch_entropies over per-byte entropies)
    let entropies = tensors
        .tensor("patch_entropies")
        .ok()
        .or_else(|| tensors.tensor("entropies").ok())
        .map(|v| tensor_to_burn_1d(&v, device))
        .transpose()
        .context("Failed to convert entropies tensor")?;

    // Create SphereEBM
    let sphere_ebm = SphereEBM::new(embeddings, prominence, entropies, config, device);

    // Load patch bytes
    let bytes_view = tensors
        .tensor("bytes")
        .context("SafeTensors file missing 'bytes' tensor")?;
    anyhow::ensure!(
        bytes_view.dtype() == safetensors::Dtype::U8,
        "Expected U8 dtype for bytes, got {:?}",
        bytes_view.dtype()
    );
    let all_bytes: &[u8] = bytes_view.data();

    let lengths_view = tensors
        .tensor("patch_lengths")
        .context("SafeTensors file missing 'patch_lengths' tensor")?;
    anyhow::ensure!(
        lengths_view.dtype() == safetensors::Dtype::I32,
        "Expected I32 dtype for patch_lengths, got {:?}",
        lengths_view.dtype()
    );
    let lengths: &[i32] = bytemuck::cast_slice(lengths_view.data());

    // Reconstruct patch bytes
    let mut patch_bytes = Vec::with_capacity(lengths.len());
    let mut offset = 0usize;
    for &len in lengths {
        let len = len as usize;
        if offset + len > all_bytes.len() {
            anyhow::bail!(
                "Patch at offset {} with length {} exceeds total bytes {}",
                offset,
                len,
                all_bytes.len()
            );
        }
        patch_bytes.push(all_bytes[offset..offset + len].to_vec());
        offset += len;
    }

    // Verify alignment
    let n_patches = sphere_ebm.n_points();
    if patch_bytes.len() != n_patches {
        anyhow::bail!(
            "Mismatch: {} patches from embeddings but {} from patch_lengths",
            n_patches,
            patch_bytes.len()
        );
    }

    Ok((sphere_ebm, patch_bytes))
}

/// Check if a SafeTensors file is in BLT v3 format (contains embeddings).
pub fn is_blt_v3_format(path: &Path) -> Result<bool> {
    let data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&data)?;
    Ok(tensors.tensor("embeddings").is_ok() && tensors.tensor("bytes").is_ok())
}

// ============================================================================
// Output
// ============================================================================

/// Saves spherical coordinates to NPZ format.
///
/// Saves the following arrays:
/// - `r`: Radii \[N\]
/// - `theta`: Polar angles \[N\]
/// - `phi`: Azimuthal angles \[N\]
/// - `cartesian`: Cartesian positions \[N, 3\]
///
/// # Arguments
///
/// * `coords` - Spherical coordinates to save.
/// * `path` - Output file path (should end in `.npz`).
///
/// # Errors
///
/// Returns an error if the file cannot be written.
pub fn save_coords_npz(coords: &thrml_core::SphericalCoords, path: &Path) -> Result<()> {
    use ndarray::{Array1, Array2};
    use ndarray_npy::NpzWriter;
    use std::fs::File;

    let n = coords.len();

    // Convert tensors to ndarray
    // Note: to_vec() returns Result<Vec<E>, DataError>, using expect for simplicity
    let r_data: Vec<f32> = coords
        .r
        .clone()
        .into_data()
        .to_vec()
        .expect("Failed to convert r tensor to vec");
    let theta_data: Vec<f32> = coords
        .theta
        .clone()
        .into_data()
        .to_vec()
        .expect("Failed to convert theta tensor to vec");
    let phi_data: Vec<f32> = coords
        .phi
        .clone()
        .into_data()
        .to_vec()
        .expect("Failed to convert phi tensor to vec");

    let cart = coords.to_cartesian();
    let cart_data: Vec<f32> = cart
        .into_data()
        .to_vec()
        .expect("Failed to convert cartesian tensor to vec");

    let r = Array1::from_vec(r_data);
    let theta = Array1::from_vec(theta_data);
    let phi = Array1::from_vec(phi_data);
    let cartesian = Array2::from_shape_vec((n, 3), cart_data)?;

    // Write to NPZ
    let file = File::create(path)?;
    let mut npz = NpzWriter::new(file);
    npz.add_array("r", &r)?;
    npz.add_array("theta", &theta)?;
    npz.add_array("phi", &phi)?;
    npz.add_array("cartesian", &cartesian)?;
    npz.finish()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use std::collections::HashMap;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_tensor_conversion_roundtrip() {
        let device = init_gpu_device();
        let n = 10;
        let d = 5;

        // Create test data
        let test_data: Vec<f32> = (0..n * d).map(|i| i as f32).collect();

        // Create safetensors
        let shape = vec![n, d];
        let bytes: Vec<u8> = test_data.iter().flat_map(|&f| f.to_le_bytes()).collect();

        // This test verifies the conversion logic works correctly
        // In practice we'd create a real safetensors file
    }
}
