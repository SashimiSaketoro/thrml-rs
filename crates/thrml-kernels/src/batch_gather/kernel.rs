//! CubeCL kernel definition for fused batch gather with linear indexing.
//!
//! This kernel gathers values from a weight tensor using multiple indices
//! without creating intermediate stride tensors.
//!
//! # Kernel Logic
//!
//! ```ignore
//! for each batch_idx in parallel:
//!     // Compute linear index from multiple indices
//!     linear_idx = batch_idx * batch_stride
//!
//!     for i in 0..n_indices:
//!         idx_val = indices[batch_idx * n_indices + i]
//!         linear_idx = linear_idx + idx_val * strides[i]
//!
//!     output[batch_idx] = weights[linear_idx]
//! ```
//!
//! # Performance
//!
//! This kernel eliminates intermediate tensor allocations from the unfused version:
//! 1. Batch indices tensor
//! 2. Stride tensors for each dimension
//! 3. Linear index computation intermediates
//! 4. Reshaped weight tensor

use cubecl::{cube, prelude::*};

// ============================================================================
// Specialized Kernels for 3-6 Indices
// ============================================================================

/// Fused batch gather kernel with 3 index dimensions.
#[cube(launch)]
pub fn batch_gather_kernel_3<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    #[comptime] stride0: u32,
    #[comptime] stride1: u32,
    #[comptime] stride2: u32,
    #[comptime] batch_stride: u32,
) {
    let batch_idx = ABSOLUTE_POS;
    if batch_idx >= output.len() {
        terminate!();
    }

    let mut linear_idx = batch_idx * batch_stride;
    let idx_base = batch_idx * 3u32;

    let idx0: u32 = u32::cast_from(indices[idx_base]);
    let idx1: u32 = u32::cast_from(indices[idx_base + 1u32]);
    let idx2: u32 = u32::cast_from(indices[idx_base + 2u32]);

    linear_idx = linear_idx + idx0 * stride0 + idx1 * stride1 + idx2 * stride2;
    output[batch_idx] = weights[linear_idx];
}

/// Fused batch gather kernel with 4 index dimensions.
#[cube(launch)]
pub fn batch_gather_kernel_4<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    #[comptime] stride0: u32,
    #[comptime] stride1: u32,
    #[comptime] stride2: u32,
    #[comptime] stride3: u32,
    #[comptime] batch_stride: u32,
) {
    let batch_idx = ABSOLUTE_POS;
    if batch_idx >= output.len() {
        terminate!();
    }

    let mut linear_idx = batch_idx * batch_stride;
    let idx_base = batch_idx * 4u32;

    let idx0: u32 = u32::cast_from(indices[idx_base]);
    let idx1: u32 = u32::cast_from(indices[idx_base + 1u32]);
    let idx2: u32 = u32::cast_from(indices[idx_base + 2u32]);
    let idx3: u32 = u32::cast_from(indices[idx_base + 3u32]);

    linear_idx = linear_idx + idx0 * stride0 + idx1 * stride1 + idx2 * stride2 + idx3 * stride3;
    output[batch_idx] = weights[linear_idx];
}

/// Fused batch gather kernel with 5 index dimensions.
#[cube(launch)]
pub fn batch_gather_kernel_5<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    #[comptime] stride0: u32,
    #[comptime] stride1: u32,
    #[comptime] stride2: u32,
    #[comptime] stride3: u32,
    #[comptime] stride4: u32,
    #[comptime] batch_stride: u32,
) {
    let batch_idx = ABSOLUTE_POS;
    if batch_idx >= output.len() {
        terminate!();
    }

    let mut linear_idx = batch_idx * batch_stride;
    let idx_base = batch_idx * 5u32;

    let idx0: u32 = u32::cast_from(indices[idx_base]);
    let idx1: u32 = u32::cast_from(indices[idx_base + 1u32]);
    let idx2: u32 = u32::cast_from(indices[idx_base + 2u32]);
    let idx3: u32 = u32::cast_from(indices[idx_base + 3u32]);
    let idx4: u32 = u32::cast_from(indices[idx_base + 4u32]);

    linear_idx = linear_idx
        + idx0 * stride0
        + idx1 * stride1
        + idx2 * stride2
        + idx3 * stride3
        + idx4 * stride4;
    output[batch_idx] = weights[linear_idx];
}

/// Fused batch gather kernel with 6 index dimensions.
#[cube(launch)]
pub fn batch_gather_kernel_6<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    #[comptime] stride0: u32,
    #[comptime] stride1: u32,
    #[comptime] stride2: u32,
    #[comptime] stride3: u32,
    #[comptime] stride4: u32,
    #[comptime] stride5: u32,
    #[comptime] batch_stride: u32,
) {
    let batch_idx = ABSOLUTE_POS;
    if batch_idx >= output.len() {
        terminate!();
    }

    let mut linear_idx = batch_idx * batch_stride;
    let idx_base = batch_idx * 6u32;

    let idx0: u32 = u32::cast_from(indices[idx_base]);
    let idx1: u32 = u32::cast_from(indices[idx_base + 1u32]);
    let idx2: u32 = u32::cast_from(indices[idx_base + 2u32]);
    let idx3: u32 = u32::cast_from(indices[idx_base + 3u32]);
    let idx4: u32 = u32::cast_from(indices[idx_base + 4u32]);
    let idx5: u32 = u32::cast_from(indices[idx_base + 5u32]);

    linear_idx = linear_idx
        + idx0 * stride0
        + idx1 * stride1
        + idx2 * stride2
        + idx3 * stride3
        + idx4 * stride4
        + idx5 * stride5;
    output[batch_idx] = weights[linear_idx];
}

// ============================================================================
// Dynamic Fallback Kernel for 7+ Indices
// ============================================================================

/// Dynamic kernel for 7+ indices. Uses runtime loop (slower but flexible).
#[cube(launch)]
pub fn batch_gather_kernel_dynamic<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    strides: &Tensor<I>,
    #[comptime] n_indices: u32,
    #[comptime] batch_stride: u32,
) {
    let batch_idx = ABSOLUTE_POS;
    if batch_idx >= output.len() {
        terminate!();
    }

    let mut linear_idx = batch_idx * batch_stride;
    let idx_base = batch_idx * n_indices;

    for i in 0..n_indices {
        let idx_val = indices[idx_base + i];
        let stride_val = strides[i];
        linear_idx = linear_idx + u32::cast_from(idx_val) * u32::cast_from(stride_val);
    }

    output[batch_idx] = weights[linear_idx];
}

// ============================================================================
// Original Kernels for 1-2 Indices (kept for backwards compatibility)
// ============================================================================

/// Fused batch gather kernel with 2 index dimensions.
///
/// Each thread handles one batch element, computing the linear index
/// from the batch index and multi-dimensional indices.
///
/// # Arguments
/// * `weights` - Flattened weight tensor
/// * `indices` - Flattened index tensor [batch_size * n_indices]
/// * `output` - Output tensor [batch_size]
/// * `stride0` - Stride for first index dimension (comptime)
/// * `stride1` - Stride for second index dimension (comptime)
/// * `batch_stride` - Stride for batch dimension (comptime)
#[cube(launch)]
pub fn batch_gather_kernel_2<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    #[comptime] stride0: u32,
    #[comptime] stride1: u32,
    #[comptime] batch_stride: u32,
) {
    let batch_idx = ABSOLUTE_POS;
    let batch_size = output.len();

    if batch_idx >= batch_size {
        terminate!();
    }

    // Start with batch contribution to linear index
    let mut linear_idx = batch_idx * batch_stride;

    // Add contribution from each index
    // indices layout: [batch_size, 2] flattened to [batch_size * 2]
    let idx_base = batch_idx * 2u32;

    // First index contribution - cast from I to u32
    let idx0_val = indices[idx_base];
    let idx0: u32 = u32::cast_from(idx0_val);
    linear_idx = linear_idx + idx0 * stride0;

    // Second index contribution
    let idx1_val = indices[idx_base + 1u32];
    let idx1: u32 = u32::cast_from(idx1_val);
    linear_idx = linear_idx + idx1 * stride1;

    // Gather the value
    output[batch_idx] = weights[linear_idx];
}

/// Fused batch gather kernel with 1 index dimension.
///
/// Simplified version for single-index gather operations.
#[cube(launch)]
pub fn batch_gather_kernel_1<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    #[comptime] stride0: u32,
    #[comptime] batch_stride: u32,
) {
    let batch_idx = ABSOLUTE_POS;
    let batch_size = output.len();

    if batch_idx >= batch_size {
        terminate!();
    }

    // Start with batch contribution to linear index
    let mut linear_idx = batch_idx * batch_stride;

    // Add contribution from single index
    let idx0_val = indices[batch_idx];
    let idx0: u32 = u32::cast_from(idx0_val);
    linear_idx = linear_idx + idx0 * stride0;

    // Gather the value
    output[batch_idx] = weights[linear_idx];
}
