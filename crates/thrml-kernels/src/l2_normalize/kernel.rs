//! CubeCL kernel for L2 normalization.

use cubecl::prelude::*;

/// L2 normalization kernel - normalizes each row to unit length.
///
/// # Arguments
/// * `input` - Input tensor `[n_rows, dim]` (flattened)
/// * `output` - Output tensor `[n_rows, dim]` (flattened)
/// * `dim` - Dimension of each row
#[cube(launch)]
pub fn l2_normalize_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] dim: u32,
) {
    // Epsilon for numerical stability (hardcoded to avoid Hash requirement)
    let epsilon = F::new(1e-8);
    let row_idx = ABSOLUTE_POS;
    let n_rows = output.len() / dim;

    if row_idx >= n_rows {
        terminate!();
    }

    // Compute L2 norm for this row
    let row_start = row_idx * dim;
    let mut sum_sq = F::new(0.0);

    for i in 0..dim {
        let val = input[row_start + i];
        sum_sq += val * val;
    }

    let norm = F::sqrt(sum_sq) + epsilon;

    // Normalize
    for i in 0..dim {
        output[row_start + i] = input[row_start + i] / norm;
    }
}

/// L2 normalization kernel that also outputs norms.
///
/// # Arguments
/// * `input` - Input tensor `[n_rows, dim]` (flattened)
/// * `output` - Output normalized tensor `[n_rows, dim]` (flattened)
/// * `norms` - Output norms `[n_rows]`
/// * `dim` - Dimension of each row
#[cube(launch)]
pub fn l2_normalize_with_norms_kernel<F: Float>(
    input: &Tensor<F>,
    output: &mut Tensor<F>,
    norms: &mut Tensor<F>,
    #[comptime] dim: u32,
) {
    // Epsilon for numerical stability (hardcoded to avoid Hash requirement)
    let epsilon = F::new(1e-8);
    let row_idx = ABSOLUTE_POS;
    let n_rows = norms.len();

    if row_idx >= n_rows {
        terminate!();
    }

    // Compute L2 norm for this row
    let row_start = row_idx * dim;
    let mut sum_sq = F::new(0.0);

    for i in 0..dim {
        let val = input[row_start + i];
        sum_sq += val * val;
    }

    let norm = F::sqrt(sum_sq) + epsilon;
    norms[row_idx] = norm;

    // Normalize
    for i in 0..dim {
        output[row_start + i] = input[row_start + i] / norm;
    }
}
