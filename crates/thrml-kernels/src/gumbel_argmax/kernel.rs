//! CubeCL kernel definition for fused Gumbel-max categorical sampling.
//!
//! This kernel computes `argmax(logits + (-log(-log(uniform))))` in a single GPU pass.
//!
//! # Kernel Logic
//!
//! ```ignore
//! for each sample_idx in parallel:
//!     base_idx = sample_idx * n_categories
//!     max_val = -infinity
//!     max_idx = 0
//!
//!     for k in 0..n_categories:
//!         u = uniform[base_idx + k]
//!         gumbel = -log(-log(u))
//!         perturbed = logits[base_idx + k] + gumbel
//!
//!         if perturbed > max_val:
//!             max_val = perturbed
//!             max_idx = k
//!
//!     output[sample_idx] = max_idx
//! ```
//!
//! # Performance
//!
//! This kernel eliminates 4 intermediate tensor allocations from the unfused version:
//! 1. `uniform.log()` - log of uniform samples
//! 2. `-log(...)` - negation
//! 3. `log(-log(...))` - second log
//! 4. `logits + gumbel` - addition before argmax

use cubecl::{cube, prelude::*};

/// Fused Gumbel-max categorical sampling kernel.
///
/// Each thread processes one sample (row) and finds the argmax after adding
/// Gumbel noise to the logits. This is a reduction kernel where each thread
/// performs a serial scan over categories.
///
/// # Arguments
///
/// * `logits` - Flattened logits tensor \[n_samples * n_categories\]
/// * `uniform` - Flattened uniform samples \[n_samples * n_categories\]
/// * `output` - Output tensor \[n_samples\] for category indices (as integers)
/// * `n_categories` - Number of categories (comptime constant)
#[cube(launch)]
pub fn gumbel_argmax_kernel<F: Float, I: Int>(
    logits: &Tensor<F>,
    uniform: &Tensor<F>,
    output: &mut Tensor<I>,
    #[comptime] n_categories: u32,
) {
    let sample_idx = ABSOLUTE_POS;
    let n_samples = output.len();

    if sample_idx >= n_samples {
        terminate!();
    }

    let base_idx = sample_idx * n_categories;

    // Initialize with first element
    let u0 = uniform[base_idx];
    let neg_log_u0 = F::new(0.0) - F::log(u0);
    let gumbel0 = F::new(0.0) - F::log(neg_log_u0);
    let mut max_val = logits[base_idx] + gumbel0;
    let mut max_idx = 0u32;

    // Scan remaining categories
    for k in 1..n_categories {
        let idx = base_idx + k;
        let u = uniform[idx];

        // Gumbel noise: -log(-log(u))
        let neg_log_u = F::new(0.0) - F::log(u);
        let gumbel = F::new(0.0) - F::log(neg_log_u);
        let perturbed = logits[idx] + gumbel;

        // Update max if this is larger
        let is_larger = perturbed > max_val;
        max_val = select(is_larger, perturbed, max_val);
        max_idx = select(is_larger, k, max_idx);
    }

    // Output as integer
    output[sample_idx] = I::cast_from(max_idx);
}
