//! CubeCL kernel definition for fused sigmoid-Bernoulli sampling.
//!
//! This file contains the GPU kernel that computes
//! `uniform < sigmoid(2 * gamma) ? 1.0 : 0.0` in a single GPU pass.
//!
//! # Kernel Logic
//!
//! ```ignore
//! for each idx in parallel:
//!     two_gamma = 2.0 * gamma[idx]
//!     prob = 1.0 / (1.0 + exp(-two_gamma))  // sigmoid
//!
//!     if uniform[idx] < prob:
//!         output[idx] = 1.0
//!     else:
//!         output[idx] = 0.0
//! ```
//!
//! # Performance
//!
//! This kernel eliminates 3 intermediate tensor allocations from the unfused version:
//! 1. `gamma * 2.0` - scaling
//! 2. `sigmoid(...)` - sigmoid computation
//! 3. `uniform.lower_equal(probs)` - comparison result

use cubecl::{cube, prelude::*};

/// Fused sigmoid-Bernoulli sampling kernel.
///
/// Each thread processes one element, computing:
/// 1. Scale gamma by 2: `two_gamma = 2 * gamma[idx]`
/// 2. Compute sigmoid: `prob = 1 / (1 + exp(-two_gamma))`
/// 3. Sample Bernoulli: `output = uniform < prob ? 1.0 : 0.0`
#[cube(launch)]
pub fn sigmoid_bernoulli_kernel<F: Float>(
    gamma: &Tensor<F>,
    uniform: &Tensor<F>,
    output: &mut Tensor<F>,
) {
    let idx = ABSOLUTE_POS;

    if idx >= output.len() {
        terminate!();
    }

    // Compute sigmoid(2 * gamma)
    let two_gamma = F::new(2.0) * gamma[idx];
    let neg_two_gamma = F::new(0.0) - two_gamma;
    let exp_neg = F::exp(neg_two_gamma);
    let prob = F::new(1.0) / (F::new(1.0) + exp_neg);

    // Bernoulli sampling: 1.0 if uniform < prob, else 0.0
    let sample = select(uniform[idx] < prob, F::new(1.0), F::new(0.0));
    output[idx] = sample;
}
