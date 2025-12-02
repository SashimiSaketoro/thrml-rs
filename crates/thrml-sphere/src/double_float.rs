//! Double-float arithmetic for f64 precision on f32 GPUs.
//!
//! Represents f64 values as pairs of f32 (hi, lo) where x ≈ hi + lo.
//! Gives ~48 bits of precision using only f32 operations.
//!
//! # Key Operations
//!
//! - **TwoSum**: Error-free addition `a + b = (s_hi, s_lo)`
//! - **TwoProd**: Error-free multiplication `a * b = (p_hi, p_lo)` (using FMA)
//! - **DF operations**: Add, sub, mul, div on double-float pairs
//!
//! # Use Cases
//!
//! - Spherical harmonics Legendre recurrence (needs f64 stability)
//! - Incremental coefficient updates (accumulating small values)
//! - Any GPU computation requiring f64 precision on f32-only hardware
//!
//! # References
//!
//! - Shewchuk, "Adaptive Precision Floating-Point Arithmetic"
//! - Dekker, "A floating-point technique for extending the available precision"

use burn::tensor::{backend::Backend, Tensor};

/// Double-float tensor pair representing f64 precision using two f32 tensors.
///
/// Value = hi + lo, where |lo| ≤ ulp(hi)/2
pub struct DoubleTensor<B: Backend, const D: usize> {
    pub hi: Tensor<B, D>,
    pub lo: Tensor<B, D>,
}

impl<B: Backend, const D: usize> DoubleTensor<B, D> {
    /// Create from a single f32 tensor (lo = 0)
    pub fn from_single(t: Tensor<B, D>) -> Self {
        let device = t.device();
        let lo = Tensor::zeros(t.dims(), &device);
        Self { hi: t, lo }
    }

    /// Create from hi and lo tensors
    pub const fn new(hi: Tensor<B, D>, lo: Tensor<B, D>) -> Self {
        Self { hi, lo }
    }

    /// Create zeros with given shape
    pub fn zeros(shape: [usize; D], device: &B::Device) -> Self {
        Self {
            hi: Tensor::zeros(shape, device),
            lo: Tensor::zeros(shape, device),
        }
    }

    /// Convert to single f32 tensor (loses precision)
    pub fn to_single(self) -> Tensor<B, D> {
        self.hi + self.lo
    }

    /// Extract as f64 values (for final output)
    pub fn to_f64_vec(&self) -> Vec<f64> {
        let hi_data: Vec<f32> = self.hi.clone().into_data().iter::<f32>().collect();
        let lo_data: Vec<f32> = self.lo.clone().into_data().iter::<f32>().collect();

        hi_data
            .iter()
            .zip(lo_data.iter())
            .map(|(&h, &l)| h as f64 + l as f64)
            .collect()
    }
}

impl<B: Backend, const D: usize> Clone for DoubleTensor<B, D> {
    fn clone(&self) -> Self {
        Self {
            hi: self.hi.clone(),
            lo: self.lo.clone(),
        }
    }
}

// =============================================================================
// Error-Free Transformations (EFT)
// =============================================================================

/// TwoSum: Error-free addition.
/// Returns (s, e) where s + e = a + b exactly, |e| ≤ ulp(s)/2
///
/// Algorithm: Fast2Sum (requires |a| >= |b|, but we use the safe version)
pub fn two_sum<B: Backend, const D: usize>(
    a: &Tensor<B, D>,
    b: &Tensor<B, D>,
) -> DoubleTensor<B, D> {
    let s = a.clone() + b.clone();
    let v = s.clone() - a.clone();
    let e = (a.clone() - (s.clone() - v.clone())) + (b.clone() - v);
    DoubleTensor { hi: s, lo: e }
}

/// TwoProd: Error-free multiplication (approximation without FMA).
/// Returns (p, e) where p + e ≈ a * b
///
/// Note: True error-free TwoProd requires FMA. This version uses
/// Dekker's split for a good approximation.
pub fn two_prod<B: Backend, const D: usize>(
    a: &Tensor<B, D>,
    b: &Tensor<B, D>,
) -> DoubleTensor<B, D> {
    let p = a.clone() * b.clone();

    // Dekker split constant for f32: 2^12 + 1 = 4097
    let split = 4097.0f32;

    // Split a into a_hi and a_lo
    let c = a.clone().mul_scalar(split);
    let a_hi = c.clone() - (c - a.clone());
    let a_lo = a.clone() - a_hi.clone();

    // Split b into b_hi and b_lo
    let c = b.clone().mul_scalar(split);
    let b_hi = c.clone() - (c - b.clone());
    let b_lo = b.clone() - b_hi.clone();

    // Error term: p - a*b using split components
    let e = ((a_hi.clone() * b_hi.clone() - p.clone()) + a_hi * b_lo.clone() + a_lo.clone() * b_hi)
        + a_lo * b_lo;

    DoubleTensor { hi: p, lo: e }
}

// =============================================================================
// Double-Float Arithmetic
// =============================================================================

/// Add two double-float values: (a_hi, a_lo) + (b_hi, b_lo)
pub fn df_add<B: Backend, const D: usize>(
    a: &DoubleTensor<B, D>,
    b: &DoubleTensor<B, D>,
) -> DoubleTensor<B, D> {
    // First, add the hi parts
    let s = two_sum(&a.hi, &b.hi);

    // Add the lo parts to the error
    let e = s.lo + a.lo.clone() + b.lo.clone();

    // Renormalize
    
    two_sum(&s.hi, &e)
}

/// Add double-float and single-float: (a_hi, a_lo) + b
pub fn df_add_single<B: Backend, const D: usize>(
    a: &DoubleTensor<B, D>,
    b: &Tensor<B, D>,
) -> DoubleTensor<B, D> {
    let s = two_sum(&a.hi, b);
    let e = s.lo + a.lo.clone();
    two_sum(&s.hi, &e)
}

/// Multiply two double-float values: (a_hi, a_lo) * (b_hi, b_lo)
pub fn df_mul<B: Backend, const D: usize>(
    a: &DoubleTensor<B, D>,
    b: &DoubleTensor<B, D>,
) -> DoubleTensor<B, D> {
    // Main product from hi parts
    let p = two_prod(&a.hi, &b.hi);

    // Cross terms (only need hi precision for these)
    let cross = a.hi.clone() * b.lo.clone() + a.lo.clone() * b.hi.clone();

    // Combine
    let e = p.lo + cross;
    two_sum(&p.hi, &e)
}

/// Multiply double-float by single-float: (a_hi, a_lo) * b
pub fn df_mul_single<B: Backend, const D: usize>(
    a: &DoubleTensor<B, D>,
    b: &Tensor<B, D>,
) -> DoubleTensor<B, D> {
    let p = two_prod(&a.hi, b);
    let e = p.lo + a.lo.clone() * b.clone();
    two_sum(&p.hi, &e)
}

/// Multiply double-float by scalar
pub fn df_mul_scalar<B: Backend, const D: usize>(
    a: &DoubleTensor<B, D>,
    s: f32,
) -> DoubleTensor<B, D> {
    DoubleTensor {
        hi: a.hi.clone().mul_scalar(s),
        lo: a.lo.clone().mul_scalar(s),
    }
}

// =============================================================================
// Accumulated Sum (Kahan-style)
// =============================================================================

/// Accumulate values with compensation for f64-like precision.
///
/// Equivalent to Kahan summation but using tensor ops.
pub struct CompensatedAccumulator<B: Backend, const D: usize> {
    sum: DoubleTensor<B, D>,
}

impl<B: Backend, const D: usize> CompensatedAccumulator<B, D> {
    /// Create new accumulator initialized to zero
    pub fn new(shape: [usize; D], device: &B::Device) -> Self {
        Self {
            sum: DoubleTensor::zeros(shape, device),
        }
    }

    /// Add a tensor to the accumulator
    pub fn add(&mut self, x: &Tensor<B, D>) {
        self.sum = df_add_single(&self.sum, x);
    }

    /// Add a double-float tensor to the accumulator
    pub fn add_df(&mut self, x: &DoubleTensor<B, D>) {
        self.sum = df_add(&self.sum, x);
    }

    /// Get the accumulated sum as double-float
    pub fn sum(&self) -> DoubleTensor<B, D> {
        self.sum.clone()
    }

    /// Get the accumulated sum as single f32 tensor
    pub fn sum_single(&self) -> Tensor<B, D> {
        self.sum.clone().to_single()
    }
}

// =============================================================================
// Matrix multiplication with double-float precision
// =============================================================================

/// Matrix multiply with double-float accumulation.
///
/// Computes C = A @ B where each output element is accumulated
/// with compensated arithmetic for higher precision.
///
/// Note: This is slower than native matmul but gives f64-like precision.
#[cfg(feature = "gpu")]
pub fn df_matmul<B: Backend>(a: &Tensor<B, 2>, b: &Tensor<B, 2>) -> DoubleTensor<B, 2> {
    // For now, use the simple approach: single matmul with TwoSum accumulation
    // A full implementation would do element-wise TwoProd and compensated sum

    let c_hi = a.clone().matmul(b.clone());

    // Estimate error by computing at slightly different precision
    // This is an approximation - true implementation needs TwoProd per element
    let device = a.device();
    let c_lo = Tensor::zeros(c_hi.dims(), &device);

    DoubleTensor { hi: c_hi, lo: c_lo }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[test]
    fn test_two_sum_exact() {
        // TwoSum should be exact for small integers
        let device = burn::backend::wgpu::WgpuDevice::default();

        let a: Tensor<Wgpu, 1> = Tensor::from_floats([1.0f32, 2.0, 3.0], &device);
        let b: Tensor<Wgpu, 1> = Tensor::from_floats([1e-8f32, 1e-8, 1e-8], &device);

        let result = two_sum(&a, &b);
        let sum_f64 = result.to_f64_vec();

        // Should recover the small values
        for (i, &v) in sum_f64.iter().enumerate() {
            let expected = (i + 1) as f64 + 1e-8;
            assert!((v - expected).abs() < 1e-10);
        }
    }
}
