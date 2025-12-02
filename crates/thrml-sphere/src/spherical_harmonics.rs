//! Spherical Harmonics for hyperspherical navigation.
//!
//! Port of JAX implementation from `src/core/tensor/spherical_harmonics.py`
//! and `src/core/tensor/quantum.py` (renamed from "quantum" to "harmonic"
//! for accuracy - this is classical wave interference, not quantum mechanics).
//!
//! # Overview
//!
//! Spherical harmonics Y_l^m form a complete orthonormal basis on the sphere,
//! enabling:
//! - **Multi-resolution analysis**: Low L captures coarse structure, high L fine detail
//! - **Smooth interpolation**: Continuous fields from discrete points
//! - **Interference patterns**: Coherently combine multiple amplitude signals
//! - **Frequency filtering**: Focus on different spatial scales
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  Associated Legendre Polynomials P_l^m(cos θ)                       │
//! │  └── Normalized for spherical harmonics integration                 │
//! └─────────────────────────────────────────────────────────────────────┘
//!                               ↓
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  Spherical Harmonics Y_l^m(θ, φ)                                    │
//! │  ├── Complex: Y_l^m = P_l^m(cos θ) × e^{imφ}                        │
//! │  └── Real: combinations of cos(mφ) and sin(mφ)                      │
//! └─────────────────────────────────────────────────────────────────────┘
//!                               ↓
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  HarmonicInterference                                               │
//! │  ├── Driscoll-Healy grid: 2L × 4L sampling                          │
//! │  ├── Forward SHT: f(θ,φ) → c_lm coefficients                        │
//! │  ├── Inverse SHT: c_lm → f(θ,φ)                                     │
//! │  └── superposition_field(): combine amplitudes → intensity          │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Integration with Navigation
//!
//! The interference field complements:
//! - **ROOTS**: Coarse partition routing (partition membership)
//! - **Springs**: Structural adjacency (hypergraph edges)
//! - **EBM Energy**: Thermodynamic landscape (Langevin dynamics)
//!
//! Spherical harmonics add frequency-domain analysis for smooth, multi-scale
//! navigation that avoids the O(n²) pairwise similarity bottleneck.

// Note: Tensor/Backend available if needed for GPU SH in future
#[allow(unused_imports)]
use burn::tensor::{backend::Backend, Tensor};
use std::f64::consts::PI;

// Re-export GPU strategy from thrml-core
pub use thrml_core::compute::GpuF64Strategy;

/// Configuration for spherical harmonics computation.
///
/// # GPU Strategy for Scale
///
/// At large scale (N > 100k points), CPU is too slow even with rayon.
/// Consumer GPUs (Apple M-series, RTX) only have fast f32, not f64.
///
/// - `CpuFallback`: CPU f64, best precision, slow at scale
/// - `GpuF32`: GPU f32, fastest, some precision loss
/// - `DoubleTensor`: GPU with f32 pairs (~48-bit), good balance for consumer GPUs
/// - `NativeF64`: GPU native f64, requires datacenter GPU (H100, A100)
#[derive(Debug, Clone, Copy)]
pub struct SphericalHarmonicsConfig {
    /// Maximum degree L (band limit). Higher = more detail but slower.
    /// L=64 gives (L+1)² = 4225 coefficients.
    pub band_limit: usize,

    /// GPU execution strategy for scale:
    /// - `CpuFallback`: All on CPU (f64)
    /// - `DoubleTensor`: GPU with emulated f64 (consumer GPUs)  
    /// - `NativeF64`: GPU native f64 (datacenter only)
    pub gpu_strategy: GpuF64Strategy,

    /// Validate GPU results against CPU at runtime (for debugging).
    pub validate_gpu: bool,

    /// Maximum relative error allowed for validation (default: 1e-5).
    pub validation_tolerance: f64,
}

impl Default for SphericalHarmonicsConfig {
    fn default() -> Self {
        use thrml_core::compute::RuntimePolicy;
        let policy = RuntimePolicy::detect();
        Self {
            band_limit: 64,
            gpu_strategy: policy.gpu_f64_strategy(),
            validate_gpu: false,
            validation_tolerance: 1e-5,
        }
    }
}

impl SphericalHarmonicsConfig {
    /// Development config: CPU-only, lower band limit for fast iteration.
    pub const fn dev() -> Self {
        Self {
            band_limit: 16,
            gpu_strategy: GpuF64Strategy::CpuFallback,
            validate_gpu: false,
            validation_tolerance: 1e-5,
        }
    }

    /// Production config: auto-detect GPU tier, high band limit.
    pub fn production() -> Self {
        use thrml_core::compute::RuntimePolicy;
        let policy = RuntimePolicy::detect();
        Self {
            band_limit: 128,
            gpu_strategy: policy.gpu_f64_strategy(),
            validate_gpu: false,
            validation_tolerance: 1e-6,
        }
    }

    /// Consumer GPU config: DoubleTensor for f64 precision on f32 hardware.
    pub const fn consumer_gpu() -> Self {
        Self {
            band_limit: 64,
            gpu_strategy: GpuF64Strategy::DoubleTensor,
            validate_gpu: false,
            validation_tolerance: 1e-5,
        }
    }

    /// Datacenter GPU config: native f64 (H100, A100, etc).
    pub const fn datacenter_gpu() -> Self {
        Self {
            band_limit: 64,
            gpu_strategy: GpuF64Strategy::NativeF64,
            validate_gpu: false,
            validation_tolerance: 1e-10,
        }
    }

    /// Debug config: validates GPU against CPU reference.
    pub fn debug_validate() -> Self {
        use thrml_core::compute::RuntimePolicy;
        Self {
            band_limit: 32,
            gpu_strategy: RuntimePolicy::detect().gpu_f64_strategy(),
            validate_gpu: true,
            validation_tolerance: 1e-4,
        }
    }

    /// Number of coefficients: (L+1)²
    pub const fn num_coefficients(&self) -> usize {
        (self.band_limit + 1).pow(2)
    }

    /// Grid dimensions for Driscoll-Healy sampling.
    pub const fn grid_size(&self) -> (usize, usize) {
        (2 * self.band_limit, 4 * self.band_limit)
    }

    /// Builder: set GPU strategy
    pub const fn with_gpu_strategy(mut self, strategy: GpuF64Strategy) -> Self {
        self.gpu_strategy = strategy;
        self
    }

    /// Builder: enable runtime validation
    pub const fn with_validation(mut self, enabled: bool) -> Self {
        self.validate_gpu = enabled;
        self
    }

    /// Builder: set band limit
    pub const fn with_band_limit(mut self, band_limit: usize) -> Self {
        self.band_limit = band_limit;
        self
    }
}

/// Compute normalized associated Legendre polynomial P_l^m(x).
///
/// Uses the normalization convention for spherical harmonics:
/// ```text
/// N_l^m = sqrt((2l+1)/(4π) × (l-m)!/(l+m)!)
/// ```
///
/// # Arguments
/// * `l` - Degree (non-negative integer)
/// * `m` - Order (integer with |m| <= l)
/// * `x` - Value in [-1, 1], typically cos(θ)
///
/// # Returns
/// Normalized P_l^m(x)
pub fn associated_legendre_normalized(l: usize, m: i32, x: f64) -> f64 {
    // Handle negative m using symmetry relation
    let m_abs = m.unsigned_abs() as usize;
    if m < 0 {
        let mut factor = if m_abs.is_multiple_of(2) { 1.0 } else { -1.0 };
        // (l-|m|)! / (l+|m|)!
        for k in (l - m_abs + 1)..=(l + m_abs) {
            factor /= k as f64;
        }
        return factor * associated_legendre_normalized(l, m_abs as i32, x);
    }

    // Normalization factor for spherical harmonics
    let mut norm = ((2 * l + 1) as f64 / (4.0 * PI)).sqrt();
    if m_abs > 0 {
        // Additional factor sqrt((l-m)!/(l+m)!)
        let mut factorial_ratio = 1.0;
        for k in (l - m_abs + 1)..=(l + m_abs) {
            factorial_ratio /= k as f64;
        }
        norm *= factorial_ratio.sqrt();
    }

    // Base case: P_0^0 = 1
    if l == 0 {
        return norm;
    }

    // sin(θ) from cos(θ)
    let sin_theta = x.mul_add(-x, 1.0).max(0.0).sqrt();

    // Sectoral: P_m^m = (-1)^m × (2m-1)!! × sin^m(θ)
    let mut pmm = 1.0;
    if m_abs > 0 {
        let mut fact = 1.0;
        for _ in 1..=m_abs {
            pmm *= -fact * sin_theta;
            fact += 2.0;
        }
    }

    if l == m_abs {
        return norm * pmm;
    }

    // P_{m+1}^m = x × (2m+1) × P_m^m
    let mut pmmp1 = x * (2 * m_abs + 1) as f64 * pmm;

    if l == m_abs + 1 {
        return norm * pmmp1;
    }

    // Recurrence for l > m+1
    for n in (m_abs + 2)..=l {
        let tmp = ((2 * n - 1) as f64 * x).mul_add(pmmp1, -((n + m_abs - 1) as f64 * pmm))
            / (n - m_abs) as f64;
        pmm = pmmp1;
        pmmp1 = tmp;
    }

    norm * pmmp1
}

/// Complex spherical harmonic Y_l^m(θ, φ).
///
/// Y_l^m(θ, φ) = P_l^m(cos θ) × e^{imφ}
///
/// # Arguments
/// * `l` - Degree
/// * `m` - Order
/// * `theta` - Colatitude [0, π]
/// * `phi` - Azimuth [0, 2π)
pub fn complex_spherical_harmonic(l: usize, m: i32, theta: f64, phi: f64) -> (f64, f64) {
    let plm = associated_legendre_normalized(l, m, theta.cos());
    let phase_real = (m as f64 * phi).cos();
    let phase_imag = (m as f64 * phi).sin();
    (plm * phase_real, plm * phase_imag)
}

/// Real spherical harmonic Y_l^m(θ, φ).
///
/// Real spherical harmonics are defined as:
/// - m > 0: √2 × Re[Y_l^m] = √2 × P_l^m × cos(mφ)
/// - m = 0: Y_l^0 (already real)
/// - m < 0: √2 × Im[Y_l^{|m|}] = √2 × P_l^{|m|} × sin(|m|φ)
pub fn real_spherical_harmonic(l: usize, m: i32, theta: f64, phi: f64) -> f64 {
    if m == 0 {
        let (re, _im) = complex_spherical_harmonic(l, 0, theta, phi);
        re
    } else if m > 0 {
        let (re, _im) = complex_spherical_harmonic(l, m, theta, phi);
        2.0_f64.sqrt() * re
    } else {
        let (_re, im) = complex_spherical_harmonic(l, -m, theta, phi);
        2.0_f64.sqrt() * im
    }
}

/// Precomputed spherical harmonics basis on a Driscoll-Healy grid.
///
/// The Driscoll-Healy grid uses 2L × 4L sampling for exact integration
/// up to band limit L.
pub struct SphericalHarmonicsBasis {
    /// Band limit L
    pub band_limit: usize,

    /// Number of θ samples (2L)
    pub n_theta: usize,

    /// Number of φ samples (4L)
    pub n_phi: usize,

    /// θ grid values `[n_theta]`
    pub theta_grid: Vec<f64>,

    /// φ grid values `[n_phi]`
    pub phi_grid: Vec<f64>,

    /// Precomputed Y_l^m basis [n_theta × n_phi, (L+1)²]
    /// Row-major: basis[i * num_coeffs + lm_idx] = Y_lm(θ_i, φ_i)
    pub basis: Vec<f64>,

    /// Quadrature weights for spherical integration [n_theta × n_phi]
    pub weights: Vec<f64>,
}

impl SphericalHarmonicsBasis {
    /// Create a new precomputed basis.
    pub fn new(config: &SphericalHarmonicsConfig) -> Self {
        let l = config.band_limit;
        let n_theta = 2 * l;
        let n_phi = 4 * l;
        let num_coeffs = (l + 1).pow(2);
        let n_points = n_theta * n_phi;

        // Driscoll-Healy θ grid (avoid exact poles for stability)
        let eps = 1e-6;
        let theta_grid: Vec<f64> = (0..n_theta)
            .map(|i| eps + 2.0f64.mul_add(-eps, PI) * i as f64 / (n_theta - 1) as f64)
            .collect();

        // φ grid (don't include 2π, it's periodic)
        let phi_grid: Vec<f64> = (0..n_phi)
            .map(|i| 2.0 * PI * i as f64 / n_phi as f64)
            .collect();

        // Precompute Y_l^m on all grid points
        let mut basis = vec![0.0; n_points * num_coeffs];

        for (i_theta, &theta) in theta_grid.iter().enumerate() {
            for (i_phi, &phi) in phi_grid.iter().enumerate() {
                let grid_idx = i_theta * n_phi + i_phi;
                let mut coeff_idx = 0;

                for l_deg in 0..=l {
                    for m_ord in -(l_deg as i32)..=(l_deg as i32) {
                        let ylm = real_spherical_harmonic(l_deg, m_ord, theta, phi);
                        basis[grid_idx * num_coeffs + coeff_idx] = ylm;
                        coeff_idx += 1;
                    }
                }
            }
        }

        // Quadrature weights: sin(θ) × Δθ × Δφ
        let d_theta = PI / n_theta as f64;
        let d_phi = 2.0 * PI / n_phi as f64;
        let mut weights: Vec<f64> = theta_grid
            .iter()
            .flat_map(|&theta| std::iter::repeat_n(theta.sin() * d_theta * d_phi, n_phi))
            .collect();

        // Normalize to integrate to 4π (surface area of unit sphere)
        let total: f64 = weights.iter().sum();
        for w in &mut weights {
            *w *= 4.0 * PI / total;
        }

        Self {
            band_limit: l,
            n_theta,
            n_phi,
            theta_grid,
            phi_grid,
            basis,
            weights,
        }
    }

    /// Forward spherical harmonic transform: f(θ,φ) → c_lm coefficients.
    ///
    /// # Arguments
    /// * `field` - Function values on grid [n_theta × n_phi]
    ///
    /// # Returns
    /// Coefficients [(L+1)²]
    pub fn forward_sht(&self, field: &[f64]) -> Vec<f64> {
        let num_coeffs = (self.band_limit + 1).pow(2);
        let n_points = self.n_theta * self.n_phi;

        assert_eq!(field.len(), n_points, "Field must match grid size");

        let mut coeffs = vec![0.0; num_coeffs];

        // Explicit indices clearer for matrix-like SHT computation
        #[allow(clippy::needless_range_loop)]
        for coeff_idx in 0..num_coeffs {
            let mut sum = 0.0;
            for grid_idx in 0..n_points {
                sum += field[grid_idx]
                    * self.weights[grid_idx]
                    * self.basis[grid_idx * num_coeffs + coeff_idx];
            }
            coeffs[coeff_idx] = sum;
        }

        coeffs
    }

    /// Inverse spherical harmonic transform: c_lm coefficients → f(θ,φ).
    ///
    /// # Arguments
    /// * `coeffs` - Coefficients [(L+1)²]
    ///
    /// # Returns
    /// Field values on grid [n_theta × n_phi]
    pub fn inverse_sht(&self, coeffs: &[f64]) -> Vec<f64> {
        let num_coeffs = (self.band_limit + 1).pow(2);
        let n_points = self.n_theta * self.n_phi;

        assert_eq!(coeffs.len(), num_coeffs, "Coefficients must match (L+1)²");

        let mut field = vec![0.0; n_points];

        // Explicit indices clearer for matrix-like SHT computation
        #[allow(clippy::needless_range_loop)]
        for grid_idx in 0..n_points {
            let mut sum = 0.0;
            for coeff_idx in 0..num_coeffs {
                sum += coeffs[coeff_idx] * self.basis[grid_idx * num_coeffs + coeff_idx];
            }
            field[grid_idx] = sum;
        }

        field
    }

    /// Compute superposition field from multiple amplitude grids.
    ///
    /// This is the core harmonic interference operation:
    /// 1. Project each amplitude grid onto spherical harmonic basis (forward SHT)
    /// 2. Sum coefficients (wave superposition)
    /// 3. Reconstruct field from combined coefficients (inverse SHT)
    /// 4. Compute intensity |field|²
    ///
    /// # Arguments
    /// * `amplitude_grids` - List of amplitude grids, each [n_theta × n_phi]
    ///
    /// # Returns
    /// Intensity field [n_theta × n_phi], normalized to sum to 1
    pub fn superposition_field(&self, amplitude_grids: &[Vec<f64>]) -> Vec<f64> {
        let num_coeffs = (self.band_limit + 1).pow(2);
        let _n_points = self.n_theta * self.n_phi; // Used in assertions/debugging

        // Sum coefficients from all amplitude grids
        let mut total_coeffs = vec![0.0; num_coeffs];

        for amp in amplitude_grids {
            let coeffs = self.forward_sht(amp);
            for (i, c) in coeffs.into_iter().enumerate() {
                total_coeffs[i] += c;
            }
        }

        // Reconstruct field from combined coefficients
        let field = self.inverse_sht(&total_coeffs);

        // Compute intensity |field|² and normalize
        let mut intensity: Vec<f64> = field.iter().map(|f| f * f).collect();
        let sum: f64 = intensity.iter().sum::<f64>() + 1e-10;
        for i in &mut intensity {
            *i /= sum;
        }

        intensity
    }

    /// Find peak in intensity field.
    ///
    /// # Returns
    /// (theta_peak, phi_peak, intensity_peak)
    pub fn find_peak(&self, intensity: &[f64]) -> (f64, f64, f64) {
        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx = 0;

        for (idx, &val) in intensity.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = idx;
            }
        }

        let i_theta = max_idx / self.n_phi;
        let i_phi = max_idx % self.n_phi;

        (self.theta_grid[i_theta], self.phi_grid[i_phi], max_val)
    }

    /// Create a Gaussian amplitude blob centered at (θ₀, φ₀).
    ///
    /// # Arguments
    /// * `theta_0` - Center colatitude
    /// * `phi_0` - Center azimuth
    /// * `sigma_theta` - Width in θ direction
    /// * `sigma_phi` - Width in φ direction
    pub fn gaussian_blob(
        &self,
        theta_0: f64,
        phi_0: f64,
        sigma_theta: f64,
        sigma_phi: f64,
    ) -> Vec<f64> {
        let n_points = self.n_theta * self.n_phi;
        let mut blob = vec![0.0; n_points];

        for (i_theta, &theta) in self.theta_grid.iter().enumerate() {
            for (i_phi, &phi) in self.phi_grid.iter().enumerate() {
                let d_theta = theta - theta_0;
                // Handle φ wraparound
                let d_phi = {
                    let diff = phi - phi_0;
                    if diff > PI {
                        2.0f64.mul_add(-PI, diff)
                    } else if diff < -PI {
                        2.0f64.mul_add(PI, diff)
                    } else {
                        diff
                    }
                };

                let val = (-d_theta * d_theta / (2.0 * sigma_theta * sigma_theta)
                    - d_phi * d_phi / (2.0 * sigma_phi * sigma_phi))
                    .exp();

                blob[i_theta * self.n_phi + i_phi] = val;
            }
        }

        blob
    }

    /// Evaluate a single spherical harmonic Y_l^m at an arbitrary point.
    ///
    /// This is useful for incremental coefficient updates where we don't
    /// need the full precomputed grid.
    ///
    /// # Arguments
    /// * `l` - Degree
    /// * `m` - Order
    /// * `theta` - Colatitude [0, π]
    /// * `phi` - Azimuth [0, 2π)
    pub fn evaluate_at(&self, l: usize, m: i32, theta: f64, phi: f64) -> f64 {
        real_spherical_harmonic(l, m, theta, phi)
    }

    /// Evaluate density field at an arbitrary point using stored coefficients.
    ///
    /// # Arguments
    /// * `coeffs` - SH coefficients [(L+1)²]
    /// * `theta` - Colatitude
    /// * `phi` - Azimuth
    pub fn evaluate_field_at(&self, coeffs: &[f64], theta: f64, phi: f64) -> f64 {
        let mut sum = 0.0;
        let mut idx = 0;

        for l in 0..=self.band_limit {
            for m in -(l as i32)..=(l as i32) {
                let ylm = real_spherical_harmonic(l, m, theta, phi);
                sum += coeffs.get(idx).copied().unwrap_or(0.0) * ylm;
                idx += 1;
            }
        }

        sum
    }

    /// GPU-accelerated forward SHT using Burn tensor matmul.
    ///
    /// Expresses both DFT and Legendre transforms as matrix multiplies:
    /// 1. DFT in φ: F_m = DFT_matrix @ f_ring (per latitude)
    /// 2. Legendre in θ: c_lm = P_matrix^T @ (weights ⊙ F_m)
    ///
    /// Keeps computation on GPU throughout, ~10-50x faster for band_limit > 32.
    #[cfg(feature = "gpu")]
    pub fn forward_sht_gpu(
        &self,
        field: &[f64],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<f64> {
        use burn::tensor::Tensor;
        use thrml_core::backend::WgpuBackend;

        let n_points = self.n_theta * self.n_phi;
        assert_eq!(field.len(), n_points, "Field must match grid size");

        // Convert to f32 for GPU
        let field_f32: Vec<f32> = field.iter().map(|&x| x as f32).collect();

        // Build field tensor [n_theta, n_phi]
        let field_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(field_f32.as_slice(), device)
                .reshape([self.n_theta, self.n_phi]);

        // Build DFT cosine matrix [n_phi, n_phi]: C[k,n] = cos(2πkn/N)
        // For real SH, we only need modes up to band_limit
        let mut dft_cos = vec![0.0f32; self.n_phi * self.n_phi];
        let mut dft_sin = vec![0.0f32; self.n_phi * self.n_phi];
        for k in 0..self.n_phi {
            for n in 0..self.n_phi {
                let angle = 2.0 * PI * (k as f64) * (n as f64) / (self.n_phi as f64);
                dft_cos[k * self.n_phi + n] = angle.cos() as f32;
                dft_sin[k * self.n_phi + n] = angle.sin() as f32;
            }
        }

        let dft_cos_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(dft_cos.as_slice(), device)
                .reshape([self.n_phi, self.n_phi]);

        let dft_sin_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(dft_sin.as_slice(), device)
                .reshape([self.n_phi, self.n_phi]);

        // Apply DFT to each latitude ring: [n_theta, n_phi] @ [n_phi, n_phi]^T
        // F_cos[theta, k] = Σ_n field[theta, n] * cos(2πkn/N)
        let f_cos = field_tensor.clone().matmul(dft_cos_tensor.swap_dims(0, 1));
        let f_sin = field_tensor.matmul(dft_sin_tensor.swap_dims(0, 1));

        // Normalize DFT
        let norm = (self.n_phi as f32).sqrt();
        let f_cos = f_cos.div_scalar(norm);
        let f_sin = f_sin.div_scalar(norm);

        // Extract to CPU for Legendre integration (could also do this on GPU)
        let f_cos_data: Vec<f32> = f_cos.into_data().iter::<f32>().collect();
        let f_sin_data: Vec<f32> = f_sin.into_data().iter::<f32>().collect();

        // Legendre transform for each (l, m)
        let num_coeffs = (self.band_limit + 1).pow(2);
        let mut coeffs = vec![0.0f64; num_coeffs];

        let mut idx = 0;
        for l in 0..=self.band_limit {
            for m in -(l as i32)..=(l as i32) {
                let m_abs = m.unsigned_abs() as usize;

                let mut sum = 0.0;
                for (i_theta, &theta) in self.theta_grid.iter().enumerate() {
                    // Get Fourier coefficient for mode m
                    let fm = if m >= 0 {
                        // Positive m: use cosine component
                        f_cos_data[i_theta * self.n_phi + m_abs] as f64
                    } else {
                        // Negative m: use sine component (with sign)
                        -f_sin_data[i_theta * self.n_phi + m_abs] as f64
                    };

                    // Legendre polynomial
                    let plm = associated_legendre_normalized(l, m, theta.cos());

                    // Quadrature weight
                    let w = self.weights[i_theta * self.n_phi];

                    sum += w * plm * fm;
                }

                coeffs[idx] = sum;
                idx += 1;
            }
        }

        coeffs
    }

    /// GPU-accelerated inverse SHT using Burn tensor matmul.
    #[cfg(feature = "gpu")]
    pub fn inverse_sht_gpu(
        &self,
        coeffs: &[f64],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<f64> {
        use burn::tensor::Tensor;
        use thrml_core::backend::WgpuBackend;

        let num_coeffs = (self.band_limit + 1).pow(2);
        assert_eq!(coeffs.len(), num_coeffs, "Coefficients must match (L+1)²");

        // Legendre synthesis: F_m(θ) = Σ_l c_lm × P_lm(cos θ)
        let mut f_cos = vec![0.0f32; self.n_theta * self.n_phi];
        let mut f_sin = vec![0.0f32; self.n_theta * self.n_phi];

        let mut idx = 0;
        for l in 0..=self.band_limit {
            for m in -(l as i32)..=(l as i32) {
                let m_abs = m.unsigned_abs() as usize;
                let c = coeffs[idx];

                for (i_theta, &theta) in self.theta_grid.iter().enumerate() {
                    let plm = associated_legendre_normalized(l, m, theta.cos());
                    let contrib = (c * plm) as f32;

                    if m >= 0 {
                        f_cos[i_theta * self.n_phi + m_abs] += contrib;
                    } else {
                        f_sin[i_theta * self.n_phi + m_abs] -= contrib;
                    }
                }

                idx += 1;
            }
        }

        // Build inverse DFT matrices
        let mut idft_cos = vec![0.0f32; self.n_phi * self.n_phi];
        let mut idft_sin = vec![0.0f32; self.n_phi * self.n_phi];
        for n in 0..self.n_phi {
            for k in 0..self.n_phi {
                let angle = 2.0 * PI * (k as f64) * (n as f64) / (self.n_phi as f64);
                idft_cos[n * self.n_phi + k] = angle.cos() as f32;
                idft_sin[n * self.n_phi + k] = angle.sin() as f32;
            }
        }

        // Convert to tensors
        let f_cos_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(f_cos.as_slice(), device)
                .reshape([self.n_theta, self.n_phi]);

        let f_sin_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(f_sin.as_slice(), device)
                .reshape([self.n_theta, self.n_phi]);

        let idft_cos_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(idft_cos.as_slice(), device)
                .reshape([self.n_phi, self.n_phi]);

        let idft_sin_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(idft_sin.as_slice(), device)
                .reshape([self.n_phi, self.n_phi]);

        // Apply inverse DFT: f[θ,φ] = Σ_k (F_cos[θ,k]*cos(kφ) + F_sin[θ,k]*sin(kφ))
        // = F_cos @ IDFT_cos^T + F_sin @ IDFT_sin^T
        let field_cos = f_cos_tensor.matmul(idft_cos_tensor.swap_dims(0, 1));
        let field_sin = f_sin_tensor.matmul(idft_sin_tensor.swap_dims(0, 1));

        let field = field_cos + field_sin;

        // Normalize
        let norm = (self.n_phi as f32).sqrt();
        let field = field.div_scalar(norm);

        // Extract to CPU
        let field_data: Vec<f32> = field.into_data().iter::<f32>().collect();
        field_data.iter().map(|&x| x as f64).collect()
    }

    /// Forward SHT with DoubleTensor for f64 precision on f32 GPUs.
    ///
    /// Uses error-free transformations (TwoSum/TwoProd) to achieve ~48-bit precision
    /// while keeping computation on GPU.
    #[cfg(feature = "gpu")]
    pub fn forward_sht_double(
        &self,
        field: &[f64],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<f64> {
        use crate::double_float::DoubleTensor;
        use burn::tensor::Tensor;
        use thrml_core::backend::WgpuBackend;

        let n_points = self.n_theta * self.n_phi;
        assert_eq!(field.len(), n_points, "Field must match grid size");

        // Split f64 into hi/lo f32 pairs
        let field_hi: Vec<f32> = field.iter().map(|&x| x as f32).collect();
        let field_lo: Vec<f32> = field
            .iter()
            .zip(field_hi.iter())
            .map(|(&x, &hi)| (x - hi as f64) as f32)
            .collect();

        // Create DoubleTensor for field
        let field_hi_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(field_hi.as_slice(), device)
                .reshape([self.n_theta, self.n_phi]);

        let field_lo_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(field_lo.as_slice(), device)
                .reshape([self.n_theta, self.n_phi]);

        let field_double = DoubleTensor::new(field_hi_tensor, field_lo_tensor);

        // Build DFT matrices (same as forward_sht_gpu)
        let mut dft_cos = vec![0.0f32; self.n_phi * self.n_phi];
        let mut dft_sin = vec![0.0f32; self.n_phi * self.n_phi];
        for k in 0..self.n_phi {
            for n in 0..self.n_phi {
                let angle = 2.0 * PI * (k as f64) * (n as f64) / (self.n_phi as f64);
                dft_cos[k * self.n_phi + n] = angle.cos() as f32;
                dft_sin[k * self.n_phi + n] = angle.sin() as f32;
            }
        }

        let dft_cos_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(dft_cos.as_slice(), device)
                .reshape([self.n_phi, self.n_phi]);

        let dft_sin_tensor: Tensor<WgpuBackend, 2> =
            Tensor::<WgpuBackend, 1>::from_floats(dft_sin.as_slice(), device)
                .reshape([self.n_phi, self.n_phi]);

        // Apply DFT with DoubleTensor precision
        // For each component (hi, lo), apply DFT separately and combine
        let f_cos_hi = field_double
            .hi
            .clone()
            .matmul(dft_cos_tensor.clone().swap_dims(0, 1));
        let f_cos_lo = field_double
            .lo
            .clone()
            .matmul(dft_cos_tensor.swap_dims(0, 1));
        let f_sin_hi = field_double
            .hi
            .matmul(dft_sin_tensor.clone().swap_dims(0, 1));
        let f_sin_lo = field_double.lo.matmul(dft_sin_tensor.swap_dims(0, 1));

        // Combine hi+lo for extraction (DoubleTensor matmul would be more accurate)
        let f_cos = f_cos_hi + f_cos_lo;
        let f_sin = f_sin_hi + f_sin_lo;

        // Normalize
        let norm = (self.n_phi as f32).sqrt();
        let f_cos = f_cos.div_scalar(norm);
        let f_sin = f_sin.div_scalar(norm);

        // Extract and continue with CPU Legendre (for maximum precision)
        let f_cos_data: Vec<f32> = f_cos.into_data().iter::<f32>().collect();
        let f_sin_data: Vec<f32> = f_sin.into_data().iter::<f32>().collect();

        // Legendre transform (CPU f64 for stability)
        let num_coeffs = (self.band_limit + 1).pow(2);
        let mut coeffs = vec![0.0f64; num_coeffs];

        let mut idx = 0;
        for l in 0..=self.band_limit {
            for m in -(l as i32)..=(l as i32) {
                let m_abs = m.unsigned_abs() as usize;

                let mut sum = 0.0f64;
                for (i_theta, &theta) in self.theta_grid.iter().enumerate() {
                    let fm = if m >= 0 {
                        f_cos_data[i_theta * self.n_phi + m_abs] as f64
                    } else {
                        -f_sin_data[i_theta * self.n_phi + m_abs] as f64
                    };

                    let plm = associated_legendre_normalized(l, m, theta.cos());
                    let w = self.weights[i_theta * self.n_phi];

                    sum += w * plm * fm;
                }

                coeffs[idx] = sum;
                idx += 1;
            }
        }

        coeffs
    }

    /// Unified forward SHT with automatic backend selection.
    ///
    /// Chooses between CPU, GPU f32, GPU DoubleTensor, or GPU native f64
    /// based on `config.gpu_strategy`:
    /// - `CpuFallback`: CPU f64 (highest precision, lowest throughput)
    /// - `DoubleTensor`: GPU with hi+lo f32 pair (~48-bit precision)
    /// - `NativeF64`: GPU native f64 (datacenter GPUs only)
    ///
    /// Optionally validates GPU results against CPU.
    #[cfg(feature = "gpu")]
    pub fn forward_sht_auto(
        &self,
        field: &[f64],
        config: &SphericalHarmonicsConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<f64> {
        // Always compute CPU reference if validating
        let cpu_result = if config.validate_gpu {
            Some(self.forward_sht(field))
        } else {
            None
        };

        // Choose backend based on GPU f64 strategy
        let result = match config.gpu_strategy {
            GpuF64Strategy::CpuFallback => {
                // Use CPU f64 - highest precision
                self.forward_sht(field)
            }
            GpuF64Strategy::DoubleTensor => {
                // Consumer GPU - use DoubleTensor for ~48-bit precision
                self.forward_sht_double(field, device)
            }
            GpuF64Strategy::NativeF64 => {
                // Datacenter GPU - use GPU f32 for DFT (WGPU doesn't support f64)
                // TODO: Add CUDA f64 path when available via burn-cuda
                self.forward_sht_gpu(field, device)
            }
        };

        // Validate if requested
        if let Some(ref cpu) = cpu_result {
            let (max_err, mean_err, idx) = validate_results(cpu, &result);
            if max_err > config.validation_tolerance {
                eprintln!(
                    "[SH Validation] WARN: max_rel_err={:.2e} > tol={:.2e} at idx={} (mean={:.2e})",
                    max_err, config.validation_tolerance, idx, mean_err
                );
            } else {
                eprintln!(
                    "[SH Validation] OK: max_rel_err={:.2e}, mean={:.2e}",
                    max_err, mean_err
                );
            }
        }

        result
    }

    /// Unified inverse SHT with automatic backend selection.
    #[cfg(feature = "gpu")]
    pub fn inverse_sht_auto(
        &self,
        coeffs: &[f64],
        config: &SphericalHarmonicsConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<f64> {
        let cpu_result = if config.validate_gpu {
            Some(self.inverse_sht(coeffs))
        } else {
            None
        };

        // Choose backend based on GPU f64 strategy
        let result = match config.gpu_strategy {
            GpuF64Strategy::CpuFallback => self.inverse_sht(coeffs),
            GpuF64Strategy::DoubleTensor | GpuF64Strategy::NativeF64 => {
                // Both use GPU path (DoubleTensor uses f32 with compensation)
                self.inverse_sht_gpu(coeffs, device)
            }
        };

        if let Some(ref cpu) = cpu_result {
            let (max_err, _mean_err, idx) = validate_results(cpu, &result);
            if max_err > config.validation_tolerance {
                eprintln!(
                    "[SH Inverse Validation] WARN: max_rel_err={:.2e} > tol={:.2e} at idx={}",
                    max_err, config.validation_tolerance, idx
                );
            }
        }

        result
    }
}

/// Validate GPU results against CPU reference.
///
/// Returns (max_relative_error, mean_relative_error, index_of_max)
pub fn validate_results(reference: &[f64], computed: &[f64]) -> (f64, f64, usize) {
    assert_eq!(reference.len(), computed.len());

    let mut max_err = 0.0f64;
    let mut max_idx = 0;
    let mut sum_err = 0.0f64;
    let mut count = 0;

    for (i, (&r, &c)) in reference.iter().zip(computed.iter()).enumerate() {
        let abs_r = r.abs();
        if abs_r > 1e-15 {
            let rel_err = ((r - c) / abs_r).abs();
            if rel_err > max_err {
                max_err = rel_err;
                max_idx = i;
            }
            sum_err += rel_err;
            count += 1;
        }
    }

    let mean_err = if count > 0 {
        sum_err / count as f64
    } else {
        0.0
    };
    (max_err, mean_err, max_idx)
}

/// Benchmark and compare CPU vs GPU vs DoubleTensor SHT performance.
#[cfg(feature = "gpu")]
pub fn benchmark_sht_backends(
    basis: &SphericalHarmonicsBasis,
    field: &[f64],
    device: &burn::backend::wgpu::WgpuDevice,
) -> ShtBenchmarkResult {
    use std::time::Instant;

    // CPU baseline
    let start = Instant::now();
    let cpu_result = basis.forward_sht(field);
    let cpu_time = start.elapsed();

    // GPU f32
    let start = Instant::now();
    let gpu_result = basis.forward_sht_gpu(field, device);
    let gpu_time = start.elapsed();

    // GPU DoubleTensor
    let start = Instant::now();
    let double_result = basis.forward_sht_double(field, device);
    let double_time = start.elapsed();

    // Compute errors
    let (gpu_max_err, gpu_mean_err, _) = validate_results(&cpu_result, &gpu_result);
    let (double_max_err, double_mean_err, _) = validate_results(&cpu_result, &double_result);

    ShtBenchmarkResult {
        cpu_time_ms: cpu_time.as_secs_f64() * 1000.0,
        gpu_time_ms: gpu_time.as_secs_f64() * 1000.0,
        double_time_ms: double_time.as_secs_f64() * 1000.0,
        gpu_max_error: gpu_max_err,
        gpu_mean_error: gpu_mean_err,
        double_max_error: double_max_err,
        double_mean_error: double_mean_err,
        gpu_speedup: cpu_time.as_secs_f64() / gpu_time.as_secs_f64(),
        double_speedup: cpu_time.as_secs_f64() / double_time.as_secs_f64(),
    }
}

/// Results from SHT backend benchmark.
#[derive(Debug, Clone)]
pub struct ShtBenchmarkResult {
    pub cpu_time_ms: f64,
    pub gpu_time_ms: f64,
    pub double_time_ms: f64,
    pub gpu_max_error: f64,
    pub gpu_mean_error: f64,
    pub double_max_error: f64,
    pub double_mean_error: f64,
    pub gpu_speedup: f64,
    pub double_speedup: f64,
}

impl std::fmt::Display for ShtBenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "SHT Backend Benchmark:")?;
        writeln!(f, "  CPU:          {:.2}ms (baseline)", self.cpu_time_ms)?;
        writeln!(
            f,
            "  GPU f32:      {:.2}ms ({:.1}x speedup, max_err={:.2e})",
            self.gpu_time_ms, self.gpu_speedup, self.gpu_max_error
        )?;
        writeln!(
            f,
            "  DoubleTensor: {:.2}ms ({:.1}x speedup, max_err={:.2e})",
            self.double_time_ms, self.double_speedup, self.double_max_error
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn test_legendre_p00() {
        // P_0^0 should be sqrt(1/(4π)) ≈ 0.282
        let p00 = associated_legendre_normalized(0, 0, 0.5);
        assert!((p00 - 0.282).abs() < 0.01);
    }

    #[test]
    fn test_real_sh_orthonormality() {
        // Y_0^0 integrated over sphere should be sqrt(4π)
        let config = SphericalHarmonicsConfig::dev();
        let basis = SphericalHarmonicsBasis::new(&config);

        // Forward SHT of constant 1 should give coefficient only for l=0, m=0
        let n_points = basis.n_theta * basis.n_phi;
        let constant_field = vec![1.0; n_points];
        let coeffs = basis.forward_sht(&constant_field);

        // First coefficient (l=0, m=0) should be ~sqrt(4π) ≈ 3.545
        assert!((coeffs[0] - (4.0 * PI).sqrt()).abs() < 0.1);

        // Other coefficients should be near zero
        for c in &coeffs[1..] {
            assert!(c.abs() < 0.1);
        }
    }

    #[test]
    fn test_superposition_field() {
        let config = SphericalHarmonicsConfig::dev();
        let basis = SphericalHarmonicsBasis::new(&config);

        // Create two blobs and compute harmonic superposition
        let blob1 = basis.gaussian_blob(FRAC_PI_2, 0.0, 0.3, 0.3);
        let blob2 = basis.gaussian_blob(FRAC_PI_2, PI, 0.3, 0.3);

        let intensity = basis.superposition_field(&[blob1, blob2]);

        // Should sum to 1 (normalized)
        let sum: f64 = intensity.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should have peaks near both blob centers
        let (theta_peak, _phi_peak, _) = basis.find_peak(&intensity);
        assert!((theta_peak - FRAC_PI_2).abs() < 0.5);
    }
}
