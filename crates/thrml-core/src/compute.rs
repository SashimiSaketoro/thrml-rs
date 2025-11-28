//! Hybrid Compute Backend for Unified Memory Systems
//!
//! ## Architecture
//!
//! On unified memory systems (Apple Silicon, AMD APU), CPU and GPU share the same
//! memory pool. Running precision-sensitive tasks on CPU can **increase** parallelism
//! by not competing for GPU compute units.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED MEMORY SYSTEM                        │
//! │                                                                 │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │              SHARED MEMORY POOL (e.g., 24GB)            │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │            ▲                              ▲                     │
//! │            │ zero-copy                    │ zero-copy           │
//! │            │                              │                     │
//! │  ┌─────────┴─────────┐          ┌────────┴────────┐            │
//! │  │   CPU CORES       │          │   GPU CORES     │            │
//! │  │  (precision ops)  │  ←───→   │  (bulk ops)     │            │
//! │  │                   │ parallel │                 │            │
//! │  │ • Ising sampling  │          │ • Similarity    │            │
//! │  │ • f64 arithmetic  │          │ • Energy        │            │
//! │  │ • Small matrix    │          │ • Large matmul  │            │
//! │  └───────────────────┘          └─────────────────┘            │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Precision Strategy
//!
//! | Scenario                    | fp32 OK? | Recommendation       |
//! |-----------------------------|----------|---------------------|
//! | Sphere optimization         | YES      | Use GPU fp32        |
//! | Ising max-cut sampling      | MARGINAL | Use CPU fp64        |
//! | Spherical harmonics L>64    | NO       | Use CPU fp64        |
//! | Long Langevin chains        | MARGINAL | Periodic renorm     |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use thrml_core::{ComputeBackend, OpType, HybridConfig};
//!
//! // Auto-detect backend (hybrid on macOS, GPU elsewhere)
//! let backend = ComputeBackend::default();
//!
//! // Check if operation should use CPU
//! if backend.use_cpu(OpType::IsingSampling, None) {
//!     // Run precision-sensitive f64 implementation
//! } else {
//!     // Run GPU f32 implementation
//! }
//!
//! // Full configuration for Apple Silicon
//! let config = HybridConfig::apple_silicon();
//! ```

/// Compute operation type for CPU/GPU routing decisions.
///
/// Operations are classified by their precision requirements and parallelism
/// characteristics. Used by [`ComputeBackend`] to determine optimal execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpType {
    /// Ising model operations (Gibbs sampling, max-cut).
    /// Precision-sensitive, benefits from f64 on CPU.
    IsingSampling,

    /// Spherical harmonics computation.
    /// Requires f64 for band limits > 64.
    SphericalHarmonics,

    /// Inverse trigonometric functions (acos, atan2).
    /// Sensitive near poles and for small angles.
    ArcTrig,

    /// Complex number arithmetic.
    /// Phase accumulation benefits from f64.
    ComplexArithmetic,

    /// Small matrix operations (below threshold).
    /// May be faster on CPU due to GPU kernel overhead.
    SmallMatmul,

    /// Similarity matrix computation.
    /// Highly parallel, ideal for GPU.
    Similarity,

    /// Large matrix operations.
    /// Ideal for GPU bulk compute.
    LargeMatmul,

    /// Energy function computation.
    /// Parallel over points, GPU-friendly.
    EnergyCompute,

    /// Langevin dynamics step.
    /// GPU-accelerated with periodic renormalization.
    LangevinStep,

    /// Gradient computation for training.
    /// Precision-sensitive due to accumulation - prefer CPU f64.
    /// f32 can cause overflow in contrastive divergence gradients.
    GradientCompute,

    /// Batch energy forward pass.
    /// Parallel over targets, can use GPU f32 for speed.
    /// Gradients computed separately on CPU.
    BatchEnergyForward,

    /// Loss reduction (softmax, logsumexp).
    /// Overflow-prone with large logits - prefer CPU f64.
    /// Use log-sum-exp trick if GPU is required.
    LossReduction,
}

/// Compute backend selection strategy.
///
/// Determines how operations are distributed between CPU and GPU.
/// The optimal choice depends on hardware architecture:
///
/// | Hardware | Recommended | Reason |
/// |----------|-------------|--------|
/// | Apple Silicon (M1-M4) | [`UnifiedHybrid`] | Zero-copy unified memory |
/// | NVIDIA discrete | [`GpuOnly`] | Memory transfer overhead |
/// | AMD APU | [`UnifiedHybrid`] | Shared memory pool |
/// | CPU-only server | [`CpuOnly`] | No GPU available |
///
/// [`UnifiedHybrid`]: ComputeBackend::UnifiedHybrid
/// [`GpuOnly`]: ComputeBackend::GpuOnly
/// [`CpuOnly`]: ComputeBackend::CpuOnly
///
/// # Example
///
/// ```rust,ignore
/// use thrml_core::{ComputeBackend, OpType};
///
/// // Auto-detect (hybrid on macOS, GPU elsewhere)
/// let backend = ComputeBackend::default();
///
/// // Explicit Apple Silicon configuration
/// let backend = ComputeBackend::apple_silicon();
///
/// // Check if an operation should use CPU
/// assert!(backend.use_cpu(OpType::IsingSampling, None));
/// ```
#[derive(Debug, Clone)]
pub enum ComputeBackend {
    /// All operations on GPU (discrete GPU systems).
    ///
    /// Best for NVIDIA/AMD discrete GPUs where CPU-GPU memory transfer
    /// is the bottleneck. Runs everything on GPU even if precision suffers.
    GpuOnly,

    /// All operations on CPU (fallback).
    ///
    /// For systems without GPU or for debugging. Provides highest precision
    /// but lowest throughput for bulk operations.
    CpuOnly,

    /// Hybrid: precision ops on CPU, bulk ops on GPU.
    ///
    /// Optimal for unified memory systems (Apple Silicon, AMD APU) where
    /// CPU and GPU share the same memory pool with zero-copy access.
    UnifiedHybrid {
        /// Operations routed to CPU (precision-sensitive).
        cpu_ops: Vec<OpType>,
        /// Size threshold below which matmul runs on CPU.
        small_matmul_threshold: usize,
    },

    /// Adaptive: automatically detect and optimize.
    ///
    /// Routes precision-sensitive operations to CPU regardless of system.
    Adaptive,
}

// This impl is not derivable because it has conditional logic for macOS vs other platforms
#[allow(clippy::derivable_impls)]
impl Default for ComputeBackend {
    fn default() -> Self {
        // Default to hybrid for Apple Silicon
        #[cfg(target_os = "macos")]
        {
            ComputeBackend::UnifiedHybrid {
                cpu_ops: vec![
                    OpType::IsingSampling,
                    OpType::SphericalHarmonics,
                    OpType::ArcTrig,
                    OpType::ComplexArithmetic,
                    OpType::SmallMatmul,
                    // Training ops: Metal lacks native f64, use CPU for precision
                    OpType::GradientCompute,
                    OpType::LossReduction,
                ],
                small_matmul_threshold: 1000,
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Non-macOS: use Adaptive to handle AMD RDNA (no f64) vs NVIDIA (has f64)
            // Adaptive routes precision-sensitive ops to CPU, safe for all GPUs
            ComputeBackend::Adaptive
        }
    }
}

impl ComputeBackend {
    /// Creates a GPU-only backend.
    ///
    /// Use for discrete NVIDIA/AMD GPUs where all operations should run on GPU.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::ComputeBackend;
    ///
    /// let backend = ComputeBackend::gpu_only();
    /// ```
    pub fn gpu_only() -> Self {
        ComputeBackend::GpuOnly
    }

    /// Creates a CPU-only backend.
    ///
    /// Useful for debugging, testing, or systems without GPU support.
    /// Provides maximum precision but lower throughput.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::ComputeBackend;
    ///
    /// let backend = ComputeBackend::cpu_only();
    /// ```
    pub fn cpu_only() -> Self {
        ComputeBackend::CpuOnly
    }

    /// Creates a hybrid backend optimized for Apple Silicon (M1/M2/M3/M4).
    ///
    /// Routes precision-sensitive operations to CPU while bulk operations
    /// use GPU. Takes advantage of unified memory for zero-copy access.
    ///
    /// CPU operations: Ising sampling, spherical harmonics, arc trig,
    /// complex arithmetic, small matrix ops (< 2000 elements).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::ComputeBackend;
    ///
    /// let backend = ComputeBackend::apple_silicon();
    /// ```
    pub fn apple_silicon() -> Self {
        ComputeBackend::UnifiedHybrid {
            cpu_ops: vec![
                OpType::IsingSampling,
                OpType::SphericalHarmonics,
                OpType::ArcTrig,
                OpType::ComplexArithmetic,
                OpType::SmallMatmul,
                // Training ops: Metal lacks native f64, use CPU for precision
                OpType::GradientCompute,
                OpType::LossReduction,
            ],
            small_matmul_threshold: 2000,
        }
    }

    /// Creates a hybrid backend with custom CPU operation routing.
    ///
    /// # Arguments
    ///
    /// * `cpu_ops` - Operations to route to CPU.
    /// * `small_matmul_threshold` - Matrix size below which to use CPU.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::{ComputeBackend, OpType};
    ///
    /// // Only route Ising sampling to CPU
    /// let backend = ComputeBackend::hybrid(
    ///     vec![OpType::IsingSampling],
    ///     1000,
    /// );
    /// ```
    pub fn hybrid(cpu_ops: Vec<OpType>, small_matmul_threshold: usize) -> Self {
        ComputeBackend::UnifiedHybrid {
            cpu_ops,
            small_matmul_threshold,
        }
    }

    /// Determines if an operation should run on CPU.
    ///
    /// # Arguments
    ///
    /// * `op` - The operation type to check.
    /// * `size` - Optional size parameter for threshold-based decisions.
    ///
    /// # Returns
    ///
    /// `true` if the operation should run on CPU, `false` for GPU.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::{ComputeBackend, OpType};
    ///
    /// let backend = ComputeBackend::apple_silicon();
    ///
    /// // Ising sampling routes to CPU for precision
    /// assert!(backend.use_cpu(OpType::IsingSampling, None));
    ///
    /// // Large matrix operations go to GPU
    /// assert!(!backend.use_cpu(OpType::LargeMatmul, Some(10000)));
    /// ```
    pub fn use_cpu(&self, op: OpType, size: Option<usize>) -> bool {
        match self {
            ComputeBackend::GpuOnly => false,
            ComputeBackend::CpuOnly => true,
            ComputeBackend::UnifiedHybrid {
                cpu_ops,
                small_matmul_threshold,
            } => {
                if cpu_ops.contains(&op) {
                    return true;
                }
                // Check small matrix threshold
                if op == OpType::SmallMatmul || op == OpType::IsingSampling {
                    if let Some(n) = size {
                        return n < *small_matmul_threshold;
                    }
                }
                false
            }
            ComputeBackend::Adaptive => {
                // Adaptive logic: use CPU for precision-sensitive ops
                // This includes training ops that can overflow in f32
                matches!(
                    op,
                    OpType::IsingSampling
                        | OpType::SphericalHarmonics
                        | OpType::ArcTrig
                        | OpType::ComplexArithmetic
                        | OpType::GradientCompute  // f32 accumulation can overflow
                        | OpType::LossReduction // logsumexp needs f64 for large batches
                )
            }
        }
    }

    /// Check if an operation should run on GPU
    pub fn use_gpu(&self, op: OpType, size: Option<usize>) -> bool {
        !self.use_cpu(op, size)
    }

    /// Try GPU operation with automatic CPU fallback.
    ///
    /// Executes `gpu_fn` first. If it fails, notifies user and runs `cpu_fn`.
    /// Returns the result and whether fallback occurred.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (result, fell_back) = backend.try_gpu_with_fallback(
    ///     || gpu_energy_compute(&tensor),
    ///     || cpu_energy_compute_f64(&data),
    ///     "energy computation",
    /// );
    /// ```
    pub fn try_gpu_with_fallback<T, E, F, G>(
        &self,
        gpu_fn: F,
        cpu_fn: G,
        op_name: &str,
    ) -> (T, bool)
    where
        F: FnOnce() -> Result<T, E>,
        G: FnOnce() -> T,
        E: std::fmt::Display,
    {
        match gpu_fn() {
            Ok(result) => (result, false),
            Err(e) => {
                eprintln!("[thrml] GPU {} failed, falling back to CPU: {}", op_name, e);
                (cpu_fn(), true)
            }
        }
    }

    /// Try GPU operation, fallback to CPU if this backend prefers CPU for the op.
    ///
    /// Unlike `try_gpu_with_fallback`, this checks the routing first and only
    /// attempts GPU if the backend says to use GPU.
    pub fn run_routed<T, F, G>(&self, op: OpType, size: Option<usize>, gpu_fn: F, cpu_fn: G) -> T
    where
        F: FnOnce() -> T,
        G: FnOnce() -> T,
    {
        if self.use_cpu(op, size) {
            cpu_fn()
        } else {
            gpu_fn()
        }
    }
}

/// Precision mode for numerical operations.
///
/// | Scenario                      | fp32 OK? | Recommendation |
/// |-------------------------------|----------|----------------|
/// | Sphere optimization           | YES      | GPU fp32       |
/// | Geodesic distances (general)  | YES      | GPU fp32       |
/// | Spherical harmonics L≤64      | YES      | GPU fp32       |
/// | Spherical harmonics L>128     | NO       | CPU fp64       |
/// | Long Langevin chains (>100K)  | MARGINAL | Re-normalize   |
#[derive(Debug, Clone)]
pub enum PrecisionMode {
    /// GPU fp32 - default for sphere optimization
    GpuFast,

    /// CPU fp64 - for spherical harmonics L>64
    CpuPrecise,

    /// Adaptive: route operations by sensitivity
    Adaptive {
        /// Max L before switching to fp64
        sh_band_limit_threshold: usize,
        /// Min angular distance before using Haversine
        small_angle_threshold: f32,
        /// Max Langevin steps before re-normalization
        langevin_renorm_interval: usize,
    },
}

impl Default for PrecisionMode {
    fn default() -> Self {
        PrecisionMode::Adaptive {
            sh_band_limit_threshold: 64,
            small_angle_threshold: 0.01,
            langevin_renorm_interval: 10000,
        }
    }
}

impl PrecisionMode {
    /// Fast GPU mode (fp32)
    pub fn gpu_fast() -> Self {
        PrecisionMode::GpuFast
    }

    /// Precise CPU mode (fp64)
    pub fn cpu_precise() -> Self {
        PrecisionMode::CpuPrecise
    }

    /// Should use f64 for the given band limit?
    pub fn use_f64_for_sh(&self, band_limit: usize) -> bool {
        match self {
            PrecisionMode::GpuFast => false,
            PrecisionMode::CpuPrecise => true,
            PrecisionMode::Adaptive {
                sh_band_limit_threshold,
                ..
            } => band_limit > *sh_band_limit_threshold,
        }
    }

    /// Should use Haversine formula for small angles?
    pub fn use_haversine(&self, angle: f32) -> bool {
        match self {
            PrecisionMode::GpuFast => false,
            PrecisionMode::CpuPrecise => true,
            PrecisionMode::Adaptive {
                small_angle_threshold,
                ..
            } => angle < *small_angle_threshold,
        }
    }

    /// Should re-normalize at this step?
    pub fn should_renormalize(&self, step: usize) -> bool {
        match self {
            PrecisionMode::GpuFast => false,
            PrecisionMode::CpuPrecise => step.is_multiple_of(1000),
            PrecisionMode::Adaptive {
                langevin_renorm_interval,
                ..
            } => step.is_multiple_of(*langevin_renorm_interval),
        }
    }
}

/// Configuration for hybrid compute strategy.
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Backend selection strategy
    pub backend: ComputeBackend,
    /// Precision mode
    pub precision: PrecisionMode,
    /// Number of CPU threads for precision ops
    pub cpu_threads: usize,
    /// Enable async overlap between CPU and GPU
    pub enable_overlap: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            backend: ComputeBackend::default(),
            precision: PrecisionMode::default(),
            cpu_threads: num_cpus::get().min(8),
            enable_overlap: true,
        }
    }
}

impl HybridConfig {
    /// Configuration for Apple Silicon (M1/M2/M3/M4)
    pub fn apple_silicon() -> Self {
        Self {
            backend: ComputeBackend::apple_silicon(),
            precision: PrecisionMode::Adaptive {
                sh_band_limit_threshold: 64,
                small_angle_threshold: 0.01,
                langevin_renorm_interval: 10000,
            },
            cpu_threads: num_cpus::get().min(8),
            enable_overlap: true,
        }
    }

    /// Configuration for discrete NVIDIA GPU
    pub fn nvidia_discrete() -> Self {
        Self {
            backend: ComputeBackend::GpuOnly,
            precision: PrecisionMode::GpuFast,
            cpu_threads: 1,
            enable_overlap: false,
        }
    }

    /// Configuration for CPU-only systems
    pub fn cpu_only() -> Self {
        Self {
            backend: ComputeBackend::CpuOnly,
            precision: PrecisionMode::CpuPrecise,
            cpu_threads: num_cpus::get(),
            enable_overlap: false,
        }
    }
}

// =============================================================================
// Test Utilities
// =============================================================================

/// Test helper: run a test function on both CPU and GPU backends.
///
/// This is useful for precision validation tests where you want to
/// verify behavior on CPU (f64 precision) and GPU (f32 precision).
///
/// # Example
///
/// ```rust,ignore
/// use thrml_core::compute::test_both_backends;
///
/// test_both_backends(|backend| {
///     let tolerance = if backend.use_cpu(OpType::EnergyCompute, None) {
///         1e-10  // CPU f64 precision
///     } else {
///         1e-3   // GPU f32 precision
///     };
///     // Run your test with appropriate tolerance
/// });
/// ```
pub fn test_both_backends<F: Fn(&ComputeBackend)>(test_fn: F) {
    // Test on CPU for precision validation
    test_fn(&ComputeBackend::CpuOnly);

    // Test on GPU for production path validation
    test_fn(&ComputeBackend::gpu_only());
}

/// Get recommended tolerance for a given backend and operation.
///
/// Returns tighter tolerances for CPU (f64) and looser for GPU (f32).
pub fn recommended_tolerance(backend: &ComputeBackend, op: OpType) -> f64 {
    if backend.use_cpu(op, None) {
        1e-10 // f64 precision
    } else {
        1e-3 // f32 precision (GPU)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_backend_default() {
        let backend = ComputeBackend::default();

        #[cfg(target_os = "macos")]
        {
            assert!(backend.use_cpu(OpType::IsingSampling, None));
            assert!(backend.use_gpu(OpType::Similarity, None));
        }

        #[cfg(not(target_os = "macos"))]
        {
            // On non-macOS, default is Adaptive (routes precision-sensitive ops to CPU)
            assert!(backend.use_cpu(OpType::IsingSampling, None)); // Adaptive routes this to CPU
            assert!(!backend.use_cpu(OpType::Similarity, None)); // But general ops go to GPU
        }
    }

    #[test]
    fn test_compute_backend_cpu_only() {
        let backend = ComputeBackend::cpu_only();
        assert!(backend.use_cpu(OpType::IsingSampling, None));
        assert!(backend.use_cpu(OpType::LargeMatmul, None));
        assert!(backend.use_cpu(OpType::Similarity, None));
    }

    #[test]
    fn test_compute_backend_gpu_only() {
        let backend = ComputeBackend::gpu_only();
        assert!(!backend.use_cpu(OpType::IsingSampling, None));
        assert!(!backend.use_cpu(OpType::SmallMatmul, None));
    }

    #[test]
    fn test_compute_backend_adaptive() {
        let backend = ComputeBackend::Adaptive;
        // Adaptive routes precision-sensitive ops to CPU
        assert!(backend.use_cpu(OpType::IsingSampling, None));
        assert!(backend.use_cpu(OpType::SphericalHarmonics, None));
        assert!(backend.use_cpu(OpType::ArcTrig, None));
        // But bulk ops go to GPU
        assert!(!backend.use_cpu(OpType::Similarity, None));
        assert!(!backend.use_cpu(OpType::LargeMatmul, None));
    }

    #[test]
    fn test_precision_mode() {
        let mode = PrecisionMode::default();
        assert!(!mode.use_f64_for_sh(32));
        assert!(mode.use_f64_for_sh(128));
    }

    #[test]
    fn test_hybrid_config() {
        let config = HybridConfig::apple_silicon();
        assert!(config.enable_overlap);
        assert!(config.cpu_threads > 0);
    }
}
