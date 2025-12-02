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

    /// Categorical/softmax sampling (Gumbel-max, Gibbs for discrete vars).
    /// Precision-sensitive: logit accumulation can overflow in f32.
    CategoricalSampling,

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
    /// Precision-sensitive: accumulation can overflow in f32.
    GradientCompute,

    /// Loss reduction (sum, mean).
    /// Precision-sensitive for large batch accumulation.
    LossReduction,

    /// Batched energy computation for forward pass.
    /// Highly parallel, ideal for GPU.
    BatchEnergyForward,
}

// =============================================================================
// Hardware Tier Classification
// =============================================================================

/// Hardware tier classification for precision/backend selection.
///
/// Classification based on FP64 capabilities:
/// - Consumer GPUs (Apple, NVIDIA RTX 30-50, AMD RDNA): weak/no FP64
/// - Datacenter GPUs (NVIDIA H100/B200): strong FP64
///
/// Use `RuntimePolicy::detect()` for automatic hardware detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareTier {
    /// Apple Silicon M1-M5: Metal backend, no native FP64 on GPU.
    /// Unified memory enables zero-copy CPU/GPU data sharing.
    AppleSilicon,

    /// NVIDIA consumer GPUs: RTX 3080-5090 (Ampere/Ada/Blackwell consumer).
    /// Compute capability 8.x, 12.x - FP64 is ~1/64 of FP32 throughput.
    NvidiaConsumer,

    /// AMD RDNA 3/4: RX 7000-9000 series.
    /// Optimized for FP32/FP16/INT8, minimal FP64 support.
    AmdRdna,

    /// NVIDIA H100/H200 (Hopper): Compute capability 9.0.
    /// Strong FP64 support for HPC workloads.
    NvidiaHopper,

    /// NVIDIA B200/GB200 (Blackwell datacenter): Compute capability 10.0.
    /// Strong FP64 + FP8/BF16 tensor cores for mixed precision.
    NvidiaBlackwell,

    /// NVIDIA DGX Spark / GB10 Grace Blackwell: Unified memory architecture.
    /// 128GB LPDDR5x unified memory (like Apple Silicon but with Blackwell GPU).
    /// 40 TFLOPS FP64 - good precision support on GPU.
    /// Optimized for in-memory inference rather than raw compute.
    NvidiaSpark,

    /// CPU-only fallback (no GPU available or disabled).
    CpuOnly,

    /// Unknown GPU - use conservative defaults.
    Unknown,
}

/// PCI Vendor IDs for GPU detection.
///
/// These are standard PCI vendor IDs used to identify GPU manufacturers.
pub mod vendor_ids {
    /// NVIDIA Corporation
    pub const NVIDIA: u32 = 0x10DE;
    /// Apple Inc.
    pub const APPLE: u32 = 0x106B;
    /// Advanced Micro Devices (AMD)
    pub const AMD: u32 = 0x1002;
    /// Intel Corporation
    pub const INTEL: u32 = 0x8086;
}

// =============================================================================
// Precision Profile
// =============================================================================

/// Precision profile for numerical operations.
///
/// Maps hardware capabilities to dtype/backend strategy:
/// - `CpuFp64Strict`: All precision-sensitive math on CPU fp64
/// - `GpuMixed`: GPU fp32 for bulk, CPU fp64 for precision-sensitive
/// - `GpuHpcFp64`: GPU fp64/complex128 for HPC-class GPUs
///
/// # Recommended Profiles by Hardware
///
/// | Hardware | Profile | Reason |
/// |----------|---------|--------|
/// | Apple Silicon M1-M5 | `CpuFp64Strict` | No native GPU FP64 |
/// | NVIDIA RTX 30-50 | `GpuMixed` | FP64 ~1/64 of FP32 |
/// | AMD RDNA 3/4 | `GpuMixed` | Minimal FP64 support |
/// | NVIDIA H100/H200 | `GpuHpcFp64` | Strong FP64 support |
/// | NVIDIA B200/GB200 | `GpuHpcFp64` | Strong FP64 + FP8 |
/// | DGX Spark / GB10 | `GpuHpcFp64` | 40 TFLOPS FP64, unified memory |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrecisionProfile {
    /// All precision-sensitive operations on CPU with fp64/complex128.
    /// GPU used only for bulk ops that tolerate fp32.
    /// Appropriate for: Apple Silicon, NVIDIA consumer, AMD RDNA
    #[default]
    CpuFp64Strict,

    /// Mixed precision: GPU fp32/complex64 for most ops,
    /// CPU fp64/complex128 for accumulation and critical kernels.
    /// Appropriate for: Consumer GPUs with good fp32 performance
    GpuMixed,

    /// Full GPU fp64/complex128 for core thermodynamic operations.
    /// CPU used only for orchestration.
    /// Appropriate for: H100, B200, datacenter GPUs with strong FP64
    GpuHpcFp64,
}

// =============================================================================
// Runtime Policy
// =============================================================================

use burn::tensor::DType;

/// Runtime policy combining hardware tier, precision profile, and dtype settings.
///
/// This is the main configuration struct for precision/backend decisions.
/// Use `RuntimePolicy::detect()` for automatic configuration based on
/// detected hardware, or construct manually for explicit control.
///
/// # Example
///
/// ```rust,ignore
/// use thrml_core::{RuntimePolicy, HardwareTier, PrecisionProfile};
///
/// // Auto-detect hardware and create appropriate policy
/// let policy = RuntimePolicy::detect();
/// println!("Detected: {:?} with {:?}", policy.tier, policy.profile);
///
/// // Or create explicit policy for specific hardware
/// let policy = RuntimePolicy::nvidia_hopper();
/// assert_eq!(policy.real_dtype, burn::tensor::DType::F64);
/// ```
#[derive(Debug, Clone)]
pub struct RuntimePolicy {
    /// Detected or specified hardware tier
    pub tier: HardwareTier,
    /// Precision strategy for this hardware
    pub profile: PrecisionProfile,
    /// Primary real dtype for computations (F32 or F64)
    pub real_dtype: DType,
    /// Complex dtype for wave/phase computations (F32=complex64, F64=complex128)
    pub complex_dtype: DType,
    /// Whether to use GPU at all
    pub use_gpu: bool,
    /// Allow mixed precision (e.g., fp32 proposals + fp64 corrections)
    pub allow_mixed_precision: bool,
    /// Maximum acceptable relative error (for validation)
    pub max_rel_error: f64,
}

impl Default for RuntimePolicy {
    fn default() -> Self {
        Self::detect()
    }
}

impl RuntimePolicy {
    /// Apple Silicon M1-M5 policy.
    ///
    /// Routes precision-sensitive operations to CPU while using GPU
    /// for bulk operations. Takes advantage of unified memory.
    pub const fn apple_silicon() -> Self {
        Self {
            tier: HardwareTier::AppleSilicon,
            profile: PrecisionProfile::CpuFp64Strict,
            real_dtype: DType::F32,    // GPU ops use f32
            complex_dtype: DType::F32, // complex64 on GPU, complex128 on CPU
            use_gpu: true,
            allow_mixed_precision: true,
            max_rel_error: 1e-6,
        }
    }

    /// NVIDIA consumer GPU policy (RTX 3080-5090).
    ///
    /// Uses GPU for most operations but routes precision-sensitive
    /// ops to CPU due to weak FP64 throughput (~1/64 of FP32).
    pub const fn nvidia_consumer() -> Self {
        Self {
            tier: HardwareTier::NvidiaConsumer,
            profile: PrecisionProfile::GpuMixed,
            real_dtype: DType::F32,
            complex_dtype: DType::F32,
            use_gpu: true,
            allow_mixed_precision: true,
            max_rel_error: 1e-4,
        }
    }

    /// AMD RDNA 3/4 policy (RX 7000-9000 series).
    ///
    /// Similar to NVIDIA consumer - good FP32/FP16, weak FP64.
    pub const fn amd_rdna() -> Self {
        Self {
            tier: HardwareTier::AmdRdna,
            profile: PrecisionProfile::GpuMixed,
            real_dtype: DType::F32,
            complex_dtype: DType::F32,
            use_gpu: true,
            allow_mixed_precision: true,
            max_rel_error: 1e-4,
        }
    }

    /// NVIDIA H100/H200 (Hopper) policy.
    ///
    /// Full GPU fp64 for core thermodynamic operations.
    /// Strong FP64 support allows GPU-native double precision.
    pub const fn nvidia_hopper() -> Self {
        Self {
            tier: HardwareTier::NvidiaHopper,
            profile: PrecisionProfile::GpuHpcFp64,
            real_dtype: DType::F64,
            complex_dtype: DType::F64, // complex128
            use_gpu: true,
            allow_mixed_precision: true,
            max_rel_error: 1e-10,
        }
    }

    /// NVIDIA B200/GB200 (Blackwell datacenter) policy.
    ///
    /// Full GPU fp64 with optional FP8/BF16 for mixed precision proposals.
    pub const fn nvidia_blackwell() -> Self {
        Self {
            tier: HardwareTier::NvidiaBlackwell,
            profile: PrecisionProfile::GpuHpcFp64,
            real_dtype: DType::F64,
            complex_dtype: DType::F64,
            use_gpu: true,
            allow_mixed_precision: true, // Can use FP8/BF16 for proposals
            max_rel_error: 1e-10,
        }
    }

    /// Policy for NVIDIA DGX Spark / GB10 Grace Blackwell.
    ///
    /// Unique architecture: unified LPDDR5x memory (like Apple Silicon)
    /// but with Blackwell GPU capable of 40 TFLOPS FP64.
    ///
    /// - 128GB unified memory: no discrete VRAM limit
    /// - Good FP64 support: can run precision ops on GPU
    /// - Lower bandwidth than HBM: optimized for inference, not training
    ///
    /// Uses `GpuHpcFp64` profile since GPU f64 is viable.
    pub const fn nvidia_spark() -> Self {
        Self {
            tier: HardwareTier::NvidiaSpark,
            profile: PrecisionProfile::GpuHpcFp64,
            real_dtype: DType::F64,
            complex_dtype: DType::F64,
            use_gpu: true,
            allow_mixed_precision: true,
            // Slightly looser tolerance than H100/B200 due to unified memory
            max_rel_error: 1e-9,
        }
    }

    /// CPU-only policy.
    ///
    /// Maximum precision, no GPU usage. Useful for debugging or
    /// systems without GPU support.
    pub const fn cpu_only() -> Self {
        Self {
            tier: HardwareTier::CpuOnly,
            profile: PrecisionProfile::CpuFp64Strict,
            real_dtype: DType::F64,
            complex_dtype: DType::F64,
            use_gpu: false,
            allow_mixed_precision: false,
            max_rel_error: 1e-12,
        }
    }

    /// Conservative default for unknown hardware.
    ///
    /// Uses CpuFp64Strict profile to be safe, but enables GPU for
    /// bulk operations that don't require high precision.
    pub const fn conservative_default() -> Self {
        Self {
            tier: HardwareTier::Unknown,
            profile: PrecisionProfile::CpuFp64Strict,
            real_dtype: DType::F32,
            complex_dtype: DType::F32,
            use_gpu: true,
            allow_mixed_precision: true,
            max_rel_error: 1e-4,
        }
    }

    /// Create policy for a specific hardware tier.
    pub const fn for_tier(tier: HardwareTier) -> Self {
        match tier {
            HardwareTier::AppleSilicon => Self::apple_silicon(),
            HardwareTier::NvidiaConsumer => Self::nvidia_consumer(),
            HardwareTier::AmdRdna => Self::amd_rdna(),
            HardwareTier::NvidiaHopper => Self::nvidia_hopper(),
            HardwareTier::NvidiaBlackwell => Self::nvidia_blackwell(),
            HardwareTier::NvidiaSpark => Self::nvidia_spark(),
            HardwareTier::CpuOnly => Self::cpu_only(),
            HardwareTier::Unknown => Self::conservative_default(),
        }
    }

    /// Platform-based fallback when GPU detection fails.
    const fn platform_fallback_tier() -> HardwareTier {
        #[cfg(target_os = "macos")]
        {
            HardwareTier::AppleSilicon
        }

        #[cfg(not(target_os = "macos"))]
        {
            HardwareTier::Unknown
        }
    }

    /// Auto-detect hardware and create appropriate policy.
    ///
    /// Detection priority:
    /// 1. WGPU backend - query AdapterInfo vendor/device
    /// 2. Platform heuristics as fallback
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let policy = RuntimePolicy::detect();
    /// println!("Running on {:?}", policy.tier);
    /// ```
    pub fn detect() -> Self {
        #[cfg(feature = "gpu")]
        {
            if let Some(gpu) = crate::backend::detect_gpu_info() {
                let tier = Self::classify_hardware(&gpu);
                return Self::for_tier(tier);
            }
        }

        // Fallback: use platform detection
        Self::for_tier(Self::platform_fallback_tier())
    }

    /// Classify hardware tier from GPU info.
    #[cfg(feature = "gpu")]
    fn classify_hardware(gpu: &crate::backend::GpuInfo) -> HardwareTier {
        match gpu.vendor_id {
            vendor_ids::APPLE => HardwareTier::AppleSilicon,

            vendor_ids::NVIDIA => {
                // Distinguish consumer vs datacenter vs unified by device name
                let name_upper = gpu.name.to_uppercase();

                if name_upper.contains("H100") || name_upper.contains("H200") {
                    HardwareTier::NvidiaHopper
                } else if name_upper.contains("B200") || name_upper.contains("GB200") {
                    // Datacenter Blackwell (discrete HBM)
                    HardwareTier::NvidiaBlackwell
                } else if name_upper.contains("GB10")
                    || name_upper.contains("GRACE")
                    || name_upper.contains("DGX SPARK")
                {
                    // GB10 Grace Blackwell (unified LPDDR5x) - DGX Spark
                    HardwareTier::NvidiaSpark
                } else if name_upper.contains("A100") || name_upper.contains("A800") {
                    // Ampere datacenter - treat like Hopper
                    HardwareTier::NvidiaHopper
                } else {
                    // RTX consumer cards (3080, 4090, 5090, etc.)
                    HardwareTier::NvidiaConsumer
                }
            }

            vendor_ids::AMD => HardwareTier::AmdRdna,

            vendor_ids::INTEL => {
                // Intel Arc or integrated - treat conservatively
                HardwareTier::Unknown
            }

            _ => HardwareTier::Unknown,
        }
    }

    /// Check if this policy is for HPC-class GPU hardware.
    ///
    /// Returns true for NVIDIA Hopper (H100/H200), Blackwell (B200/GB200),
    /// and Spark (GB10). These GPUs have strong FP64 support and can run
    /// precision ops on GPU.
    pub const fn is_hpc_tier(&self) -> bool {
        matches!(
            self.tier,
            HardwareTier::NvidiaHopper | HardwareTier::NvidiaBlackwell | HardwareTier::NvidiaSpark
        )
    }

    /// Check if CUDA f64 is available and should be used.
    ///
    /// Returns true when:
    /// 1. Hardware is HPC-class (Hopper/Blackwell)
    /// 2. CUDA feature is enabled
    /// 3. Profile is GpuHpcFp64
    pub fn cuda_f64_available(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.is_hpc_tier() && self.profile == PrecisionProfile::GpuHpcFp64
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// Get the recommended dtype for precision-sensitive operations.
    ///
    /// - For HPC tiers: F64 (GPU or CPU depending on CUDA availability)
    /// - For consumer tiers with CpuFp64Strict: F64 on CPU
    /// - Otherwise: F32
    pub fn precision_dtype(&self) -> DType {
        if self.profile == PrecisionProfile::GpuHpcFp64
            || self.profile == PrecisionProfile::CpuFp64Strict
        {
            DType::F64
        } else {
            DType::F32
        }
    }

    /// Check if DoubleTensor should be used for f64-like precision on GPU.
    ///
    /// Returns true when:
    /// 1. GPU is enabled
    /// 2. Hardware is consumer-grade (no native f64 on GPU)
    /// 3. f64 precision is requested
    ///
    /// For datacenter GPUs (H100, B200, Spark), native f64 is available
    /// and DoubleTensor is not needed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let policy = RuntimePolicy::detect();
    /// if policy.needs_double_tensor_for_f64() {
    ///     // Use DoubleTensor (hi + lo f32 pair) for ~48-bit precision
    /// } else {
    ///     // Use native f64 on GPU or CPU
    /// }
    /// ```
    pub const fn needs_double_tensor_for_f64(&self) -> bool {
        self.use_gpu && !self.is_hpc_tier()
    }

    /// Check if this tier is consumer-grade (needs DoubleTensor for f64).
    ///
    /// Consumer tiers: Apple Silicon, NVIDIA RTX consumer, AMD RDNA, Unknown.
    /// These have weak or no native f64 support on GPU.
    pub const fn is_consumer_tier(&self) -> bool {
        matches!(
            self.tier,
            HardwareTier::AppleSilicon
                | HardwareTier::NvidiaConsumer
                | HardwareTier::AmdRdna
                | HardwareTier::Unknown
        )
    }

    /// Get the recommended strategy for GPU f64 operations.
    ///
    /// Returns:
    /// - `GpuF64Strategy::NativeF64` for HPC tiers (H100, B200, Spark)
    /// - `GpuF64Strategy::DoubleTensor` for consumer tiers with GPU
    /// - `GpuF64Strategy::CpuFallback` for CPU-only or when GPU disabled
    pub fn gpu_f64_strategy(&self) -> GpuF64Strategy {
        if !self.use_gpu || self.tier == HardwareTier::CpuOnly {
            GpuF64Strategy::CpuFallback
        } else if self.is_hpc_tier() {
            GpuF64Strategy::NativeF64
        } else {
            GpuF64Strategy::DoubleTensor
        }
    }
}

/// Strategy for achieving f64-like precision on GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuF64Strategy {
    /// Use native f64 on GPU (datacenter GPUs: H100, B200, Spark).
    NativeF64,
    /// Use DoubleTensor (hi + lo f32 pair) for ~48-bit precision.
    /// For consumer GPUs: Apple Silicon, RTX consumer, AMD RDNA.
    DoubleTensor,
    /// Fall back to CPU f64 (no GPU or GPU disabled).
    CpuFallback,
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
    /// All operations on GPU in f32 (discrete GPU systems).
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

    /// HPC GPU with f64 support (H100, B200, etc. via CUDA).
    ///
    /// All operations on GPU using f64 precision. Requires CUDA backend.
    /// This enables true double-precision computation on datacenter GPUs
    /// without falling back to CPU.
    ///
    /// Only available when built with `cuda` feature.
    #[cfg(feature = "cuda")]
    GpuHpcF64,
}

// Default uses RuntimePolicy::detect() for hardware-aware configuration
impl Default for ComputeBackend {
    fn default() -> Self {
        let policy = RuntimePolicy::detect();
        Self::from_policy(&policy)
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
    pub const fn gpu_only() -> Self {
        Self::GpuOnly
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
    pub const fn cpu_only() -> Self {
        Self::CpuOnly
    }

    /// Creates a ComputeBackend from a RuntimePolicy.
    ///
    /// This is the recommended way to create a backend configuration
    /// based on detected or specified hardware capabilities.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::{ComputeBackend, RuntimePolicy};
    ///
    /// let policy = RuntimePolicy::detect();
    /// let backend = ComputeBackend::from_policy(&policy);
    /// ```
    pub fn from_policy(policy: &RuntimePolicy) -> Self {
        match policy.profile {
            PrecisionProfile::CpuFp64Strict => Self::UnifiedHybrid {
                cpu_ops: vec![
                    OpType::IsingSampling,
                    OpType::CategoricalSampling,
                    OpType::SphericalHarmonics,
                    OpType::ArcTrig,
                    OpType::ComplexArithmetic,
                    OpType::SmallMatmul,
                    OpType::GradientCompute,
                    OpType::LossReduction,
                ],
                small_matmul_threshold: 2000,
            },
            PrecisionProfile::GpuMixed => Self::UnifiedHybrid {
                cpu_ops: vec![
                    OpType::IsingSampling,
                    OpType::CategoricalSampling,
                    OpType::SphericalHarmonics,
                    OpType::GradientCompute,
                    OpType::LossReduction,
                ],
                small_matmul_threshold: 1000,
            },
            PrecisionProfile::GpuHpcFp64 => {
                // HPC GPUs (H100, B200) have strong FP64 support.
                // When CUDA is available, we can run precision ops on GPU in f64.
                // Without CUDA, fall back to CPU f64 for precision ops.
                #[cfg(feature = "cuda")]
                {
                    if policy.use_gpu {
                        // CUDA available - can use GPU f64 for all ops
                        Self::GpuHpcF64
                    } else {
                        Self::CpuOnly
                    }
                }

                #[cfg(not(feature = "cuda"))]
                {
                    // No CUDA - use WGPU for bulk ops, CPU for precision ops
                    // This is similar to CpuFp64Strict but acknowledges HPC capability
                    if policy.use_gpu {
                        ComputeBackend::UnifiedHybrid {
                            cpu_ops: vec![
                                OpType::IsingSampling,
                                OpType::CategoricalSampling,
                                OpType::SphericalHarmonics,
                                OpType::ArcTrig,
                                OpType::ComplexArithmetic,
                                OpType::GradientCompute,
                                OpType::LossReduction,
                            ],
                            small_matmul_threshold: 512, // Lower threshold - HPC GPUs handle more
                        }
                    } else {
                        ComputeBackend::CpuOnly
                    }
                }
            }
        }
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
        Self::UnifiedHybrid {
            cpu_ops: vec![
                OpType::IsingSampling,
                OpType::CategoricalSampling,
                OpType::SphericalHarmonics,
                OpType::ArcTrig,
                OpType::ComplexArithmetic,
                OpType::SmallMatmul,
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
    pub const fn hybrid(cpu_ops: Vec<OpType>, small_matmul_threshold: usize) -> Self {
        Self::UnifiedHybrid {
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
            Self::GpuOnly => false,
            Self::CpuOnly => true,
            Self::UnifiedHybrid {
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
            Self::Adaptive => {
                // Adaptive logic: use CPU for precision-sensitive ops
                // This includes training ops that can overflow in f32
                matches!(
                    op,
                    OpType::IsingSampling
                        | OpType::CategoricalSampling
                        | OpType::SphericalHarmonics
                        | OpType::ArcTrig
                        | OpType::ComplexArithmetic
                        | OpType::GradientCompute  // f32 accumulation can overflow
                        | OpType::LossReduction // logsumexp needs f64 for large batches
                )
            }
            #[cfg(feature = "cuda")]
            Self::GpuHpcF64 => {
                // HPC GPU with f64 - all ops run on GPU in double precision
                false
            }
        }
    }

    /// Check if this backend uses GPU f64 (HPC CUDA mode).
    ///
    /// When true, precision-sensitive operations should use CUDA f64
    /// tensors instead of WGPU f32 or CPU f64.
    pub const fn uses_gpu_f64(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            matches!(self, Self::GpuHpcF64)
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
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
        Self::Adaptive {
            sh_band_limit_threshold: 64,
            small_angle_threshold: 0.01,
            langevin_renorm_interval: 10000,
        }
    }
}

impl PrecisionMode {
    /// Fast GPU mode (fp32)
    pub const fn gpu_fast() -> Self {
        Self::GpuFast
    }

    /// Precise CPU mode (fp64)
    pub const fn cpu_precise() -> Self {
        Self::CpuPrecise
    }

    /// Should use f64 for the given band limit?
    pub const fn use_f64_for_sh(&self, band_limit: usize) -> bool {
        match self {
            Self::GpuFast => false,
            Self::CpuPrecise => true,
            Self::Adaptive {
                sh_band_limit_threshold,
                ..
            } => band_limit > *sh_band_limit_threshold,
        }
    }

    /// Should use Haversine formula for small angles?
    pub fn use_haversine(&self, angle: f32) -> bool {
        match self {
            Self::GpuFast => false,
            Self::CpuPrecise => true,
            Self::Adaptive {
                small_angle_threshold,
                ..
            } => angle < *small_angle_threshold,
        }
    }

    /// Should re-normalize at this step?
    pub const fn should_renormalize(&self, step: usize) -> bool {
        match self {
            Self::GpuFast => false,
            Self::CpuPrecise => step.is_multiple_of(1000),
            Self::Adaptive {
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
    pub const fn nvidia_discrete() -> Self {
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

    /// Create HybridConfig from a RuntimePolicy (recommended).
    ///
    /// This creates a complete configuration based on detected or
    /// specified hardware capabilities.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::{HybridConfig, RuntimePolicy};
    ///
    /// let policy = RuntimePolicy::detect();
    /// let config = HybridConfig::from_policy(&policy);
    /// ```
    pub fn from_policy(policy: &RuntimePolicy) -> Self {
        Self {
            backend: ComputeBackend::from_policy(policy),
            precision: match policy.profile {
                PrecisionProfile::CpuFp64Strict => PrecisionMode::CpuPrecise,
                PrecisionProfile::GpuMixed => PrecisionMode::Adaptive {
                    sh_band_limit_threshold: 64,
                    small_angle_threshold: 0.01,
                    langevin_renorm_interval: 10000,
                },
                PrecisionProfile::GpuHpcFp64 => PrecisionMode::GpuFast,
            },
            cpu_threads: num_cpus::get().min(8),
            enable_overlap: policy.use_gpu,
        }
    }

    /// Auto-detect hardware and create appropriate config.
    ///
    /// This is a convenience method equivalent to:
    /// `HybridConfig::from_policy(&RuntimePolicy::detect())`
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::HybridConfig;
    ///
    /// let config = HybridConfig::detect();
    /// println!("Backend: {:?}", config.backend);
    /// ```
    pub fn detect() -> Self {
        Self::from_policy(&RuntimePolicy::detect())
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

    // =========================================================================
    // RuntimePolicy tests - lock in the tier → profile contract
    // =========================================================================

    #[test]
    fn test_runtime_policy_apple_silicon() {
        let policy = RuntimePolicy::apple_silicon();
        assert_eq!(policy.tier, HardwareTier::AppleSilicon);
        assert_eq!(policy.profile, PrecisionProfile::CpuFp64Strict);
        assert!(policy.use_gpu);
        assert!(policy.allow_mixed_precision); // GPU f32 + CPU f64 mixing
                                               // GPU uses f32, CPU f64 for precision ops
        assert_eq!(policy.real_dtype, DType::F32);
    }

    #[test]
    fn test_runtime_policy_nvidia_consumer() {
        let policy = RuntimePolicy::nvidia_consumer();
        assert_eq!(policy.tier, HardwareTier::NvidiaConsumer);
        assert_eq!(policy.profile, PrecisionProfile::GpuMixed);
        assert!(policy.use_gpu);
        assert!(policy.allow_mixed_precision);
    }

    #[test]
    fn test_runtime_policy_nvidia_hopper() {
        let policy = RuntimePolicy::nvidia_hopper();
        assert_eq!(policy.tier, HardwareTier::NvidiaHopper);
        assert_eq!(policy.profile, PrecisionProfile::GpuHpcFp64);
        assert!(policy.use_gpu);
        // HPC can do f64 on GPU
        assert_eq!(policy.real_dtype, DType::F64);
        assert!(policy.is_hpc_tier());
    }

    #[test]
    fn test_runtime_policy_nvidia_blackwell() {
        let policy = RuntimePolicy::nvidia_blackwell();
        assert_eq!(policy.tier, HardwareTier::NvidiaBlackwell);
        assert_eq!(policy.profile, PrecisionProfile::GpuHpcFp64);
        assert!(policy.is_hpc_tier());
    }

    #[test]
    fn test_runtime_policy_nvidia_spark() {
        let policy = RuntimePolicy::nvidia_spark();
        assert_eq!(policy.tier, HardwareTier::NvidiaSpark);
        assert_eq!(policy.profile, PrecisionProfile::GpuHpcFp64);
        assert!(policy.is_hpc_tier());
        // Spark has unified memory like Apple Silicon but with strong f64
        assert_eq!(policy.real_dtype, DType::F64);
    }

    #[test]
    fn test_runtime_policy_cpu_only() {
        let policy = RuntimePolicy::cpu_only();
        assert_eq!(policy.tier, HardwareTier::CpuOnly);
        assert_eq!(policy.profile, PrecisionProfile::CpuFp64Strict);
        assert!(!policy.use_gpu);
        assert_eq!(policy.real_dtype, DType::F64);
    }

    #[test]
    fn test_runtime_policy_for_tier_roundtrip() {
        // Verify for_tier produces the same policy as direct constructors
        let tiers = [
            HardwareTier::AppleSilicon,
            HardwareTier::NvidiaConsumer,
            HardwareTier::NvidiaHopper,
            HardwareTier::NvidiaBlackwell,
            HardwareTier::NvidiaSpark,
            HardwareTier::CpuOnly,
        ];

        for tier in tiers {
            let policy = RuntimePolicy::for_tier(tier);
            assert_eq!(policy.tier, tier);
        }
    }

    #[test]
    fn test_compute_backend_from_policy() {
        // CpuFp64Strict → UnifiedHybrid with CPU ops for precision
        let policy = RuntimePolicy::apple_silicon();
        let backend = ComputeBackend::from_policy(&policy);
        assert!(backend.use_cpu(OpType::IsingSampling, None));
        assert!(!backend.use_cpu(OpType::Similarity, None));

        // GpuHpcFp64 with CUDA → can do precision ops on GPU
        let policy = RuntimePolicy::nvidia_hopper();
        let backend = ComputeBackend::from_policy(&policy);

        #[cfg(feature = "cuda")]
        {
            // With CUDA, HPC can do precision ops on GPU (f64 support)
            assert!(!backend.use_cpu(OpType::IsingSampling, None));
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Without CUDA feature, HPC still routes precision ops to CPU
            assert!(backend.use_cpu(OpType::IsingSampling, None));
        }
    }

    #[test]
    fn test_precision_profile_mapping() {
        // Verify precision_dtype returns expected values
        let apple = RuntimePolicy::apple_silicon();
        assert_eq!(apple.precision_dtype(), DType::F64); // CpuFp64Strict → F64

        let consumer = RuntimePolicy::nvidia_consumer();
        assert_eq!(consumer.precision_dtype(), DType::F32); // GpuMixed → F32

        let hpc = RuntimePolicy::nvidia_hopper();
        assert_eq!(hpc.precision_dtype(), DType::F64); // GpuHpcFp64 → F64
    }

    #[test]
    fn test_platform_fallback() {
        // Platform fallback should be deterministic
        let fallback = RuntimePolicy::platform_fallback_tier();
        #[cfg(target_os = "macos")]
        assert_eq!(fallback, HardwareTier::AppleSilicon);
        #[cfg(not(target_os = "macos"))]
        assert_eq!(fallback, HardwareTier::Unknown);
    }
}
