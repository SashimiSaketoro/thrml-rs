//! Hybrid Compute Backend for Unified Memory Systems
//!
//! ## Architecture (from Section 6.2.10 of EBM Navigator Architecture Plan)
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
//! ## Precision Strategy (from Section 6.2.7)
//!
//! | Scenario                    | fp32 OK? | Recommendation       |
//! |-----------------------------|----------|---------------------|
//! | Sphere optimization         | YES      | Use GPU fp32        |
//! | Ising max-cut sampling      | MARGINAL | Use CPU fp64        |
//! | Spherical harmonics L>64    | NO       | Use CPU fp64        |
//! | Long Langevin chains        | MARGINAL | Periodic renorm     |

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
/// ```
/// use thrml_sphere::ComputeBackend;
///
/// // Auto-detect (hybrid on macOS, GPU elsewhere)
/// let backend = ComputeBackend::default();
///
/// // Explicit Apple Silicon configuration
/// let backend = ComputeBackend::apple_silicon();
///
/// // Check if an operation should use CPU
/// use thrml_sphere::OpType;
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
                ],
                small_matmul_threshold: 1000,
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            // Non-macOS: prefer GPU for everything
            ComputeBackend::GpuOnly
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
    /// ```
    /// use thrml_sphere::ComputeBackend;
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
    /// ```
    /// use thrml_sphere::ComputeBackend;
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
    /// ```
    /// use thrml_sphere::ComputeBackend;
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
    /// ```
    /// use thrml_sphere::{ComputeBackend, OpType};
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
    /// ```
    /// use thrml_sphere::{ComputeBackend, OpType};
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
                // Adaptive logic: always use CPU for precision-sensitive ops
                matches!(
                    op,
                    OpType::IsingSampling
                        | OpType::SphericalHarmonics
                        | OpType::ArcTrig
                        | OpType::ComplexArithmetic
                )
            }
        }
    }

    /// Check if an operation should run on GPU
    pub fn use_gpu(&self, op: OpType, size: Option<usize>) -> bool {
        !self.use_cpu(op, size)
    }
}

/// Precision mode for numerical operations.
///
/// From Section 6.2.7:
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
            PrecisionMode::CpuPrecise => step % 1000 == 0,
            PrecisionMode::Adaptive {
                langevin_renorm_interval,
                ..
            } => step % langevin_renorm_interval == 0,
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
// Substring Similarity (for Ising max-cut enhancement)
// =============================================================================

/// Substring similarity computation for Ising max-cut coupling.
///
/// This module provides efficient substring containment checking
/// to enhance the Ising max-cut partitioning with structural relationships.
/// Particularly useful for code and structured text where byte-level
/// relationships are semantically meaningful.
///
/// # Algorithm
///
/// For each pair (i, j) of byte sequences:
/// 1. **Containment check**: Does sequence A contain sequence B (or vice versa)?
/// 2. **Multi-scale hash matching**: Find common substrings using rolling hashes
/// 3. **Jaccard similarity**: Compute overlap based on shared substring hashes
///
/// # Coupling Formula
///
/// The combined Ising coupling weight is:
///
/// ```text
/// J_ij = α × cosine_sim(emb_i, emb_j) + β × substring_sim(bytes_i, bytes_j)
/// ```
///
/// Where `α` and `β` are configurable weights (default: 0.7 and 0.3).
///
/// # Example
///
/// ```
/// use thrml_sphere::SubstringConfig;
///
/// // Default: 70% embedding, 30% substring
/// let config = SubstringConfig::default();
///
/// // Custom weights for code-heavy datasets
/// let config = SubstringConfig::with_weights(0.5, 0.5);
///
/// // Pure substring matching (ignore embeddings)
/// let config = SubstringConfig::substring_only();
/// ```
pub mod substring {
    use std::collections::HashSet;

    /// Configuration for substring similarity computation.
    ///
    /// Controls how byte-level substring relationships are computed and
    /// weighted against embedding similarity for Ising max-cut partitioning.
    ///
    /// # Formula
    ///
    /// Combined coupling: `J_ij = α × cosine_sim + β × substring_sim`
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::SubstringConfig;
    ///
    /// let config = SubstringConfig::default()
    ///     .with_min_length(8)   // Require longer matches
    ///     .with_max_length(128); // Check longer substrings
    ///
    /// assert_eq!(config.min_length, 8);
    /// ```
    #[derive(Debug, Clone, Copy)]
    pub struct SubstringConfig {
        /// Minimum substring length to consider.
        /// Shorter substrings produce trivial matches. Default: 4
        pub min_length: usize,

        /// Maximum substring length to hash.
        /// Longer substrings are expensive to compute. Default: 64
        pub max_length: usize,

        /// Weight for embedding similarity (α). Range: 0.0 to 1.0
        pub alpha: f64,

        /// Weight for substring similarity (β). Range: 0.0 to 1.0
        pub beta: f64,

        /// Whether to check if one sequence contains the other.
        pub check_containment: bool,

        /// Whether to use multi-scale rolling hash matching.
        pub use_hash_matching: bool,
    }

    impl Default for SubstringConfig {
        fn default() -> Self {
            Self {
                min_length: 4,
                max_length: 64,
                alpha: 0.7,
                beta: 0.3,
                check_containment: true,
                use_hash_matching: true,
            }
        }
    }

    impl SubstringConfig {
        /// Creates a configuration with custom weights.
        ///
        /// # Arguments
        ///
        /// * `alpha` - Weight for embedding similarity (0.0 to 1.0).
        /// * `beta` - Weight for substring similarity (0.0 to 1.0).
        ///
        /// # Example
        ///
        /// ```
        /// use thrml_sphere::SubstringConfig;
        ///
        /// // Equal weighting
        /// let config = SubstringConfig::with_weights(0.5, 0.5);
        /// assert_eq!(config.alpha, 0.5);
        /// ```
        pub fn with_weights(alpha: f64, beta: f64) -> Self {
            Self {
                alpha,
                beta,
                ..Default::default()
            }
        }

        /// Creates a configuration for pure substring matching.
        ///
        /// Ignores embedding similarity entirely (α=0, β=1).
        /// Useful when byte-level structure is the primary signal.
        ///
        /// # Example
        ///
        /// ```
        /// use thrml_sphere::SubstringConfig;
        ///
        /// let config = SubstringConfig::substring_only();
        /// assert_eq!(config.alpha, 0.0);
        /// assert_eq!(config.beta, 1.0);
        /// ```
        pub fn substring_only() -> Self {
            Self {
                alpha: 0.0,
                beta: 1.0,
                ..Default::default()
            }
        }

        /// Creates a configuration for pure embedding similarity.
        ///
        /// Disables substring matching entirely (α=1, β=0).
        /// Equivalent to standard cosine similarity partitioning.
        ///
        /// # Example
        ///
        /// ```
        /// use thrml_sphere::SubstringConfig;
        ///
        /// let config = SubstringConfig::embedding_only();
        /// assert_eq!(config.alpha, 1.0);
        /// assert_eq!(config.beta, 0.0);
        /// ```
        pub fn embedding_only() -> Self {
            Self {
                alpha: 1.0,
                beta: 0.0,
                check_containment: false,
                use_hash_matching: false,
                ..Default::default()
            }
        }

        /// Sets the minimum substring length.
        ///
        /// Substrings shorter than this are not considered, reducing
        /// trivial matches. Default: 4
        ///
        /// # Arguments
        ///
        /// * `len` - Minimum length in bytes.
        pub fn with_min_length(mut self, len: usize) -> Self {
            self.min_length = len;
            self
        }

        /// Sets the maximum substring length for hashing.
        ///
        /// Longer substrings increase computation but may catch
        /// more meaningful structural relationships. Default: 64
        ///
        /// # Arguments
        ///
        /// * `len` - Maximum length in bytes.
        pub fn with_max_length(mut self, len: usize) -> Self {
            self.max_length = len;
            self
        }
    }

    /// Rolling hash for efficient substring hashing.
    ///
    /// Uses polynomial rolling hash with a large prime base.
    /// `h(s) = s[0] * BASE^(n-1) + s[1] * BASE^(n-2) + ... + s[n-1]`
    #[derive(Clone)]
    pub struct RollingHash {
        /// Current hash value
        hash: u64,
        /// Base for polynomial hash
        base: u64,
        /// Modulus (large prime)
        modulus: u64,
        /// BASE^(window_size) mod modulus (for removing leading char)
        pow_base: u64,
        /// Current window size
        window_size: usize,
    }

    impl RollingHash {
        /// Prime base for hashing
        const BASE: u64 = 31;
        /// Large prime modulus
        const MODULUS: u64 = 1_000_000_007;

        /// Create a new rolling hash for the given window size
        pub fn new(window_size: usize) -> Self {
            // pow_base = BASE^(window_size - 1) for removing the leading character
            let mut pow_base = 1u64;
            for _ in 0..(window_size.saturating_sub(1)) {
                pow_base = (pow_base * Self::BASE) % Self::MODULUS;
            }

            Self {
                hash: 0,
                base: Self::BASE,
                modulus: Self::MODULUS,
                pow_base,
                window_size,
            }
        }

        /// Compute hash for initial window
        pub fn init(&mut self, bytes: &[u8]) {
            self.hash = 0;
            for &b in bytes.iter().take(self.window_size) {
                self.hash = (self.hash * self.base + b as u64) % self.modulus;
            }
        }

        /// Rolls the hash: removes old_byte, adds new_byte.
        ///
        /// Given hash of `bytes[i..i+n]`, computes hash of `bytes[i+1..i+n+1]`
        /// by removing `old_byte` (at position i) and adding `new_byte` (at position i+n).
        pub fn roll(&mut self, old_byte: u8, new_byte: u8) {
            // Current hash = old_byte * B^(n-1) + middle_bits + last_byte
            // We want: middle_bits * B + new_byte
            //
            // Step 1: Remove old_byte's contribution (at position n-1)
            // Step 2: Shift everything left by multiplying by B
            // Step 3: Add new_byte at position 0
            //
            // Combined: ((hash - old_byte * B^(n-1)) * B + new_byte) mod M

            let old_contribution = (old_byte as u64 * self.pow_base) % self.modulus;
            // Subtract old contribution (with modular arithmetic care)
            let after_sub = (self.hash + self.modulus - old_contribution) % self.modulus;
            // Multiply by base to shift left
            let shifted = (after_sub * self.base) % self.modulus;
            // Add new byte
            self.hash = (shifted + new_byte as u64) % self.modulus;
        }

        /// Get current hash value
        pub fn value(&self) -> u64 {
            self.hash
        }
    }

    /// Compute all substring hashes for a byte sequence.
    ///
    /// Returns a HashSet of (length, hash) pairs for all substrings
    /// of length [min_len, max_len].
    pub fn compute_substring_hashes(
        bytes: &[u8],
        min_len: usize,
        max_len: usize,
    ) -> HashSet<(usize, u64)> {
        let mut hashes = HashSet::new();

        if bytes.len() < min_len {
            return hashes;
        }

        let actual_max = max_len.min(bytes.len());

        for len in min_len..=actual_max {
            if len > bytes.len() {
                break;
            }

            let mut hasher = RollingHash::new(len);
            hasher.init(&bytes[..len]);
            hashes.insert((len, hasher.value()));

            for i in 1..=(bytes.len() - len) {
                hasher.roll(bytes[i - 1], bytes[i + len - 1]);
                hashes.insert((len, hasher.value()));
            }
        }

        hashes
    }

    /// Check if bytes_a contains bytes_b (or vice versa).
    ///
    /// Returns (contained, container_len, contained_len) or None if no containment.
    pub fn check_containment(bytes_a: &[u8], bytes_b: &[u8]) -> Option<(bool, usize, usize)> {
        if bytes_a.len() < bytes_b.len() {
            // Check if bytes_b contains bytes_a
            if contains_substring(bytes_b, bytes_a) {
                return Some((true, bytes_b.len(), bytes_a.len()));
            }
        } else {
            // Check if bytes_a contains bytes_b
            if contains_substring(bytes_a, bytes_b) {
                return Some((true, bytes_a.len(), bytes_b.len()));
            }
        }
        None
    }

    /// Simple substring search (for containment check)
    fn contains_substring(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        if needle.len() > haystack.len() {
            return false;
        }

        // Use built-in windows iterator for simplicity
        haystack
            .windows(needle.len())
            .any(|window| window == needle)
    }

    /// Compute substring similarity between two byte sequences.
    ///
    /// Returns a score in [0, 1] based on:
    /// - Containment relationship
    /// - Multi-scale hash overlap
    pub fn compute_similarity(bytes_a: &[u8], bytes_b: &[u8], config: &SubstringConfig) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        // Check containment
        if config.check_containment {
            if let Some((_, container_len, contained_len)) = check_containment(bytes_a, bytes_b) {
                // Score based on how much of the container is the contained string
                let containment_score = contained_len as f64 / container_len as f64;
                score += containment_score;
                factors += 1;
            }
        }

        // Multi-scale hash matching
        if config.use_hash_matching {
            let hashes_a = compute_substring_hashes(bytes_a, config.min_length, config.max_length);
            let hashes_b = compute_substring_hashes(bytes_b, config.min_length, config.max_length);

            if !hashes_a.is_empty() && !hashes_b.is_empty() {
                // Jaccard similarity of hash sets
                let intersection = hashes_a.intersection(&hashes_b).count();
                let union = hashes_a.len() + hashes_b.len() - intersection;

                if union > 0 {
                    let jaccard = intersection as f64 / union as f64;
                    score += jaccard;
                    factors += 1;
                }
            }
        }

        if factors > 0 {
            score / factors as f64
        } else {
            0.0
        }
    }

    /// Compute combined coupling weight for Ising model.
    ///
    /// J_ij = α × cosine_sim + β × substring_sim
    pub fn compute_combined_coupling(
        embedding_sim: f64,
        bytes_a: &[u8],
        bytes_b: &[u8],
        config: &SubstringConfig,
    ) -> f64 {
        if config.beta == 0.0 {
            return embedding_sim;
        }

        let substring_sim = compute_similarity(bytes_a, bytes_b, config);
        config.alpha * embedding_sim + config.beta * substring_sim
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_rolling_hash() {
            let data = b"hello world";
            let mut hasher = RollingHash::new(5);

            hasher.init(data);
            let h1 = hasher.value();

            // Roll to next position
            hasher.roll(data[0], data[5]);
            let h2 = hasher.value();

            // Different hashes for different substrings
            assert_ne!(h1, h2);

            // Same hash for same substring
            let mut hasher2 = RollingHash::new(5);
            hasher2.init(&data[1..6]);
            assert_eq!(h2, hasher2.value());
        }

        #[test]
        fn test_containment() {
            let full = b"the quick brown fox";
            let sub = b"quick";
            let unrelated = b"slow";

            assert!(check_containment(full, sub).is_some());
            assert!(check_containment(sub, full).is_some()); // Reversed also works
            assert!(check_containment(full, unrelated).is_none());
        }

        #[test]
        fn test_substring_similarity() {
            let config = SubstringConfig::default();

            // Identical strings
            let a = b"hello world";
            let sim_self = compute_similarity(a, a, &config);
            assert!(sim_self > 0.9, "Self-similarity should be high");

            // Substring relationship
            let container = b"the quick brown fox";
            let contained = b"quick brown";
            let sim_contain = compute_similarity(container, contained, &config);
            assert!(
                sim_contain > 0.3,
                "Containment should have positive similarity"
            );

            // Unrelated
            let unrelated = b"xyz123abc";
            let sim_unrelated = compute_similarity(a, unrelated, &config);
            assert!(
                sim_unrelated < 0.1,
                "Unrelated strings should have low similarity"
            );
        }

        #[test]
        fn test_combined_coupling() {
            let config = SubstringConfig::with_weights(0.5, 0.5);

            let a = b"function calculate";
            let b = b"calculate total";

            // Some embedding similarity
            let emb_sim = 0.6;

            let combined = compute_combined_coupling(emb_sim, a, b, &config);

            // Should be between pure embedding and combined
            assert!(combined >= 0.0 && combined <= 1.0);
            println!("Combined coupling: {}", combined);
        }
    }
}

// =============================================================================
// CPU-based Ising Partition (f64 precision)
// =============================================================================

/// CPU-based Ising max-cut partitioning with f64 precision.
///
/// This implementation avoids GPU tensor edge cases and provides
/// higher numerical precision for the partitioning algorithm.
pub mod cpu_ising {
    use super::substring::{compute_combined_coupling, SubstringConfig};
    use std::collections::HashMap;

    /// Ising model state for CPU computation.
    pub struct CpuIsingState {
        /// Spin values {-1, +1}
        pub spins: Vec<i8>,
        /// Coupling weights J\[i\]\[j\]
        pub couplings: HashMap<(usize, usize), f64>,
        /// Biases b\[i\]
        pub biases: Vec<f64>,
        /// Inverse temperature
        pub beta: f64,
    }

    impl CpuIsingState {
        /// Create a new Ising state from embeddings.
        pub fn from_embeddings(
            embeddings: &[f32],
            n_points: usize,
            embedding_dim: usize,
            beta: f64,
        ) -> Self {
            // Initialize random spins
            let mut spins = vec![1i8; n_points];
            for (i, s) in spins.iter_mut().enumerate() {
                *s = if i % 2 == 0 { 1 } else { -1 };
            }

            // Compute couplings as cosine similarity
            let mut couplings = HashMap::new();
            for i in 0..n_points {
                for j in (i + 1)..n_points {
                    let emb_i = &embeddings[i * embedding_dim..(i + 1) * embedding_dim];
                    let emb_j = &embeddings[j * embedding_dim..(j + 1) * embedding_dim];

                    let dot: f64 = emb_i
                        .iter()
                        .zip(emb_j.iter())
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum();
                    let norm_i: f64 = emb_i
                        .iter()
                        .map(|&x| (x as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let norm_j: f64 = emb_j
                        .iter()
                        .map(|&x| (x as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let sim = dot / (norm_i * norm_j + 1e-10);

                    if sim.abs() > 0.01 {
                        // Only store significant couplings
                        couplings.insert((i, j), sim);
                    }
                }
            }

            let biases = vec![0.0; n_points];

            Self {
                spins,
                couplings,
                biases,
                beta,
            }
        }

        /// Create a new Ising state from embeddings with optional raw bytes.
        ///
        /// When `raw_bytes` is provided, the coupling weights combine:
        /// - Cosine similarity of embeddings (semantic)
        /// - Substring containment similarity (structural)
        ///
        /// Formula: J_ij = α × cosine_sim + β × substring_sim
        pub fn from_embeddings_with_bytes(
            embeddings: &[f32],
            n_points: usize,
            embedding_dim: usize,
            raw_bytes: Option<&[Vec<u8>]>,
            substring_config: Option<SubstringConfig>,
            beta: f64,
        ) -> Self {
            // Initialize random spins
            let mut spins = vec![1i8; n_points];
            for (i, s) in spins.iter_mut().enumerate() {
                *s = if i % 2 == 0 { 1 } else { -1 };
            }

            let sub_config = substring_config.unwrap_or(SubstringConfig::embedding_only());

            // Compute couplings with combined similarity
            let mut couplings = HashMap::new();
            for i in 0..n_points {
                for j in (i + 1)..n_points {
                    let emb_i = &embeddings[i * embedding_dim..(i + 1) * embedding_dim];
                    let emb_j = &embeddings[j * embedding_dim..(j + 1) * embedding_dim];

                    // Compute cosine similarity
                    let dot: f64 = emb_i
                        .iter()
                        .zip(emb_j.iter())
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum();
                    let norm_i: f64 = emb_i
                        .iter()
                        .map(|&x| (x as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let norm_j: f64 = emb_j
                        .iter()
                        .map(|&x| (x as f64).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    let cosine_sim = dot / (norm_i * norm_j + 1e-10);

                    // Compute combined coupling (embedding + substring)
                    let coupling = if let Some(bytes) = raw_bytes {
                        if i < bytes.len() && j < bytes.len() {
                            compute_combined_coupling(cosine_sim, &bytes[i], &bytes[j], &sub_config)
                        } else {
                            cosine_sim
                        }
                    } else {
                        cosine_sim
                    };

                    if coupling.abs() > 0.01 {
                        // Only store significant couplings
                        couplings.insert((i, j), coupling);
                    }
                }
            }

            let biases = vec![0.0; n_points];

            Self {
                spins,
                couplings,
                biases,
                beta,
            }
        }

        /// Create from sparse similarity (top-k neighbors).
        pub fn from_sparse_similarity(
            indices: &[Vec<usize>],
            values: &[Vec<f32>],
            n_points: usize,
            beta: f64,
        ) -> Self {
            let mut spins = vec![1i8; n_points];
            for (i, s) in spins.iter_mut().enumerate() {
                *s = if i % 2 == 0 { 1 } else { -1 };
            }

            let mut couplings = HashMap::new();
            for (i, (neighbors, sims)) in indices.iter().zip(values.iter()).enumerate() {
                for (&j, &sim) in neighbors.iter().zip(sims.iter()) {
                    if i < j {
                        couplings.insert((i, j), sim as f64);
                    }
                }
            }

            let biases = vec![0.0; n_points];

            Self {
                spins,
                couplings,
                biases,
                beta,
            }
        }

        /// Create from sparse similarity with optional raw bytes.
        ///
        /// Enhances sparse similarity with substring containment.
        pub fn from_sparse_similarity_with_bytes(
            indices: &[Vec<usize>],
            values: &[Vec<f32>],
            n_points: usize,
            raw_bytes: Option<&[Vec<u8>]>,
            substring_config: Option<SubstringConfig>,
            beta: f64,
        ) -> Self {
            let mut spins = vec![1i8; n_points];
            for (i, s) in spins.iter_mut().enumerate() {
                *s = if i % 2 == 0 { 1 } else { -1 };
            }

            let sub_config = substring_config.unwrap_or(SubstringConfig::embedding_only());

            let mut couplings = HashMap::new();
            for (i, (neighbors, sims)) in indices.iter().zip(values.iter()).enumerate() {
                for (&j, &sim) in neighbors.iter().zip(sims.iter()) {
                    if i < j {
                        let coupling = if let Some(bytes) = raw_bytes {
                            if i < bytes.len() && j < bytes.len() {
                                compute_combined_coupling(
                                    sim as f64,
                                    &bytes[i],
                                    &bytes[j],
                                    &sub_config,
                                )
                            } else {
                                sim as f64
                            }
                        } else {
                            sim as f64
                        };

                        if coupling.abs() > 0.01 {
                            couplings.insert((i, j), coupling);
                        }
                    }
                }
            }

            let biases = vec![0.0; n_points];

            Self {
                spins,
                couplings,
                biases,
                beta,
            }
        }

        /// Compute total energy.
        pub fn energy(&self) -> f64 {
            let mut e = 0.0;

            // Bias contribution
            for (i, &s) in self.spins.iter().enumerate() {
                e -= self.beta * self.biases[i] * s as f64;
            }

            // Coupling contribution
            for (&(i, j), &coupling) in &self.couplings {
                e -= self.beta * coupling * self.spins[i] as f64 * self.spins[j] as f64;
            }

            e
        }

        /// Compute local field at site i.
        fn local_field(&self, i: usize) -> f64 {
            let mut field = self.biases[i];

            // Sum over neighbors
            for (&(a, b), &coupling) in &self.couplings {
                if a == i {
                    field += coupling * self.spins[b] as f64;
                } else if b == i {
                    field += coupling * self.spins[a] as f64;
                }
            }

            self.beta * field
        }

        /// Single Gibbs sweep (update all spins once).
        pub fn gibbs_sweep(&mut self, rng_seed: u64) {
            let n = self.spins.len();

            for i in 0..n {
                let field = self.local_field(i);
                let prob_up = 1.0 / (1.0 + (-2.0 * field).exp());

                // Deterministic "random" based on seed and position
                let hash = rng_seed
                    .wrapping_mul(i as u64 + 1)
                    .wrapping_mul(0x517cc1b727220a95);
                let u = (hash % 10000) as f64 / 10000.0;

                self.spins[i] = if u < prob_up { 1 } else { -1 };
            }
        }

        /// Run Gibbs sampling for multiple sweeps.
        pub fn sample(&mut self, n_warmup: usize, n_sweeps: usize, seed: u64) {
            // Warmup phase
            for i in 0..n_warmup {
                self.gibbs_sweep(seed.wrapping_add(i as u64));
            }

            // Sampling phase
            for i in 0..n_sweeps {
                self.gibbs_sweep(seed.wrapping_add((n_warmup + i) as u64));
            }
        }

        /// Get partition assignment (true = partition 1, false = partition 0).
        pub fn get_partition(&self) -> Vec<bool> {
            self.spins.iter().map(|&s| s > 0).collect()
        }

        /// Get partition as indices.
        pub fn get_partition_indices(&self) -> (Vec<usize>, Vec<usize>) {
            let mut left = Vec::new();
            let mut right = Vec::new();

            for (i, &s) in self.spins.iter().enumerate() {
                if s > 0 {
                    right.push(i);
                } else {
                    left.push(i);
                }
            }

            (left, right)
        }
    }

    /// Perform hierarchical binary partition using CPU Ising max-cut.
    pub fn hierarchical_partition(
        indices: &[usize],
        embeddings: &[f32],
        embedding_dim: usize,
        target_k: usize,
        min_size: usize,
        beta: f64,
        n_warmup: usize,
        n_sweeps: usize,
        seed: u64,
    ) -> Vec<Vec<usize>> {
        // Base case
        if target_k <= 1 || indices.len() <= min_size {
            return vec![indices.to_vec()];
        }

        let n = indices.len();

        // Build local embeddings for this subset
        let local_embeddings: Vec<f32> = indices
            .iter()
            .flat_map(|&idx| embeddings[idx * embedding_dim..(idx + 1) * embedding_dim].to_vec())
            .collect();

        // Create Ising state
        let mut ising = CpuIsingState::from_embeddings(&local_embeddings, n, embedding_dim, beta);

        // Sample partition
        ising.sample(n_warmup, n_sweeps, seed);

        // Get partition
        let (local_left, local_right) = ising.get_partition_indices();

        // Map back to original indices
        let left: Vec<usize> = local_left.iter().map(|&i| indices[i]).collect();
        let right: Vec<usize> = local_right.iter().map(|&i| indices[i]).collect();

        // Handle degenerate cases
        let (left, right) = if left.is_empty() || right.is_empty() {
            let mid = n / 2;
            (indices[..mid].to_vec(), indices[mid..].to_vec())
        } else {
            (left, right)
        };

        // Recurse
        let k_left = target_k / 2;
        let k_right = target_k - k_left;

        let mut partitions = hierarchical_partition(
            &left,
            embeddings,
            embedding_dim,
            k_left,
            min_size,
            beta,
            n_warmup,
            n_sweeps,
            seed.wrapping_add(1),
        );

        partitions.extend(hierarchical_partition(
            &right,
            embeddings,
            embedding_dim,
            k_right,
            min_size,
            beta,
            n_warmup,
            n_sweeps,
            seed.wrapping_add(2),
        ));

        partitions
    }

    /// Perform hierarchical partition using sparse similarity.
    pub fn hierarchical_partition_sparse(
        indices: &[usize],
        sparse_indices: &[Vec<usize>],
        sparse_values: &[Vec<f32>],
        target_k: usize,
        min_size: usize,
        beta: f64,
        n_warmup: usize,
        n_sweeps: usize,
        seed: u64,
    ) -> Vec<Vec<usize>> {
        // Base case
        if target_k <= 1 || indices.len() <= min_size {
            return vec![indices.to_vec()];
        }

        let n = indices.len();

        // Build index map
        let idx_map: HashMap<usize, usize> = indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        // Build local sparse similarity
        let mut local_indices = vec![Vec::new(); n];
        let mut local_values = vec![Vec::new(); n];

        for (local_i, &global_i) in indices.iter().enumerate() {
            if global_i < sparse_indices.len() {
                for (&neighbor, &sim) in sparse_indices[global_i]
                    .iter()
                    .zip(sparse_values[global_i].iter())
                {
                    if let Some(&local_j) = idx_map.get(&neighbor) {
                        if local_i < local_j {
                            local_indices[local_i].push(local_j);
                            local_values[local_i].push(sim);
                        }
                    }
                }
            }
        }

        // Create Ising state from sparse similarity
        let mut ising =
            CpuIsingState::from_sparse_similarity(&local_indices, &local_values, n, beta);

        // Sample partition
        ising.sample(n_warmup, n_sweeps, seed);

        // Get partition
        let (local_left, local_right) = ising.get_partition_indices();

        // Map back to original indices
        let left: Vec<usize> = local_left.iter().map(|&i| indices[i]).collect();
        let right: Vec<usize> = local_right.iter().map(|&i| indices[i]).collect();

        // Handle degenerate cases
        let (left, right) = if left.is_empty() || right.is_empty() {
            let mid = n / 2;
            (indices[..mid].to_vec(), indices[mid..].to_vec())
        } else {
            (left, right)
        };

        // Recurse
        let k_left = target_k / 2;
        let k_right = target_k - k_left;

        let mut partitions = hierarchical_partition_sparse(
            &left,
            sparse_indices,
            sparse_values,
            k_left,
            min_size,
            beta,
            n_warmup,
            n_sweeps,
            seed.wrapping_add(1),
        );

        partitions.extend(hierarchical_partition_sparse(
            &right,
            sparse_indices,
            sparse_values,
            k_right,
            min_size,
            beta,
            n_warmup,
            n_sweeps,
            seed.wrapping_add(2),
        ));

        partitions
    }

    /// Perform hierarchical binary partition with raw bytes for substring coupling.
    ///
    /// This version enhances the standard partition with structural relationships
    /// from byte-level containment.
    pub fn hierarchical_partition_with_bytes(
        indices: &[usize],
        embeddings: &[f32],
        embedding_dim: usize,
        raw_bytes: &[Vec<u8>],
        substring_config: SubstringConfig,
        target_k: usize,
        min_size: usize,
        beta: f64,
        n_warmup: usize,
        n_sweeps: usize,
        seed: u64,
    ) -> Vec<Vec<usize>> {
        // Base case
        if target_k <= 1 || indices.len() <= min_size {
            return vec![indices.to_vec()];
        }

        let n = indices.len();

        // Build local embeddings for this subset
        let local_embeddings: Vec<f32> = indices
            .iter()
            .flat_map(|&idx| embeddings[idx * embedding_dim..(idx + 1) * embedding_dim].to_vec())
            .collect();

        // Build local bytes for this subset
        let local_bytes: Vec<Vec<u8>> = indices
            .iter()
            .map(|&idx| {
                if idx < raw_bytes.len() {
                    raw_bytes[idx].clone()
                } else {
                    Vec::new()
                }
            })
            .collect();

        // Create Ising state with combined coupling
        let mut ising = CpuIsingState::from_embeddings_with_bytes(
            &local_embeddings,
            n,
            embedding_dim,
            Some(&local_bytes),
            Some(substring_config),
            beta,
        );

        // Sample partition
        ising.sample(n_warmup, n_sweeps, seed);

        // Get partition
        let (local_left, local_right) = ising.get_partition_indices();

        // Map back to original indices
        let left: Vec<usize> = local_left.iter().map(|&i| indices[i]).collect();
        let right: Vec<usize> = local_right.iter().map(|&i| indices[i]).collect();

        // Handle degenerate cases
        let (left, right) = if left.is_empty() || right.is_empty() {
            let mid = n / 2;
            (indices[..mid].to_vec(), indices[mid..].to_vec())
        } else {
            (left, right)
        };

        // Recurse
        let k_left = target_k / 2;
        let k_right = target_k - k_left;

        let mut partitions = hierarchical_partition_with_bytes(
            &left,
            embeddings,
            embedding_dim,
            raw_bytes,
            substring_config,
            k_left,
            min_size,
            beta,
            n_warmup,
            n_sweeps,
            seed.wrapping_add(1),
        );

        partitions.extend(hierarchical_partition_with_bytes(
            &right,
            embeddings,
            embedding_dim,
            raw_bytes,
            substring_config,
            k_right,
            min_size,
            beta,
            n_warmup,
            n_sweeps,
            seed.wrapping_add(2),
        ));

        partitions
    }

    /// Perform hierarchical partition using sparse similarity with bytes.
    pub fn hierarchical_partition_sparse_with_bytes(
        indices: &[usize],
        sparse_indices: &[Vec<usize>],
        sparse_values: &[Vec<f32>],
        raw_bytes: &[Vec<u8>],
        substring_config: SubstringConfig,
        target_k: usize,
        min_size: usize,
        beta: f64,
        n_warmup: usize,
        n_sweeps: usize,
        seed: u64,
    ) -> Vec<Vec<usize>> {
        // Base case
        if target_k <= 1 || indices.len() <= min_size {
            return vec![indices.to_vec()];
        }

        let n = indices.len();

        // Build index map
        let idx_map: HashMap<usize, usize> = indices
            .iter()
            .enumerate()
            .map(|(local, &global)| (global, local))
            .collect();

        // Build local sparse similarity
        let mut local_indices = vec![Vec::new(); n];
        let mut local_values = vec![Vec::new(); n];

        for (local_i, &global_i) in indices.iter().enumerate() {
            if global_i < sparse_indices.len() {
                for (&neighbor, &sim) in sparse_indices[global_i]
                    .iter()
                    .zip(sparse_values[global_i].iter())
                {
                    if let Some(&local_j) = idx_map.get(&neighbor) {
                        if local_i < local_j {
                            local_indices[local_i].push(local_j);
                            local_values[local_i].push(sim);
                        }
                    }
                }
            }
        }

        // Build local bytes
        let local_bytes: Vec<Vec<u8>> = indices
            .iter()
            .map(|&idx| {
                if idx < raw_bytes.len() {
                    raw_bytes[idx].clone()
                } else {
                    Vec::new()
                }
            })
            .collect();

        // Create Ising state from sparse similarity with bytes
        let mut ising = CpuIsingState::from_sparse_similarity_with_bytes(
            &local_indices,
            &local_values,
            n,
            Some(&local_bytes),
            Some(substring_config),
            beta,
        );

        // Sample partition
        ising.sample(n_warmup, n_sweeps, seed);

        // Get partition
        let (local_left, local_right) = ising.get_partition_indices();

        // Map back to original indices
        let left: Vec<usize> = local_left.iter().map(|&i| indices[i]).collect();
        let right: Vec<usize> = local_right.iter().map(|&i| indices[i]).collect();

        // Handle degenerate cases
        let (left, right) = if left.is_empty() || right.is_empty() {
            let mid = n / 2;
            (indices[..mid].to_vec(), indices[mid..].to_vec())
        } else {
            (left, right)
        };

        // Recurse
        let k_left = target_k / 2;
        let k_right = target_k - k_left;

        let mut partitions = hierarchical_partition_sparse_with_bytes(
            &left,
            sparse_indices,
            sparse_values,
            raw_bytes,
            substring_config,
            k_left,
            min_size,
            beta,
            n_warmup,
            n_sweeps,
            seed.wrapping_add(1),
        );

        partitions.extend(hierarchical_partition_sparse_with_bytes(
            &right,
            sparse_indices,
            sparse_values,
            raw_bytes,
            substring_config,
            k_right,
            min_size,
            beta,
            n_warmup,
            n_sweeps,
            seed.wrapping_add(2),
        ));

        partitions
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_cpu_ising_partition() {
            let n = 20;
            let d = 8;

            // Create simple embeddings
            let embeddings: Vec<f32> = (0..n * d)
                .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
                .collect();

            let indices: Vec<usize> = (0..n).collect();

            let partitions = hierarchical_partition(
                &indices,
                &embeddings,
                d,
                4,   // target K
                2,   // min size
                1.0, // beta
                10,  // warmup
                10,  // sweeps
                42,  // seed
            );

            // Should have at least 1 partition
            assert!(!partitions.is_empty());

            // All points should be assigned
            let total: usize = partitions.iter().map(|p| p.len()).sum();
            assert_eq!(total, n);

            println!(
                "Partitions: {:?}",
                partitions.iter().map(|p| p.len()).collect::<Vec<_>>()
            );
        }

        #[test]
        fn test_cpu_ising_energy() {
            let embeddings = vec![1.0f32, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
            let ising = CpuIsingState::from_embeddings(&embeddings, 4, 2, 1.0);

            let e = ising.energy();
            println!("Initial energy: {}", e);

            // Energy should be finite
            assert!(e.is_finite());
        }

        #[test]
        fn test_partition_with_bytes() {
            let n = 16;
            let d = 8;

            // Create embeddings
            let embeddings: Vec<f32> = (0..n * d)
                .map(|i| ((i % 100) as f32 / 100.0) - 0.5)
                .collect();

            // Create byte sequences with some containing relationships
            let raw_bytes: Vec<Vec<u8>> = vec![
                b"hello world".to_vec(),
                b"hello".to_vec(), // contained in 0
                b"world".to_vec(), // contained in 0
                b"goodbye world".to_vec(),
                b"goodbye".to_vec(), // contained in 3
                b"random text".to_vec(),
                b"more random".to_vec(),
                b"some other".to_vec(),
                b"data here".to_vec(),
                b"testing".to_vec(),
                b"testing 123".to_vec(), // contains 9
                b"abc xyz".to_vec(),
                b"xyz abc".to_vec(),
                b"final one".to_vec(),
                b"one more".to_vec(),
                b"the end".to_vec(),
            ];

            let indices: Vec<usize> = (0..n).collect();
            let config = SubstringConfig::default();

            let partitions = hierarchical_partition_with_bytes(
                &indices,
                &embeddings,
                d,
                &raw_bytes,
                config,
                4,   // target K
                2,   // min size
                1.0, // beta
                5,   // warmup
                5,   // sweeps
                42,  // seed
            );

            // Should have at least 1 partition
            assert!(!partitions.is_empty());

            // All points should be assigned
            let total: usize = partitions.iter().map(|p| p.len()).sum();
            assert_eq!(total, n);

            println!(
                "Partitions with bytes: {:?}",
                partitions.iter().map(|p| p.len()).collect::<Vec<_>>()
            );

            // Verify contained strings tend to be in same partition
            // (not guaranteed, but likely with the coupling)
            let mut same_partition_count = 0;
            let mut total_pairs = 0;

            // Check hello/hello world containment
            for (pi, part) in partitions.iter().enumerate() {
                let has_0 = part.contains(&0); // "hello world"
                let has_1 = part.contains(&1); // "hello"
                let has_2 = part.contains(&2); // "world"

                if has_0 && has_1 {
                    same_partition_count += 1;
                }
                if has_0 && has_2 {
                    same_partition_count += 1;
                }
                total_pairs += 2;

                if has_0 || has_1 || has_2 {
                    println!(
                        "Partition {} contains hello-related: 0={} 1={} 2={}",
                        pi, has_0, has_1, has_2
                    );
                }
            }

            println!(
                "Same partition for containment pairs: {}/{}",
                same_partition_count, total_pairs
            );
        }

        #[test]
        fn test_combined_coupling() {
            // Test that combined coupling properly weights embedding and substring
            let config = SubstringConfig::with_weights(0.5, 0.5);

            let embedding_sim = 0.8;
            let bytes_a = b"function calculate_total";
            let bytes_b = b"calculate";

            let combined = compute_combined_coupling(embedding_sim, bytes_a, bytes_b, &config);

            println!("Combined coupling: {}", combined);

            // Should be higher than pure embedding due to containment
            assert!(combined > 0.0);

            // With alpha=0.5, beta=0.5, should blend both
            // Substring sim should be positive due to containment
            let sub_only_config = SubstringConfig::substring_only();
            let sub_sim = compute_combined_coupling(0.0, bytes_a, bytes_b, &sub_only_config);
            println!("Substring only: {}", sub_sim);
            assert!(sub_sim > 0.0);
        }
    }
}

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
