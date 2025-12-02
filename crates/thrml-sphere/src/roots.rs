//! ROOTS Layer - Compressed inner-shell index for coarse navigation.
//!
//! ## Architecture (from EBM Navigator Architecture Plan Section 7)
//!
//! ROOTS provides a compressed representation of the sphere using a three-stage hierarchy:
//!
//! ```text
//! Stage 1: COARSE PARTITION (Max-Cut on similarity graph)
//!          ┌─────────────────────────────────────────┐
//!          │  Sphere → K macro-regions via Ising    │
//!          │  J_ij = cosine_sim(embed_i, embed_j)   │
//!          │  Sample s ∈ {-1,+1}^N via Gibbs        │
//!          │  → Binary partition, recurse for K>2   │
//!          └─────────────────────────────────────────┘
//!                            ↓
//! Stage 2: ROOTS INDEX (Compressed representatives)
//!          ┌─────────────────────────────────────────┐
//!          │  Each partition → centroid embedding   │
//!          │  + n-gram(1) byte distribution         │
//!          │  + prominence statistics               │
//!          │  = Ultra-compact index (~KB total)     │
//!          └─────────────────────────────────────────┘
//!                            ↓
//! Stage 3: CLASSIFIER EBM (Learned patch assignment)
//!          ┌─────────────────────────────────────────┐
//!          │  Input query → Which patch?            │
//!          │  E(q, patch_k) = learned energy        │
//!          │  Gumbel-Softmax for differentiable     │
//!          │  routing during training               │
//!          └─────────────────────────────────────────┘
//! ```
//!
//! ## Key Insight: Ising-MaxCut Connection (Section 7.1)
//!
//! The Ising model energy IS the max-cut objective:
//! - `E(s) = -β · Σ J_ij · s_i · s_j`
//! - When `s_i ≠ s_j`: contribution = +J_ij (cut edge)
//! - When `s_i = s_j`: contribution = -J_ij (uncut edge)
//! - **Minimizing E = Maximizing cut!**
//!
//! ## Scalability
//!
//! For terabyte-scale data, uses:
//! - **Sparse similarity** via `cosine_similarity_topk` (top-k neighbors only)
//! - **Recursive binary partitioning** (O(N log K) vs O(N²))
//! - **Streaming centroid updates** for memory efficiency

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::similarity::{cosine_similarity_topk, SparseSimilarity};
use thrml_samplers::rng::RngKey;

use crate::compute::{cpu_ising, substring::SubstringConfig, HybridConfig};
use crate::sphere_ebm::SphereEBM;

// Fused kernel imports (when feature enabled)
#[cfg(feature = "fused-kernels")]
use thrml_core::backend::CubeWgpuBackend;
#[cfg(feature = "fused-kernels")]
use thrml_kernels::cosine_similarity_fused;

// ============================================================================
// Configuration
// ============================================================================

/// Partition method for ROOTS index construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum PartitionMethod {
    /// Pure Ising max-cut: similarity-based graph partitioning only.
    /// May create "cantaloupe slice" elongated partitions.
    IsingMaxCut,

    /// Pure alternating-axis: median splits alternate between θ and φ.
    /// Guaranteed balanced tiles but ignores semantic similarity.
    AlternatingAxis,

    /// Hybrid (recommended): Alternating axis direction + max-cut boundary.
    /// - Alternating axis ensures "checkerboard" tiles (no cantaloupe slices)
    /// - Max-cut finds semantically optimal boundary within each axis
    ///
    /// Best of both: geometric balance + semantic clustering.
    #[default]
    AlternatingMaxCut,
}


/// Configuration for ROOTS index construction.
#[derive(Clone, Copy, Debug)]
pub struct RootsConfig {
    /// Partition method. Default: AlternatingMaxCut
    /// - IsingMaxCut: pure similarity-based (may create elongated partitions)
    /// - AlternatingAxis: pure geometric (ignores semantic similarity)
    /// - AlternatingMaxCut: hybrid - axis direction + max-cut boundary (recommended)
    pub partition_method: PartitionMethod,

    /// Number of partitions (K). Default: 256
    /// More partitions = finer granularity but larger index
    pub n_partitions: usize,

    /// Minimum points per partition. Default: 10
    /// Partitions smaller than this get merged with neighbors
    pub min_partition_size: usize,

    /// Inverse temperature for Ising sampling. Default: 1.0
    /// Higher β = sharper partitions (more deterministic)
    pub beta: f32,

    /// Gibbs sampling warmup steps. Default: 100
    pub gibbs_warmup: usize,

    /// Gibbs sampling steps. Default: 50
    pub gibbs_steps: usize,

    /// Number of samples for moment estimation. Default: 10
    pub gibbs_samples: usize,

    /// Top-k neighbors for sparse similarity. Default: 50
    /// Only edges to top-k similar points are included in Ising graph
    /// Set to 0 for dense similarity (small datasets only!)
    pub similarity_k: usize,

    /// Activation threshold for peak detection. Default: 0.2
    /// Only activations above this are considered
    pub activation_threshold: f32,

    /// Minimum angular separation between peaks (radians). Default: π/16
    pub min_peak_separation: f32,

    /// Whether to store member indices (memory intensive for large N)
    /// Default: true for small data, consider false for TB-scale
    pub store_member_indices: bool,

    /// Maximum points for dense similarity. Default: 10K
    /// Above this, switches to sparse top-k similarity
    pub dense_threshold: usize,

    /// Whether to compute n-gram byte distributions. Default: true
    pub compute_ngrams: bool,

    /// Optional substring similarity configuration.
    ///
    /// When enabled, Ising coupling combines:
    /// - Cosine similarity of embeddings (semantic)
    /// - Substring containment similarity (structural)
    ///
    /// Formula: `J_ij = α × cosine_sim + β × substring_sim`
    pub substring_config: Option<SubstringConfig>,

    /// Polar cone exclusion angle (radians from pole). Default: π/12 (15°)
    ///
    /// Points with θ < angle or θ > (π - angle) are placed in special
    /// "instruction" partitions at the poles, not partitioned with content.
    /// This creates a content torus in the equatorial band.
    ///
    /// Set to 0.0 to disable pole exclusion (partition entire sphere).
    pub pole_exclusion_angle: f32,
}

impl Default for RootsConfig {
    fn default() -> Self {
        Self {
            partition_method: PartitionMethod::AlternatingMaxCut,
            n_partitions: 256,
            min_partition_size: 10,
            beta: 1.0,
            gibbs_warmup: 100,
            gibbs_steps: 50,
            gibbs_samples: 10,
            similarity_k: 50,
            activation_threshold: 0.2,
            min_peak_separation: std::f32::consts::PI / 16.0,
            store_member_indices: true,
            dense_threshold: 10_000,
            compute_ngrams: true,
            substring_config: None, // Disabled by default
            pole_exclusion_angle: std::f32::consts::PI / 12.0, // 15° cones at poles
        }
    }
}

impl RootsConfig {
    /// Sets the number of partitions for the ROOTS index.
    ///
    /// More partitions provide finer granularity but increase index size.
    /// Recommended values:
    /// - Small datasets (<10K points): 16-64
    /// - Medium datasets (10K-100K): 64-256
    /// - Large datasets (100K-1M): 256-1024
    /// - Terabyte scale: 1024-4096
    ///
    /// # Arguments
    ///
    /// * `n` - Target number of partitions. Actual count may differ slightly
    ///   due to recursive binary partitioning.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// let config = RootsConfig::default().with_partitions(256);
    /// assert_eq!(config.n_partitions, 256);
    /// ```
    pub const fn with_partitions(mut self, n: usize) -> Self {
        self.n_partitions = n;
        self
    }

    /// Sets the partition method.
    ///
    /// # Arguments
    ///
    /// * `method` - Partition method to use
    ///   - `IsingMaxCut`: Similarity-based graph partitioning (semantic clustering)
    ///   - `AlternatingAxis`: θ/φ alternating splits (better beam search locality)
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::{RootsConfig, PartitionMethod};
    ///
    /// // Use alternating-axis for better spherical navigation
    /// let config = RootsConfig::default().with_partition_method(PartitionMethod::AlternatingAxis);
    /// ```
    pub const fn with_partition_method(mut self, method: PartitionMethod) -> Self {
        self.partition_method = method;
        self
    }

    /// Sets the polar cone exclusion angle.
    ///
    /// Points within this angle of the poles are placed in special
    /// "instruction" partitions rather than being partitioned with content.
    /// This reserves the polar cones for behavioral/instruction embeddings.
    ///
    /// # Arguments
    ///
    /// * `angle` - Exclusion angle in radians from pole. Default: π/12 (15°)
    ///   Set to 0.0 to disable and partition the entire sphere.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// // 30° cones at poles for instruction embeddings
    /// let config = RootsConfig::default()
    ///     .with_pole_exclusion(std::f32::consts::PI / 6.0);
    /// ```
    pub const fn with_pole_exclusion(mut self, angle: f32) -> Self {
        self.pole_exclusion_angle = angle;
        self
    }

    /// Sets the activation threshold for peak detection.
    ///
    /// During query routing, partitions with activation below this threshold
    /// are not considered as peaks for cone spawning.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum activation value (0.0 to 1.0). Default: 0.2
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// // More selective peak detection
    /// let config = RootsConfig::default().with_threshold(0.5);
    /// ```
    pub const fn with_threshold(mut self, threshold: f32) -> Self {
        self.activation_threshold = threshold;
        self
    }

    /// Sets the minimum number of points per partition.
    ///
    /// Partitions smaller than this threshold are merged with neighbors
    /// during the recursive partitioning process.
    ///
    /// # Arguments
    ///
    /// * `size` - Minimum partition size. Default: 10
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// // Allow smaller partitions for fine-grained indexing
    /// let config = RootsConfig::default().with_min_partition_size(5);
    /// ```
    pub const fn with_min_partition_size(mut self, size: usize) -> Self {
        self.min_partition_size = size;
        self
    }

    /// Sets the inverse temperature (β) for Ising max-cut sampling.
    ///
    /// Higher values produce sharper, more deterministic partitions.
    /// Lower values allow more exploration during sampling.
    ///
    /// # Arguments
    ///
    /// * `beta` - Inverse temperature. Typical range: 0.5 to 2.0. Default: 1.0
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// // Sharper partitioning
    /// let config = RootsConfig::default().with_beta(1.5);
    /// ```
    pub const fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Sets the Gibbs sampling parameters for Ising max-cut.
    ///
    /// # Arguments
    ///
    /// * `warmup` - Burn-in steps before collecting samples. Default: 100
    /// * `steps` - Gibbs steps between samples. Default: 50
    /// * `samples` - Number of samples to average. Default: 10
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// // Faster but noisier partitioning for dev/testing
    /// let config = RootsConfig::default().with_gibbs(20, 10, 5);
    /// ```
    pub const fn with_gibbs(mut self, warmup: usize, steps: usize, samples: usize) -> Self {
        self.gibbs_warmup = warmup;
        self.gibbs_steps = steps;
        self.gibbs_samples = samples;
        self
    }

    /// Sets the top-k parameter for sparse similarity computation.
    ///
    /// For large datasets, only the top-k most similar neighbors are
    /// considered when building the Ising graph, reducing memory from
    /// O(N²) to O(N×k).
    ///
    /// # Arguments
    ///
    /// * `k` - Number of nearest neighbors. Set to 0 for dense similarity.
    ///   Default: 50
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// // Dense similarity for small datasets
    /// let config = RootsConfig::default().with_similarity_k(0);
    ///
    /// // Sparse similarity for large datasets
    /// let config = RootsConfig::default().with_similarity_k(100);
    /// ```
    pub const fn with_similarity_k(mut self, k: usize) -> Self {
        self.similarity_k = k;
        self
    }

    /// Disables storage of member indices in partitions.
    ///
    /// For terabyte-scale datasets, storing member indices for each
    /// partition can consume significant memory. Disable when you only
    /// need routing, not member lookup.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// let config = RootsConfig::terabyte_scale()
    ///     .without_member_indices();
    /// assert!(!config.store_member_indices);
    /// ```
    pub const fn without_member_indices(mut self) -> Self {
        self.store_member_indices = false;
        self
    }

    /// Disables byte n-gram distribution computation.
    ///
    /// N-gram distributions are useful for content fingerprinting but
    /// add memory overhead. Disable for pure embedding-based routing.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// let config = RootsConfig::default().without_ngrams();
    /// assert!(!config.compute_ngrams);
    /// ```
    pub const fn without_ngrams(mut self) -> Self {
        self.compute_ngrams = false;
        self
    }

    /// Creates a configuration optimized for terabyte-scale datasets.
    ///
    /// This preset:
    /// - Uses 4096 partitions for fine-grained indexing
    /// - Requires 1000+ points per partition to avoid fragmentation
    /// - Disables member indices to save memory
    /// - Uses sparse similarity (top-100 neighbors)
    /// - Skips n-gram computation
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// let config = RootsConfig::terabyte_scale();
    /// assert_eq!(config.n_partitions, 4096);
    /// assert!(!config.store_member_indices);
    /// ```
    pub fn terabyte_scale() -> Self {
        Self {
            partition_method: PartitionMethod::AlternatingMaxCut,
            n_partitions: 4096,
            min_partition_size: 1000,
            beta: 1.5,
            gibbs_warmup: 50,
            gibbs_steps: 25,
            gibbs_samples: 5,
            similarity_k: 100,
            activation_threshold: 0.15,
            min_peak_separation: std::f32::consts::PI / 32.0,
            store_member_indices: false,
            dense_threshold: 5_000,
            compute_ngrams: false,
            substring_config: None,
            pole_exclusion_angle: std::f32::consts::PI / 12.0, // 15° cones
        }
    }

    /// Creates a configuration for development and testing.
    ///
    /// This preset:
    /// - Uses only 16 partitions for fast iteration
    /// - Allows small partitions (2+ points)
    /// - Uses dense similarity (no top-k filtering)
    /// - Stores all member indices for debugging
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// let config = RootsConfig::dev();
    /// assert_eq!(config.n_partitions, 16);
    /// assert!(config.store_member_indices);
    /// ```
    pub fn dev() -> Self {
        Self {
            partition_method: PartitionMethod::AlternatingMaxCut,
            n_partitions: 16,
            min_partition_size: 2,
            beta: 1.0,
            gibbs_warmup: 20,
            gibbs_steps: 10,
            gibbs_samples: 5,
            similarity_k: 0,
            activation_threshold: 0.2,
            min_peak_separation: std::f32::consts::PI / 8.0,
            store_member_indices: true,
            dense_threshold: 10_000,
            compute_ngrams: false,
            substring_config: None,
            pole_exclusion_angle: std::f32::consts::PI / 12.0, // 15° cones
        }
    }

    /// Enables substring-enhanced coupling for Ising partitioning.
    ///
    /// When enabled, the Ising coupling weights combine:
    /// - **Semantic**: Cosine similarity of embeddings
    /// - **Structural**: Substring containment similarity of raw bytes
    ///
    /// Formula: `J_ij = α × cosine_sim(emb_i, emb_j) + β × substring_sim(bytes_i, bytes_j)`
    ///
    /// This is particularly useful for code and structured text where
    /// byte-level relationships matter (e.g., "calculate_total" should
    /// cluster with "function calculate_total").
    ///
    /// # Arguments
    ///
    /// * `config` - Substring similarity configuration with weights α and β.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::{RootsConfig, SubstringConfig};
    ///
    /// // Custom weights: 60% embedding, 40% substring
    /// let config = RootsConfig::default()
    ///     .with_substring_coupling(SubstringConfig::with_weights(0.6, 0.4));
    /// ```
    pub const fn with_substring_coupling(mut self, config: SubstringConfig) -> Self {
        self.substring_config = Some(config);
        self
    }

    /// Enables default substring coupling (70% embedding, 30% substring).
    ///
    /// Convenience method equivalent to:
    /// ```ignore
    /// config.with_substring_coupling(SubstringConfig::default())
    /// ```
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::RootsConfig;
    ///
    /// let config = RootsConfig::default()
    ///     .with_partitions(256)
    ///     .with_default_substring_coupling();
    ///
    /// assert!(config.substring_config.is_some());
    /// ```
    pub fn with_default_substring_coupling(mut self) -> Self {
        self.substring_config = Some(SubstringConfig::default());
        self
    }
}

// ============================================================================
// Statistics and Partition Types
// ============================================================================

/// Statistics about prominence distribution within a partition.
///
/// Tracks mean, standard deviation, min, max, and count for prominence
/// values of points assigned to a partition. Used for:
/// - Understanding partition quality
/// - Weighting partitions during routing
/// - Detecting outlier partitions
#[derive(Clone, Debug, Default)]
pub struct ProminenceStats {
    /// Mean prominence value.
    pub mean: f32,
    /// Standard deviation of prominence values.
    pub std: f32,
    /// Maximum prominence value in the partition.
    pub max: f32,
    /// Minimum prominence value in the partition.
    pub min: f32,
    /// Number of points contributing to statistics.
    pub count: usize,
}

impl ProminenceStats {
    /// Computes statistics from a slice of prominence values.
    ///
    /// # Arguments
    ///
    /// * `values` - Slice of prominence values. Can be empty.
    ///
    /// # Returns
    ///
    /// A `ProminenceStats` struct with computed statistics.
    /// Returns default (zeros) for empty input.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::ProminenceStats;
    ///
    /// let values = vec![0.5, 0.8, 0.6, 0.9];
    /// let stats = ProminenceStats::from_slice(&values);
    ///
    /// assert_eq!(stats.count, 4);
    /// assert!((stats.mean - 0.7).abs() < 0.01);
    /// assert_eq!(stats.max, 0.9);
    /// ```
    pub fn from_slice(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }

        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();
        let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = values.iter().cloned().fold(f32::INFINITY, f32::min);

        Self {
            mean,
            std,
            max,
            min,
            count: values.len(),
        }
    }

    /// Merges another `ProminenceStats` into this one using Welford's algorithm.
    ///
    /// This enables online/streaming computation of statistics without storing
    /// all individual values. The combined statistics are mathematically exact.
    ///
    /// # Arguments
    ///
    /// * `other` - Statistics to merge into this instance.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::ProminenceStats;
    ///
    /// let mut stats1 = ProminenceStats::from_slice(&[0.5, 0.6]);
    /// let stats2 = ProminenceStats::from_slice(&[0.7, 0.8]);
    /// stats1.merge(&stats2);
    ///
    /// assert_eq!(stats1.count, 4);
    /// ```
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;

        // Combined mean
        let combined_mean = delta.mul_add(other.count as f32 / combined_count as f32, self.mean);

        // Combined variance (Welford's parallel algorithm)
        let m2_self = self.std * self.std * self.count as f32;
        let m2_other = other.std * other.std * other.count as f32;
        let m2_combined = (delta * delta).mul_add(self.count as f32 * other.count as f32 / combined_count as f32, m2_self + m2_other);

        self.mean = combined_mean;
        self.std = (m2_combined / combined_count as f32).sqrt();
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.count = combined_count;
    }
}

// ============================================================================
// Polar Zone Types
// ============================================================================

/// Zone classification for partitions on the sphere.
///
/// The sphere is divided into three semantic zones based on polar angle θ:
///
/// ```text
///          N (θ < 15°)    ← INSTRUCTION zone
///          │               (behavioral anchors, system prompts)
///     ╱────┼────╲   
///    ╱     │     ╲
///   ╱      │      ╲
///  │═══════════════│  ← CONTENT zone (torus)
///  │   Knowledge   │    (documents, facts, reference material)
///  │               │
///   ╲      │      ╱
///    ╲     │     ╱
///     ╲────┼────╱   
///          │
///          S (θ > 165°)  ← QA_PAIRS zone
///                         (fine-tuning examples, demonstrations)
/// ```
///
/// This separation enables:
/// - Targeted retrieval by zone type
/// - Clean partitioning without elongation near poles
/// - Semantic separation: "how to behave" (N) vs "what to know" (torus) vs "how to respond" (S)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PartitionZone {
    /// Content zone - equatorial torus for knowledge/documents
    #[default]
    Content,
    /// Instruction zone (north pole, θ near 0) - behavioral anchors, system prompts
    Instruction,
    /// QA pairs zone (south pole, θ near π) - fine-tuning examples, demonstrations
    QAPairs,
}

impl PartitionZone {
    /// Returns true if this is a polar (non-content) zone.
    pub const fn is_polar(&self) -> bool {
        !matches!(self, Self::Content)
    }

    /// Returns true if this is the instruction zone (north pole).
    pub const fn is_instruction(&self) -> bool {
        matches!(self, Self::Instruction)
    }

    /// Returns true if this is the QA pairs zone (south pole).
    pub const fn is_qa_pairs(&self) -> bool {
        matches!(self, Self::QAPairs)
    }

    /// Returns true if this is the content zone (torus).
    pub const fn is_content(&self) -> bool {
        matches!(self, Self::Content)
    }

    /// Energy weight for this zone during optimization.
    /// Lower weight = less resistance = faster settling.
    pub const fn energy_weight(&self, config: &ZoneEnergyConfig) -> f32 {
        match self {
            Self::Content => config.content_weight,
            Self::Instruction => config.instruction_weight,
            Self::QAPairs => config.qa_pairs_weight,
        }
    }
}

/// Configuration for zone-specific energy weighting during optimization.
///
/// Lower weights = less resistance = faster settling during Langevin dynamics.
/// This allows polar zones to stabilize quickly while content torus optimizes normally.
#[derive(Clone, Copy, Debug)]
pub struct ZoneEnergyConfig {
    /// Energy weight for content zone (default: 1.0)
    pub content_weight: f32,
    /// Energy weight for instruction zone (default: 0.5 - settles faster)
    pub instruction_weight: f32,
    /// Energy weight for QA pairs zone (default: 0.5 - settles faster)
    pub qa_pairs_weight: f32,
}

impl Default for ZoneEnergyConfig {
    fn default() -> Self {
        Self {
            content_weight: 1.0,
            instruction_weight: 0.5, // Polar zones settle faster
            qa_pairs_weight: 0.5,
        }
    }
}

impl ZoneEnergyConfig {
    /// All zones have equal energy weight.
    pub const fn uniform() -> Self {
        Self {
            content_weight: 1.0,
            instruction_weight: 1.0,
            qa_pairs_weight: 1.0,
        }
    }

    /// Set instruction zone weight.
    pub const fn with_instruction_weight(mut self, weight: f32) -> Self {
        self.instruction_weight = weight;
        self
    }

    /// Set QA pairs zone weight.
    pub const fn with_qa_pairs_weight(mut self, weight: f32) -> Self {
        self.qa_pairs_weight = weight;
        self
    }

    /// Set content zone weight.
    pub const fn with_content_weight(mut self, weight: f32) -> Self {
        self.content_weight = weight;
        self
    }
}

/// Configuration for instruction embeddings at the poles.
///
/// The polar cones can hold fixed "instruction" embeddings that act as
/// behavioral anchors during retrieval. These are not partitioned with
/// content but instead provide context/weighting for queries.
#[derive(Clone, Debug, Default)]
pub struct InstructionConfig {
    /// Fixed embeddings for north pole (optional).
    /// When set, queries can be weighted toward these during retrieval.
    pub north_pole_embeddings: Option<Vec<f32>>,

    /// Fixed embeddings for south pole (optional).
    pub south_pole_embeddings: Option<Vec<f32>>,

    /// Weight for instruction embeddings during retrieval (0.0 to 1.0).
    /// Higher values bias results toward instruction-aligned content.
    pub instruction_weight: f32,
}

impl InstructionConfig {
    /// Creates a new instruction config with default settings.
    pub const fn new() -> Self {
        Self {
            north_pole_embeddings: None,
            south_pole_embeddings: None,
            instruction_weight: 0.1, // Light influence by default
        }
    }

    /// Sets north pole instruction embeddings.
    pub fn with_north_pole(mut self, embeddings: Vec<f32>) -> Self {
        self.north_pole_embeddings = Some(embeddings);
        self
    }

    /// Sets south pole instruction embeddings.
    pub fn with_south_pole(mut self, embeddings: Vec<f32>) -> Self {
        self.south_pole_embeddings = Some(embeddings);
        self
    }

    /// Sets instruction weight.
    pub const fn with_weight(mut self, weight: f32) -> Self {
        self.instruction_weight = weight.clamp(0.0, 1.0);
        self
    }
}

/// Byte unigram distribution (n-gram of size 1).
///
/// Stores a histogram of raw byte values [0-255] for content fingerprinting.
/// Useful for:
/// - Detecting content type (text vs binary)
/// - Computing byte-level entropy
/// - Fast approximate content matching
///
/// # Memory
///
/// Fixed size: 256 × 8 bytes = 2KB per instance.
#[derive(Clone, Debug)]
pub struct ByteNgramDist {
    /// Histogram counts for each byte value [0-255].
    pub counts: [u64; 256],
    /// Total number of bytes counted.
    pub total: u64,
}

impl Default for ByteNgramDist {
    fn default() -> Self {
        Self {
            counts: [0; 256],
            total: 0,
        }
    }
}

impl ByteNgramDist {
    /// Adds bytes to the distribution histogram.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Byte slice to incorporate into the distribution.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::ByteNgramDist;
    ///
    /// let mut dist = ByteNgramDist::default();
    /// dist.add_bytes(b"hello world");
    ///
    /// assert_eq!(dist.total, 11);
    /// assert_eq!(dist.counts[b'l' as usize], 3);
    /// ```
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.counts[b as usize] += 1;
            self.total += 1;
        }
    }

    /// Returns the normalized probability distribution.
    ///
    /// Each element `dist[i]` represents P(byte = i), summing to 1.0.
    ///
    /// # Returns
    ///
    /// Array of 256 f32 values representing byte probabilities.
    /// Returns all zeros if no bytes have been added.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::ByteNgramDist;
    ///
    /// let mut dist = ByteNgramDist::default();
    /// dist.add_bytes(b"aaab");
    ///
    /// let probs = dist.as_distribution();
    /// assert!((probs[b'a' as usize] - 0.75).abs() < 0.01);
    /// assert!((probs[b'b' as usize] - 0.25).abs() < 0.01);
    /// ```
    pub fn as_distribution(&self) -> [f32; 256] {
        let mut dist = [0.0f32; 256];
        if self.total > 0 {
            let scale = 1.0 / self.total as f32;
            for (i, &count) in self.counts.iter().enumerate() {
                dist[i] = count as f32 * scale;
            }
        }
        dist
    }

    /// Merges another distribution into this one.
    ///
    /// # Arguments
    ///
    /// * `other` - Distribution to merge into this instance.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::ByteNgramDist;
    ///
    /// let mut dist1 = ByteNgramDist::default();
    /// dist1.add_bytes(b"hello");
    ///
    /// let mut dist2 = ByteNgramDist::default();
    /// dist2.add_bytes(b"world");
    ///
    /// dist1.merge(&dist2);
    /// assert_eq!(dist1.total, 10);
    /// ```
    pub fn merge(&mut self, other: &Self) {
        for i in 0..256 {
            self.counts[i] += other.counts[i];
        }
        self.total += other.total;
    }

    /// Computes Shannon entropy of the byte distribution in bits.
    ///
    /// Higher entropy indicates more uniform/random content.
    /// Lower entropy indicates structured/repetitive content.
    ///
    /// # Returns
    ///
    /// Entropy value in bits. Range: 0.0 (single byte) to 8.0 (uniform).
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::ByteNgramDist;
    ///
    /// let mut uniform = ByteNgramDist::default();
    /// for i in 0..=255u8 {
    ///     uniform.add_bytes(&[i]);
    /// }
    /// assert!((uniform.entropy() - 8.0).abs() < 0.01); // Max entropy
    ///
    /// let mut single = ByteNgramDist::default();
    /// single.add_bytes(b"aaaa");
    /// assert!(single.entropy() < 0.01); // Min entropy
    /// ```
    pub fn entropy(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let mut entropy = 0.0f32;
        let total = self.total as f32;
        for &count in &self.counts {
            if count > 0 {
                let p = count as f32 / total;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

/// A single partition in the ROOTS index.
///
/// Each partition contains:
/// - A centroid (mean embedding of members)
/// - Statistics about member prominence
/// - Optional byte n-gram distribution
/// - Optional member indices (can be disabled at scale)
#[derive(Clone, Debug)]
pub struct RootsPartition {
    /// Partition ID (0 to K-1).
    pub id: usize,

    /// Zone classification (Content, Instruction, or QAPairs).
    /// Polar partitions are instruction zones, not content zones.
    pub zone: PartitionZone,

    /// Centroid embedding \[D\] - mean of member embeddings.
    pub centroid: Vec<f32>,

    /// Running sum for incremental centroid update \[D\].
    centroid_sum: Vec<f32>,

    /// Number of points contributing to centroid
    pub point_count: usize,

    /// Byte unigram distribution (if computed)
    pub ngram_dist: Option<ByteNgramDist>,

    /// Prominence statistics for members
    pub prominence_stats: ProminenceStats,

    /// Radius range (min, max) of members in optimized sphere
    pub radius_range: (f32, f32),

    /// Mean angular position (θ, φ) on sphere (for cone targeting)
    pub mean_position: Option<(f32, f32)>,

    /// Indices of points belonging to this partition (optional for scale)
    /// At TB scale, this may be None to save memory
    pub member_indices: Option<Vec<usize>>,
}

impl RootsPartition {
    /// Create a new empty partition (defaults to Content zone)
    pub fn new(id: usize, embedding_dim: usize, store_indices: bool) -> Self {
        Self::with_zone(id, embedding_dim, store_indices, PartitionZone::Content)
    }

    /// Create a new empty partition with explicit zone
    pub fn with_zone(
        id: usize,
        embedding_dim: usize,
        store_indices: bool,
        zone: PartitionZone,
    ) -> Self {
        Self {
            id,
            zone,
            centroid: vec![0.0; embedding_dim],
            centroid_sum: vec![0.0; embedding_dim],
            point_count: 0,
            ngram_dist: None,
            prominence_stats: ProminenceStats::default(),
            radius_range: (f32::MAX, f32::MIN),
            mean_position: None,
            member_indices: if store_indices {
                Some(Vec::new())
            } else {
                None
            },
        }
    }

    /// Returns true if this is an instruction (polar) partition.
    pub const fn is_instruction(&self) -> bool {
        self.zone.is_instruction()
    }

    /// Returns true if this is a content partition.
    pub const fn is_content(&self) -> bool {
        self.zone.is_content()
    }

    /// Add a point to this partition (streaming update)
    pub fn add_point(&mut self, embedding: &[f32], prominence: f32, index: usize) {
        // Update centroid sum
        for (sum, &val) in self.centroid_sum.iter_mut().zip(embedding.iter()) {
            *sum += val;
        }
        self.point_count += 1;

        // Update prominence stats incrementally
        if self.point_count == 1 {
            self.prominence_stats.mean = prominence;
            self.prominence_stats.min = prominence;
            self.prominence_stats.max = prominence;
            self.prominence_stats.count = 1;
        } else {
            let n = self.point_count as f32;
            let delta = prominence - self.prominence_stats.mean;
            self.prominence_stats.mean += delta / n;
            self.prominence_stats.min = self.prominence_stats.min.min(prominence);
            self.prominence_stats.max = self.prominence_stats.max.max(prominence);
            self.prominence_stats.count = self.point_count;
        }

        // Store index if enabled
        if let Some(ref mut indices) = self.member_indices {
            indices.push(index);
        }
    }

    /// Add raw bytes to the n-gram distribution
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        if self.ngram_dist.is_none() {
            self.ngram_dist = Some(ByteNgramDist::default());
        }
        if let Some(ref mut dist) = self.ngram_dist {
            dist.add_bytes(bytes);
        }
    }

    /// Finalize centroid computation
    pub fn finalize(&mut self) {
        if self.point_count > 0 {
            let scale = 1.0 / self.point_count as f32;
            for (c, &sum) in self.centroid.iter_mut().zip(self.centroid_sum.iter()) {
                *c = sum * scale;
            }
        }
    }

    /// Number of points in this partition.
    pub const fn size(&self) -> usize {
        self.point_count
    }

    /// Embedding dimension.
    pub const fn dim(&self) -> usize {
        self.centroid.len()
    }
}

// ============================================================================
// H-ROOTS: Hierarchical Tree Structure
// ============================================================================
//
// Instead of flattening partitions into a Vec, we preserve the binary tree
// structure from hierarchical Ising max-cut. This enables:
// - O(log K) query routing via beam search (vs O(K) linear scan)
// - Pruning entire subtrees based on centroid similarity
// - Natural hierarchical organization by semantic similarity
//
// At 1B vectors with K=10,000 partitions:
// - Tree depth: ~14 levels
// - Query ops: ~14 × 2 × 768 ≈ 21k (vs 7.68M for flat scan)
// - Speedup: ~380x

/// A node in the hierarchical ROOTS tree.
///
/// The tree preserves the binary partitioning structure from Ising max-cut,
/// enabling efficient O(log K) query routing instead of O(K) linear scan.
///
/// # Structure
///
/// ```text
///                    Internal (root)
///                   /              \
///            Internal              Internal
///           /        \            /        \
///        Leaf        Leaf      Leaf        Leaf
///     (partition)  (partition) ...         ...
/// ```
#[derive(Clone, Debug)]
pub enum RootsNode {
    /// Internal node: A signpost pointing to children.
    ///
    /// Contains aggregate statistics for beam search pruning.
    Internal {
        /// Centroid of all points below this node (mean of child centroids).
        /// Used for coarse similarity check during beam search.
        centroid: Vec<f32>,

        /// Radius range (min, max) of all points in this subtree.
        /// Used for pruning based on radial constraints.
        radius_range: (f32, f32),

        /// Prominence range (min, max) of all points in this subtree.
        /// Used for quality-based pruning.
        prominence_range: (f32, f32),

        /// Total number of points in this subtree.
        point_count: usize,

        /// Left and right children (binary partition from Ising).
        children: Box<(RootsNode, RootsNode)>,
    },

    /// Leaf node: Actual data bucket containing a partition (boxed to reduce enum size).
    Leaf {
        /// The actual partition with centroid, stats, and member indices.
        partition: Box<RootsPartition>,
    },
}

impl RootsNode {
    /// Create a leaf node from a partition.
    pub fn leaf(partition: RootsPartition) -> Self {
        Self::Leaf { partition: Box::new(partition) }
    }

    /// Create an internal node from two children.
    ///
    /// Automatically computes aggregate centroid and ranges.
    pub fn internal(left: Self, right: Self, embedding_dim: usize) -> Self {
        // Compute aggregate centroid (weighted by point count)
        let left_count = left.point_count();
        let right_count = right.point_count();
        let total_count = left_count + right_count;

        let mut centroid = vec![0.0f32; embedding_dim];
        if total_count > 0 {
            let left_centroid = left.centroid();
            let right_centroid = right.centroid();

            let left_weight = left_count as f32 / total_count as f32;
            let right_weight = right_count as f32 / total_count as f32;

            for i in 0..embedding_dim {
                centroid[i] = left_centroid[i].mul_add(left_weight, right_centroid[i] * right_weight);
            }
        }

        // Compute aggregate ranges
        let left_radius = left.radius_range();
        let right_radius = right.radius_range();
        let radius_range = (
            left_radius.0.min(right_radius.0),
            left_radius.1.max(right_radius.1),
        );

        let left_prom = left.prominence_range();
        let right_prom = right.prominence_range();
        let prominence_range = (left_prom.0.min(right_prom.0), left_prom.1.max(right_prom.1));

        Self::Internal {
            centroid,
            radius_range,
            prominence_range,
            point_count: total_count,
            children: Box::new((left, right)),
        }
    }

    /// Get the centroid of this node.
    pub fn centroid(&self) -> &[f32] {
        match self {
            Self::Internal { centroid, .. } => centroid,
            Self::Leaf { partition } => &partition.centroid,
        }
    }

    /// Get the total point count in this subtree.
    pub fn point_count(&self) -> usize {
        match self {
            Self::Internal { point_count, .. } => *point_count,
            Self::Leaf { partition } => partition.point_count,
        }
    }

    /// Get the radius range of this subtree.
    pub fn radius_range(&self) -> (f32, f32) {
        match self {
            Self::Internal { radius_range, .. } => *radius_range,
            Self::Leaf { partition } => partition.radius_range,
        }
    }

    /// Get the prominence range of this subtree.
    pub fn prominence_range(&self) -> (f32, f32) {
        match self {
            Self::Internal {
                prominence_range, ..
            } => *prominence_range,
            Self::Leaf { partition } => (
                partition.prominence_stats.min,
                partition.prominence_stats.max,
            ),
        }
    }

    /// Check if this is a leaf node.
    pub const fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf { .. })
    }

    /// Get the depth of this tree.
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Internal { children, .. } => 1 + children.0.depth().max(children.1.depth()),
        }
    }

    /// Count the number of leaf nodes (partitions).
    pub fn leaf_count(&self) -> usize {
        match self {
            Self::Leaf { .. } => 1,
            Self::Internal { children, .. } => {
                children.0.leaf_count() + children.1.leaf_count()
            }
        }
    }

    /// Collect all partitions from the tree (flattening).
    pub fn collect_partitions(&self) -> Vec<&RootsPartition> {
        let mut result = Vec::new();
        self.collect_partitions_recursive(&mut result);
        result
    }

    fn collect_partitions_recursive<'a>(&'a self, result: &mut Vec<&'a RootsPartition>) {
        match self {
            Self::Leaf { partition } => {
                result.push(partition);
            }
            Self::Internal { children, .. } => {
                children.0.collect_partitions_recursive(result);
                children.1.collect_partitions_recursive(result);
            }
        }
    }

    /// Perform beam search activation on the tree.
    ///
    /// Returns activation peaks sorted by strength (descending).
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding `[D]`
    /// * `threshold` - Minimum similarity threshold for pruning
    /// * `max_results` - Maximum number of peaks to return
    ///
    /// # Complexity
    ///
    /// O(log K × D) average case with good pruning, O(K × D) worst case.
    pub fn beam_search(
        &self,
        query: &[f32],
        threshold: f32,
        max_results: usize,
    ) -> Vec<ActivationPeak> {
        let mut results = Vec::new();
        self.beam_search_recursive(query, threshold, 1.0, &mut results);

        // Sort by strength descending
        results.sort_by(|a, b| {
            b.strength
                .partial_cmp(&a.strength)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Truncate to max_results
        results.truncate(max_results);
        results
    }

    fn beam_search_recursive(
        &self,
        query: &[f32],
        threshold: f32,
        _inherited_score: f32,
        results: &mut Vec<ActivationPeak>,
    ) {
        // Compute similarity to this node's centroid
        let centroid = self.centroid();
        let sim = cosine_similarity_cpu(query, centroid);

        // Early pruning: if similarity is too low, skip this entire subtree
        if sim < threshold {
            return;
        }

        match self {
            Self::Internal { children, .. } => {
                // Recurse into both children with the current similarity as context
                children
                    .0
                    .beam_search_recursive(query, threshold, sim, results);
                children
                    .1
                    .beam_search_recursive(query, threshold, sim, results);
            }
            Self::Leaf { partition } => {
                // Found a leaf - add to results
                results.push(ActivationPeak {
                    partition_id: partition.id,
                    center: partition.mean_position,
                    strength: sim,
                    spread: partition.prominence_stats.std,
                    member_indices: partition.member_indices.clone().unwrap_or_default(),
                });
            }
        }
    }
}

/// Compute cosine similarity between two vectors (CPU, for tree traversal).
fn cosine_similarity_cpu(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-8);
    dot / denom
}

// ============================================================================
// Ising-based Partitioning (Section 7.1, 7.5)
// ============================================================================
//
// Uses CPU-based f64 Ising max-cut for unified memory systems (Apple Silicon)
// This provides:
// - Higher numerical precision (f64 vs f32)
// - Avoids GPU tensor edge cases (0-size tensors)
// - Zero-copy on unified memory (M4 Pro, etc.)
// - Better parallelism (CPU + GPU working simultaneously)

/// Result of partitioning, including zone classification.
struct PartitionResult {
    /// Indices of points in this partition
    indices: Vec<usize>,
    /// Zone classification for this partition
    zone: PartitionZone,
}

/// Perform hierarchical partitioning using the configured method.
///
/// Routes to either:
/// - **AlternatingMaxCut**: Hybrid axis direction + max-cut boundary (recommended)
/// - **AlternatingAxis**: θ/φ alternating splits (better beam search locality)
/// - **IsingMaxCut**: Similarity-based graph partitioning (semantic clustering)
///
/// Polar cone exclusion: Points near poles (θ < angle or θ > π-angle) are
/// placed in special "instruction" partitions rather than content partitions.
fn run_ising_partition(
    indices: &[usize],
    embeddings: &[f32],
    embedding_dim: usize,
    sparse_sim: Option<&SparseSimilarity>,
    config: &RootsConfig,
    key: RngKey,
    _compute_config: &HybridConfig,
) -> Vec<PartitionResult> {
    use std::f32::consts::PI;

    // === POLAR CONE EXCLUSION ===
    // Separate points near poles (instruction zone) from equatorial band (content zone)
    let pole_angle = config.pole_exclusion_angle;

    let (north_pole_indices, south_pole_indices, content_indices): (
        Vec<usize>,
        Vec<usize>,
        Vec<usize>,
    ) = if pole_angle > 0.0 {
        let mut north = Vec::new();
        let mut south = Vec::new();
        let mut content = Vec::new();

        for &idx in indices {
            let emb = &embeddings[idx * embedding_dim..(idx + 1) * embedding_dim];
            let theta = cpu_ising::embedding_to_spherical(emb).0;

            if theta < pole_angle {
                north.push(idx);
            } else if theta > PI - pole_angle {
                south.push(idx);
            } else {
                content.push(idx);
            }
        }
        (north, south, content)
    } else {
        // No pole exclusion - all points are content
        (Vec::new(), Vec::new(), indices.to_vec())
    };

    // Partition only the content zone (equatorial torus)
    let partitions = if content_indices.is_empty() {
        Vec::new()
    } else {
        match config.partition_method {
            PartitionMethod::AlternatingMaxCut => {
                // Hybrid: alternating axis direction + max-cut boundary
                cpu_ising::hierarchical_partition_alternating_maxcut(
                    &content_indices,
                    embeddings,
                    embedding_dim,
                    config.n_partitions,
                    config.min_partition_size,
                    config.beta as f64,
                    config.gibbs_warmup,
                    config.gibbs_steps,
                    key.0,
                    0, // Start at depth 0
                )
            }
            PartitionMethod::AlternatingAxis => {
                // Pure alternating-axis: median splits only
                cpu_ising::hierarchical_partition_alternating(
                    &content_indices,
                    embeddings,
                    embedding_dim,
                    config.n_partitions,
                    config.min_partition_size,
                    0, // Start at depth 0
                )
            }
            PartitionMethod::IsingMaxCut => {
                // Pure Ising max-cut: similarity-based only
                if let Some(sparse) = sparse_sim {
                    cpu_ising::hierarchical_partition_sparse(
                        &content_indices,
                        &sparse.indices,
                        &sparse.values,
                        config.n_partitions,
                        config.min_partition_size,
                        config.beta as f64,
                        config.gibbs_warmup,
                        config.gibbs_steps,
                        key.0,
                    )
                } else {
                    cpu_ising::hierarchical_partition(
                        &content_indices,
                        embeddings,
                        embedding_dim,
                        config.n_partitions,
                        config.min_partition_size,
                        config.beta as f64,
                        config.gibbs_warmup,
                        config.gibbs_steps,
                        key.0,
                    )
                }
            }
        }
    };

    // Convert content partitions to PartitionResults
    let mut results: Vec<PartitionResult> = partitions
        .into_iter()
        .map(|indices| PartitionResult {
            indices,
            zone: PartitionZone::Content,
        })
        .collect();

    // === ADD POLAR INSTRUCTION PARTITIONS ===
    // These are special partitions at the poles for behavioral/instruction embeddings
    if !north_pole_indices.is_empty() {
        results.push(PartitionResult {
            indices: north_pole_indices,
            zone: PartitionZone::Instruction,
        });
    }
    if !south_pole_indices.is_empty() {
        results.push(PartitionResult {
            indices: south_pole_indices,
            zone: PartitionZone::QAPairs,
        });
    }

    results
}

/// Run Ising Max-Cut partitioning with bytes enhancement.
///
/// When `config.substring_config` is set, coupling weights combine
/// cosine similarity with substring containment similarity.
fn run_ising_partition_with_bytes(
    indices: &[usize],
    embeddings: &[f32],
    embedding_dim: usize,
    raw_bytes: &[Vec<u8>],
    sparse_sim: Option<&SparseSimilarity>,
    config: &RootsConfig,
    key: RngKey,
    _compute_config: &HybridConfig,
) -> Vec<PartitionResult> {
    // Get substring config, or fall back to embedding-only
    let sub_config = config
        .substring_config
        .unwrap_or_else(SubstringConfig::embedding_only);

    // Use the bytes-enhanced partition functions
    // Note: bytes-enhanced path doesn't yet support polar cone exclusion
    // TODO: Add polar exclusion to bytes-enhanced partitioning
    let partitions = if let Some(sparse) = sparse_sim {
        // Use sparse similarity path with bytes
        cpu_ising::hierarchical_partition_sparse_with_bytes(
            indices,
            &sparse.indices,
            &sparse.values,
            raw_bytes,
            sub_config,
            config.n_partitions,
            config.min_partition_size,
            config.beta as f64,
            config.gibbs_warmup,
            config.gibbs_steps,
            key.0,
        )
    } else {
        // Use dense similarity path with bytes
        cpu_ising::hierarchical_partition_with_bytes(
            indices,
            embeddings,
            embedding_dim,
            raw_bytes,
            sub_config,
            config.n_partitions,
            config.min_partition_size,
            config.beta as f64,
            config.gibbs_warmup,
            config.gibbs_steps,
            key.0,
        )
    };

    // Convert to PartitionResult (all content zones for bytes path)
    partitions
        .into_iter()
        .map(|indices| PartitionResult {
            indices,
            zone: PartitionZone::Content,
        })
        .collect()
}

// ============================================================================
// PatchClassifierEBM (Section 7.3)
// ============================================================================

/// Energy-based classifier for routing queries to partitions.
///
/// From Section 7.3:
/// ```text
/// E(q, k) = ||proj(q) - centroid_k||² / τ_k - log(prior_k)
/// ```
///
/// - Mahalanobis-like distance to patch centroid
/// - Per-patch temperature allows varying "softness"
/// - Prior can encode patch size/importance
#[derive(Clone)]
pub struct PatchClassifierEBM {
    /// Patch centroids \[K, D\].
    pub centroids: Tensor<WgpuBackend, 2>,

    /// Learnable temperature per patch \[K\].
    pub temperatures: Tensor<WgpuBackend, 1>,

    /// Log-prior per patch \[K\] (based on partition size).
    pub log_priors: Tensor<WgpuBackend, 1>,

    /// Optional learnable query projection \[D, D\].
    pub query_projection: Option<Tensor<WgpuBackend, 2>>,

    /// Number of partitions.
    pub n_partitions: usize,

    /// Embedding dimension.
    pub embedding_dim: usize,
}

impl PatchClassifierEBM {
    /// Create a new PatchClassifierEBM from ROOTS partitions.
    pub fn from_partitions(
        partitions: &[RootsPartition],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let k = partitions.len();
        let d = partitions.first().map(|p| p.dim()).unwrap_or(768);

        // Build centroids matrix [K, D]
        let centroids_data: Vec<f32> = partitions.iter().flat_map(|p| p.centroid.clone()).collect();

        let centroids: Tensor<WgpuBackend, 2> = {
            let tensor_1d: Tensor<WgpuBackend, 1> =
                Tensor::from_data(centroids_data.as_slice(), device);
            tensor_1d.reshape([k as i32, d as i32])
        };

        // Initialize temperatures to 1.0
        let temperatures: Tensor<WgpuBackend, 1> = Tensor::ones([k], device);

        // Compute log-priors from partition sizes
        let total_points: usize = partitions.iter().map(|p| p.point_count).sum();
        let log_priors_data: Vec<f32> = partitions
            .iter()
            .map(|p| (p.point_count as f32 / total_points.max(1) as f32).ln())
            .collect();
        let log_priors: Tensor<WgpuBackend, 1> =
            Tensor::from_data(log_priors_data.as_slice(), device);

        Self {
            centroids,
            temperatures,
            log_priors,
            query_projection: None,
            n_partitions: k,
            embedding_dim: d,
        }
    }

    /// Computes energy for a query against all partitions.
    ///
    /// `E(q, k) = ||proj(q) - centroid_k||² / τ_k - log(prior_k)`
    ///
    /// Returns \[K\] energy values (lower = better match).
    pub fn energy(&self, query: &Tensor<WgpuBackend, 1>) -> Tensor<WgpuBackend, 1> {
        // Apply projection if present
        let projected = self.query_projection.as_ref().map_or_else(
            || query.clone(),
            |proj| {
                // query [D] @ proj [D, D] -> [D]
                let query_2d = query.clone().unsqueeze_dim::<2>(0); // [1, D]
                let result = query_2d.matmul(proj.clone()); // [1, D]
                result.squeeze_dim::<1>(0) // [D]
            },
        );

        // Compute squared distances to centroids
        // diff = projected - centroids [K, D]
        let projected_expanded = projected.unsqueeze_dim::<2>(0); // [1, D]
        let diff = self.centroids.clone() - projected_expanded; // [K, D]

        // squared_dist = sum(diff²) per row -> [K]
        let squared_dist: Tensor<WgpuBackend, 1> =
            diff.powf_scalar(2.0).sum_dim(1).squeeze_dim::<1>(1);

        // Scale by temperature and subtract log-prior
        // E(q, k) = ||diff||² / τ_k - log(prior_k)
        let scaled = squared_dist / self.temperatures.clone();
        scaled - self.log_priors.clone()
    }

    /// Compute routing probabilities via softmax.
    ///
    /// P(k | q) = softmax(-E(q, :) / τ_global)
    pub fn route_probabilities(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        temperature: f32,
    ) -> Tensor<WgpuBackend, 1> {
        let energy = self.energy(query);
        let logits = -energy / temperature;

        // Softmax
        let max_logit = logits.clone().max();
        let exp_logits = (logits - max_logit).exp();
        let sum_exp = exp_logits.clone().sum();
        exp_logits / sum_exp
    }

    /// Hard routing: return partition with lowest energy.
    pub fn route(&self, query: &Tensor<WgpuBackend, 1>) -> usize {
        let energy = self.energy(query);
        let energy_data: Vec<f32> = energy.into_data().to_vec().expect("energy to vec");

        energy_data
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

// ============================================================================
// Main ROOTS Index
// ============================================================================

/// The ROOTS index - compressed representation of the full sphere.
///
/// # H-ROOTS: Hierarchical Tree Structure
///
/// The index now supports both flat (legacy) and hierarchical (tree) modes:
///
/// - **Flat mode** (`partitions`): O(K) linear scan for activation
/// - **Tree mode** (`root`): O(log K) beam search for activation
///
/// Tree mode is automatically enabled when using the `*_hierarchical` constructors.
///
/// # Memory Footprint (Section 7.4)
///
/// For K=256 partitions, D=768 embedding dim:
/// - Centroids: 256 * 768 * 4 bytes = 786KB
/// - N-gram dist: 256 * 1KB = 256KB (optional)
/// - Stats + metadata: ~50KB
/// - Total: ~1MB (vs 3GB for 1M full embeddings)
/// - Compression ratio: 3000:1
///
/// # Web Scale (1B vectors)
///
/// With hierarchical tree mode:
/// - Tree depth: ~14 levels
/// - Query ops: ~14 × 2 × 768 ≈ 21k (vs 7.68M for flat scan)
/// - Speedup: ~380x
#[derive(Clone)]
pub struct RootsIndex {
    /// All partitions (flattened from tree for backward compatibility)
    pub partitions: Vec<RootsPartition>,

    /// Hierarchical tree root (H-ROOTS).
    ///
    /// When present, enables O(log K) beam search activation instead of O(K) linear scan.
    /// Use `activate_tree()` to query with the tree, or `activate()` for legacy flat scan.
    pub root: Option<RootsNode>,

    /// Centroids as a tensor [K, D] for fast similarity computation (flat mode)
    pub centroids_matrix: Tensor<WgpuBackend, 2>,

    /// Patch classifier EBM for learned routing
    pub classifier: Option<PatchClassifierEBM>,

    /// Configuration used to build this index
    pub config: RootsConfig,

    /// Total number of points in the source sphere
    pub n_points: usize,

    /// Embedding dimension
    pub embedding_dim: usize,

    /// Sparse similarity (kept for potential re-partitioning)
    sparse_similarity: Option<SparseSimilarity>,

    /// Instruction configuration for polar zones
    pub instruction_config: InstructionConfig,

    /// Index of north pole partition (if any)
    pub north_pole_partition: Option<usize>,

    /// Index of south pole partition (if any)
    pub south_pole_partition: Option<usize>,
}

/// An activation peak detected in ROOTS.
#[derive(Clone, Debug)]
pub struct ActivationPeak {
    /// ID of the partition at this peak
    pub partition_id: usize,
    /// Centroid position (θ, φ) if computed
    pub center: Option<(f32, f32)>,
    /// Activation strength (similarity to query)
    pub strength: f32,
    /// Spread of the activation (how concentrated)
    pub spread: f32,
    /// Indices of points in this partition (may be empty if not stored)
    pub member_indices: Vec<usize>,
}

/// Statistics about a ROOTS index.
#[derive(Clone, Debug)]
pub struct RootsStats {
    pub n_partitions: usize,
    pub n_points: usize,
    pub embedding_dim: usize,
    pub min_partition_size: usize,
    pub max_partition_size: usize,
    pub mean_partition_size: f32,
    pub memory_bytes: usize,
    pub has_ngrams: bool,
    pub has_classifier: bool,
}

impl std::fmt::Display for RootsStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ROOTS Index: {} partitions, {} points, {}D embeddings\n\
             Partition sizes: min={}, max={}, mean={:.1}\n\
             Memory: {:.2} MB | N-grams: {} | Classifier: {}",
            self.n_partitions,
            self.n_points,
            self.embedding_dim,
            self.min_partition_size,
            self.max_partition_size,
            self.mean_partition_size,
            self.memory_bytes as f32 / (1024.0 * 1024.0),
            if self.has_ngrams { "yes" } else { "no" },
            if self.has_classifier { "yes" } else { "no" },
        )
    }
}

impl RootsIndex {
    /// Build ROOTS index from a SphereEBM using Ising Max-Cut partitioning.
    pub fn from_sphere_ebm(
        sphere_ebm: &SphereEBM,
        config: RootsConfig,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        use burn::tensor::Transaction;

        let n_points = sphere_ebm.n_points();
        let embedding_dim = sphere_ebm.embedding_dim();

        // Extract embeddings and prominence to CPU (batched GPU->CPU transfer)
        let [emb_tensor_data, prom_tensor_data] = Transaction::default()
            .register(sphere_ebm.embeddings.clone())
            .register(sphere_ebm.prominence.clone())
            .execute()
            .try_into()
            .expect("Transaction should return 2 tensors");
        let emb_data: Vec<f32> = emb_tensor_data.to_vec().expect("emb to vec");
        let prom_data: Vec<f32> = prom_tensor_data.to_vec().expect("prom to vec");

        // Compute sparse or dense similarity
        let sparse_sim = if n_points > config.dense_threshold && config.similarity_k > 0 {
            Some(cosine_similarity_topk(
                &sphere_ebm.embeddings,
                config.similarity_k,
                device,
            ))
        } else {
            None
        };

        // Use hybrid compute configuration (CPU Ising for Apple Silicon)
        let compute_config = HybridConfig::default();

        // Run hierarchical partitioning using CPU Ising max-cut
        // This uses f64 precision and avoids GPU tensor edge cases
        let indices: Vec<usize> = (0..n_points).collect();
        let partition_results = run_ising_partition(
            &indices,
            &emb_data,
            embedding_dim,
            sparse_sim.as_ref(),
            &config,
            key,
            &compute_config,
        );

        // Build partitions with zone information
        let _actual_k = partition_results.len();
        let mut partitions: Vec<RootsPartition> = partition_results
            .iter()
            .enumerate()
            .map(|(id, pr)| {
                RootsPartition::with_zone(id, embedding_dim, config.store_member_indices, pr.zone)
            })
            .collect();

        for (partition_id, pr) in partition_results.iter().enumerate() {
            for &idx in &pr.indices {
                let emb = &emb_data[idx * embedding_dim..(idx + 1) * embedding_dim];
                partitions[partition_id].add_point(emb, prom_data[idx], idx);
            }
        }

        // Finalize centroids
        for p in &mut partitions {
            p.finalize();
        }

        // Remove empty partitions
        partitions.retain(|p| p.point_count > 0);

        // Build centroids matrix
        let k = partitions.len();
        let centroids_data: Vec<f32> = partitions.iter().flat_map(|p| p.centroid.clone()).collect();

        let centroids_matrix: Tensor<WgpuBackend, 2> = {
            let tensor_1d: Tensor<WgpuBackend, 1> =
                Tensor::from_data(centroids_data.as_slice(), device);
            tensor_1d.reshape([k as i32, embedding_dim as i32])
        };

        // Build classifier
        let classifier = Some(PatchClassifierEBM::from_partitions(&partitions, device));

        // Identify polar partitions (last two if pole_exclusion_angle > 0)
        let (north_pole_partition, south_pole_partition) =
            Self::identify_pole_partitions(&partitions);

        Self {
            partitions,
            root: None, // Flat mode - use from_sphere_ebm_hierarchical for tree mode
            centroids_matrix,
            classifier,
            config,
            n_points,
            embedding_dim,
            sparse_similarity: sparse_sim,
            instruction_config: InstructionConfig::new(),
            north_pole_partition,
            south_pole_partition,
        }
    }

    /// Identify north and south pole partitions by their zone marker.
    fn identify_pole_partitions(partitions: &[RootsPartition]) -> (Option<usize>, Option<usize>) {
        let mut north = None;
        let mut south = None;

        for (i, p) in partitions.iter().enumerate() {
            match p.zone {
                PartitionZone::Instruction => north = Some(i),
                PartitionZone::QAPairs => south = Some(i),
                PartitionZone::Content => {}
            }
        }

        (north, south)
    }

    /// Build ROOTS index from a SphereEBM with raw bytes for substring-enhanced partitioning.
    ///
    /// When `raw_bytes` is provided and `config.substring_config` is set,
    /// the Ising coupling weights combine:
    /// - Cosine similarity of embeddings (semantic)
    /// - Substring containment similarity (structural)
    ///
    /// This creates partitions that group semantically similar AND structurally
    /// related content (e.g., code containing the same function names).
    ///
    /// # Arguments
    /// * `sphere_ebm` - The sphere model with embeddings and prominence
    /// * `raw_bytes` - Raw byte sequences for each point (index-aligned)
    /// * `config` - Configuration with optional substring settings
    /// * `key` - Random key for Gibbs sampling
    /// * `device` - GPU device
    pub fn from_sphere_ebm_with_bytes(
        sphere_ebm: &SphereEBM,
        raw_bytes: &[Vec<u8>],
        config: RootsConfig,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        use burn::tensor::Transaction;

        let n_points = sphere_ebm.n_points();
        let embedding_dim = sphere_ebm.embedding_dim();

        // Validate bytes alignment
        if raw_bytes.len() != n_points {
            eprintln!(
                "Warning: raw_bytes length ({}) != n_points ({}). \
                 Bytes beyond n_points will be ignored.",
                raw_bytes.len(),
                n_points
            );
        }

        // Extract embeddings and prominence to CPU (batched GPU->CPU transfer)
        let [emb_tensor_data, prom_tensor_data] = Transaction::default()
            .register(sphere_ebm.embeddings.clone())
            .register(sphere_ebm.prominence.clone())
            .execute()
            .try_into()
            .expect("Transaction should return 2 tensors");
        let emb_data: Vec<f32> = emb_tensor_data.to_vec().expect("emb to vec");
        let prom_data: Vec<f32> = prom_tensor_data.to_vec().expect("prom to vec");

        // Compute sparse or dense similarity
        let sparse_sim = if n_points > config.dense_threshold && config.similarity_k > 0 {
            Some(cosine_similarity_topk(
                &sphere_ebm.embeddings,
                config.similarity_k,
                device,
            ))
        } else {
            None
        };

        // Use hybrid compute configuration
        let compute_config = HybridConfig::default();

        // Run hierarchical partitioning with bytes enhancement
        let indices: Vec<usize> = (0..n_points).collect();
        let partition_results = run_ising_partition_with_bytes(
            &indices,
            &emb_data,
            embedding_dim,
            raw_bytes,
            sparse_sim.as_ref(),
            &config,
            key,
            &compute_config,
        );

        // Build partitions with zone information
        let _actual_k = partition_results.len();
        let mut partitions: Vec<RootsPartition> = partition_results
            .iter()
            .enumerate()
            .map(|(id, pr)| {
                RootsPartition::with_zone(id, embedding_dim, config.store_member_indices, pr.zone)
            })
            .collect();

        for (partition_id, pr) in partition_results.iter().enumerate() {
            for &idx in &pr.indices {
                let emb = &emb_data[idx * embedding_dim..(idx + 1) * embedding_dim];
                partitions[partition_id].add_point(emb, prom_data[idx], idx);
            }
        }

        // Finalize centroids
        for p in &mut partitions {
            p.finalize();
        }

        // Remove empty partitions
        partitions.retain(|p| p.point_count > 0);

        // Build centroids matrix
        let k = partitions.len();
        let centroids_data: Vec<f32> = partitions.iter().flat_map(|p| p.centroid.clone()).collect();

        let centroids_matrix: Tensor<WgpuBackend, 2> = {
            let tensor_1d: Tensor<WgpuBackend, 1> =
                Tensor::from_data(centroids_data.as_slice(), device);
            tensor_1d.reshape([k as i32, embedding_dim as i32])
        };

        // Build classifier
        let classifier = Some(PatchClassifierEBM::from_partitions(&partitions, device));

        // Identify polar partitions
        let (north_pole_partition, south_pole_partition) =
            Self::identify_pole_partitions(&partitions);

        Self {
            partitions,
            root: None, // Flat mode - use from_sphere_ebm_with_bytes_hierarchical for tree mode
            centroids_matrix,
            classifier,
            config,
            n_points,
            embedding_dim,
            sparse_similarity: sparse_sim,
            instruction_config: InstructionConfig::new(),
            north_pole_partition,
            south_pole_partition,
        }
    }

    // =========================================================================
    // H-ROOTS: Hierarchical Tree Constructors
    // =========================================================================

    /// Build ROOTS index with hierarchical tree structure (H-ROOTS).
    ///
    /// This constructor preserves the binary tree from hierarchical Ising max-cut,
    /// enabling O(log K) query routing instead of O(K) linear scan.
    ///
    /// # When to use
    ///
    /// - **Large datasets** (N > 100K): Tree provides significant speedup
    /// - **Web scale** (N > 1M): Essential for real-time queries
    /// - **Small datasets** (N < 10K): Use flat mode (`from_sphere_ebm`)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let roots = RootsIndex::from_sphere_ebm_hierarchical(
    ///     &sphere_ebm,
    ///     RootsConfig::default().with_partitions(1024),
    ///     RngKey::new(42),
    ///     &device,
    /// );
    ///
    /// // Use tree-based activation (O(log K))
    /// let peaks = roots.activate_tree(&query, 0.1, 10);
    /// ```
    pub fn from_sphere_ebm_hierarchical(
        sphere_ebm: &SphereEBM,
        config: RootsConfig,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        use burn::tensor::Transaction;

        let n_points = sphere_ebm.n_points();
        let embedding_dim = sphere_ebm.embedding_dim();

        // Extract embeddings and prominence to CPU (batched GPU->CPU transfer)
        let [emb_tensor_data, prom_tensor_data] = Transaction::default()
            .register(sphere_ebm.embeddings.clone())
            .register(sphere_ebm.prominence.clone())
            .execute()
            .try_into()
            .expect("Transaction should return 2 tensors");
        let emb_data: Vec<f32> = emb_tensor_data.to_vec().expect("emb to vec");
        let prom_data: Vec<f32> = prom_tensor_data.to_vec().expect("prom to vec");

        // Compute sparse or dense similarity
        let sparse_sim = if n_points > config.dense_threshold && config.similarity_k > 0 {
            Some(cosine_similarity_topk(
                &sphere_ebm.embeddings,
                config.similarity_k,
                device,
            ))
        } else {
            None
        };

        // Build hierarchical tree using tree-preserving partition
        let indices: Vec<usize> = (0..n_points).collect();
        let mut partition_counter = 0;

        let root = if let Some(ref sparse) = sparse_sim {
            cpu_ising::hierarchical_partition_sparse_tree(
                &indices,
                &sparse.indices,
                &sparse.values,
                &emb_data,
                embedding_dim,
                &prom_data,
                config.n_partitions,
                config.min_partition_size,
                config.beta as f64,
                config.gibbs_warmup,
                config.gibbs_steps,
                key.0,
                &mut partition_counter,
                config.store_member_indices,
            )
        } else {
            cpu_ising::hierarchical_partition_tree(
                &indices,
                &emb_data,
                embedding_dim,
                &prom_data,
                config.n_partitions,
                config.min_partition_size,
                config.beta as f64,
                config.gibbs_warmup,
                config.gibbs_steps,
                key.0,
                &mut partition_counter,
                config.store_member_indices,
            )
        };

        // Collect partitions from tree for backward compatibility
        let partitions: Vec<RootsPartition> =
            root.collect_partitions().into_iter().cloned().collect();

        // Build centroids matrix
        let k = partitions.len();
        let centroids_data: Vec<f32> = partitions.iter().flat_map(|p| p.centroid.clone()).collect();

        let centroids_matrix: Tensor<WgpuBackend, 2> = {
            let tensor_1d: Tensor<WgpuBackend, 1> =
                Tensor::from_data(centroids_data.as_slice(), device);
            tensor_1d.reshape([k as i32, embedding_dim as i32])
        };

        // Build classifier
        let classifier = Some(PatchClassifierEBM::from_partitions(&partitions, device));

        Self {
            partitions,
            root: Some(root),
            centroids_matrix,
            classifier,
            config,
            n_points,
            embedding_dim,
            sparse_similarity: sparse_sim,
            instruction_config: InstructionConfig::new(),
            north_pole_partition: None, // Tree mode doesn't track poles separately yet
            south_pole_partition: None,
        }
    }

    /// Build ROOTS index with hierarchical tree and substring-enhanced partitioning.
    ///
    /// Combines the benefits of:
    /// - **H-ROOTS tree**: O(log K) query routing
    /// - **Substring coupling**: Structural byte-level relationships
    pub fn from_sphere_ebm_with_bytes_hierarchical(
        sphere_ebm: &SphereEBM,
        raw_bytes: &[Vec<u8>],
        config: RootsConfig,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        use burn::tensor::Transaction;

        let n_points = sphere_ebm.n_points();
        let embedding_dim = sphere_ebm.embedding_dim();

        // Extract embeddings and prominence to CPU (batched GPU->CPU transfer)
        let [emb_tensor_data, prom_tensor_data] = Transaction::default()
            .register(sphere_ebm.embeddings.clone())
            .register(sphere_ebm.prominence.clone())
            .execute()
            .try_into()
            .expect("Transaction should return 2 tensors");
        let emb_data: Vec<f32> = emb_tensor_data.to_vec().expect("emb to vec");
        let prom_data: Vec<f32> = prom_tensor_data.to_vec().expect("prom to vec");

        // Compute sparse or dense similarity
        let sparse_sim = if n_points > config.dense_threshold && config.similarity_k > 0 {
            Some(cosine_similarity_topk(
                &sphere_ebm.embeddings,
                config.similarity_k,
                device,
            ))
        } else {
            None
        };

        // Get substring config or default to embedding-only
        let sub_config = config
            .substring_config
            .unwrap_or_else(crate::compute::substring::SubstringConfig::embedding_only);

        // Build hierarchical tree using tree-preserving partition
        let indices: Vec<usize> = (0..n_points).collect();
        let mut partition_counter = 0;

        let root = if let Some(ref sparse) = sparse_sim {
            cpu_ising::hierarchical_partition_sparse_with_bytes_tree(
                &indices,
                &sparse.indices,
                &sparse.values,
                &emb_data,
                embedding_dim,
                &prom_data,
                raw_bytes,
                sub_config,
                config.n_partitions,
                config.min_partition_size,
                config.beta as f64,
                config.gibbs_warmup,
                config.gibbs_steps,
                key.0,
                &mut partition_counter,
                config.store_member_indices,
            )
        } else {
            cpu_ising::hierarchical_partition_with_bytes_tree(
                &indices,
                &emb_data,
                embedding_dim,
                &prom_data,
                raw_bytes,
                sub_config,
                config.n_partitions,
                config.min_partition_size,
                config.beta as f64,
                config.gibbs_warmup,
                config.gibbs_steps,
                key.0,
                &mut partition_counter,
                config.store_member_indices,
            )
        };

        // Collect partitions from tree for backward compatibility
        let partitions: Vec<RootsPartition> =
            root.collect_partitions().into_iter().cloned().collect();

        // Build centroids matrix
        let k = partitions.len();
        let centroids_data: Vec<f32> = partitions.iter().flat_map(|p| p.centroid.clone()).collect();

        let centroids_matrix: Tensor<WgpuBackend, 2> = {
            let tensor_1d: Tensor<WgpuBackend, 1> =
                Tensor::from_data(centroids_data.as_slice(), device);
            tensor_1d.reshape([k as i32, embedding_dim as i32])
        };

        // Build classifier
        let classifier = Some(PatchClassifierEBM::from_partitions(&partitions, device));

        Self {
            partitions,
            root: Some(root),
            centroids_matrix,
            classifier,
            config,
            n_points,
            embedding_dim,
            sparse_similarity: sparse_sim,
            instruction_config: InstructionConfig::new(),
            north_pole_partition: None, // Tree mode doesn't track poles separately yet
            south_pole_partition: None,
        }
    }

    // =========================================================================
    // Instruction Zone API (Phase 2)
    // =========================================================================

    /// Returns the number of content partitions (excludes instruction zones).
    pub fn n_content_partitions(&self) -> usize {
        self.partitions.iter().filter(|p| p.is_content()).count()
    }

    /// Returns the number of instruction partitions (pole zones).
    pub fn n_instruction_partitions(&self) -> usize {
        self.partitions
            .iter()
            .filter(|p| p.is_instruction())
            .count()
    }

    /// Returns true if this index has instruction (polar) partitions.
    pub const fn has_instruction_zones(&self) -> bool {
        self.north_pole_partition.is_some() || self.south_pole_partition.is_some()
    }

    /// Gets content partitions only (excludes polar instruction zones).
    pub fn content_partitions(&self) -> impl Iterator<Item = &RootsPartition> {
        self.partitions.iter().filter(|p| p.is_content())
    }

    /// Gets instruction (polar) partitions only.
    pub fn instruction_partitions(&self) -> impl Iterator<Item = &RootsPartition> {
        self.partitions.iter().filter(|p| p.is_instruction())
    }

    /// Gets the north pole partition, if present.
    pub fn north_pole(&self) -> Option<&RootsPartition> {
        self.north_pole_partition.map(|i| &self.partitions[i])
    }

    /// Gets the south pole partition, if present.
    pub fn south_pole(&self) -> Option<&RootsPartition> {
        self.south_pole_partition.map(|i| &self.partitions[i])
    }

    /// Sets instruction embeddings for the poles.
    pub fn set_instruction_config(&mut self, config: InstructionConfig) {
        self.instruction_config = config;
    }

    /// Gets the instruction configuration.
    pub const fn instruction_config(&self) -> &InstructionConfig {
        &self.instruction_config
    }

    /// Computes activation (relevance) for each partition given a query.
    ///
    /// `activation[i] = similarity(query, centroid[i]) * prominence_mean[i]`
    ///
    /// Uses fused CubeCL kernel when `fused-kernels` feature is enabled.
    pub fn activate(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        #[cfg(feature = "fused-kernels")]
        {
            // Use fused cosine similarity kernel
            let query_data: Vec<f32> = query.clone().into_data().to_vec().expect("query to vec");
            let centroids_data: Vec<f32> = self.centroids_matrix.clone().into_data().to_vec().expect("centroids to vec");
            let [n_centroids, d] = self.centroids_matrix.dims();
            
            let device = thrml_core::backend::init_gpu_device();
            let query_cube: Tensor<CubeWgpuBackend, 1> = 
                Tensor::from_floats(query_data.as_slice(), &device);
            let centroids_flat: Tensor<CubeWgpuBackend, 1> = 
                Tensor::from_floats(centroids_data.as_slice(), &device);
            let centroids_cube: Tensor<CubeWgpuBackend, 2> = centroids_flat.reshape([n_centroids, d]);
            
            let sims_cube = cosine_similarity_fused(query_cube, centroids_cube);
            
            // Convert back to WgpuBackend
            let sims_data: Vec<f32> = sims_cube.into_data().to_vec().expect("sims to vec");
            let wgpu_device = query.device();
            Tensor::from_floats(sims_data.as_slice(), &wgpu_device)
        }

        #[cfg(not(feature = "fused-kernels"))]
        {
            let k = self.partitions.len();
            
            // Normalize query
            let query_norm = query.clone().powf_scalar(2.0).sum().sqrt();
            let query_normalized = query.clone() / (query_norm + 1e-8);

            // Normalize centroids
            let centroid_norms = self
                .centroids_matrix
                .clone()
                .powf_scalar(2.0)
                .sum_dim(1)
                .sqrt()
                .clamp(1e-8, f32::MAX);
            let centroids_normalized = self.centroids_matrix.clone() / centroid_norms;

            // Similarity = centroids @ query^T → [K]
            let query_2d = query_normalized.unsqueeze_dim::<2>(1);
            let similarity = centroids_normalized.matmul(query_2d);

            similarity.reshape([k as i32])
        }
    }

    /// Activate using the hierarchical tree (O(log K) beam search).
    ///
    /// This is the preferred method for web-scale indexes. Falls back to flat
    /// activation if no tree is present.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding `[D]`
    /// * `threshold` - Similarity threshold for pruning (default: 0.1)
    /// * `max_results` - Maximum number of peaks to return
    ///
    /// # Returns
    ///
    /// Activation peaks sorted by strength (descending).
    ///
    /// # Complexity
    ///
    /// - With tree: O(log K × D) average case
    /// - Without tree: O(K × D) (falls back to flat activation)
    pub fn activate_tree(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        threshold: f32,
        max_results: usize,
    ) -> Vec<ActivationPeak> {
        // Convert query tensor to CPU slice for tree traversal
        let query_data: Vec<f32> = query.clone().into_data().to_vec().expect("query to vec");

        if let Some(ref root) = self.root {
            // Tree mode: O(log K) beam search
            root.beam_search(&query_data, threshold, max_results)
        } else {
            // Fallback: flat activation + peak detection
            let activations = self.activate(query, &query.device());
            self.detect_peaks(&activations)
                .into_iter()
                .take(max_results)
                .collect()
        }
    }

    /// Check if this index has a hierarchical tree.
    pub const fn has_tree(&self) -> bool {
        self.root.is_some()
    }

    /// Get the tree depth (0 if no tree).
    pub fn tree_depth(&self) -> usize {
        self.root.as_ref().map(|r| r.depth()).unwrap_or(0)
    }

    /// Route a query to the best partition using the classifier EBM.
    pub fn route(&self, query: &Tensor<WgpuBackend, 1>) -> usize {
        if let Some(ref classifier) = self.classifier {
            classifier.route(query)
        } else {
            // Fallback: use activation
            let activations = self.activate(query, &query.device());
            let act_data: Vec<f32> = activations.into_data().to_vec().expect("act to vec");

            act_data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    /// Detect activation peaks for cone spawning (Section 5.4).
    ///
    /// Returns peaks sorted by strength descending.
    pub fn detect_peaks(&self, activations: &Tensor<WgpuBackend, 1>) -> Vec<ActivationPeak> {
        let act_data: Vec<f32> = activations
            .clone()
            .into_data()
            .to_vec()
            .expect("act to vec");

        // Filter to above-threshold activations
        let mut candidates: Vec<(usize, f32)> = act_data
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > self.config.activation_threshold)
            .map(|(i, &a)| (i, a))
            .collect();

        // Sort by strength descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut peaks = Vec::new();

        for (partition_id, strength) in candidates {
            // Check if dominated by nearby peak (simple proximity check)
            let dominated = peaks
                .iter()
                .any(|p: &ActivationPeak| (p.partition_id as i32 - partition_id as i32).abs() < 2);

            if !dominated && partition_id < self.partitions.len() {
                let partition = &self.partitions[partition_id];

                // Compute spread (variance of activation in neighborhood)
                let neighborhood_start = partition_id.saturating_sub(2);
                let neighborhood_end = (partition_id + 3).min(act_data.len());
                let neighborhood: Vec<f32> =
                    act_data[neighborhood_start..neighborhood_end].to_vec();
                let mean = neighborhood.iter().sum::<f32>() / neighborhood.len() as f32;
                let spread = (neighborhood.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
                    / neighborhood.len() as f32)
                    .sqrt();

                peaks.push(ActivationPeak {
                    partition_id,
                    center: partition.mean_position,
                    strength,
                    spread,
                    member_indices: partition.member_indices.clone().unwrap_or_default(),
                });
            }
        }

        peaks
    }

    /// Get statistics about the ROOTS index.
    pub fn stats(&self) -> RootsStats {
        let partition_sizes: Vec<usize> = self.partitions.iter().map(|p| p.size()).collect();
        let total_size = partition_sizes.iter().sum::<usize>();
        let min_size = partition_sizes.iter().cloned().min().unwrap_or(0);
        let max_size = partition_sizes.iter().cloned().max().unwrap_or(0);
        let mean_size = if partition_sizes.is_empty() {
            0.0
        } else {
            total_size as f32 / partition_sizes.len() as f32
        };

        RootsStats {
            n_partitions: self.partitions.len(),
            n_points: self.n_points,
            embedding_dim: self.embedding_dim,
            min_partition_size: min_size,
            max_partition_size: max_size,
            mean_partition_size: mean_size,
            memory_bytes: self.memory_bytes(),
            has_ngrams: self.partitions.iter().any(|p| p.ngram_dist.is_some()),
            has_classifier: self.classifier.is_some(),
        }
    }

    /// Approximate memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        let k = self.partitions.len();
        let d = self.embedding_dim;

        // Centroids matrix: K * D * 4 bytes
        let centroids = k * d * 4;

        // Per partition overhead
        let per_partition: usize = self
            .partitions
            .iter()
            .map(|p| {
                let mut size = d * 4 * 2 // centroid + centroid_sum
                    + 64; // stats and metadata

                if p.ngram_dist.is_some() {
                    size += 256 * 8 + 8; // counts + total
                }

                if let Some(ref indices) = p.member_indices {
                    size += indices.len() * 8;
                }

                size
            })
            .sum();

        // Classifier
        let classifier_size = self.classifier.as_ref().map_or(0, |c| {
            c.n_partitions * c.embedding_dim * 4 // centroids
            + c.n_partitions * 4 * 2 // temperatures + log_priors
        });

        // Sparse similarity (if stored)
        let sparse_size = self
            .sparse_similarity
            .as_ref()
            .map(|s| s.memory_bytes())
            .unwrap_or(0);

        centroids + per_partition + classifier_size + sparse_size
    }

    /// Get the number of partitions.
    pub const fn n_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Get partition by ID.
    pub fn get_partition(&self, id: usize) -> Option<&RootsPartition> {
        self.partitions.get(id)
    }

    /// Extract semantic similarity edges above a threshold.
    ///
    /// Uses the sparse similarity computed during ROOTS construction.
    /// Returns edges as (src, dst, similarity) tuples where similarity > threshold.
    ///
    /// # Arguments
    /// * `threshold` - Minimum cosine similarity (0.0 to 1.0). Recommended: 0.7
    ///
    /// # Returns
    /// Vector of (src_idx, dst_idx, similarity) for edges above threshold
    ///
    /// # Example
    /// ```rust,ignore
    /// let roots = RootsIndex::from_sphere_ebm(&sphere, config, key, &device);
    /// let semantic_edges = roots.extract_semantic_edges(0.7);
    /// // Write to sphere_edges table with label="semantic"
    /// ```
    pub fn extract_semantic_edges(&self, threshold: f32) -> Vec<(usize, usize, f32)> {
        let mut edges = Vec::new();

        if let Some(ref sparse_sim) = self.sparse_similarity {
            for (i, (neighbors, sims)) in sparse_sim
                .indices
                .iter()
                .zip(sparse_sim.values.iter())
                .enumerate()
            {
                for (&j, &sim) in neighbors.iter().zip(sims.iter()) {
                    if sim >= threshold && i < j {
                        // Only emit each edge once (i < j avoids duplicates)
                        edges.push((i, j, sim));
                    }
                }
            }
        }

        edges
    }

    /// Check if sparse similarity is available.
    pub const fn has_sparse_similarity(&self) -> bool {
        self.sparse_similarity.is_some()
    }

    /// Flatten the hierarchical tree into a list of nodes for serialization.
    ///
    /// Returns None if tree was built in flat mode (no root node).
    pub fn flatten_tree(&self) -> Option<Vec<FlatTreeNode>> {
        let root = self.root.as_ref()?;
        let mut nodes = Vec::new();
        let mut node_id_counter = 0;

        self.flatten_node_recursive(root, None, &mut nodes, &mut node_id_counter);

        Some(nodes)
    }

    #[allow(clippy::only_used_in_recursion)] // Method for organization, may access self fields later
    fn flatten_node_recursive(
        &self,
        node: &RootsNode,
        parent_id: Option<usize>,
        nodes: &mut Vec<FlatTreeNode>,
        id_counter: &mut usize,
    ) -> usize {
        let my_id = *id_counter;
        *id_counter += 1;

        match node {
            RootsNode::Leaf { partition } => {
                nodes.push(FlatTreeNode {
                    node_id: my_id,
                    parent_id,
                    is_leaf: true,
                    partition_id: Some(partition.id),
                    left_child: None,
                    right_child: None,
                    centroid: partition.centroid.clone(),
                    point_count: partition.point_count,
                    radius_range: partition.radius_range,
                    prom_range: (
                        partition.prominence_stats.min,
                        partition.prominence_stats.max,
                    ),
                });
            }
            RootsNode::Internal {
                centroid,
                radius_range,
                prominence_range,
                point_count,
                children,
            } => {
                // Reserve slot for this node
                let placeholder_idx = nodes.len();
                nodes.push(FlatTreeNode {
                    node_id: my_id,
                    parent_id,
                    is_leaf: false,
                    partition_id: None,
                    left_child: None,  // Will be filled
                    right_child: None, // Will be filled
                    centroid: centroid.clone(),
                    point_count: *point_count,
                    radius_range: *radius_range,
                    prom_range: *prominence_range,
                });

                // Recurse into children
                let left_id =
                    self.flatten_node_recursive(&children.0, Some(my_id), nodes, id_counter);
                let right_id =
                    self.flatten_node_recursive(&children.1, Some(my_id), nodes, id_counter);

                // Update our node with child IDs
                nodes[placeholder_idx].left_child = Some(left_id);
                nodes[placeholder_idx].right_child = Some(right_id);
            }
        }

        my_id
    }

    /// Reconstruct hierarchical tree from flattened nodes.
    ///
    /// Used when loading from database.
    pub fn unflatten_tree(
        flat_nodes: &[FlatTreeNode],
        partitions: &[RootsPartition],
    ) -> Option<RootsNode> {
        if flat_nodes.is_empty() {
            return None;
        }

        // Build a map from node_id to flat node
        let node_map: std::collections::HashMap<usize, &FlatTreeNode> =
            flat_nodes.iter().map(|n| (n.node_id, n)).collect();

        // Find the root (node with no parent)
        let root_flat = flat_nodes.iter().find(|n| n.parent_id.is_none())?;

        Some(Self::unflatten_node_recursive(
            root_flat, &node_map, partitions,
        ))
    }

    fn unflatten_node_recursive(
        flat: &FlatTreeNode,
        node_map: &std::collections::HashMap<usize, &FlatTreeNode>,
        partitions: &[RootsPartition],
    ) -> RootsNode {
        if flat.is_leaf {
            // Find the partition by ID
            let partition = partitions
                .iter()
                .find(|p| Some(p.id) == flat.partition_id)
                .cloned()
                .unwrap_or_else(|| {
                    // Fallback: create minimal partition from flat node data
                    let mut p = RootsPartition::new(
                        flat.partition_id.unwrap_or(0),
                        flat.centroid.len(),
                        false,
                    );
                    p.centroid = flat.centroid.clone();
                    p.point_count = flat.point_count;
                    p.radius_range = flat.radius_range;
                    p
                });
            RootsNode::Leaf { partition: Box::new(partition) }
        } else {
            // Recurse into children
            let left_flat = node_map.get(&flat.left_child.unwrap()).unwrap();
            let right_flat = node_map.get(&flat.right_child.unwrap()).unwrap();

            let left = Self::unflatten_node_recursive(left_flat, node_map, partitions);
            let right = Self::unflatten_node_recursive(right_flat, node_map, partitions);

            RootsNode::Internal {
                centroid: flat.centroid.clone(),
                radius_range: flat.radius_range,
                prominence_range: flat.prom_range,
                point_count: flat.point_count,
                children: Box::new((left, right)),
            }
        }
    }
}

/// Flattened tree node for serialization.
///
/// The hierarchical tree is converted to a flat list with parent/child
/// references for storage in SQLite.
#[derive(Debug, Clone)]
pub struct FlatTreeNode {
    /// Unique node ID
    pub node_id: usize,
    /// Parent node ID (None for root)
    pub parent_id: Option<usize>,
    /// True if leaf (partition), false if internal (signpost)
    pub is_leaf: bool,
    /// Partition ID (only for leaves)
    pub partition_id: Option<usize>,
    /// Left child node ID (only for internal)
    pub left_child: Option<usize>,
    /// Right child node ID (only for internal)
    pub right_child: Option<usize>,
    /// Centroid embedding
    pub centroid: Vec<f32>,
    /// Point count in subtree
    pub point_count: usize,
    /// Radius range (min, max)
    pub radius_range: (f32, f32),
    /// Prominence range (min, max)
    pub prom_range: (f32, f32),
}

// ============================================================================
// Contrastive Divergence Training (Section 8)
// ============================================================================

/// Training configuration for the PatchClassifierEBM.
///
/// From Section 8.5:
/// ```text
/// L = E(q, k_correct) - log(Σ_k exp(-E(q, k)))
///   = E(q, k_correct) + log Z
/// ```
#[derive(Clone, Debug)]
pub struct ClassifierTrainingConfig {
    /// Learning rate
    pub lr: f32,
    /// Number of negative samples per positive (if using hard negatives)
    pub n_negatives: usize,
    /// Temperature for Gumbel-Softmax (anneal from 1.0 to 0.1)
    pub temperature: f32,
    /// Batch size
    pub batch_size: usize,
    /// Weight decay for regularization
    pub weight_decay: f32,
}

impl Default for ClassifierTrainingConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            n_negatives: 4,
            temperature: 1.0,
            batch_size: 32,
            weight_decay: 1e-4,
        }
    }
}

/// Training metrics for logging.
#[derive(Clone, Debug, Default)]
pub struct TrainingMetrics {
    pub step: usize,
    pub loss: f32,
    pub e_positive: f32,
    pub e_negative: f32,
    pub routing_accuracy: f32,
    pub temperature: f32,
    pub lr: f32,
}

impl PatchClassifierEBM {
    /// Compute contrastive loss for a batch of queries.
    ///
    /// L = E(q, k_correct) + log Z
    ///   = E(q, k_correct) + log(Σ_k exp(-E(q, k)))
    ///
    /// Returns (loss, positive_energy, negative_energy)
    pub fn contrastive_loss(
        &self,
        queries: &Tensor<WgpuBackend, 2>, // [batch, D]
        correct_partitions: &[usize],     // [batch]
        temperature: f32,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, f32, f32) {
        let batch_size = queries.dims()[0];
        let d = queries.dims()[1];

        let mut total_e_pos = 0.0f32;
        let mut total_e_neg = 0.0f32;
        let mut losses = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let query: Tensor<WgpuBackend, 1> =
                queries.clone().slice([i..i + 1, 0..d]).reshape([d as i32]);

            let energies = self.energy(&query);
            let energy_data: Vec<f32> = energies
                .clone()
                .into_data()
                .to_vec()
                .expect("energy to vec");

            // E(q, k_correct)
            let e_pos = energy_data[correct_partitions[i]];
            total_e_pos += e_pos;

            // log Z = log(Σ_k exp(-E(q, k) / τ))
            let scaled_energies: Vec<f32> = energy_data.iter().map(|&e| -e / temperature).collect();
            let max_e: f32 = scaled_energies
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let log_z = max_e
                + scaled_energies
                    .iter()
                    .map(|&e| (e - max_e).exp())
                    .sum::<f32>()
                    .ln();

            // Approximate negative energy from partition function
            let avg_neg_e: f32 = energy_data.iter().sum::<f32>() / energy_data.len() as f32;
            total_e_neg += avg_neg_e;

            // Loss = E_pos + log Z
            let loss = log_z.mul_add(temperature, e_pos);
            losses.push(loss);
        }

        let loss_tensor: Tensor<WgpuBackend, 1> = Tensor::from_data(losses.as_slice(), device);

        (
            loss_tensor,
            total_e_pos / batch_size as f32,
            total_e_neg / batch_size as f32,
        )
    }

    /// Compute routing accuracy for a batch.
    pub fn routing_accuracy(
        &self,
        queries: &Tensor<WgpuBackend, 2>,
        correct_partitions: &[usize],
    ) -> f32 {
        let batch_size = queries.dims()[0];
        let d = queries.dims()[1];

        let mut correct = 0usize;

        for (i, &correct_partition) in correct_partitions.iter().enumerate().take(batch_size) {
            let query: Tensor<WgpuBackend, 1> =
                queries.clone().slice([i..i + 1, 0..d]).reshape([d as i32]);

            let routed = self.route(&query);
            if routed == correct_partition {
                correct += 1;
            }
        }

        correct as f32 / batch_size as f32
    }

    /// Perform one training step (gradient descent on centroids).
    ///
    /// Updates centroids to minimize contrastive loss:
    /// ∂L/∂centroid_k = ⟨∂E/∂centroid_k⟩_{k=k*} - ⟨∂E/∂centroid_k⟩_{k~P}
    pub fn train_step(
        &mut self,
        queries: &Tensor<WgpuBackend, 2>,
        correct_partitions: &[usize],
        config: &ClassifierTrainingConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> TrainingMetrics {
        let batch_size = queries.dims()[0];
        let d = queries.dims()[1];

        // Compute current centroids as Vec for gradient updates
        let centroid_data: Vec<f32> = self
            .centroids
            .clone()
            .into_data()
            .to_vec()
            .expect("centroid to vec");

        let mut grad_centroids = vec![0.0f32; self.n_partitions * self.embedding_dim];
        let mut total_loss = 0.0f32;
        let mut total_e_pos = 0.0f32;
        let mut total_e_neg = 0.0f32;

        for (i, &correct_k) in correct_partitions.iter().enumerate().take(batch_size) {
            let query: Tensor<WgpuBackend, 1> =
                queries.clone().slice([i..i + 1, 0..d]).reshape([d as i32]);
            let query_data: Vec<f32> = query.into_data().to_vec().expect("query to vec");

            // Compute energies and softmax probabilities
            let mut energies = Vec::with_capacity(self.n_partitions);
            let temps_data: Vec<f32> = self
                .temperatures
                .clone()
                .into_data()
                .to_vec()
                .expect("temps to vec");
            let priors_data: Vec<f32> = self
                .log_priors
                .clone()
                .into_data()
                .to_vec()
                .expect("priors to vec");

            for k in 0..self.n_partitions {
                let centroid_k =
                    &centroid_data[k * self.embedding_dim..(k + 1) * self.embedding_dim];
                let diff: Vec<f32> = query_data
                    .iter()
                    .zip(centroid_k.iter())
                    .map(|(q, c)| q - c)
                    .collect();
                let sq_dist: f32 = diff.iter().map(|x| x * x).sum();
                let e = sq_dist / temps_data[k] - priors_data[k];
                energies.push(e);
            }

            // Softmax probabilities
            let scaled: Vec<f32> = energies.iter().map(|&e| -e / config.temperature).collect();
            let max_s: f32 = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_scaled: Vec<f32> = scaled.iter().map(|&s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exp_scaled.iter().sum();
            let probs: Vec<f32> = exp_scaled.iter().map(|&e| e / sum_exp).collect();

            // Positive phase gradient: ∂E/∂centroid_k* = -2(q - c_k*) / τ_k*
            let centroid_correct = &centroid_data
                [correct_k * self.embedding_dim..(correct_k + 1) * self.embedding_dim];
            for (j, (&q, &c)) in query_data.iter().zip(centroid_correct.iter()).enumerate() {
                let grad = -2.0 * (q - c) / temps_data[correct_k];
                grad_centroids[correct_k * self.embedding_dim + j] += grad;
            }

            // Negative phase gradient: E_k~P[∂E/∂centroid_k] = Σ_k P(k) * ∂E/∂centroid_k
            for k in 0..self.n_partitions {
                let centroid_k =
                    &centroid_data[k * self.embedding_dim..(k + 1) * self.embedding_dim];
                for (j, (&q, &c)) in query_data.iter().zip(centroid_k.iter()).enumerate() {
                    let grad = -2.0 * (q - c) / temps_data[k];
                    grad_centroids[k * self.embedding_dim + j] -= probs[k] * grad;
                }
            }

            total_loss += config.temperature.mul_add(max_s + sum_exp.ln(), energies[correct_k]);
            total_e_pos += energies[correct_k];
            total_e_neg += energies.iter().sum::<f32>() / energies.len() as f32;
        }

        // Average gradients and apply update
        let scale = config.lr / batch_size as f32;
        let mut new_centroid_data = centroid_data;
        for (i, grad) in grad_centroids.iter().enumerate() {
            // Gradient descent: c = c - lr * grad
            // Weight decay: c = c * (1 - lr * λ)
            new_centroid_data[i] -= scale * grad;
            new_centroid_data[i] *= config.lr.mul_add(-config.weight_decay, 1.0);
        }

        // Update centroids tensor
        let new_centroids: Tensor<WgpuBackend, 2> = {
            let tensor_1d: Tensor<WgpuBackend, 1> =
                Tensor::from_data(new_centroid_data.as_slice(), device);
            tensor_1d.reshape([self.n_partitions as i32, self.embedding_dim as i32])
        };
        self.centroids = new_centroids;

        // Compute accuracy
        let accuracy = self.routing_accuracy(queries, correct_partitions);

        TrainingMetrics {
            step: 0,
            loss: total_loss / batch_size as f32,
            e_positive: total_e_pos / batch_size as f32,
            e_negative: total_e_neg / batch_size as f32,
            routing_accuracy: accuracy,
            temperature: config.temperature,
            lr: config.lr,
        }
    }
}

/// Learning rate schedule: Warmup + Cosine Decay
///
/// From Section 8.8:
/// ```text
/// lr(t) =
///     lr_max * t / warmup_steps           if t < warmup_steps
///     lr_min + (lr_max - lr_min) * 0.5 * (1 + cos(π * (t - warmup) / (total - warmup)))  otherwise
/// ```
#[derive(Clone, Debug)]
pub struct LRSchedule {
    pub lr_max: f32,
    pub lr_min: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl LRSchedule {
    pub const fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            lr_max,
            lr_min,
            warmup_steps,
            total_steps,
        }
    }

    pub fn get(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.lr_max * step as f32 / self.warmup_steps.max(1) as f32
        } else {
            let t = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let cos_t = (t * std::f32::consts::PI).cos();
            ((self.lr_max - self.lr_min) * 0.5).mul_add(1.0 + cos_t, self.lr_min)
        }
    }
}

/// Temperature annealing schedule for Gumbel-Softmax
///
/// From Section 8.7:
/// Cosine annealing: τ_end + (τ_start - τ_end) * 0.5 * (1 + cos(πt))
#[derive(Clone, Debug)]
pub struct TemperatureSchedule {
    pub tau_start: f32,
    pub tau_end: f32,
    pub total_steps: usize,
}

impl TemperatureSchedule {
    pub const fn new(tau_start: f32, tau_end: f32, total_steps: usize) -> Self {
        Self {
            tau_start,
            tau_end,
            total_steps,
        }
    }

    pub fn get(&self, step: usize) -> f32 {
        let t = step as f32 / self.total_steps.max(1) as f32;
        let cos_t = (t * std::f32::consts::PI).cos();
        ((self.tau_start - self.tau_end) * 0.5).mul_add(1.0 + cos_t, self.tau_end)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScaleProfile;
    use crate::SphereConfig;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_roots_config_defaults() {
        let config = RootsConfig::default();
        assert_eq!(config.n_partitions, 256);
        assert_eq!(config.min_partition_size, 10);
        assert!((config.beta - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_prominence_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = ProminenceStats::from_slice(&values);

        assert!((stats.mean - 3.0).abs() < 0.01);
        assert!((stats.min - 1.0).abs() < 0.01);
        assert!((stats.max - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_byte_ngram_dist() {
        let mut dist = ByteNgramDist::default();
        dist.add_bytes(&[0, 1, 2, 1, 1, 255]);

        assert_eq!(dist.total, 6);
        assert_eq!(dist.counts[1], 3);
        assert_eq!(dist.counts[255], 1);

        let normalized = dist.as_distribution();
        assert!((normalized[1] - 0.5).abs() < 0.01);

        let entropy = dist.entropy();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_roots_from_sphere_ebm() {
        let device = init_gpu_device();
        let n = 100;
        let d = 32;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev);
        let sphere_ebm = SphereEBM::new(embeddings, prominence, None, sphere_config, &device);

        // Use dev config for faster test
        let roots_config = RootsConfig::dev().with_partitions(8);

        let key = RngKey::new(42);
        let roots = RootsIndex::from_sphere_ebm(&sphere_ebm, roots_config, key, &device);

        assert!(
            !roots.partitions.is_empty(),
            "Should have at least 1 partition"
        );

        // All points should be assigned
        let total_assigned: usize = roots.partitions.iter().map(|p| p.point_count).sum();
        assert_eq!(total_assigned, n, "All points should be assigned");

        // Should have classifier
        assert!(roots.classifier.is_some(), "Should have classifier");

        println!("{}", roots.stats());
    }

    #[test]
    fn test_roots_activation() {
        let device = init_gpu_device();
        let n = 50;
        let d = 16;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev);
        let sphere_ebm =
            SphereEBM::new(embeddings.clone(), prominence, None, sphere_config, &device);

        let roots_config = RootsConfig::dev().with_partitions(4);
        let roots =
            RootsIndex::from_sphere_ebm(&sphere_ebm, roots_config, RngKey::new(42), &device);

        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d]);

        let activations = roots.activate(&query, &device);
        let act_data: Vec<f32> = activations.into_data().to_vec().expect("act to vec");

        assert_eq!(act_data.len(), roots.partitions.len());

        let max_act = act_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_act > 0.0, "Should have positive activation somewhere");
    }

    #[test]
    fn test_patch_classifier_ebm() {
        let device = init_gpu_device();
        let n = 50;
        let d = 16;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev);
        let sphere_ebm =
            SphereEBM::new(embeddings.clone(), prominence, None, sphere_config, &device);

        let roots_config = RootsConfig::dev().with_partitions(4);
        let roots =
            RootsIndex::from_sphere_ebm(&sphere_ebm, roots_config, RngKey::new(42), &device);

        let classifier = roots.classifier.unwrap();

        // Test energy computation
        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d]);
        let energy = classifier.energy(&query);

        let energy_data: Vec<f32> = energy.into_data().to_vec().expect("energy to vec");
        assert_eq!(energy_data.len(), roots.partitions.len());

        // Test routing
        let routed = classifier.route(&query);
        assert!(routed < roots.partitions.len());

        // Test probabilities
        let probs = classifier.route_probabilities(&query, 1.0);
        let probs_data: Vec<f32> = probs.into_data().to_vec().expect("probs to vec");

        // Should sum to 1
        let sum: f32 = probs_data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Probabilities should sum to 1");
    }

    #[test]
    fn test_roots_with_bytes() {
        let device = init_gpu_device();
        let n = 32;
        let d = 16;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        // Create byte sequences with containment relationships
        let raw_bytes: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                match i % 4 {
                    0 => format!("function_calculate_total_{}", i).into_bytes(),
                    1 => "calculate_total".to_string().into_bytes(), // contained in 0
                    2 => format!("def process_data_{}", i).into_bytes(),
                    _ => "process_data".to_string().into_bytes(), // contained in 2
                }
            })
            .collect();

        let sphere_config = SphereConfig::from(ScaleProfile::Dev);
        let sphere_ebm = SphereEBM::new(embeddings, prominence, None, sphere_config, &device);

        // Create config WITH substring coupling enabled
        let roots_config = RootsConfig::dev()
            .with_partitions(4)
            .with_default_substring_coupling();

        let roots = RootsIndex::from_sphere_ebm_with_bytes(
            &sphere_ebm,
            &raw_bytes,
            roots_config,
            RngKey::new(42),
            &device,
        );

        // Should have partitions
        assert!(!roots.partitions.is_empty(), "Should have partitions");

        // All points should be assigned
        let total_assigned: usize = roots.partitions.iter().map(|p| p.point_count).sum();
        assert_eq!(total_assigned, n, "All points should be assigned");

        // Should have classifier
        assert!(roots.classifier.is_some(), "Should have classifier");

        println!(
            "ROOTS with bytes: {} partitions, config substring enabled: {}",
            roots.partitions.len(),
            roots_config.substring_config.is_some()
        );
    }

    #[test]
    fn test_substring_config_builder() {
        // Test that substring config can be set via builder
        let config = RootsConfig::default()
            .with_partitions(64)
            .with_default_substring_coupling();

        assert!(config.substring_config.is_some());
        let sub = config.substring_config.unwrap();
        assert_eq!(sub.min_length, 4);
        assert!((sub.alpha - 0.7).abs() < 0.01);
        assert!((sub.beta - 0.3).abs() < 0.01);

        // Test custom weights
        let custom = SubstringConfig::with_weights(0.5, 0.5);
        let config2 = RootsConfig::default().with_substring_coupling(custom);

        let sub2 = config2.substring_config.unwrap();
        assert!((sub2.alpha - 0.5).abs() < 0.01);
        assert!((sub2.beta - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_peak_detection() {
        let device = init_gpu_device();

        let activations: Tensor<WgpuBackend, 1> = Tensor::from_data(
            [0.1f32, 0.8, 0.2, 0.7, 0.1, 0.3, 0.9, 0.2].as_slice(),
            &device,
        );

        let config = RootsConfig::dev().with_threshold(0.5);
        let roots = RootsIndex {
            partitions: (0..8)
                .map(|id| RootsPartition {
                    id,
                    zone: PartitionZone::Content,
                    centroid: vec![0.0],
                    centroid_sum: vec![0.0],
                    point_count: 1,
                    ngram_dist: None,
                    prominence_stats: ProminenceStats::default(),
                    radius_range: (0.0, 1.0),
                    mean_position: None,
                    member_indices: Some(vec![id]),
                })
                .collect(),
            root: None, // Flat mode for this test
            centroids_matrix: Tensor::zeros([8, 1], &device),
            classifier: None,
            config,
            n_points: 8,
            embedding_dim: 1,
            sparse_similarity: None,
            instruction_config: InstructionConfig::new(),
            north_pole_partition: None,
            south_pole_partition: None,
        };

        let peaks = roots.detect_peaks(&activations);

        assert!(!peaks.is_empty(), "Should find at least one peak");
        assert!(peaks[0].strength >= 0.7, "Strongest peak should be >= 0.7");
    }

    #[test]
    fn test_terabyte_config() {
        let config = RootsConfig::terabyte_scale();

        assert_eq!(config.n_partitions, 4096);
        assert!(!config.store_member_indices);
        assert!(config.similarity_k > 0);
    }
}
