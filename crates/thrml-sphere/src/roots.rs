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

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for ROOTS index construction.
#[derive(Clone, Copy, Debug)]
pub struct RootsConfig {
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
}

impl Default for RootsConfig {
    fn default() -> Self {
        Self {
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
    pub fn with_partitions(mut self, n: usize) -> Self {
        self.n_partitions = n;
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
    pub fn with_threshold(mut self, threshold: f32) -> Self {
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
    pub fn with_min_partition_size(mut self, size: usize) -> Self {
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
    pub fn with_beta(mut self, beta: f32) -> Self {
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
    pub fn with_gibbs(mut self, warmup: usize, steps: usize, samples: usize) -> Self {
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
    pub fn with_similarity_k(mut self, k: usize) -> Self {
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
    pub fn without_member_indices(mut self) -> Self {
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
    pub fn without_ngrams(mut self) -> Self {
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
    pub fn with_substring_coupling(mut self, config: SubstringConfig) -> Self {
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
    pub fn merge(&mut self, other: &ProminenceStats) {
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
        let combined_mean = self.mean + delta * (other.count as f32 / combined_count as f32);

        // Combined variance (Welford's parallel algorithm)
        let m2_self = self.std * self.std * self.count as f32;
        let m2_other = other.std * other.std * other.count as f32;
        let m2_combined = m2_self
            + m2_other
            + delta * delta * (self.count as f32 * other.count as f32 / combined_count as f32);

        self.mean = combined_mean;
        self.std = (m2_combined / combined_count as f32).sqrt();
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.count = combined_count;
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
    pub fn merge(&mut self, other: &ByteNgramDist) {
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
    /// Create a new empty partition
    pub fn new(id: usize, embedding_dim: usize, store_indices: bool) -> Self {
        Self {
            id,
            centroid: vec![0.0; embedding_dim],
            centroid_sum: vec![0.0; embedding_dim],
            point_count: 0,
            ngram_dist: None,
            prominence_stats: ProminenceStats::default(),
            radius_range: (f32::MAX, f32::MIN),
            mean_position: None,
            member_indices: if store_indices { Some(Vec::new()) } else { None },
        }
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
    pub fn size(&self) -> usize {
        self.point_count
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.centroid.len()
    }
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

/// Perform Ising max-cut partitioning using the appropriate backend.
///
/// On Apple Silicon (unified memory), uses CPU f64 Ising for better precision.
/// On discrete GPU systems, uses GPU-based partitioning.
fn run_ising_partition(
    indices: &[usize],
    embeddings: &[f32],
    embedding_dim: usize,
    sparse_sim: Option<&SparseSimilarity>,
    config: &RootsConfig,
    key: RngKey,
    _compute_config: &HybridConfig,
) -> Vec<Vec<usize>> {
    // Use CPU Ising for precision-sensitive max-cut
    // (This is the recommended path for Apple Silicon with unified memory)
    
    if let Some(sparse) = sparse_sim {
        // Use sparse similarity path
        cpu_ising::hierarchical_partition_sparse(
            indices,
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
        // Use dense similarity path
        cpu_ising::hierarchical_partition(
            indices,
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
) -> Vec<Vec<usize>> {
    // Get substring config, or fall back to embedding-only
    let sub_config = config.substring_config.unwrap_or(SubstringConfig::embedding_only());

    // Use the bytes-enhanced partition functions
    if let Some(sparse) = sparse_sim {
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
    }
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
        let centroids_data: Vec<f32> = partitions
            .iter()
            .flat_map(|p| p.centroid.clone())
            .collect();

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
        let projected = if let Some(ref proj) = self.query_projection {
            // query [D] @ proj [D, D] -> [D]
            let query_2d = query.clone().unsqueeze_dim::<2>(0); // [1, D]
            let result = query_2d.matmul(proj.clone()); // [1, D]
            result.squeeze_dim::<1>(0) // [D]
        } else {
            query.clone()
        };

        // Compute squared distances to centroids
        // diff = projected - centroids [K, D]
        let projected_expanded = projected.unsqueeze_dim::<2>(0); // [1, D]
        let diff = self.centroids.clone() - projected_expanded; // [K, D]

        // squared_dist = sum(diff²) per row -> [K]
        let squared_dist: Tensor<WgpuBackend, 1> = diff
            .powf_scalar(2.0)
            .sum_dim(1)
            .squeeze_dim::<1>(1);

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
/// # Memory Footprint (Section 7.4)
///
/// For K=256 partitions, D=768 embedding dim:
/// - Centroids: 256 * 768 * 4 bytes = 786KB
/// - N-gram dist: 256 * 1KB = 256KB (optional)
/// - Stats + metadata: ~50KB
/// - Total: ~1MB (vs 3GB for 1M full embeddings)
/// - Compression ratio: 3000:1
#[derive(Clone)]
pub struct RootsIndex {
    /// All partitions
    pub partitions: Vec<RootsPartition>,

    /// Centroids as a tensor [K, D] for fast similarity computation
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
        let n_points = sphere_ebm.n_points();
        let embedding_dim = sphere_ebm.embedding_dim();

        // Extract embeddings and prominence to CPU
        let emb_data: Vec<f32> = sphere_ebm
            .embeddings
            .clone()
            .into_data()
            .to_vec()
            .expect("emb to vec");
        let prom_data: Vec<f32> = sphere_ebm
            .prominence
            .clone()
            .into_data()
            .to_vec()
            .expect("prom to vec");

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
        let partition_assignments = run_ising_partition(
            &indices,
            &emb_data,
            embedding_dim,
            sparse_sim.as_ref(),
            &config,
            key,
            &compute_config,
        );

        // Build partitions
        let actual_k = partition_assignments.len();
        let mut partitions: Vec<RootsPartition> = (0..actual_k)
            .map(|id| RootsPartition::new(id, embedding_dim, config.store_member_indices))
            .collect();

        for (partition_id, member_indices) in partition_assignments.iter().enumerate() {
            for &idx in member_indices {
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
        let centroids_data: Vec<f32> = partitions
            .iter()
            .flat_map(|p| p.centroid.clone())
            .collect();

        let centroids_matrix: Tensor<WgpuBackend, 2> = {
            let tensor_1d: Tensor<WgpuBackend, 1> =
                Tensor::from_data(centroids_data.as_slice(), device);
            tensor_1d.reshape([k as i32, embedding_dim as i32])
        };

        // Build classifier
        let classifier = Some(PatchClassifierEBM::from_partitions(&partitions, device));

        Self {
            partitions,
            centroids_matrix,
            classifier,
            config,
            n_points,
            embedding_dim,
            sparse_similarity: sparse_sim,
        }
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

        // Extract embeddings and prominence to CPU
        let emb_data: Vec<f32> = sphere_ebm
            .embeddings
            .clone()
            .into_data()
            .to_vec()
            .expect("emb to vec");
        let prom_data: Vec<f32> = sphere_ebm
            .prominence
            .clone()
            .into_data()
            .to_vec()
            .expect("prom to vec");

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
        let partition_assignments = run_ising_partition_with_bytes(
            &indices,
            &emb_data,
            embedding_dim,
            raw_bytes,
            sparse_sim.as_ref(),
            &config,
            key,
            &compute_config,
        );

        // Build partitions
        let actual_k = partition_assignments.len();
        let mut partitions: Vec<RootsPartition> = (0..actual_k)
            .map(|id| RootsPartition::new(id, embedding_dim, config.store_member_indices))
            .collect();

        for (partition_id, member_indices) in partition_assignments.iter().enumerate() {
            for &idx in member_indices {
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
        let centroids_data: Vec<f32> = partitions
            .iter()
            .flat_map(|p| p.centroid.clone())
            .collect();

        let centroids_matrix: Tensor<WgpuBackend, 2> = {
            let tensor_1d: Tensor<WgpuBackend, 1> =
                Tensor::from_data(centroids_data.as_slice(), device);
            tensor_1d.reshape([k as i32, embedding_dim as i32])
        };

        // Build classifier
        let classifier = Some(PatchClassifierEBM::from_partitions(&partitions, device));

        Self {
            partitions,
            centroids_matrix,
            classifier,
            config,
            n_points,
            embedding_dim,
            sparse_similarity: sparse_sim,
        }
    }

    /// Computes activation (relevance) for each partition given a query.
    ///
    /// `activation[i] = similarity(query, centroid[i]) * prominence_mean[i]`
    pub fn activate(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
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
            let dominated = peaks.iter().any(|p: &ActivationPeak| {
                (p.partition_id as i32 - partition_id as i32).abs() < 2
            });

            if !dominated && partition_id < self.partitions.len() {
                let partition = &self.partitions[partition_id];

                // Compute spread (variance of activation in neighborhood)
                let neighborhood_start = partition_id.saturating_sub(2);
                let neighborhood_end = (partition_id + 3).min(act_data.len());
                let neighborhood: Vec<f32> = act_data[neighborhood_start..neighborhood_end].to_vec();
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
        let classifier_size = if let Some(ref c) = self.classifier {
            c.n_partitions * c.embedding_dim * 4 // centroids
            + c.n_partitions * 4 * 2 // temperatures + log_priors
        } else {
            0
        };

        // Sparse similarity (if stored)
        let sparse_size = self
            .sparse_similarity
            .as_ref()
            .map(|s| s.memory_bytes())
            .unwrap_or(0);

        centroids + per_partition + classifier_size + sparse_size
    }

    /// Get the number of partitions.
    pub fn n_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Get partition by ID.
    pub fn get_partition(&self, id: usize) -> Option<&RootsPartition> {
        self.partitions.get(id)
    }
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
            let query: Tensor<WgpuBackend, 1> = queries
                .clone()
                .slice([i..i + 1, 0..d])
                .reshape([d as i32]);

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
            let scaled_energies: Vec<f32> = energy_data
                .iter()
                .map(|&e| -e / temperature)
                .collect();
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
            let loss = e_pos + log_z * temperature;
            losses.push(loss);
        }

        let loss_tensor: Tensor<WgpuBackend, 1> =
            Tensor::from_data(losses.as_slice(), device);

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

        for i in 0..batch_size {
            let query: Tensor<WgpuBackend, 1> = queries
                .clone()
                .slice([i..i + 1, 0..d])
                .reshape([d as i32]);

            let routed = self.route(&query);
            if routed == correct_partitions[i] {
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

        for i in 0..batch_size {
            let query: Tensor<WgpuBackend, 1> = queries
                .clone()
                .slice([i..i + 1, 0..d])
                .reshape([d as i32]);
            let query_data: Vec<f32> = query
                .into_data()
                .to_vec()
                .expect("query to vec");

            let correct_k = correct_partitions[i];

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
            let scaled: Vec<f32> = energies
                .iter()
                .map(|&e| -e / config.temperature)
                .collect();
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

            total_loss += energies[correct_k]
                + config.temperature
                    * (max_s + sum_exp.ln());
            total_e_pos += energies[correct_k];
            total_e_neg += energies.iter().sum::<f32>() / energies.len() as f32;
        }

        // Average gradients and apply update
        let scale = config.lr / batch_size as f32;
        let mut new_centroid_data = centroid_data.clone();
        for (i, grad) in grad_centroids.iter().enumerate() {
            // Gradient descent: c = c - lr * grad
            // Weight decay: c = c * (1 - lr * λ)
            new_centroid_data[i] -= scale * grad;
            new_centroid_data[i] *= 1.0 - config.lr * config.weight_decay;
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
    pub fn new(lr_max: f32, lr_min: f32, warmup_steps: usize, total_steps: usize) -> Self {
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
            self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1.0 + cos_t)
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
    pub fn new(tau_start: f32, tau_end: f32, total_steps: usize) -> Self {
        Self {
            tau_start,
            tau_end,
            total_steps,
        }
    }

    pub fn get(&self, step: usize) -> f32 {
        let t = step as f32 / self.total_steps.max(1) as f32;
        let cos_t = (t * std::f32::consts::PI).cos();
        self.tau_end + (self.tau_start - self.tau_end) * 0.5 * (1.0 + cos_t)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use crate::config::ScaleProfile;
    use crate::SphereConfig;
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

        assert!(roots.partitions.len() >= 1, "Should have at least 1 partition");

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
        let roots = RootsIndex::from_sphere_ebm(&sphere_ebm, roots_config, RngKey::new(42), &device);

        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);

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
        let roots = RootsIndex::from_sphere_ebm(&sphere_ebm, roots_config, RngKey::new(42), &device);

        let classifier = roots.classifier.unwrap();

        // Test energy computation
        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);
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
                    1 => format!("calculate_total").into_bytes(), // contained in 0
                    2 => format!("def process_data_{}", i).into_bytes(),
                    _ => format!("process_data").into_bytes(),    // contained in 2
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
        let config2 = RootsConfig::default()
            .with_substring_coupling(custom);

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
            centroids_matrix: Tensor::zeros([8, 1], &device),
            classifier: None,
            config,
            n_points: 8,
            embedding_dim: 1,
            sparse_similarity: None,
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
