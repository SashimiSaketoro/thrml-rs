//! Text and byte-level similarity utilities.
//!
//! This module provides efficient text similarity primitives based on
//! n-gram hashing and substring matching. These are useful for comparing
//! text, code, or any byte sequences based on structural similarity.
//!
//! ## Features
//!
//! - **Rolling Hash**: Efficient n-gram hashing with O(1) sliding window updates
//! - **N-gram Hashing**: Compute hash sets for all n-grams in a byte sequence
//! - **Jaccard Similarity**: Compare hash sets to measure overlap
//! - **Subsequence Matching**: Check if one sequence contains another
//!
//! ## Example
//!
//! ```
//! use thrml_core::text::{ngram_hashes, jaccard_similarity};
//!
//! let text1 = b"hello world";
//! let text2 = b"hello there";
//!
//! let hashes1 = ngram_hashes(text1, 3, 5);
//! let hashes2 = ngram_hashes(text2, 3, 5);
//!
//! let similarity = jaccard_similarity(&hashes1, &hashes2);
//! // Similarity > 0 because they share "hel", "ell", "llo", etc.
//! assert!(similarity > 0.0);
//! ```

use std::collections::HashSet;

/// Rolling hash for efficient n-gram hashing.
///
/// Uses polynomial rolling hash with a large prime base, allowing O(1)
/// updates when sliding a window across bytes.
///
/// Hash formula: `h(s) = s[0] * BASE^(n-1) + s[1] * BASE^(n-2) + ... + s[n-1]`
///
/// # Example
///
/// ```
/// use thrml_core::text::RollingHash;
///
/// let mut hasher = RollingHash::new(3);  // 3-gram
/// hasher.init(b"abc");
///
/// let hash1 = hasher.value();
/// hasher.roll(b'a', b'd');  // Now hashing "bcd"
/// let hash2 = hasher.value();
///
/// assert_ne!(hash1, hash2);
/// ```
#[derive(Clone, Debug)]
pub struct RollingHash {
    /// Current hash value
    hash: u64,
    /// Base for polynomial hash
    base: u64,
    /// Modulus (large prime)
    modulus: u64,
    /// BASE^(window_size-1) mod modulus (for removing leading char)
    pow_base: u64,
    /// Current window size
    window_size: usize,
}

impl RollingHash {
    /// Prime base for hashing (small prime for good distribution)
    pub const BASE: u64 = 31;
    /// Large prime modulus
    pub const MODULUS: u64 = 1_000_000_007;

    /// Create a new rolling hash for the given window size.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Number of bytes in each n-gram
    pub fn new(window_size: usize) -> Self {
        // Compute BASE^(window_size - 1) for removing the leading character
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

    /// Initialize hash with the first window of bytes.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Byte slice (must have at least `window_size` bytes)
    pub fn init(&mut self, bytes: &[u8]) {
        self.hash = 0;
        for &b in bytes.iter().take(self.window_size) {
            self.hash = (self.hash * self.base + b as u64) % self.modulus;
        }
    }

    /// Roll the hash: remove `old_byte`, add `new_byte`.
    ///
    /// Given hash of `bytes[i..i+n]`, computes hash of `bytes[i+1..i+n+1]`
    /// in O(1) time.
    ///
    /// # Arguments
    ///
    /// * `old_byte` - Byte leaving the window (leftmost)
    /// * `new_byte` - Byte entering the window (rightmost)
    pub const fn roll(&mut self, old_byte: u8, new_byte: u8) {
        // Current hash = old_byte * B^(n-1) + ... + last_byte
        // New hash = middle_bytes * B + new_byte
        //
        // Step 1: Remove old_byte's contribution
        let old_contribution = (old_byte as u64 * self.pow_base) % self.modulus;
        let after_sub = (self.hash + self.modulus - old_contribution) % self.modulus;

        // Step 2: Shift left (multiply by base)
        let shifted = (after_sub * self.base) % self.modulus;

        // Step 3: Add new byte
        self.hash = (shifted + new_byte as u64) % self.modulus;
    }

    /// Get the current hash value.
    pub const fn value(&self) -> u64 {
        self.hash
    }

    /// Get the window size.
    pub const fn window_size(&self) -> usize {
        self.window_size
    }
}

/// Compute all n-gram hashes for a byte sequence.
///
/// Returns a HashSet of hashes for all n-grams of lengths in `[min_n, max_n]`.
///
/// # Arguments
///
/// * `bytes` - Input byte sequence
/// * `min_n` - Minimum n-gram length
/// * `max_n` - Maximum n-gram length
///
/// # Example
///
/// ```
/// use thrml_core::text::ngram_hashes;
///
/// let hashes = ngram_hashes(b"hello", 2, 3);
/// // Contains hashes for: "he", "el", "ll", "lo", "hel", "ell", "llo"
/// assert!(!hashes.is_empty());
/// ```
pub fn ngram_hashes(bytes: &[u8], min_n: usize, max_n: usize) -> HashSet<u64> {
    let mut hashes = HashSet::new();

    if bytes.len() < min_n || min_n == 0 {
        return hashes;
    }

    let actual_max = max_n.min(bytes.len());

    for n in min_n..=actual_max {
        if n > bytes.len() {
            break;
        }

        let mut hasher = RollingHash::new(n);
        hasher.init(&bytes[..n]);
        hashes.insert(hasher.value());

        for i in 1..=(bytes.len() - n) {
            hasher.roll(bytes[i - 1], bytes[i + n - 1]);
            hashes.insert(hasher.value());
        }
    }

    hashes
}

/// Compute all n-gram hashes with their lengths (for multi-scale comparison).
///
/// Returns a HashSet of (length, hash) pairs, useful when you want to
/// weight matches by n-gram length.
///
/// # Example
///
/// ```
/// use thrml_core::text::ngram_hashes_with_length;
///
/// let hashes = ngram_hashes_with_length(b"abc", 2, 3);
/// // Contains: (2, hash("ab")), (2, hash("bc")), (3, hash("abc"))
/// assert_eq!(hashes.len(), 3);
/// ```
pub fn ngram_hashes_with_length(bytes: &[u8], min_n: usize, max_n: usize) -> HashSet<(usize, u64)> {
    let mut hashes = HashSet::new();

    if bytes.len() < min_n || min_n == 0 {
        return hashes;
    }

    let actual_max = max_n.min(bytes.len());

    for n in min_n..=actual_max {
        if n > bytes.len() {
            break;
        }

        let mut hasher = RollingHash::new(n);
        hasher.init(&bytes[..n]);
        hashes.insert((n, hasher.value()));

        for i in 1..=(bytes.len() - n) {
            hasher.roll(bytes[i - 1], bytes[i + n - 1]);
            hashes.insert((n, hasher.value()));
        }
    }

    hashes
}

/// Compute Jaccard similarity between two hash sets.
///
/// Jaccard(A, B) = |A ∩ B| / |A ∪ B|
///
/// Returns a value in [0.0, 1.0] where 1.0 means identical sets.
///
/// # Example
///
/// ```
/// use thrml_core::text::{ngram_hashes, jaccard_similarity};
/// use std::collections::HashSet;
///
/// let a: HashSet<u64> = [1, 2, 3].into_iter().collect();
/// let b: HashSet<u64> = [2, 3, 4].into_iter().collect();
///
/// let sim = jaccard_similarity(&a, &b);
/// // Intersection: {2, 3}, Union: {1, 2, 3, 4}
/// // Jaccard = 2/4 = 0.5
/// assert!((sim - 0.5).abs() < 1e-6);
/// ```
pub fn jaccard_similarity(a: &HashSet<u64>, b: &HashSet<u64>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0; // Both empty = identical
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let intersection = a.intersection(b).count();
    let union = a.union(b).count();

    intersection as f32 / union as f32
}

/// Check if `haystack` contains `needle` as a contiguous subsequence.
///
/// # Example
///
/// ```
/// use thrml_core::text::contains_subsequence;
///
/// assert!(contains_subsequence(b"hello world", b"lo wo"));
/// assert!(!contains_subsequence(b"hello", b"world"));
/// assert!(contains_subsequence(b"abc", b""));  // Empty always contained
/// ```
pub fn contains_subsequence(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }

    // Simple sliding window check
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

/// Check mutual containment between two byte sequences.
///
/// Returns `Some((is_a_in_b, container_len, contained_len))` if one contains the other,
/// or `None` if neither contains the other.
///
/// # Example
///
/// ```
/// use thrml_core::text::check_containment;
///
/// // "hello" contains "ell"
/// let result = check_containment(b"hello", b"ell");
/// assert!(result.is_some());
/// let (a_in_b, container, contained) = result.unwrap();
/// assert!(!a_in_b);  // "ell" is in "hello", not vice versa
/// assert_eq!(container, 5);  // "hello" is the container
/// assert_eq!(contained, 3);  // "ell" is contained
/// ```
pub fn check_containment(a: &[u8], b: &[u8]) -> Option<(bool, usize, usize)> {
    if a.len() < b.len() {
        // Check if b contains a
        if contains_subsequence(b, a) {
            return Some((true, b.len(), a.len()));
        }
    } else {
        // Check if a contains b
        if contains_subsequence(a, b) {
            return Some((false, a.len(), b.len()));
        }
    }
    None
}

/// Compute hybrid similarity: weighted combination of two similarity scores.
///
/// `hybrid = alpha * sim_a + beta * sim_b`
///
/// Useful for combining embedding similarity with structural (text) similarity.
///
/// # Arguments
///
/// * `sim_a` - First similarity score (e.g., embedding cosine similarity)
/// * `sim_b` - Second similarity score (e.g., text Jaccard similarity)
/// * `alpha` - Weight for first similarity
/// * `beta` - Weight for second similarity
///
/// # Example
///
/// ```
/// use thrml_core::text::hybrid_similarity;
///
/// let embedding_sim = 0.8;
/// let text_sim = 0.4;
///
/// // 70% embedding, 30% text
/// let hybrid = hybrid_similarity(embedding_sim, text_sim, 0.7, 0.3);
/// assert!((hybrid - 0.68).abs() < 1e-6);  // 0.7*0.8 + 0.3*0.4 = 0.68
/// ```
pub fn hybrid_similarity(sim_a: f32, sim_b: f32, alpha: f32, beta: f32) -> f32 {
    alpha.mul_add(sim_a, beta * sim_b)
}

/// Configuration for text similarity computation.
///
/// Controls n-gram range and combination weights for hybrid similarity.
#[derive(Clone, Copy, Debug)]
pub struct TextSimilarityConfig {
    /// Minimum n-gram length
    pub min_n: usize,
    /// Maximum n-gram length
    pub max_n: usize,
    /// Weight for embedding similarity (alpha)
    pub embedding_weight: f32,
    /// Weight for text similarity (beta)
    pub text_weight: f32,
}

impl Default for TextSimilarityConfig {
    fn default() -> Self {
        Self {
            min_n: 4,
            max_n: 64,
            embedding_weight: 0.7,
            text_weight: 0.3,
        }
    }
}

impl TextSimilarityConfig {
    /// Create config for pure text similarity (no embedding).
    pub fn text_only() -> Self {
        Self {
            embedding_weight: 0.0,
            text_weight: 1.0,
            ..Default::default()
        }
    }

    /// Create config for pure embedding similarity (no text).
    pub fn embedding_only() -> Self {
        Self {
            embedding_weight: 1.0,
            text_weight: 0.0,
            ..Default::default()
        }
    }

    /// Builder: set n-gram range.
    pub const fn with_ngram_range(mut self, min_n: usize, max_n: usize) -> Self {
        self.min_n = min_n;
        self.max_n = max_n;
        self
    }

    /// Builder: set weights.
    pub const fn with_weights(mut self, embedding_weight: f32, text_weight: f32) -> Self {
        self.embedding_weight = embedding_weight;
        self.text_weight = text_weight;
        self
    }
}

/// Compute text similarity between two byte sequences.
///
/// Uses n-gram Jaccard similarity based on the config.
pub fn text_similarity(a: &[u8], b: &[u8], config: &TextSimilarityConfig) -> f32 {
    let hashes_a = ngram_hashes(a, config.min_n, config.max_n);
    let hashes_b = ngram_hashes(b, config.min_n, config.max_n);
    jaccard_similarity(&hashes_a, &hashes_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_hash_basic() {
        let mut hasher = RollingHash::new(3);
        hasher.init(b"abc");
        let hash1 = hasher.value();

        // Different input should give different hash
        hasher.init(b"xyz");
        let hash2 = hasher.value();
        assert_ne!(hash1, hash2);

        // Same input should give same hash
        hasher.init(b"abc");
        assert_eq!(hasher.value(), hash1);
    }

    #[test]
    fn test_rolling_hash_roll() {
        // Hash "abc", then roll to "bcd"
        let mut hasher = RollingHash::new(3);
        hasher.init(b"abc");
        hasher.roll(b'a', b'd');
        let rolled_hash = hasher.value();

        // Direct hash of "bcd"
        let mut hasher2 = RollingHash::new(3);
        hasher2.init(b"bcd");
        let direct_hash = hasher2.value();

        assert_eq!(rolled_hash, direct_hash);
    }

    #[test]
    fn test_ngram_hashes() {
        let hashes = ngram_hashes(b"abcd", 2, 3);
        // 2-grams: "ab", "bc", "cd" (3 hashes)
        // 3-grams: "abc", "bcd" (2 hashes)
        // Total: 5 (assuming no collisions)
        assert_eq!(hashes.len(), 5);
    }

    #[test]
    fn test_ngram_hashes_with_length() {
        let hashes = ngram_hashes_with_length(b"abc", 2, 3);
        // (2, hash("ab")), (2, hash("bc")), (3, hash("abc"))
        assert_eq!(hashes.len(), 3);

        // Check that lengths are correct
        for (len, _) in &hashes {
            assert!(*len >= 2 && *len <= 3);
        }
    }

    #[test]
    fn test_jaccard_similarity() {
        let a: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let b: HashSet<u64> = [2, 3, 4].into_iter().collect();

        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.5).abs() < 1e-6);

        // Identical sets
        let sim2 = jaccard_similarity(&a, &a);
        assert!((sim2 - 1.0).abs() < 1e-6);

        // Disjoint sets
        let c: HashSet<u64> = [5, 6, 7].into_iter().collect();
        let sim3 = jaccard_similarity(&a, &c);
        assert!((sim3 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_contains_subsequence() {
        assert!(contains_subsequence(b"hello world", b"lo wo"));
        assert!(contains_subsequence(b"hello", b"hello"));
        assert!(contains_subsequence(b"hello", b""));
        assert!(!contains_subsequence(b"hello", b"world"));
        assert!(!contains_subsequence(b"hi", b"hello")); // Needle longer
    }

    #[test]
    fn test_check_containment() {
        // "hello" contains "ell"
        let result = check_containment(b"hello", b"ell");
        assert!(result.is_some());
        let (a_in_b, container, contained) = result.unwrap();
        assert!(!a_in_b);
        assert_eq!(container, 5);
        assert_eq!(contained, 3);

        // Neither contains the other
        let result2 = check_containment(b"abc", b"xyz");
        assert!(result2.is_none());
    }

    #[test]
    fn test_hybrid_similarity() {
        let sim = hybrid_similarity(0.8, 0.4, 0.7, 0.3);
        assert!((sim - 0.68).abs() < 1e-6);
    }

    #[test]
    fn test_text_similarity() {
        let config = TextSimilarityConfig::default().with_ngram_range(2, 4);

        // Similar strings
        let sim1 = text_similarity(b"hello world", b"hello there", &config);
        assert!(sim1 > 0.0);

        // Identical strings
        let sim2 = text_similarity(b"hello", b"hello", &config);
        assert!((sim2 - 1.0).abs() < 1e-6);

        // Completely different
        let sim3 = text_similarity(b"abcdefgh", b"12345678", &config);
        assert!((sim3 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_builders() {
        let text_only = TextSimilarityConfig::text_only();
        assert_eq!(text_only.embedding_weight, 0.0);
        assert_eq!(text_only.text_weight, 1.0);

        let embedding_only = TextSimilarityConfig::embedding_only();
        assert_eq!(embedding_only.embedding_weight, 1.0);
        assert_eq!(embedding_only.text_weight, 0.0);
    }
}
