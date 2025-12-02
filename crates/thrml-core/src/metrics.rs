//! Retrieval and ranking evaluation metrics.
//!
//! This module provides standard information retrieval metrics for evaluating
//! ranking and retrieval quality. These are general-purpose primitives that
//! can be used by any ranking/retrieval system.
//!
//! ## Metrics
//!
//! - **Recall@k**: Fraction of queries where the target is in top-k results
//! - **MRR**: Mean Reciprocal Rank - average of 1/rank for correct answers
//! - **nDCG@k**: Normalized Discounted Cumulative Gain
//!
//! ## Example
//!
//! ```
//! use thrml_core::metrics::{recall_at_k, mrr, ndcg};
//!
//! // Retrieved items (indices) and the target we're looking for
//! let retrieved = vec![5, 2, 8, 1, 9];
//! let target = 8;
//!
//! assert_eq!(recall_at_k(&retrieved, target, 3), 1.0);  // Found in top 3
//! assert_eq!(recall_at_k(&retrieved, target, 2), 0.0);  // Not in top 2
//! assert_eq!(mrr(&retrieved, target), 1.0 / 3.0);       // Rank 3 -> 1/3
//! ```

use std::collections::HashSet;

/// Compute Recall@k: whether the target appears in top-k retrieved items.
///
/// Returns 1.0 if target is found in the first k items, 0.0 otherwise.
///
/// # Arguments
///
/// * `retrieved` - Ranked list of retrieved item indices (best first)
/// * `target` - The target item index we're looking for
/// * `k` - Number of top results to consider
///
/// # Example
///
/// ```
/// use thrml_core::metrics::recall_at_k;
///
/// let retrieved = vec![3, 1, 4, 1, 5];
/// assert_eq!(recall_at_k(&retrieved, 4, 3), 1.0);  // 4 is at position 3
/// assert_eq!(recall_at_k(&retrieved, 4, 2), 0.0);  // 4 is not in top 2
/// assert_eq!(recall_at_k(&retrieved, 9, 5), 0.0);  // 9 not in list
/// ```
pub fn recall_at_k(retrieved: &[usize], target: usize, k: usize) -> f32 {
    let k = k.min(retrieved.len());
    if retrieved[..k].contains(&target) {
        1.0
    } else {
        0.0
    }
}

/// Compute Mean Reciprocal Rank (MRR) for a single query.
///
/// Returns 1/rank if the target is found, 0.0 otherwise.
/// Rank is 1-indexed (first position = rank 1).
///
/// # Arguments
///
/// * `retrieved` - Ranked list of retrieved item indices (best first)
/// * `target` - The target item index we're looking for
///
/// # Example
///
/// ```
/// use thrml_core::metrics::mrr;
///
/// let retrieved = vec![3, 1, 4, 1, 5];
/// assert_eq!(mrr(&retrieved, 3), 1.0);       // Rank 1 -> 1/1
/// assert_eq!(mrr(&retrieved, 1), 0.5);       // Rank 2 -> 1/2
/// assert_eq!(mrr(&retrieved, 4), 1.0 / 3.0); // Rank 3 -> 1/3
/// assert_eq!(mrr(&retrieved, 9), 0.0);       // Not found
/// ```
pub fn mrr(retrieved: &[usize], target: usize) -> f32 {
    for (i, &item) in retrieved.iter().enumerate() {
        if item == target {
            return 1.0 / (i + 1) as f32;
        }
    }
    0.0
}

/// Find the rank (1-indexed) of a target in retrieved list.
///
/// Returns None if not found.
///
/// # Example
///
/// ```
/// use thrml_core::metrics::find_rank;
///
/// let retrieved = vec![3, 1, 4, 1, 5];
/// assert_eq!(find_rank(&retrieved, 3), Some(1));
/// assert_eq!(find_rank(&retrieved, 4), Some(3));
/// assert_eq!(find_rank(&retrieved, 9), None);
/// ```
pub fn find_rank(retrieved: &[usize], target: usize) -> Option<usize> {
    retrieved
        .iter()
        .position(|&x| x == target)
        .map(|pos| pos + 1)
}

/// Compute Normalized Discounted Cumulative Gain (nDCG@k).
///
/// For binary relevance (single target), this simplifies to:
/// - DCG@k = 1/log2(rank+1) if target is in top k, else 0
/// - IDCG = 1/log2(2) = 1.0 (perfect ranking puts target at rank 1)
/// - nDCG = DCG / IDCG
///
/// # Arguments
///
/// * `retrieved` - Ranked list of retrieved item indices (best first)
/// * `target` - The target item index (binary relevance)
/// * `k` - Number of top results to consider
///
/// # Example
///
/// ```
/// use thrml_core::metrics::ndcg;
///
/// let retrieved = vec![3, 1, 4, 1, 5];
///
/// // Target at rank 1 -> nDCG = 1.0
/// assert!((ndcg(&retrieved, 3, 5) - 1.0).abs() < 1e-6);
///
/// // Target at rank 3 -> nDCG = (1/log2(4)) / 1.0 = 0.5
/// assert!((ndcg(&retrieved, 4, 5) - 0.5).abs() < 1e-6);
///
/// // Target not in top k -> nDCG = 0.0
/// assert_eq!(ndcg(&retrieved, 5, 3), 0.0);
/// ```
pub fn ndcg(retrieved: &[usize], target: usize, k: usize) -> f32 {
    let k = k.min(retrieved.len());

    // Find rank of target in top-k
    for (i, &item) in retrieved[..k].iter().enumerate() {
        if item == target {
            let rank = i + 1;
            // DCG = 1 / log2(rank + 1)
            // IDCG = 1 / log2(2) = 1.0 for binary relevance
            let dcg = 1.0 / (rank as f32 + 1.0).log2();
            let idcg = 1.0; // Perfect ranking: target at rank 1
            return dcg / idcg;
        }
    }
    0.0
}

/// Compute nDCG@k with multiple relevant items (graded relevance).
///
/// For multiple relevant items, computes proper DCG and IDCG.
///
/// # Arguments
///
/// * `retrieved` - Ranked list of retrieved item indices (best first)
/// * `relevant` - Set of relevant item indices (all equally relevant)
/// * `k` - Number of top results to consider
///
/// # Example
///
/// ```
/// use thrml_core::metrics::ndcg_multi;
///
/// let retrieved = vec![3, 1, 4, 2, 5];
/// let relevant = vec![1, 4];  // Items 1 and 4 are relevant
///
/// // Item 1 at rank 2, item 4 at rank 3
/// // DCG = 1/log2(3) + 1/log2(4) = 0.631 + 0.5 = 1.131
/// // IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.631 = 1.631
/// // nDCG = 1.131 / 1.631 ≈ 0.693
/// let score = ndcg_multi(&retrieved, &relevant, 5);
/// assert!((score - 0.693).abs() < 0.01);
/// ```
pub fn ndcg_multi(retrieved: &[usize], relevant: &[usize], k: usize) -> f32 {
    let k = k.min(retrieved.len());
    let relevant_set: HashSet<_> = relevant.iter().copied().collect();

    if relevant_set.is_empty() {
        return 0.0;
    }

    // Compute DCG
    let mut dcg = 0.0;
    for (i, &item) in retrieved[..k].iter().enumerate() {
        if relevant_set.contains(&item) {
            let rank = i + 1;
            dcg += 1.0 / (rank as f32 + 1.0).log2();
        }
    }

    // Compute IDCG (perfect ranking: all relevant items first)
    let n_relevant_in_k = relevant_set.len().min(k);

    // Early return if no relevant items - avoids division by zero
    if n_relevant_in_k == 0 {
        return 0.0;
    }

    let mut idcg = 0.0;
    for rank in 1..=n_relevant_in_k {
        idcg += 1.0 / (rank as f32 + 1.0).log2();
    }

    dcg / idcg
}

/// Aggregated retrieval metrics over multiple queries.
#[derive(Clone, Debug, Default)]
pub struct RetrievalMetrics {
    /// Recall at various k values: (k, recall)
    pub recall_at_k: Vec<(usize, f32)>,
    /// Mean Reciprocal Rank
    pub mrr: f32,
    /// nDCG at various k values: (k, ndcg)
    pub ndcg_at_k: Vec<(usize, f32)>,
    /// Number of queries evaluated
    pub n_queries: usize,
    /// Average rank of target (if found)
    pub avg_rank: f32,
}

impl RetrievalMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if metrics indicate good retrieval quality.
    ///
    /// Returns true if MRR > threshold and any recall@k > threshold.
    pub fn is_good(&self, mrr_threshold: f32, recall_threshold: f32) -> bool {
        let has_good_recall = self
            .recall_at_k
            .iter()
            .any(|(_, recall)| *recall > recall_threshold);
        self.mrr > mrr_threshold && has_good_recall
    }
}

impl std::fmt::Display for RetrievalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Retrieval Metrics (n={})", self.n_queries)?;
        writeln!(f, "  MRR:      {:.4}", self.mrr)?;
        for (k, recall) in &self.recall_at_k {
            writeln!(f, "  Recall@{}: {:.4}", k, recall)?;
        }
        for (k, ndcg) in &self.ndcg_at_k {
            writeln!(f, "  nDCG@{}:   {:.4}", k, ndcg)?;
        }
        if self.avg_rank > 0.0 {
            writeln!(f, "  Avg Rank: {:.2}", self.avg_rank)?;
        }
        Ok(())
    }
}

/// Evaluate retrieval over multiple queries.
///
/// # Arguments
///
/// * `results` - List of (retrieved_items, target) pairs
/// * `ks` - List of k values to compute recall@k and nDCG@k for
///
/// # Example
///
/// ```
/// use thrml_core::metrics::evaluate_retrieval;
///
/// let results = vec![
///     (vec![1, 2, 3, 4, 5], 3),  // Target 3 at rank 3
///     (vec![5, 4, 3, 2, 1], 5),  // Target 5 at rank 1
///     (vec![1, 2, 3, 4, 5], 9),  // Target 9 not found
/// ];
///
/// let metrics = evaluate_retrieval(&results, &[1, 3, 5]);
///
/// assert_eq!(metrics.n_queries, 3);
/// // MRR = (1/3 + 1/1 + 0) / 3 = 0.444...
/// assert!((metrics.mrr - 0.444).abs() < 0.01);
/// ```
pub fn evaluate_retrieval(results: &[(Vec<usize>, usize)], ks: &[usize]) -> RetrievalMetrics {
    if results.is_empty() {
        return RetrievalMetrics::default();
    }

    let n = results.len() as f32;

    // Accumulate metrics
    let mut total_mrr = 0.0;
    let mut total_rank = 0.0;
    let mut n_found = 0;

    let mut recall_sums: Vec<f32> = vec![0.0; ks.len()];
    let mut ndcg_sums: Vec<f32> = vec![0.0; ks.len()];

    for (retrieved, target) in results {
        total_mrr += mrr(retrieved, *target);

        if let Some(rank) = find_rank(retrieved, *target) {
            total_rank += rank as f32;
            n_found += 1;
        }

        for (i, &k) in ks.iter().enumerate() {
            recall_sums[i] += recall_at_k(retrieved, *target, k);
            ndcg_sums[i] += ndcg(retrieved, *target, k);
        }
    }

    RetrievalMetrics {
        recall_at_k: ks
            .iter()
            .zip(recall_sums.iter())
            .map(|(&k, &sum)| (k, sum / n))
            .collect(),
        mrr: total_mrr / n,
        ndcg_at_k: ks
            .iter()
            .zip(ndcg_sums.iter())
            .map(|(&k, &sum)| (k, sum / n))
            .collect(),
        n_queries: results.len(),
        avg_rank: if n_found > 0 {
            total_rank / n_found as f32
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let retrieved = vec![5, 2, 8, 1, 9];

        assert_eq!(recall_at_k(&retrieved, 5, 1), 1.0); // Found at position 1
        assert_eq!(recall_at_k(&retrieved, 2, 1), 0.0); // Not in top 1
        assert_eq!(recall_at_k(&retrieved, 2, 2), 1.0); // Found at position 2
        assert_eq!(recall_at_k(&retrieved, 8, 3), 1.0); // Found at position 3
        assert_eq!(recall_at_k(&retrieved, 99, 5), 0.0); // Not in list
    }

    #[test]
    fn test_mrr() {
        let retrieved = vec![5, 2, 8, 1, 9];

        assert_eq!(mrr(&retrieved, 5), 1.0); // Rank 1
        assert_eq!(mrr(&retrieved, 2), 0.5); // Rank 2
        assert!((mrr(&retrieved, 8) - 1.0 / 3.0).abs() < 1e-6); // Rank 3
        assert_eq!(mrr(&retrieved, 99), 0.0); // Not found
    }

    #[test]
    fn test_ndcg() {
        let retrieved = vec![5, 2, 8, 1, 9];

        // Rank 1: nDCG = 1.0
        assert!((ndcg(&retrieved, 5, 5) - 1.0).abs() < 1e-6);

        // Rank 2: nDCG = (1/log2(3)) / 1.0 ≈ 0.631
        assert!((ndcg(&retrieved, 2, 5) - 0.631).abs() < 0.01);

        // Rank 3: nDCG = (1/log2(4)) / 1.0 = 0.5
        assert!((ndcg(&retrieved, 8, 5) - 0.5).abs() < 1e-6);

        // Not in top k
        assert_eq!(ndcg(&retrieved, 9, 3), 0.0);
    }

    #[test]
    fn test_ndcg_multi() {
        let retrieved = vec![3, 1, 4, 2, 5];
        let relevant = vec![1, 4]; // Items 1 and 4 are relevant

        let score = ndcg_multi(&retrieved, &relevant, 5);
        // Item 1 at rank 2, item 4 at rank 3
        // DCG = 1/log2(3) + 1/log2(4) ≈ 1.131
        // IDCG = 1/log2(2) + 1/log2(3) ≈ 1.631
        // nDCG ≈ 0.693
        assert!((score - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_evaluate_retrieval() {
        let results = vec![
            (vec![1, 2, 3, 4, 5], 3), // Target 3 at rank 3
            (vec![5, 4, 3, 2, 1], 5), // Target 5 at rank 1
            (vec![1, 2, 3, 4, 5], 9), // Target 9 not found
        ];

        let metrics = evaluate_retrieval(&results, &[1, 3, 5]);

        assert_eq!(metrics.n_queries, 3);

        // MRR = (1/3 + 1/1 + 0) / 3 ≈ 0.444
        assert!((metrics.mrr - 0.444).abs() < 0.01);

        // Recall@1: (0 + 1 + 0) / 3 ≈ 0.333
        assert!((metrics.recall_at_k[0].1 - 0.333).abs() < 0.01);

        // Recall@3: (1 + 1 + 0) / 3 ≈ 0.667
        assert!((metrics.recall_at_k[1].1 - 0.667).abs() < 0.01);
    }
}
