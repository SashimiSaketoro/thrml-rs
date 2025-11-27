//! Evaluation metrics for navigation quality.
//!
//! This module provides standard information retrieval metrics for evaluating
//! the quality of navigation results:
//!
//! - **Recall@k**: Fraction of queries where ground truth is in top-k results
//! - **MRR**: Mean Reciprocal Rank - average of 1/rank for correct answers
//! - **nDCG@k**: Normalized Discounted Cumulative Gain
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use thrml_sphere::{evaluate_navigator, NavigationMetrics, TrainingExample};
//!
//! // Evaluate on a test set
//! let metrics = evaluate_navigator(&navigator, &test_examples, 10, key, &device);
//!
//! println!("Recall@1:  {:.4}", metrics.recall_1);
//! println!("Recall@10: {:.4}", metrics.recall_10);
//! println!("MRR:       {:.4}", metrics.mrr);
//! println!("nDCG@10:   {:.4}", metrics.ndcg_10);
//! ```
//!
//! ## Metric Definitions
//!
//! ### Recall@k
//!
//! The fraction of queries where the correct answer appears in the top-k results:
//!
//! ```text
//! Recall@k = (1/N) * Σ_i 𝟙[correct_i ∈ top_k_i]
//! ```
//!
//! ### Mean Reciprocal Rank (MRR)
//!
//! The average of reciprocal ranks across all queries:
//!
//! ```text
//! MRR = (1/N) * Σ_i (1 / rank_i)
//! ```
//!
//! where `rank_i` is the position (1-indexed) of the correct answer.
//! If correct answer is not found, contributes 0.
//!
//! ### nDCG@k (Normalized Discounted Cumulative Gain)
//!
//! Measures ranking quality with position-based discounting:
//!
//! ```text
//! DCG@k = Σ_{i=1}^{k} rel_i / log2(i + 1)
//! nDCG@k = DCG@k / IDCG@k
//! ```
//!
//! For binary relevance (correct/incorrect), IDCG = 1/log2(2) = 1.

use thrml_samplers::RngKey;

use crate::navigator::{MultiConeNavigator, NavigatorEBM};
use crate::training::TrainingExample;

/// Evaluation metrics for navigation quality.
///
/// Contains standard information retrieval metrics computed over a test set.
///
/// # Example
///
/// ```rust,ignore
/// let metrics = evaluate_navigator(&navigator, &test_set, 10, key, &device);
///
/// if metrics.mrr > 0.5 {
///     println!("Good ranking quality!");
/// }
/// ```
#[derive(Clone, Debug, Default)]
pub struct NavigationMetrics {
    /// Recall at k=1 (hit rate).
    pub recall_1: f32,
    /// Recall at k=5.
    pub recall_5: f32,
    /// Recall at k=10.
    pub recall_10: f32,
    /// Mean Reciprocal Rank.
    pub mrr: f32,
    /// Normalized DCG at k=10.
    pub ndcg_10: f32,
    /// Number of queries evaluated.
    pub n_queries: usize,
    /// Average position of correct answer (if found).
    pub avg_rank: f32,
}

impl NavigationMetrics {
    /// Create empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if metrics indicate good performance.
    ///
    /// Returns true if MRR > 0.3 and Recall@10 > 0.5.
    pub fn is_good(&self) -> bool {
        self.mrr > 0.3 && self.recall_10 > 0.5
    }

    /// Get a summary string.
    pub fn summary(&self) -> String {
        format!(
            "R@1={:.3} R@5={:.3} R@10={:.3} MRR={:.3} nDCG@10={:.3}",
            self.recall_1, self.recall_5, self.recall_10, self.mrr, self.ndcg_10
        )
    }
}

impl std::fmt::Display for NavigationMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "NavigationMetrics ({} queries):", self.n_queries)?;
        writeln!(f, "  Recall@1:  {:.4}", self.recall_1)?;
        writeln!(f, "  Recall@5:  {:.4}", self.recall_5)?;
        writeln!(f, "  Recall@10: {:.4}", self.recall_10)?;
        writeln!(f, "  MRR:       {:.4}", self.mrr)?;
        writeln!(f, "  nDCG@10:   {:.4}", self.ndcg_10)?;
        write!(f, "  Avg Rank:  {:.2}", self.avg_rank)
    }
}

// ============================================================================
// Individual Metric Functions
// ============================================================================

/// Compute recall at k.
///
/// Returns 1.0 if the ground truth index appears in the top-k results,
/// 0.0 otherwise.
///
/// # Arguments
///
/// * `results` - Ranked list of result indices (best first)
/// * `ground_truth` - Index of the correct answer
/// * `k` - Number of top results to consider
///
/// # Example
///
/// ```
/// use thrml_sphere::evaluation::recall_at_k;
///
/// let results = vec![5, 3, 7, 1, 9];
/// assert_eq!(recall_at_k(&results, 3, 3), 1.0);  // 3 is in top 3
/// assert_eq!(recall_at_k(&results, 1, 3), 0.0);  // 1 is not in top 3
/// assert_eq!(recall_at_k(&results, 1, 5), 1.0);  // 1 is in top 5
/// ```
pub fn recall_at_k(results: &[usize], ground_truth: usize, k: usize) -> f32 {
    let top_k = results.iter().take(k);
    if top_k.into_iter().any(|&idx| idx == ground_truth) {
        1.0
    } else {
        0.0
    }
}

/// Compute reciprocal rank.
///
/// Returns 1/rank if ground truth is found, 0.0 otherwise.
/// Rank is 1-indexed (first position = rank 1).
///
/// # Arguments
///
/// * `results` - Ranked list of result indices (best first)
/// * `ground_truth` - Index of the correct answer
///
/// # Example
///
/// ```
/// use thrml_sphere::evaluation::reciprocal_rank;
///
/// let results = vec![5, 3, 7, 1, 9];
/// assert!((reciprocal_rank(&results, 5) - 1.0).abs() < 0.001);    // First = 1/1
/// assert!((reciprocal_rank(&results, 3) - 0.5).abs() < 0.001);    // Second = 1/2
/// assert!((reciprocal_rank(&results, 7) - 0.333).abs() < 0.01);   // Third = 1/3
/// assert_eq!(reciprocal_rank(&results, 99), 0.0);                  // Not found
/// ```
pub fn reciprocal_rank(results: &[usize], ground_truth: usize) -> f32 {
    for (i, &idx) in results.iter().enumerate() {
        if idx == ground_truth {
            return 1.0 / (i + 1) as f32;
        }
    }
    0.0
}

/// Compute Mean Reciprocal Rank over multiple queries.
///
/// # Arguments
///
/// * `all_results` - List of (results, ground_truth) pairs
///
/// # Example
///
/// ```
/// use thrml_sphere::evaluation::mean_reciprocal_rank;
///
/// let queries = vec![
///     (vec![1, 2, 3], 1),  // RR = 1.0 (first)
///     (vec![1, 2, 3], 2),  // RR = 0.5 (second)
///     (vec![1, 2, 3], 3),  // RR = 0.333 (third)
/// ];
/// let mrr = mean_reciprocal_rank(&queries);
/// assert!((mrr - 0.611).abs() < 0.01);  // (1 + 0.5 + 0.333) / 3
/// ```
pub fn mean_reciprocal_rank(all_results: &[(Vec<usize>, usize)]) -> f32 {
    if all_results.is_empty() {
        return 0.0;
    }

    let sum_rr: f32 = all_results
        .iter()
        .map(|(results, gt)| reciprocal_rank(results, *gt))
        .sum();

    sum_rr / all_results.len() as f32
}

/// Compute nDCG at k (binary relevance).
///
/// For binary relevance (correct answer has rel=1, others rel=0),
/// the ideal DCG is 1/log2(2) = 1.0 when the correct answer is at position 1.
///
/// # Arguments
///
/// * `results` - Ranked list of result indices (best first)
/// * `ground_truth` - Index of the correct answer
/// * `k` - Number of top results to consider
///
/// # Example
///
/// ```
/// use thrml_sphere::evaluation::ndcg_at_k;
///
/// let results = vec![5, 3, 7, 1, 9];
///
/// // Correct at position 1: DCG = 1/log2(2) = 1.0, nDCG = 1.0
/// assert!((ndcg_at_k(&results, 5, 10) - 1.0).abs() < 0.001);
///
/// // Correct at position 2: DCG = 1/log2(3) ≈ 0.631, nDCG ≈ 0.631
/// assert!((ndcg_at_k(&results, 3, 10) - 0.631).abs() < 0.01);
///
/// // Not in results: nDCG = 0
/// assert_eq!(ndcg_at_k(&results, 99, 10), 0.0);
/// ```
pub fn ndcg_at_k(results: &[usize], ground_truth: usize, k: usize) -> f32 {
    // Find position of ground truth in top-k
    let position = results.iter().take(k).position(|&idx| idx == ground_truth);

    match position {
        Some(pos) => {
            // DCG for binary relevance at position pos (0-indexed)
            // DCG = rel / log2(pos + 2)  (pos + 2 because log2(1) = 0)
            let dcg = 1.0 / (pos as f32 + 2.0).log2();

            // IDCG for binary relevance = 1/log2(2) = 1.0
            let idcg = 1.0;

            dcg / idcg
        }
        None => 0.0,
    }
}

/// Find the rank of ground truth in results (1-indexed).
///
/// Returns None if not found.
pub fn find_rank(results: &[usize], ground_truth: usize) -> Option<usize> {
    results
        .iter()
        .position(|&idx| idx == ground_truth)
        .map(|pos| pos + 1)
}

// ============================================================================
// Evaluation Functions
// ============================================================================

/// Evaluate a NavigatorEBM on a test set.
///
/// Runs navigation for each query in the test set and computes metrics.
///
/// # Arguments
///
/// * `navigator` - The navigator to evaluate
/// * `examples` - Test examples with queries and ground truth
/// * `top_k` - Number of results to retrieve per query
/// * `key` - RNG key for navigation
/// * `device` - GPU device
///
/// # Example
///
/// ```rust,ignore
/// let metrics = evaluate_navigator(&navigator, &test_set, 10, RngKey::new(42), &device);
/// println!("{}", metrics);
/// ```
pub fn evaluate_navigator(
    navigator: &NavigatorEBM,
    examples: &[TrainingExample],
    top_k: usize,
    key: RngKey,
    device: &burn::backend::wgpu::WgpuDevice,
) -> NavigationMetrics {
    if examples.is_empty() {
        return NavigationMetrics::default();
    }

    let keys = key.split(examples.len());

    let mut total_recall_1 = 0.0f32;
    let mut total_recall_5 = 0.0f32;
    let mut total_recall_10 = 0.0f32;
    let mut total_rr = 0.0f32;
    let mut total_ndcg = 0.0f32;
    let mut total_rank = 0.0f32;
    let mut found_count = 0usize;

    for (example, k) in examples.iter().zip(keys) {
        // Run navigation
        let result = navigator.navigate(
            example.query.clone(),
            example.query_radius,
            k,
            top_k.max(10), // At least 10 for recall@10
            device,
        );

        let gt = example.positive_target;
        let results = &result.target_indices;

        // Compute metrics
        total_recall_1 += recall_at_k(results, gt, 1);
        total_recall_5 += recall_at_k(results, gt, 5);
        total_recall_10 += recall_at_k(results, gt, 10);
        total_rr += reciprocal_rank(results, gt);
        total_ndcg += ndcg_at_k(results, gt, 10);

        if let Some(rank) = find_rank(results, gt) {
            total_rank += rank as f32;
            found_count += 1;
        }
    }

    let n = examples.len() as f32;

    NavigationMetrics {
        recall_1: total_recall_1 / n,
        recall_5: total_recall_5 / n,
        recall_10: total_recall_10 / n,
        mrr: total_rr / n,
        ndcg_10: total_ndcg / n,
        n_queries: examples.len(),
        avg_rank: if found_count > 0 {
            total_rank / found_count as f32
        } else {
            f32::INFINITY
        },
    }
}

/// Evaluate a MultiConeNavigator on a test set.
///
/// Similar to [`evaluate_navigator`] but uses multi-cone navigation.
///
/// # Arguments
///
/// * `navigator` - The multi-cone navigator to evaluate
/// * `examples` - Test examples with queries and ground truth
/// * `top_k` - Number of results to retrieve per query
/// * `key` - RNG key for navigation
/// * `device` - GPU device
pub fn evaluate_multi_cone_navigator(
    navigator: &mut MultiConeNavigator,
    examples: &[TrainingExample],
    top_k: usize,
    key: RngKey,
    device: &burn::backend::wgpu::WgpuDevice,
) -> NavigationMetrics {
    if examples.is_empty() {
        return NavigationMetrics::default();
    }

    let keys = key.split(examples.len());

    let mut total_recall_1 = 0.0f32;
    let mut total_recall_5 = 0.0f32;
    let mut total_recall_10 = 0.0f32;
    let mut total_rr = 0.0f32;
    let mut total_ndcg = 0.0f32;
    let mut total_rank = 0.0f32;
    let mut found_count = 0usize;

    for (example, k) in examples.iter().zip(keys) {
        // Run multi-cone navigation
        let result = navigator.navigate_multi_cone(
            example.query.clone(),
            example.query_radius,
            top_k.max(10),
            k,
            device,
        );

        let gt = example.positive_target;
        let results = &result.target_indices;

        // Compute metrics
        total_recall_1 += recall_at_k(results, gt, 1);
        total_recall_5 += recall_at_k(results, gt, 5);
        total_recall_10 += recall_at_k(results, gt, 10);
        total_rr += reciprocal_rank(results, gt);
        total_ndcg += ndcg_at_k(results, gt, 10);

        if let Some(rank) = find_rank(results, gt) {
            total_rank += rank as f32;
            found_count += 1;
        }
    }

    let n = examples.len() as f32;

    NavigationMetrics {
        recall_1: total_recall_1 / n,
        recall_5: total_recall_5 / n,
        recall_10: total_recall_10 / n,
        mrr: total_rr / n,
        ndcg_10: total_ndcg / n,
        n_queries: examples.len(),
        avg_rank: if found_count > 0 {
            total_rank / found_count as f32
        } else {
            f32::INFINITY
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let results = vec![5, 3, 7, 1, 9, 2, 4, 6, 8, 0];

        // Ground truth at position 1
        assert_eq!(recall_at_k(&results, 5, 1), 1.0);
        assert_eq!(recall_at_k(&results, 5, 5), 1.0);

        // Ground truth at position 2
        assert_eq!(recall_at_k(&results, 3, 1), 0.0);
        assert_eq!(recall_at_k(&results, 3, 2), 1.0);
        assert_eq!(recall_at_k(&results, 3, 5), 1.0);

        // Ground truth at position 5
        assert_eq!(recall_at_k(&results, 9, 4), 0.0);
        assert_eq!(recall_at_k(&results, 9, 5), 1.0);

        // Ground truth not in results
        assert_eq!(recall_at_k(&results, 99, 10), 0.0);
    }

    #[test]
    fn test_reciprocal_rank() {
        let results = vec![5, 3, 7, 1, 9];

        // Position 1 -> RR = 1.0
        assert!((reciprocal_rank(&results, 5) - 1.0).abs() < 0.001);

        // Position 2 -> RR = 0.5
        assert!((reciprocal_rank(&results, 3) - 0.5).abs() < 0.001);

        // Position 3 -> RR = 0.333
        assert!((reciprocal_rank(&results, 7) - 1.0 / 3.0).abs() < 0.001);

        // Position 4 -> RR = 0.25
        assert!((reciprocal_rank(&results, 1) - 0.25).abs() < 0.001);

        // Not found -> RR = 0
        assert_eq!(reciprocal_rank(&results, 99), 0.0);
    }

    #[test]
    fn test_mean_reciprocal_rank() {
        let queries = vec![
            (vec![1, 2, 3], 1), // RR = 1.0
            (vec![1, 2, 3], 2), // RR = 0.5
            (vec![1, 2, 3], 3), // RR = 0.333
        ];

        let mrr = mean_reciprocal_rank(&queries);
        let expected = (1.0 + 0.5 + 1.0 / 3.0) / 3.0;
        assert!((mrr - expected).abs() < 0.001);
    }

    #[test]
    fn test_mrr_empty() {
        let queries: Vec<(Vec<usize>, usize)> = vec![];
        assert_eq!(mean_reciprocal_rank(&queries), 0.0);
    }

    #[test]
    fn test_ndcg_at_k() {
        let results = vec![5, 3, 7, 1, 9];

        // Position 1: DCG = 1/log2(2) = 1.0, IDCG = 1.0, nDCG = 1.0
        assert!((ndcg_at_k(&results, 5, 10) - 1.0).abs() < 0.001);

        // Position 2: DCG = 1/log2(3) ≈ 0.631, nDCG ≈ 0.631
        let expected_pos2 = 1.0 / 3.0f32.log2();
        assert!((ndcg_at_k(&results, 3, 10) - expected_pos2).abs() < 0.001);

        // Position 3: DCG = 1/log2(4) = 0.5, nDCG = 0.5
        assert!((ndcg_at_k(&results, 7, 10) - 0.5).abs() < 0.001);

        // Not found
        assert_eq!(ndcg_at_k(&results, 99, 10), 0.0);
    }

    #[test]
    fn test_ndcg_k_limit() {
        let results = vec![5, 3, 7, 1, 9];

        // Ground truth at position 5, but k=3 means we don't see it
        assert_eq!(ndcg_at_k(&results, 9, 3), 0.0);

        // With k=5, we see it
        assert!(ndcg_at_k(&results, 9, 5) > 0.0);
    }

    #[test]
    fn test_find_rank() {
        let results = vec![5, 3, 7, 1, 9];

        assert_eq!(find_rank(&results, 5), Some(1));
        assert_eq!(find_rank(&results, 3), Some(2));
        assert_eq!(find_rank(&results, 7), Some(3));
        assert_eq!(find_rank(&results, 1), Some(4));
        assert_eq!(find_rank(&results, 9), Some(5));
        assert_eq!(find_rank(&results, 99), None);
    }

    #[test]
    fn test_navigation_metrics_display() {
        let metrics = NavigationMetrics {
            recall_1: 0.5,
            recall_5: 0.7,
            recall_10: 0.85,
            mrr: 0.6,
            ndcg_10: 0.75,
            n_queries: 100,
            avg_rank: 3.5,
        };

        let display = format!("{}", metrics);
        assert!(display.contains("0.5"));
        assert!(display.contains("100 queries"));
    }

    #[test]
    fn test_navigation_metrics_summary() {
        let metrics = NavigationMetrics {
            recall_1: 0.5,
            recall_5: 0.7,
            recall_10: 0.85,
            mrr: 0.6,
            ndcg_10: 0.75,
            n_queries: 100,
            avg_rank: 3.5,
        };

        let summary = metrics.summary();
        assert!(summary.contains("R@1="));
        assert!(summary.contains("MRR="));
    }

    #[test]
    fn test_is_good() {
        let good_metrics = NavigationMetrics {
            recall_1: 0.5,
            recall_5: 0.7,
            recall_10: 0.85,
            mrr: 0.6,
            ndcg_10: 0.75,
            n_queries: 100,
            avg_rank: 3.5,
        };
        assert!(good_metrics.is_good());

        let bad_metrics = NavigationMetrics {
            recall_1: 0.1,
            recall_5: 0.2,
            recall_10: 0.3,
            mrr: 0.15,
            ndcg_10: 0.2,
            n_queries: 100,
            avg_rank: 10.0,
        };
        assert!(!bad_metrics.is_good());
    }
}
