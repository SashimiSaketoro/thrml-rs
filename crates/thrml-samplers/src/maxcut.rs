//! Max-Cut graph partitioning via Gibbs sampling.
//!
//! This module provides CPU-based f64 implementations of max-cut algorithms
//! using Gibbs sampling. These are useful for graph partitioning tasks where
//! high precision is required.
//!
//! ## Algorithm
//!
//! The max-cut problem partitions graph nodes into two sets to maximize
//! the total weight of edges between sets. We use Ising-style Gibbs sampling:
//!
//! 1. Initialize spins σ ∈ {-1, +1}^n randomly
//! 2. For each sweep, visit each node i:
//!    - Compute local field: h_i = Σ_j J_ij * σ_j
//!    - Flip with probability: p = 1 / (1 + exp(2 * β * h_i * σ_i))
//! 3. The partition maximizes Σ_ij J_ij * (1 - σ_i * σ_j) / 2
//!
//! ## Example
//!
//! ```
//! use thrml_samplers::maxcut::{maxcut_gibbs, cut_value};
//!
//! // Simple 4-node graph with edges (0,1), (1,2), (2,3), (3,0)
//! let weights = vec![
//!     vec![0.0, 1.0, 0.0, 1.0],
//!     vec![1.0, 0.0, 1.0, 0.0],
//!     vec![0.0, 1.0, 0.0, 1.0],
//!     vec![1.0, 0.0, 1.0, 0.0],
//! ];
//!
//! let partition = maxcut_gibbs(&weights, 100, 2.0, 42);
//! let value = cut_value(&weights, &partition);
//!
//! // Max cut should be 4 (alternating partition)
//! assert!(value >= 3.0);
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Compute max-cut partition using Gibbs sampling.
///
/// Uses Ising-style Gibbs sampling to find a partition that approximately
/// maximizes the cut value. Runs in f64 precision for numerical stability.
///
/// # Arguments
///
/// * `weights` - Symmetric weight matrix J[i][j] for edge (i,j)
/// * `n_sweeps` - Number of full sweeps through all nodes
/// * `beta` - Inverse temperature (higher = more greedy, lower = more exploration)
/// * `seed` - Random seed for reproducibility
///
/// # Returns
///
/// Partition assignment: +1 or -1 for each node.
///
/// # Example
///
/// ```
/// use thrml_samplers::maxcut::maxcut_gibbs;
///
/// let weights = vec![
///     vec![0.0, 1.0, 1.0],
///     vec![1.0, 0.0, 1.0],
///     vec![1.0, 1.0, 0.0],
/// ];
///
/// let partition = maxcut_gibbs(&weights, 50, 1.0, 42);
/// assert!(partition.iter().all(|&s| s == 1 || s == -1));
/// ```
pub fn maxcut_gibbs(weights: &[Vec<f64>], n_sweeps: usize, beta: f64, seed: u64) -> Vec<i8> {
    let n = weights.len();
    if n == 0 {
        return vec![];
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize random spins
    let mut spins: Vec<i8> = (0..n)
        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
        .collect();

    // Gibbs sampling sweeps
    for _ in 0..n_sweeps {
        for i in 0..n {
            // Compute cut contribution change from flipping spin i
            // Current cut contribution from node i: edges where σ_i ≠ σ_j
            // After flip: edges where σ_i = σ_j (before flip)
            let mut delta_cut = 0.0;
            for j in 0..n {
                if i != j && weights[i][j] != 0.0 {
                    // If currently same sign: flipping adds to cut
                    // If currently different sign: flipping removes from cut
                    if spins[i] == spins[j] {
                        delta_cut += weights[i][j]; // Flip would add this edge to cut
                    } else {
                        delta_cut -= weights[i][j]; // Flip would remove this edge from cut
                    }
                }
            }

            // Metropolis-Hastings acceptance for maximizing cut
            // Accept if delta_cut > 0 (improves cut)
            // Accept with probability exp(beta * delta_cut) if delta_cut < 0
            let accept_prob = if delta_cut >= 0.0 {
                1.0
            } else {
                (beta * delta_cut).exp()
            };

            if rng.gen::<f64>() < accept_prob {
                spins[i] = -spins[i];
            }
        }
    }

    spins
}

/// Compute max-cut with multiple random restarts.
///
/// Runs `maxcut_gibbs` multiple times with different seeds and returns
/// the best partition found.
///
/// # Arguments
///
/// * `weights` - Symmetric weight matrix
/// * `n_sweeps` - Number of sweeps per restart
/// * `beta` - Inverse temperature
/// * `n_restarts` - Number of independent runs
/// * `seed` - Base random seed
///
/// # Returns
///
/// Tuple of (best_partition, best_cut_value)
///
/// # Example
///
/// ```
/// use thrml_samplers::maxcut::maxcut_multistart;
///
/// let weights = vec![
///     vec![0.0, 1.0, 1.0],
///     vec![1.0, 0.0, 1.0],
///     vec![1.0, 1.0, 0.0],
/// ];
///
/// let (partition, value) = maxcut_multistart(&weights, 50, 1.0, 5, 42);
/// assert!(value >= 2.0);  // Triangle: max cut = 2
/// ```
pub fn maxcut_multistart(
    weights: &[Vec<f64>],
    n_sweeps: usize,
    beta: f64,
    n_restarts: usize,
    seed: u64,
) -> (Vec<i8>, f64) {
    let mut best_partition = vec![];
    let mut best_value = f64::NEG_INFINITY;

    for restart in 0..n_restarts {
        let partition = maxcut_gibbs(weights, n_sweeps, beta, seed + restart as u64);
        let value = cut_value(weights, &partition);

        if value > best_value {
            best_value = value;
            best_partition = partition;
        }
    }

    (best_partition, best_value)
}

/// Compute the cut value for a given partition.
///
/// Cut value = Σ_{i<j} J[i][j] * (1 - σ_i * σ_j) / 2
///
/// This counts the total weight of edges crossing the partition.
///
/// # Arguments
///
/// * `weights` - Symmetric weight matrix
/// * `partition` - Node assignments (+1 or -1)
///
/// # Example
///
/// ```
/// use thrml_samplers::maxcut::cut_value;
///
/// let weights = vec![
///     vec![0.0, 1.0, 1.0],
///     vec![1.0, 0.0, 1.0],
///     vec![1.0, 1.0, 0.0],
/// ];
///
/// // All same partition: cut = 0
/// assert_eq!(cut_value(&weights, &[1, 1, 1]), 0.0);
///
/// // Alternating: node 0 vs nodes 1,2: cut = 2
/// assert_eq!(cut_value(&weights, &[1, -1, -1]), 2.0);
/// ```
pub fn cut_value(weights: &[Vec<f64>], partition: &[i8]) -> f64 {
    let n = weights.len();
    let mut value = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            // Edge contributes if nodes are in different partitions
            if partition[i] != partition[j] {
                value += weights[i][j];
            }
        }
    }

    value
}

/// Convert partition from {-1, +1} to {0, 1} representation.
///
/// Useful for interfacing with code expecting binary labels.
pub fn partition_to_binary(partition: &[i8]) -> Vec<u8> {
    partition
        .iter()
        .map(|&s| if s > 0 { 1 } else { 0 })
        .collect()
}

/// Convert partition from {0, 1} to {-1, +1} representation.
pub fn binary_to_partition(binary: &[u8]) -> Vec<i8> {
    binary.iter().map(|&b| if b > 0 { 1 } else { -1 }).collect()
}

/// Compute the Ising energy for a given spin configuration.
///
/// E = -Σ_{i<j} J[i][j] * σ_i * σ_j
///
/// Note: Max-cut maximizes cut_value, which minimizes Ising energy
/// (they differ by a constant).
pub fn ising_energy(weights: &[Vec<f64>], spins: &[i8]) -> f64 {
    let n = weights.len();
    let mut energy = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            energy -= weights[i][j] * (spins[i] as f64) * (spins[j] as f64);
        }
    }

    energy
}

/// Greedy max-cut: flip each node if it improves the cut.
///
/// A fast but less optimal alternative to Gibbs sampling.
/// Useful for initializing or refining solutions.
pub fn maxcut_greedy(weights: &[Vec<f64>], max_iterations: usize, seed: u64) -> Vec<i8> {
    let n = weights.len();
    if n == 0 {
        return vec![];
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize random partition
    let mut spins: Vec<i8> = (0..n)
        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
        .collect();

    for _ in 0..max_iterations {
        let mut improved = false;

        for i in 0..n {
            // Compute gain from flipping node i
            let mut gain = 0.0;
            for j in 0..n {
                if i != j {
                    // Current contribution: positive if same partition
                    // After flip: positive if different partition
                    let current = if spins[i] == spins[j] {
                        0.0
                    } else {
                        weights[i][j]
                    };
                    let after = if spins[i] != spins[j] {
                        0.0
                    } else {
                        weights[i][j]
                    };
                    gain += after - current;
                }
            }

            if gain > 0.0 {
                spins[i] = -spins[i];
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    spins
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_graph() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ]
    }

    fn cycle_4() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn test_cut_value_all_same() {
        let weights = triangle_graph();
        assert_eq!(cut_value(&weights, &[1, 1, 1]), 0.0);
        assert_eq!(cut_value(&weights, &[-1, -1, -1]), 0.0);
    }

    #[test]
    fn test_cut_value_split() {
        let weights = triangle_graph();
        // One node vs two: cuts two edges
        assert_eq!(cut_value(&weights, &[1, -1, -1]), 2.0);
        assert_eq!(cut_value(&weights, &[-1, 1, 1]), 2.0);
    }

    #[test]
    fn test_maxcut_gibbs_triangle() {
        let weights = triangle_graph();
        let partition = maxcut_gibbs(&weights, 100, 2.0, 42);

        let value = cut_value(&weights, &partition);
        // Triangle max cut = 2 (one node vs two)
        assert!(value >= 2.0 - 1e-6);
    }

    #[test]
    fn test_maxcut_gibbs_cycle() {
        let weights = cycle_4();
        let partition = maxcut_gibbs(&weights, 100, 2.0, 42);

        let value = cut_value(&weights, &partition);
        // 4-cycle max cut = 4 (alternating partition)
        assert!(value >= 3.0); // Allow some suboptimality
    }

    #[test]
    fn test_maxcut_multistart() {
        let weights = cycle_4();
        let (partition, value) = maxcut_multistart(&weights, 50, 2.0, 10, 42);

        // With multiple restarts, should find optimal
        assert!(value >= 4.0 - 1e-6);
        // Check partition is alternating
        let alternating = partition
            .windows(2)
            .all(|w| w[0] != w[1] || partition[0] != partition[3]);
        // Either alternating or at least good cut value
        assert!(alternating || value >= 4.0 - 1e-6);
    }

    #[test]
    fn test_partition_conversion() {
        let partition = vec![1, -1, 1, -1];
        let binary = partition_to_binary(&partition);
        assert_eq!(binary, vec![1, 0, 1, 0]);

        let back = binary_to_partition(&binary);
        assert_eq!(back, partition);
    }

    #[test]
    fn test_ising_energy() {
        let weights = triangle_graph();

        // All aligned: E = -3
        let e1 = ising_energy(&weights, &[1, 1, 1]);
        assert!((e1 - (-3.0)).abs() < 1e-6);

        // All anti-aligned from some perspective: E = +1
        let e2 = ising_energy(&weights, &[1, -1, -1]);
        assert!((e2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_maxcut_greedy() {
        let weights = cycle_4();
        let partition = maxcut_greedy(&weights, 100, 42);

        let value = cut_value(&weights, &partition);
        // Greedy should find a reasonable solution
        assert!(value >= 2.0);
    }
}
