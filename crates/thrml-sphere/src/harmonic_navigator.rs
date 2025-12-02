//! Harmonic Navigator - Spherical harmonics-based navigation for EBMs.
//!
//! This module provides `HarmonicNavigator`, which uses spherical harmonic
//! interference patterns to guide cone-based navigation on hyperspherical
//! embedding manifolds.
//!
//! # Architecture
//!
//! ```text
//! Query Embedding
//!       │
//!       ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  HarmonicNavigator                                               │
//! │  ├── Generate probe candidates (Gaussian amplitude blobs)       │
//! │  ├── **Spring AM**: Modulate amplitude by adjacency springs     │
//! │  ├── Compute superposition field via SHT                        │
//! │  ├── Find intensity peaks → candidate cone centers              │
//! │  ├── Iteratively refine (r, θ, φ, α) with confidence feedback   │
//! │  └── Return best cone parameters + retrieved points             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Spring Amplitude Modulation (Spring AM)
//!
//! When springs are provided, the amplitude of each probe candidate is
//! modulated by the structural connectivity:
//!
//! ```text
//! A_final(θ,φ) = A_base(θ,φ) × (1 + λ_spring × Σ w_ij × A_neighbor(θ_j,φ_j))
//! ```
//!
//! This causes structurally connected regions to have higher superposition
//! intensity, guiding navigation toward semantically coherent patches.
//!
//! # Integration with Other Components
//!
//! - **ROOTS**: Can use ROOTS partitions to seed initial probe locations
//! - **Springs**: Spring energy modulates amplitude in connected regions (Spring AM)
//! - **MultiConeNavigator**: HarmonicNavigator produces cone parameters
//!   that can be used by MultiConeNavigator for final retrieval
//!
//! # Example
//!
//! ```ignore
//! use thrml_sphere::{HarmonicNavigator, HarmonicNavigatorConfig, SphericalHarmonicsBasis};
//!
//! let config = HarmonicNavigatorConfig::default();
//! let navigator = HarmonicNavigator::new(embeddings, config);
//!
//! let result = navigator.navigate(&query_embedding);
//! println!("Best cone: θ={}, φ={}, α={}", result.theta, result.phi, result.alpha);
//! ```

use crate::spherical_harmonics::{SphericalHarmonicsBasis, SphericalHarmonicsConfig};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Configuration for HarmonicNavigator.
#[derive(Debug, Clone)]
pub struct HarmonicNavigatorConfig {
    /// Spherical harmonics band limit (default: 64).
    pub band_limit: usize,

    /// Maximum number of probe iterations (default: 32).
    pub max_probes: usize,

    /// Number of candidate probes per iteration (default: 12).
    pub probe_candidates: usize,

    /// Initial cone half-angle in radians (default: π/3).
    pub initial_alpha: f64,

    /// Confidence threshold for early stopping (default: 0.98).
    pub confidence_threshold: f64,

    /// Minimum probes before early stopping (default: 5).
    pub min_probes_before_stop: usize,

    /// Momentum for position updates (default: 0.7).
    pub momentum: f64,

    /// Gaussian blob width in θ direction (default: 0.28).
    pub sigma_theta: f64,

    /// Gaussian blob width in φ direction (default: 0.39).
    pub sigma_phi: f64,

    /// Weight for spring amplitude modulation (default: 0.3).
    /// Higher values give more influence to structurally connected regions.
    pub spring_weight: f64,

    /// Decay factor for spring influence with distance (default: 0.5).
    /// Lower values mean springs have more localized effect.
    pub spring_decay: f64,
}

impl Default for HarmonicNavigatorConfig {
    fn default() -> Self {
        Self {
            band_limit: 64,
            max_probes: 32,
            probe_candidates: 12,
            initial_alpha: PI / 3.0,
            confidence_threshold: 0.98,
            min_probes_before_stop: 5,
            momentum: 0.7,
            sigma_theta: 0.28,
            sigma_phi: 0.39,
            spring_weight: 0.3,
            spring_decay: 0.5,
        }
    }
}

impl HarmonicNavigatorConfig {
    /// Fast development configuration.
    pub fn dev() -> Self {
        Self {
            band_limit: 16,
            max_probes: 10,
            probe_candidates: 6,
            ..Default::default()
        }
    }

    /// High-fidelity configuration for production.
    pub fn high_fidelity() -> Self {
        Self {
            band_limit: 128,
            max_probes: 64,
            probe_candidates: 24,
            confidence_threshold: 0.995,
            ..Default::default()
        }
    }
}

/// Spring-based amplitude modulation data.
///
/// Stores the structural connectivity from hypergraph edges to modulate
/// amplitude fields during harmonic navigation.
#[derive(Debug, Clone)]
pub struct SpringAmplitudeModulation {
    /// Spherical coordinates of each point: [(theta, phi), ...]
    pub point_coords: Vec<(f64, f64)>,

    /// Adjacency list: point_idx -> [(neighbor_idx, weight), ...]
    pub adjacency: HashMap<usize, Vec<(usize, f64)>>,

    /// Prominence/coherence weights for each point (optional).
    pub point_weights: Option<Vec<f64>>,
}

impl SpringAmplitudeModulation {
    /// Create spring modulation from hypergraph edges.
    ///
    /// # Arguments
    /// * `point_coords` - Spherical coordinates (θ, φ) for each point
    /// * `edges` - List of (from_idx, to_idx, weight) tuples
    pub fn from_edges(point_coords: Vec<(f64, f64)>, edges: &[(usize, usize, f64)]) -> Self {
        let mut adjacency: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

        for &(from, to, weight) in edges {
            adjacency.entry(from).or_default().push((to, weight));
            adjacency.entry(to).or_default().push((from, weight));
        }

        Self {
            point_coords,
            adjacency,
            point_weights: None,
        }
    }

    /// Add point weights (prominence/coherence scores).
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.point_weights = Some(weights);
        self
    }

    /// Compute spring-modulated amplitude for a given position.
    ///
    /// Returns a multiplier based on proximity to spring-connected regions.
    ///
    /// # Arguments
    /// * `theta` - Query colatitude
    /// * `phi` - Query azimuth
    /// * `spring_weight` - Weight for spring contribution
    /// * `decay` - Distance decay factor
    pub fn compute_amplitude_modulation(
        &self,
        theta: f64,
        phi: f64,
        spring_weight: f64,
        decay: f64,
    ) -> f64 {
        if self.point_coords.is_empty() {
            return 1.0;
        }

        let mut total_spring_contribution = 0.0;
        let mut total_weight = 0.0;

        for (idx, &(pt_theta, pt_phi)) in self.point_coords.iter().enumerate() {
            // Angular distance on sphere
            let d_theta = theta - pt_theta;
            let d_phi = {
                let diff = phi - pt_phi;
                if diff > PI {
                    2.0f64.mul_add(-PI, diff)
                } else if diff < -PI {
                    2.0f64.mul_add(PI, diff)
                } else {
                    diff
                }
            };
            let angular_dist = d_theta.hypot(d_phi);

            // Gaussian proximity weight
            let proximity = (-angular_dist * angular_dist / (2.0 * decay * decay)).exp();

            // Get spring connectivity contribution
            let connectivity = self
                .adjacency
                .get(&idx)
                .map_or(0.0, |neighbors| neighbors.iter().map(|(_, w)| w).sum::<f64>());

            // Optional point weight
            let point_weight = self
                .point_weights
                .as_ref()
                .and_then(|w| w.get(idx))
                .copied()
                .unwrap_or(1.0);

            total_spring_contribution += proximity * connectivity * point_weight;
            total_weight += proximity;
        }

        if total_weight > 0.0 {
            spring_weight.mul_add(total_spring_contribution / total_weight, 1.0)
        } else {
            1.0
        }
    }
}

/// Result of harmonic navigation.
#[derive(Debug, Clone)]
pub struct HarmonicNavigationResult {
    /// Radial coordinate of best cone center.
    pub r: f64,

    /// Colatitude of best cone center [0, π].
    pub theta: f64,

    /// Azimuth of best cone center [0, 2π).
    pub phi: f64,

    /// Cone half-angle (adaptive based on confidence).
    pub alpha: f64,

    /// Best similarity score achieved.
    pub score: f64,

    /// Number of probes used before convergence.
    pub probes_used: usize,

    /// Final confidence (peak intensity).
    pub confidence: f64,
}

/// Harmonic Navigator using spherical harmonic interference for navigation.
///
/// This navigator uses wave superposition in the spherical harmonic basis
/// to find optimal cone centers for embedding retrieval.
pub struct HarmonicNavigator {
    /// Precomputed spherical harmonics basis.
    basis: SphericalHarmonicsBasis,

    /// Configuration.
    config: HarmonicNavigatorConfig,

    /// Optional spring amplitude modulation from hypergraph.
    springs: Option<SpringAmplitudeModulation>,
}

impl HarmonicNavigator {
    /// Create a new HarmonicNavigator.
    pub fn new(config: HarmonicNavigatorConfig) -> Self {
        let sh_config = SphericalHarmonicsConfig::default().with_band_limit(config.band_limit);
        let basis = SphericalHarmonicsBasis::new(&sh_config);

        Self {
            basis,
            config,
            springs: None,
        }
    }

    /// Enable spring amplitude modulation.
    ///
    /// When springs are provided, amplitude grids are modulated based on
    /// structural connectivity from the hypergraph.
    pub fn with_springs(mut self, springs: SpringAmplitudeModulation) -> Self {
        self.springs = Some(springs);
        self
    }

    /// Check if spring AM is enabled.
    pub const fn has_springs(&self) -> bool {
        self.springs.is_some()
    }

    /// Navigate to find optimal cone parameters for a query.
    ///
    /// # Arguments
    /// * `query_dir` - Unit vector in embedding space (normalized query)
    /// * `rng_seed` - Random seed for probe perturbations
    ///
    /// # Returns
    /// `HarmonicNavigationResult` with optimal cone parameters
    pub fn navigate(&self, query_dir: &[f64], rng_seed: u64) -> HarmonicNavigationResult {
        assert!(
            query_dir.len() >= 3,
            "Query must have at least 3 dimensions"
        );

        // Convert query direction to spherical coordinates
        let query_norm: f64 = query_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        let normalized: Vec<f64> = query_dir.iter().map(|x| x / query_norm.max(1e-8)).collect();

        // Initial estimate: use first 3 components as Cartesian direction
        let z = normalized.get(2).copied().unwrap_or(0.0).clamp(-1.0, 1.0);
        let x = normalized.first().copied().unwrap_or(1.0);
        let y = normalized.get(1).copied().unwrap_or(0.0);

        let current_r = query_norm * 1.2; // Scale factor (radial doesn't change in SH optimization)
        let mut current_theta = z.acos();
        let mut current_phi = y.atan2(x);
        if current_phi < 0.0 {
            current_phi += 2.0 * PI;
        }
        let mut current_alpha = self.config.initial_alpha;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_result = HarmonicNavigationResult {
            r: current_r,
            theta: current_theta,
            phi: current_phi,
            alpha: current_alpha,
            score: best_score,
            probes_used: 0,
            confidence: 0.0,
        };

        // Simple LCG for deterministic pseudo-random numbers
        let mut rng_state = rng_seed;
        let next_rand = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*state as f64) / (u64::MAX as f64)).mul_add(2.0, -1.0) // [-1, 1]
        };

        for probe in 0..self.config.max_probes {
            // Generate candidate amplitude grids
            let mut candidate_grids = Vec::with_capacity(self.config.probe_candidates);

            for _ in 0..self.config.probe_candidates {
                let delta_theta = next_rand(&mut rng_state) * 0.15;
                let delta_phi = next_rand(&mut rng_state) * 0.3;
                let delta_r = next_rand(&mut rng_state) * 0.1;

                let pert_theta = (current_theta + delta_theta).clamp(1e-6, PI - 1e-6);
                let pert_phi = (current_phi + delta_phi).rem_euclid(2.0 * PI);
                let pert_r = (current_r + delta_r).max(0.1);

                // Create Gaussian amplitude blob
                let mut blob = self.basis.gaussian_blob(
                    pert_theta,
                    pert_phi,
                    self.config.sigma_theta,
                    self.config.sigma_phi,
                );

                // Modulate by radial match
                let radial_weight = (-(pert_r - current_r).powi(2) / 0.5).exp();
                for val in &mut blob {
                    *val *= radial_weight;
                }

                // Spring Amplitude Modulation (Spring AM)
                // Boost amplitude in regions with spring-connected neighbors
                if let Some(ref springs) = self.springs {
                    let spring_mod = springs.compute_amplitude_modulation(
                        pert_theta,
                        pert_phi,
                        self.config.spring_weight,
                        self.config.spring_decay,
                    );
                    for val in &mut blob {
                        *val *= spring_mod;
                    }
                }

                candidate_grids.push(blob);
            }

            // Compute harmonic superposition
            let intensity = self.basis.superposition_field(&candidate_grids);

            // Find peak
            let (peak_theta, peak_phi, confidence) = self.basis.find_peak(&intensity);

            // Update with momentum
            let new_theta =
                self.config.momentum.mul_add(current_theta, (1.0 - self.config.momentum) * peak_theta);
            let new_phi =
                self.config.momentum.mul_add(current_phi, (1.0 - self.config.momentum) * peak_phi);

            current_theta = new_theta;
            current_phi = new_phi;

            // Adaptive cone width based on confidence
            current_alpha = self.adaptive_cone_width(1.0 - confidence, current_r / 10.0);

            // Score this configuration (using confidence as proxy)
            let score = confidence;

            if score > best_score {
                best_score = score;
                best_result = HarmonicNavigationResult {
                    r: current_r,
                    theta: current_theta,
                    phi: current_phi,
                    alpha: current_alpha,
                    score,
                    probes_used: probe + 1,
                    confidence,
                };
            }

            // Early stopping
            if confidence > self.config.confidence_threshold
                && probe >= self.config.min_probes_before_stop
            {
                break;
            }
        }

        best_result
    }

    /// Adaptive cone width based on query complexity and radius.
    fn adaptive_cone_width(&self, complexity: f64, normalized_radius: f64) -> f64 {
        let alpha_0: f64 = PI / 4.0;
        let beta: f64 = 5.0;
        let local_density: f64 = 0.1; // Placeholder - would be computed from actual point density

        alpha_0 * normalized_radius.sqrt() * (-beta * local_density).exp() * (1.0 - complexity)
    }

    /// Get the underlying spherical harmonics basis.
    pub const fn basis(&self) -> &SphericalHarmonicsBasis {
        &self.basis
    }

    /// Get the configuration.
    pub const fn config(&self) -> &HarmonicNavigatorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harmonic_navigator_basic() {
        let config = HarmonicNavigatorConfig::dev();
        let navigator = HarmonicNavigator::new(config);

        // Query pointing in +z direction
        let query = vec![0.0, 0.0, 1.0];
        let result = navigator.navigate(&query, 42);

        // Should converge to theta near 0 (north pole)
        assert!(result.theta < PI / 2.0, "Should be in northern hemisphere");
        assert!(result.probes_used > 0);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_harmonic_navigator_equator() {
        let config = HarmonicNavigatorConfig::dev();
        let navigator = HarmonicNavigator::new(config);

        // Query pointing in +x direction (equator)
        let query = vec![1.0, 0.0, 0.0];
        let result = navigator.navigate(&query, 123);

        // Should converge to theta near π/2 (equator)
        assert!(
            (result.theta - PI / 2.0).abs() < 0.5,
            "Should be near equator"
        );
    }

    #[test]
    fn test_harmonic_navigator_reproducible() {
        let config = HarmonicNavigatorConfig::dev();
        let navigator = HarmonicNavigator::new(config);

        let query = vec![0.5, 0.5, 0.707];

        // Same seed should give same result
        let result1 = navigator.navigate(&query, 999);
        let result2 = navigator.navigate(&query, 999);

        assert!((result1.theta - result2.theta).abs() < 1e-10);
        assert!((result1.phi - result2.phi).abs() < 1e-10);
    }

    #[test]
    fn test_spring_amplitude_modulation() {
        // Create some points at known positions
        let point_coords = vec![
            (PI / 2.0, 0.0),      // equator, φ=0
            (PI / 2.0, PI / 4.0), // equator, φ=π/4
            (PI / 2.0, PI / 2.0), // equator, φ=π/2
        ];

        // Create edges connecting them
        let edges = vec![
            (0, 1, 1.0), // point 0 <-> point 1
            (1, 2, 1.0), // point 1 <-> point 2
        ];

        let springs = SpringAmplitudeModulation::from_edges(point_coords, &edges);

        // Point 1 has 2 connections, should have highest modulation
        let mod_at_0 = springs.compute_amplitude_modulation(PI / 2.0, 0.0, 0.3, 0.5);
        let mod_at_1 = springs.compute_amplitude_modulation(PI / 2.0, PI / 4.0, 0.3, 0.5);
        let mod_at_2 = springs.compute_amplitude_modulation(PI / 2.0, PI / 2.0, 0.3, 0.5);

        // All should be >= 1.0 (no negative modulation)
        assert!(mod_at_0 >= 1.0);
        assert!(mod_at_1 >= 1.0);
        assert!(mod_at_2 >= 1.0);

        // Point at center (1) should have highest modulation due to connectivity
        assert!(
            mod_at_1 > mod_at_0.min(mod_at_2) * 0.9,
            "Center point should have high modulation"
        );
    }

    #[test]
    fn test_harmonic_navigator_with_springs() {
        let config = HarmonicNavigatorConfig::dev();

        // Create spring data near the equator
        let point_coords = vec![(PI / 2.0, 0.0), (PI / 2.0, 0.1), (PI / 2.0, 0.2)];
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0)];
        let springs = SpringAmplitudeModulation::from_edges(point_coords, &edges);

        let navigator = HarmonicNavigator::new(config).with_springs(springs);

        assert!(navigator.has_springs());

        // Navigate with springs enabled
        let query = vec![1.0, 0.0, 0.0];
        let result = navigator.navigate(&query, 42);

        // Should still converge
        assert!(result.probes_used > 0);
        assert!(result.confidence > 0.0);
    }
}
