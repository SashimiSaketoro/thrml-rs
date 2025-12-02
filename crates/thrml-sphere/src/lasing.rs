//! Lasing dynamics for coherence-driven radius evolution.
//!
//! This module implements coherence-driven "lasing" dynamics where
//! high-coherence patches amplify (grow toward the core) while
//! low-coherence patches decay (move toward the periphery).
//!
//! This provides an alternative to pure water-filling optimization,
//! allowing coherent information to naturally concentrate at the sphere core.

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::SphericalCoords;

/// Lasing dynamics for coherence-driven radius evolution.
///
/// In the lasing model:
/// - Nodes with coherence > baseline experience "gain" (radii shrink toward core)
/// - Nodes with coherence < baseline experience "loss" (radii grow toward periphery)
///
/// The dynamics follow:
/// ```text
/// dr/dt = -β * (coherence - baseline) * r
/// ```
///
/// This creates a self-organizing system where coherent information
/// naturally concentrates at the sphere core.
pub struct LasingDynamics {
    /// Gain coefficient (β) - controls rate of radius change
    pub beta: f32,
    /// Coherence baseline - threshold for amplification vs decay
    pub baseline: f32,
    /// Minimum allowed radius
    pub min_radius: f32,
    /// Maximum allowed radius
    pub max_radius: f32,
    /// Number of lasing steps per iteration
    pub steps: usize,
    /// Step size (dt)
    pub step_size: f32,
}

impl LasingDynamics {
    /// Create new lasing dynamics with default parameters.
    pub const fn new(min_radius: f32, max_radius: f32) -> Self {
        Self {
            beta: 0.1,
            baseline: 0.5,
            min_radius,
            max_radius,
            steps: 50,
            step_size: 0.02,
        }
    }

    /// Set gain coefficient.
    pub const fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Set coherence baseline.
    pub const fn with_baseline(mut self, baseline: f32) -> Self {
        self.baseline = baseline;
        self
    }

    /// Set number of steps.
    pub const fn with_steps(mut self, steps: usize) -> Self {
        self.steps = steps;
        self
    }

    /// Set step size.
    pub const fn with_step_size(mut self, step_size: f32) -> Self {
        self.step_size = step_size;
        self
    }

    /// Compute growth rates for each node.
    ///
    /// Growth rate = β * (coherence - baseline)
    /// - Positive for coherence > baseline (shrink radius)
    /// - Negative for coherence < baseline (grow radius)
    pub fn growth_rates(&self, coherence: &Tensor<WgpuBackend, 1>) -> Tensor<WgpuBackend, 1> {
        (coherence.clone() - self.baseline).mul_scalar(self.beta)
    }

    /// Perform one lasing step.
    ///
    /// Updates radii according to:
    /// r_new = r * exp(-growth_rate * dt)
    ///
    /// Then clamps to [min_radius, max_radius].
    pub fn step(
        &self,
        coords: SphericalCoords,
        coherence: &Tensor<WgpuBackend, 1>,
    ) -> SphericalCoords {
        let growth_rates = self.growth_rates(coherence);

        // r_new = r * exp(-growth_rate * dt)
        // Using exp(-x) so positive growth rate -> smaller radius
        let scale = (-growth_rates.mul_scalar(self.step_size)).exp();
        let new_r = coords.r.clone() * scale;

        // Clamp radii
        let clamped_r = new_r.clamp(self.min_radius, self.max_radius);

        SphericalCoords::new(clamped_r, coords.theta, coords.phi)
    }

    /// Run full lasing dynamics.
    ///
    /// Evolves radii over multiple steps based on coherence values.
    pub fn run(
        &self,
        init_coords: SphericalCoords,
        coherence: &Tensor<WgpuBackend, 1>,
    ) -> SphericalCoords {
        let mut coords = init_coords;

        for _ in 0..self.steps {
            coords = self.step(coords, coherence);
        }

        coords
    }

    /// Run with callback for monitoring.
    pub fn run_with_callback<F>(
        &self,
        init_coords: SphericalCoords,
        coherence: &Tensor<WgpuBackend, 1>,
        mut callback: F,
        callback_interval: usize,
    ) -> SphericalCoords
    where
        F: FnMut(usize, &SphericalCoords),
    {
        let mut coords = init_coords;

        for step in 0..self.steps {
            coords = self.step(coords, coherence);

            if callback_interval > 0 && step % callback_interval == 0 {
                callback(step, &coords);
            }
        }

        coords
    }
}

/// Combined lasing + Langevin dynamics.
///
/// Alternates between:
/// 1. Lasing steps (coherence-driven radius evolution)
/// 2. Langevin steps (similarity-driven position optimization)
///
/// This allows coherent patches to both concentrate at the core
/// AND cluster with similar neighbors.
pub struct LasingLangevinOptimizer {
    /// Lasing dynamics for radial evolution
    pub lasing: LasingDynamics,
    /// Number of Langevin steps per lasing step
    pub langevin_steps_per_lasing: usize,
    /// Total outer iterations
    pub outer_iterations: usize,
}

impl LasingLangevinOptimizer {
    /// Create new combined optimizer.
    pub const fn new(
        lasing: LasingDynamics,
        langevin_steps_per_lasing: usize,
        outer_iterations: usize,
    ) -> Self {
        Self {
            lasing,
            langevin_steps_per_lasing,
            outer_iterations,
        }
    }

    /// Run combined optimization.
    ///
    /// # Arguments
    /// * `init_coords` - Initial spherical coordinates
    /// * `coherence` - Coherence scores per node
    /// * `hamiltonian` - Hamiltonian for Langevin dynamics
    /// * `langevin` - Langevin sampler configuration
    /// * `key` - RNG key
    /// * `device` - GPU device
    pub fn run<H: crate::hamiltonian::SphereHamiltonian>(
        &self,
        init_coords: SphericalCoords,
        coherence: &Tensor<WgpuBackend, 1>,
        hamiltonian: &H,
        langevin: &crate::langevin::LangevinSampler,
        key: thrml_samplers::RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> SphericalCoords {
        let mut coords = init_coords;
        let keys = key.split(self.outer_iterations);

        for iter_key in keys.into_iter() {
            // 1. Lasing step (radius evolution)
            coords = self.lasing.step(coords, coherence);

            // 2. Langevin steps (position optimization with fixed radii)
            let step_keys = iter_key.split(self.langevin_steps_per_lasing);
            for step_key in step_keys {
                coords = langevin.step(hamiltonian, &coords, step_key, device);
            }

            // Clamp radii after Langevin (may have drifted)
            coords = coords.clamp_radii(self.lasing.min_radius, self.lasing.max_radius);
        }

        coords
    }
}

/// Equilibrium finder using coherence statistics.
///
/// Estimates the optimal coherence baseline such that the sphere
/// is roughly half-filled (balanced amplification/decay).
pub fn estimate_baseline(coherence: &Tensor<WgpuBackend, 1>) -> f32 {
    // Use median coherence as baseline
    let coh_data: Vec<f32> = coherence.clone().into_data().to_vec().expect("coh to vec");
    let mut sorted = coh_data;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    if n == 0 {
        return 0.5;
    }

    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_lasing_amplifies_high_coherence() {
        let device = init_gpu_device();
        let n = 10;

        // Create coords all at mid-radius
        let mid_radius = 50.0;
        let r = Tensor::from_data([mid_radius; 10].as_slice(), &device);
        let theta = Tensor::random([n], Distribution::Uniform(0.5, 2.5), &device);
        let phi = Tensor::random(
            [n],
            Distribution::Uniform(0.0, std::f64::consts::TAU),
            &device,
        );
        let init_coords = SphericalCoords::new(r, theta, phi);

        // Half high coherence, half low coherence
        let mut coh_data = vec![0.8f32; 5];
        coh_data.extend(vec![0.2f32; 5]);
        let coherence = Tensor::from_data(coh_data.as_slice(), &device);

        let lasing = LasingDynamics::new(10.0, 100.0)
            .with_baseline(0.5)
            .with_beta(0.5)
            .with_steps(20);

        let final_coords = lasing.run(init_coords, &coherence);
        let final_r: Vec<f32> = final_coords.r.into_data().to_vec().expect("r to vec");

        // High coherence nodes should have smaller radii
        let avg_high: f32 = final_r[0..5].iter().sum::<f32>() / 5.0;
        let avg_low: f32 = final_r[5..10].iter().sum::<f32>() / 5.0;

        assert!(
            avg_high < avg_low,
            "High coherence nodes should have smaller radii: {} vs {}",
            avg_high,
            avg_low
        );
    }

    #[test]
    fn test_estimate_baseline() {
        let device = init_gpu_device();

        let coherence = Tensor::from_data([0.1f32, 0.2, 0.3, 0.4, 0.5].as_slice(), &device);
        let baseline = estimate_baseline(&coherence);

        assert!(
            (baseline - 0.3).abs() < 0.01,
            "Baseline should be median: {}",
            baseline
        );
    }
}
