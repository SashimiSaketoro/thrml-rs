//! Langevin dynamics sampler for sphere optimization.
//!
//! This module provides sphere-specific Langevin samplers that work with
//! `SphericalCoords` and `SphereHamiltonian`.
//!
//! For general Langevin utilities, see `thrml_samplers::langevin`.

use crate::config::SphereConfig;
use crate::hamiltonian::SphereHamiltonian;
use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use thrml_core::SphericalCoords;
use thrml_samplers::RngKey;

// Re-export general Langevin utilities
pub use thrml_samplers::langevin::{
    langevin_step_1d, langevin_step_2d, AnnealingSchedule, LangevinConfig,
};

/// Sphere-specific Langevin dynamics sampler.
///
/// Works with `SphericalCoords` and `SphereHamiltonian` to optimize
/// positions on a hypersphere.
pub struct SphereLangevinSampler {
    /// Step size (dt) for each iteration
    pub step_size: f32,
    /// Temperature controlling noise magnitude
    pub temperature: f32,
    /// Total number of steps to run
    pub n_steps: usize,
    /// Optional gradient clipping threshold
    pub gradient_clip: Option<f32>,
}

impl SphereLangevinSampler {
    /// Create a new sphere Langevin sampler.
    pub const fn new(step_size: f32, temperature: f32, n_steps: usize) -> Self {
        Self {
            step_size,
            temperature,
            n_steps,
            gradient_clip: None,
        }
    }

    /// Create from a SphereConfig.
    pub const fn from_config(config: &SphereConfig) -> Self {
        Self::new(config.step_size, config.temperature, config.n_steps)
    }

    /// Set gradient clipping threshold.
    pub const fn with_gradient_clip(mut self, max_grad: f32) -> Self {
        self.gradient_clip = Some(max_grad);
        self
    }

    /// Perform one Langevin step on spherical coordinates.
    pub fn step<H: SphereHamiltonian>(
        &self,
        hamiltonian: &H,
        coords: &SphericalCoords,
        _key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> SphericalCoords {
        let n = coords.len();

        // Get current Cartesian positions
        let positions = coords.to_cartesian();

        // Compute forces
        let gravity_force = hamiltonian.gravity_force(coords);
        let lateral_force = hamiltonian.lateral_force_cartesian(coords);

        // Project gravity force to Cartesian (radial direction)
        let sin_theta = coords.theta.clone().sin();
        let cos_theta = coords.theta.clone().cos();
        let sin_phi = coords.phi.clone().sin();
        let cos_phi = coords.phi.clone().cos();

        let r_hat_x = sin_theta.clone() * cos_phi;
        let r_hat_y = sin_theta * sin_phi;
        let r_hat_z = cos_theta;

        let gf = gravity_force.unsqueeze_dim::<2>(1);
        let r_hat = Tensor::stack(vec![r_hat_x, r_hat_y, r_hat_z], 1);
        let gravity_cart = r_hat * gf;

        // Total force
        let mut total_force = gravity_cart + lateral_force;

        // Apply gradient clipping if enabled
        if let Some(max_grad) = self.gradient_clip {
            let force_sq = total_force.clone().powf_scalar(2.0);
            let force_mag_sq = force_sq.sum_dim(1);
            let force_mag = force_mag_sq.sqrt().clamp(1e-8, f32::MAX);
            let scale = (force_mag.recip().mul_scalar(max_grad)).clamp(0.0, 1.0);
            total_force = total_force * scale;
        }

        // Drift term
        let drift = total_force.mul_scalar(self.step_size);

        // Diffusion term
        let noise_scale = (2.0 * self.temperature * self.step_size).sqrt();
        let noise: Tensor<WgpuBackend, 2> =
            Tensor::random([n, 3], Distribution::Normal(0.0, 1.0), device);
        let diffusion = noise.mul_scalar(noise_scale);

        // Update positions
        let new_positions = positions + drift + diffusion;

        SphericalCoords::from_cartesian(new_positions)
    }

    /// Run full Langevin dynamics.
    pub fn run<H: SphereHamiltonian>(
        &self,
        hamiltonian: &H,
        init_coords: SphericalCoords,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> SphericalCoords {
        let keys = key.split(self.n_steps);
        let mut coords = init_coords;

        for step_key in keys {
            coords = self.step(hamiltonian, &coords, step_key, device);
        }

        coords
    }

    /// Run with optional progress callback.
    pub fn run_with_callback<H: SphereHamiltonian, F>(
        &self,
        hamiltonian: &H,
        init_coords: SphericalCoords,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
        mut callback: F,
        callback_interval: usize,
    ) -> SphericalCoords
    where
        F: FnMut(usize, &SphericalCoords, &Tensor<WgpuBackend, 1>),
    {
        let keys = key.split(self.n_steps);
        let mut coords = init_coords;

        for (step, step_key) in keys.into_iter().enumerate() {
            coords = self.step(hamiltonian, &coords, step_key, device);

            if callback_interval > 0 && step % callback_interval == 0 {
                let energy = hamiltonian.total_energy(&coords);
                callback(step, &coords, &energy);
            }
        }

        coords
    }
}

/// Sphere Langevin sampler with temperature annealing.
pub struct AnnealingSphereLangevinSampler {
    /// Base sampler
    pub sampler: SphereLangevinSampler,
    /// Annealing schedule
    pub schedule: AnnealingSchedule,
}

impl AnnealingSphereLangevinSampler {
    /// Create with linear annealing.
    pub const fn linear(step_size: f32, temp_init: f32, temp_final: f32, n_steps: usize) -> Self {
        Self {
            sampler: SphereLangevinSampler::new(step_size, temp_init, n_steps),
            schedule: AnnealingSchedule::Linear {
                start: temp_init,
                end: temp_final,
            },
        }
    }

    /// Run with annealing.
    pub fn run<H: SphereHamiltonian>(
        &self,
        hamiltonian: &H,
        init_coords: SphericalCoords,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> SphericalCoords {
        let keys = key.split(self.sampler.n_steps);
        let mut coords = init_coords;
        let n_steps_f = self.sampler.n_steps as f32;

        for (step, step_key) in keys.into_iter().enumerate() {
            let t = step as f32 / n_steps_f;
            let temperature = self.schedule.temperature(t);

            let step_sampler = SphereLangevinSampler {
                step_size: self.sampler.step_size,
                temperature,
                n_steps: 1,
                gradient_clip: self.sampler.gradient_clip,
            };
            coords = step_sampler.step(hamiltonian, &coords, step_key, device);
        }

        coords
    }
}

// Keep backwards compatibility aliases
pub type LangevinSampler = SphereLangevinSampler;
pub type AnnealingLangevinSampler = AnnealingSphereLangevinSampler;

// ============================================================================
// Polar Boundary Constraints (Phase 3)
// ============================================================================

// Re-export ZoneEnergyConfig from roots
pub use crate::roots::PartitionZone;
pub use crate::roots::ZoneEnergyConfig;

/// Zone targeting configuration for forced placement.
///
/// When set, embeddings are pulled toward the target zone during optimization.
/// The translucency controls how much cross-zone similarity can influence placement.
#[derive(Clone, Copy, Debug)]
pub struct ZoneTargeting {
    /// Target zone for this ingest
    pub target: PartitionZone,
    /// Attraction strength toward target zone (default: 2.0)
    pub attraction_strength: f32,
    /// Translucency: how much cross-zone similarity matters (0.0 = opaque, 1.0 = transparent)
    pub translucency: f32,
}

impl ZoneTargeting {
    /// Create targeting for instruction zone (north pole).
    pub const fn instruction() -> Self {
        Self {
            target: PartitionZone::Instruction,
            attraction_strength: 2.0,
            translucency: 0.3,
        }
    }

    /// Create targeting for content zone (torus).
    pub const fn content() -> Self {
        Self {
            target: PartitionZone::Content,
            attraction_strength: 2.0,
            translucency: 0.3,
        }
    }

    /// Create targeting for QA pairs zone (south pole).
    pub const fn qa_pairs() -> Self {
        Self {
            target: PartitionZone::QAPairs,
            attraction_strength: 2.0,
            translucency: 0.3,
        }
    }

    /// Parse from string (for CLI).
    pub fn from_str(s: &str, translucency: f32) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "instruction" | "north" => Some(Self {
                target: PartitionZone::Instruction,
                attraction_strength: 2.0,
                translucency,
            }),
            "content" | "torus" => Some(Self {
                target: PartitionZone::Content,
                attraction_strength: 2.0,
                translucency,
            }),
            "qa" | "qa_pairs" | "south" => Some(Self {
                target: PartitionZone::QAPairs,
                attraction_strength: 2.0,
                translucency,
            }),
            _ => None,
        }
    }

    /// Set attraction strength.
    pub const fn with_strength(mut self, strength: f32) -> Self {
        self.attraction_strength = strength;
        self
    }

    /// Set translucency.
    pub const fn with_translucency(mut self, translucency: f32) -> Self {
        self.translucency = translucency.clamp(0.0, 1.0);
        self
    }

    /// Get target θ range for this zone.
    /// Returns (θ_min, θ_max) in radians.
    pub fn target_theta_range(&self, pole_angle: f32) -> (f32, f32) {
        use std::f32::consts::PI;
        match self.target {
            PartitionZone::Instruction => (0.0, pole_angle),
            PartitionZone::Content => (pole_angle, PI - pole_angle),
            PartitionZone::QAPairs => (PI - pole_angle, PI),
        }
    }

    /// Get target θ center for this zone.
    pub fn target_theta_center(&self, pole_angle: f32) -> f32 {
        use std::f32::consts::PI;
        match self.target {
            PartitionZone::Instruction => pole_angle / 2.0, // Center of north cap
            PartitionZone::Content => PI / 2.0,             // Equator
            PartitionZone::QAPairs => PI - pole_angle / 2.0, // Center of south cap
        }
    }
}

/// Configuration for polar boundary constraints.
///
/// These constraints ensure content embeddings stay within the equatorial torus
/// and don't drift into the polar instruction/QA zones during optimization.
///
/// ## Zone Architecture
/// ```text
///      N (θ < 15°)    ← INSTRUCTION zone (behavioral anchors)
///      │
///  ════════════════   ← CONTENT zone (knowledge torus)
///      │
///      S (θ > 165°)   ← QA_PAIRS zone (fine-tuning examples)
/// ```
#[derive(Clone, Debug)]
pub struct PolarConstraintConfig {
    /// Exclusion angle from poles (radians). Default: π/12 (15°)
    /// Content is clamped to: pole_angle <= θ <= π - pole_angle
    pub pole_angle: f32,
    /// Whether to apply soft (gradient-based) or hard (clamp) constraint
    pub soft_constraint: bool,
    /// Repulsion strength for soft constraint (force away from poles)
    pub repulsion_strength: f32,
    /// Zone-specific energy weights for regional energy loss
    pub zone_energy: ZoneEnergyConfig,
    /// Optional zone targeting for forced placement
    pub zone_targeting: Option<ZoneTargeting>,
}

impl Default for PolarConstraintConfig {
    fn default() -> Self {
        Self {
            pole_angle: std::f32::consts::PI / 12.0, // 15°
            soft_constraint: false,                  // Default to hard clamp
            repulsion_strength: 1.0,
            zone_energy: ZoneEnergyConfig::default(),
            zone_targeting: None, // No forced targeting by default
        }
    }
}

impl PolarConstraintConfig {
    /// Create with custom pole angle (in radians).
    pub fn with_angle(angle: f32) -> Self {
        Self {
            pole_angle: angle,
            ..Default::default()
        }
    }

    /// Create with custom pole angle in degrees.
    pub fn with_angle_degrees(degrees: f32) -> Self {
        Self::with_angle(degrees.to_radians())
    }

    /// Use soft constraint (gradient repulsion from poles).
    pub const fn soft(mut self, strength: f32) -> Self {
        self.soft_constraint = true;
        self.repulsion_strength = strength;
        self
    }

    /// Set zone energy configuration.
    pub const fn with_zone_energy(mut self, config: ZoneEnergyConfig) -> Self {
        self.zone_energy = config;
        self
    }

    /// Set instruction zone energy weight.
    pub const fn with_instruction_energy(mut self, weight: f32) -> Self {
        self.zone_energy.instruction_weight = weight;
        self
    }

    /// Set QA pairs zone energy weight.
    pub const fn with_qa_energy(mut self, weight: f32) -> Self {
        self.zone_energy.qa_pairs_weight = weight;
        self
    }

    /// Set zone targeting for forced placement.
    pub const fn with_zone_targeting(mut self, targeting: ZoneTargeting) -> Self {
        self.zone_targeting = Some(targeting);
        self
    }

    /// Set zone targeting from string (for CLI).
    pub fn with_zone_targeting_str(mut self, zone: &str, translucency: f32) -> Self {
        if let Some(targeting) = ZoneTargeting::from_str(zone, translucency) {
            self.zone_targeting = Some(targeting);
        }
        self
    }
}

/// Clamps theta values to stay within the content torus (hard constraint).
///
/// Points with θ < pole_angle are pushed to θ = pole_angle.
/// Points with θ > π - pole_angle are pushed to θ = π - pole_angle.
pub fn clamp_to_content_torus(coords: SphericalCoords, pole_angle: f32) -> SphericalCoords {
    use std::f32::consts::PI;

    let theta_min = pole_angle;
    let theta_max = PI - pole_angle;

    let theta_clamped = coords.theta.clamp(theta_min, theta_max);

    SphericalCoords {
        r: coords.r,
        theta: theta_clamped,
        phi: coords.phi,
    }
}

/// Clamps theta values to stay within a target zone (for forced placement).
///
/// When zone targeting is set, points are clamped to the target zone's θ range.
/// Without targeting, defaults to content torus clamping.
pub fn clamp_to_zone(coords: SphericalCoords, config: &PolarConstraintConfig) -> SphericalCoords {
    match &config.zone_targeting {
        Some(targeting) => {
            let (theta_min, theta_max) = targeting.target_theta_range(config.pole_angle);
            let theta_clamped = coords.theta.clamp(theta_min, theta_max);
            SphericalCoords {
                r: coords.r,
                theta: theta_clamped,
                phi: coords.phi,
            }
        }
        None => clamp_to_content_torus(coords, config.pole_angle),
    }
}

/// Computes a repulsion force away from poles (soft constraint).
///
/// Returns a force tensor `[N]` in the theta direction:
/// - Positive force (toward equator) when near north pole
/// - Negative force (toward equator) when near south pole
/// - Zero force when in content zone
pub fn polar_repulsion_force(
    coords: &SphericalCoords,
    config: &PolarConstraintConfig,
) -> Tensor<WgpuBackend, 1> {
    use std::f32::consts::PI;

    let theta = &coords.theta;
    let _n = coords.len();
    let _device = theta.device();

    let theta_min = config.pole_angle;
    let theta_max = PI - config.pole_angle;

    // Distance into north pole zone (positive if in zone)
    let north_penetration = (theta_min - theta.clone()).clamp(0.0, theta_min);

    // Distance into south pole zone (positive if in zone)
    let south_penetration = (theta.clone() - theta_max).clamp(0.0, theta_min);

    // Force: positive pushes toward equator (larger θ from north, smaller θ from south)
    let north_force = north_penetration * config.repulsion_strength;
    let south_force = south_penetration * (-config.repulsion_strength);

    north_force + south_force
}

/// Sphere Langevin sampler with polar boundary constraints.
///
/// This sampler applies constraints to keep content embeddings within
/// the equatorial torus, reserving the poles for instruction embeddings.
pub struct PolarConstrainedLangevinSampler {
    /// Base sampler
    pub sampler: SphereLangevinSampler,
    /// Polar constraint configuration
    pub polar_config: PolarConstraintConfig,
}

impl PolarConstrainedLangevinSampler {
    /// Create a new polar-constrained sampler.
    pub const fn new(sampler: SphereLangevinSampler, polar_config: PolarConstraintConfig) -> Self {
        Self {
            sampler,
            polar_config,
        }
    }

    /// Create from SphereConfig with default polar constraints.
    pub fn from_config(config: &SphereConfig) -> Self {
        Self {
            sampler: SphereLangevinSampler::from_config(config),
            polar_config: PolarConstraintConfig::default(),
        }
    }

    /// Set zone targeting.
    pub const fn with_zone_targeting(mut self, targeting: ZoneTargeting) -> Self {
        self.polar_config.zone_targeting = Some(targeting);
        self
    }

    /// Run with polar constraints (respects zone targeting if set).
    pub fn run<H: SphereHamiltonian>(
        &self,
        hamiltonian: &H,
        init_coords: SphericalCoords,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> SphericalCoords {
        let keys = key.split(self.sampler.n_steps);
        let mut coords = init_coords;

        // Apply initial clamp (respects zone targeting)
        coords = clamp_to_zone(coords, &self.polar_config);

        for step_key in keys {
            coords = self.sampler.step(hamiltonian, &coords, step_key, device);

            // Apply polar constraint after each step
            if !self.polar_config.soft_constraint {
                coords = clamp_to_zone(coords, &self.polar_config);
            }
            // Note: soft constraint would be applied as additional force in step()
        }

        coords
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::WaterFillingHamiltonian;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_langevin_converges() {
        let device = init_gpu_device();
        let n = 10;

        let ideal_radii: Tensor<WgpuBackend, 1> = Tensor::from_data([50.0; 10].as_slice(), &device);

        let init_radii: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(30.0, 70.0), &device);
        let init_coords = SphericalCoords::init_random(n, init_radii, &device);

        let similarity: Tensor<WgpuBackend, 2> = Tensor::zeros([n, n], &device);
        let hamiltonian = WaterFillingHamiltonian::new(ideal_radii.clone(), similarity, 1.0);

        let sampler = SphereLangevinSampler::new(0.5, 0.0, 50);
        let key = RngKey::new(42);
        let final_coords = sampler.run(&hamiltonian, init_coords, key, &device);

        let final_r: Vec<f32> = final_coords.r.into_data().to_vec().expect("r to vec");
        let ideal_r: Vec<f32> = ideal_radii.into_data().to_vec().expect("ideal to vec");

        for (r, ideal) in final_r.iter().zip(ideal_r.iter()) {
            let error = (r - ideal).abs();
            assert!(
                error < 5.0,
                "Radius should converge to ideal: {} vs {}",
                r,
                ideal
            );
        }
    }
}
