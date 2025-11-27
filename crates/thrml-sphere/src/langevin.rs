//! Langevin dynamics sampler for sphere optimization.
//!
//! This module provides sphere-specific Langevin samplers that work with
//! `SphericalCoords` and `SphereHamiltonian`.
//!
//! For general Langevin utilities, see `thrml_samplers::langevin`.

use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use thrml_core::SphericalCoords;
use thrml_samplers::RngKey;
use crate::hamiltonian::SphereHamiltonian;
use crate::config::SphereConfig;

// Re-export general Langevin utilities
pub use thrml_samplers::langevin::{
    AnnealingSchedule,
    LangevinConfig,
    langevin_step_1d,
    langevin_step_2d,
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
    pub fn new(step_size: f32, temperature: f32, n_steps: usize) -> Self {
        Self {
            step_size,
            temperature,
            n_steps,
            gradient_clip: None,
        }
    }

    /// Create from a SphereConfig.
    pub fn from_config(config: &SphereConfig) -> Self {
        Self::new(config.step_size, config.temperature, config.n_steps)
    }
    
    /// Set gradient clipping threshold.
    pub fn with_gradient_clip(mut self, max_grad: f32) -> Self {
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

        let r_hat_x = sin_theta.clone() * cos_phi.clone();
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
    pub fn linear(
        step_size: f32,
        temp_init: f32,
        temp_final: f32,
        n_steps: usize,
    ) -> Self {
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

        let ideal_radii: Tensor<WgpuBackend, 1> = 
            Tensor::from_data([50.0; 10].as_slice(), &device);
        
        let init_radii: Tensor<WgpuBackend, 1> = 
            Tensor::random([n], Distribution::Uniform(30.0, 70.0), &device);
        let init_coords = SphericalCoords::init_random(n, init_radii, &device);

        let similarity: Tensor<WgpuBackend, 2> = Tensor::zeros([n, n], &device);
        let hamiltonian = WaterFillingHamiltonian::new(
            ideal_radii.clone(),
            similarity,
            1.0,
        );

        let sampler = SphereLangevinSampler::new(0.5, 0.0, 50);
        let key = RngKey::new(42);
        let final_coords = sampler.run(&hamiltonian, init_coords, key, &device);

        let final_r: Vec<f32> = final_coords.r.into_data().to_vec().expect("r to vec");
        let ideal_r: Vec<f32> = ideal_radii.into_data().to_vec().expect("ideal to vec");
        
        for (r, ideal) in final_r.iter().zip(ideal_r.iter()) {
            let error = (r - ideal).abs();
            assert!(error < 5.0, "Radius should converge to ideal: {} vs {}", r, ideal);
        }
    }
}
