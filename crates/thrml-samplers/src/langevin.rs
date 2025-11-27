//! Langevin dynamics sampler for continuous variables.
//!
//! Implements overdamped Langevin dynamics for sampling from continuous
//! energy-based models.
//!
//! The update rule is:
//! ```text
//! dx = -∇H(x)dt + sqrt(2Tdt)ξ
//! ```
//! where ξ is Gaussian noise.
//!
//! ## Example
//!
//! ```rust,ignore
//! use thrml_samplers::{LangevinConfig, langevin_step};
//!
//! let config = LangevinConfig::new(0.01, 0.1).with_gradient_clip(1.0);
//! let new_state = langevin_step(&state, &gradient, &config, &noise, device);
//! ```

use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use crate::RngKey;

/// Configuration for Langevin dynamics.
#[derive(Clone, Debug)]
pub struct LangevinConfig {
    /// Step size (dt) for each iteration
    pub step_size: f32,
    /// Temperature controlling noise magnitude
    pub temperature: f32,
    /// Optional gradient clipping threshold
    pub gradient_clip: Option<f32>,
}

impl LangevinConfig {
    /// Create new Langevin configuration.
    pub fn new(step_size: f32, temperature: f32) -> Self {
        Self {
            step_size,
            temperature,
            gradient_clip: None,
        }
    }
    
    /// Set gradient clipping threshold.
    pub fn with_gradient_clip(mut self, max_grad: f32) -> Self {
        self.gradient_clip = Some(max_grad);
        self
    }
    
    /// Compute noise scale: sqrt(2 * T * dt)
    pub fn noise_scale(&self) -> f32 {
        (2.0 * self.temperature * self.step_size).sqrt()
    }
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self::new(0.01, 1.0)
    }
}

/// Perform one Langevin step on a state tensor.
///
/// Updates state according to:
/// ```text
/// x_new = x - step_size * gradient + noise_scale * noise
/// ```
///
/// # Arguments
/// * `state` - Current state tensor [N, D]
/// * `gradient` - Gradient of energy w.r.t. state [N, D]
/// * `config` - Langevin configuration
/// * `device` - GPU device
///
/// # Returns
/// Updated state tensor [N, D]
pub fn langevin_step_2d(
    state: &Tensor<WgpuBackend, 2>,
    gradient: &Tensor<WgpuBackend, 2>,
    config: &LangevinConfig,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2> {
    let shape = state.dims();
    
    // Apply gradient clipping if enabled
    let grad = if let Some(max_grad) = config.gradient_clip {
        clip_gradient_2d(gradient, max_grad)
    } else {
        gradient.clone()
    };
    
    // Drift: -gradient * step_size
    let drift = grad.mul_scalar(-config.step_size);
    
    // Diffusion: noise_scale * N(0, 1)
    let noise: Tensor<WgpuBackend, 2> = Tensor::random(
        [shape[0], shape[1]],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let diffusion = noise.mul_scalar(config.noise_scale());
    
    state.clone() + drift + diffusion
}

/// Perform one Langevin step on a 1D state tensor.
pub fn langevin_step_1d(
    state: &Tensor<WgpuBackend, 1>,
    gradient: &Tensor<WgpuBackend, 1>,
    config: &LangevinConfig,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 1> {
    let n = state.dims()[0];
    
    // Apply gradient clipping if enabled
    let grad = if let Some(max_grad) = config.gradient_clip {
        clip_gradient_1d(gradient, max_grad)
    } else {
        gradient.clone()
    };
    
    // Drift: -gradient * step_size
    let drift = grad.mul_scalar(-config.step_size);
    
    // Diffusion: noise_scale * N(0, 1)
    let noise: Tensor<WgpuBackend, 1> = Tensor::random(
        [n],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let diffusion = noise.mul_scalar(config.noise_scale());
    
    state.clone() + drift + diffusion
}

/// Clip gradient tensor to maximum magnitude.
fn clip_gradient_2d(
    gradient: &Tensor<WgpuBackend, 2>,
    max_grad: f32,
) -> Tensor<WgpuBackend, 2> {
    // Compute gradient magnitudes per row
    let grad_sq = gradient.clone().powf_scalar(2.0);
    let grad_mag_sq = grad_sq.sum_dim(1); // [N, 1]
    let grad_mag = grad_mag_sq.sqrt().clamp(1e-8, f32::MAX);
    
    // Scale factor: min(1, max_grad / ||grad||)
    let scale = (grad_mag.recip().mul_scalar(max_grad)).clamp(0.0, 1.0);
    
    gradient.clone() * scale
}

/// Clip gradient tensor to maximum magnitude (1D).
fn clip_gradient_1d(
    gradient: &Tensor<WgpuBackend, 1>,
    max_grad: f32,
) -> Tensor<WgpuBackend, 1> {
    // Element-wise clipping
    gradient.clone().clamp(-max_grad, max_grad)
}

/// Annealing schedule for temperature.
#[derive(Clone, Debug)]
pub enum AnnealingSchedule {
    /// Constant temperature
    Constant(f32),
    /// Linear annealing from start to end
    Linear { start: f32, end: f32 },
    /// Exponential annealing: T(t) = start * decay^t
    Exponential { start: f32, decay: f32 },
    /// Cosine annealing
    Cosine { start: f32, end: f32 },
}

impl AnnealingSchedule {
    /// Get temperature at step t (normalized to [0, 1]).
    pub fn temperature(&self, t: f32) -> f32 {
        match self {
            AnnealingSchedule::Constant(temp) => *temp,
            AnnealingSchedule::Linear { start, end } => {
                start + (end - start) * t
            }
            AnnealingSchedule::Exponential { start, decay } => {
                start * decay.powf(t)
            }
            AnnealingSchedule::Cosine { start, end } => {
                let cos_t = (t * std::f32::consts::PI).cos();
                end + (start - end) * 0.5 * (1.0 + cos_t)
            }
        }
    }
}

/// Run multiple Langevin steps with optional annealing.
///
/// # Arguments
/// * `init_state` - Initial state [N, D]
/// * `gradient_fn` - Function computing gradient from state
/// * `config` - Base Langevin configuration
/// * `n_steps` - Number of steps to run
/// * `annealing` - Optional temperature annealing schedule
/// * `key` - RNG key for reproducibility
/// * `device` - GPU device
///
/// # Returns
/// Final state after n_steps
pub fn run_langevin_2d<F>(
    init_state: Tensor<WgpuBackend, 2>,
    gradient_fn: F,
    config: &LangevinConfig,
    n_steps: usize,
    annealing: Option<&AnnealingSchedule>,
    key: RngKey,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2>
where
    F: Fn(&Tensor<WgpuBackend, 2>) -> Tensor<WgpuBackend, 2>,
{
    let _keys = key.split(n_steps);
    let mut state = init_state;
    let n_steps_f = n_steps as f32;
    
    for step in 0..n_steps {
        // Get current temperature
        let temperature = annealing
            .map(|a| a.temperature(step as f32 / n_steps_f))
            .unwrap_or(config.temperature);
        
        let step_config = LangevinConfig {
            step_size: config.step_size,
            temperature,
            gradient_clip: config.gradient_clip,
        };
        
        let gradient = gradient_fn(&state);
        state = langevin_step_2d(&state, &gradient, &step_config, device);
    }
    
    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrml_core::backend::init_gpu_device;
    use burn::tensor::Distribution;

    #[test]
    fn test_langevin_step_shape_preserved() {
        let device = init_gpu_device();
        let n = 10;
        let d = 3;
        
        let state: Tensor<WgpuBackend, 2> = Tensor::random(
            [n, d],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let gradient: Tensor<WgpuBackend, 2> = Tensor::random(
            [n, d],
            Distribution::Normal(0.0, 0.1),
            &device,
        );
        
        let config = LangevinConfig::new(0.01, 0.1);
        let new_state = langevin_step_2d(&state, &gradient, &config, &device);
        
        assert_eq!(new_state.dims(), [n, d]);
    }
    
    #[test]
    fn test_zero_temperature_is_deterministic() {
        let device = init_gpu_device();
        let n = 5;
        
        let state: Tensor<WgpuBackend, 1> = Tensor::from_data([1.0f32, 2.0, 3.0, 4.0, 5.0].as_slice(), &device);
        let gradient: Tensor<WgpuBackend, 1> = Tensor::from_data([0.1f32, 0.1, 0.1, 0.1, 0.1].as_slice(), &device);
        
        let config = LangevinConfig::new(1.0, 0.0); // Zero temperature
        
        // With zero temperature, noise is zero, so step is deterministic
        let new_state = langevin_step_1d(&state, &gradient, &config, &device);
        let new_data: Vec<f32> = new_state.into_data().to_vec().expect("to vec");
        
        // new = old - step_size * gradient = [1, 2, 3, 4, 5] - 1.0 * [0.1, ...] = [0.9, 1.9, ...]
        assert!((new_data[0] - 0.9).abs() < 1e-5);
        assert!((new_data[1] - 1.9).abs() < 1e-5);
    }
    
    #[test]
    fn test_annealing_schedule() {
        let linear = AnnealingSchedule::Linear { start: 1.0, end: 0.1 };
        assert!((linear.temperature(0.0) - 1.0).abs() < 1e-5);
        assert!((linear.temperature(0.5) - 0.55).abs() < 1e-5);
        assert!((linear.temperature(1.0) - 0.1).abs() < 1e-5);
        
        let constant = AnnealingSchedule::Constant(0.5);
        assert!((constant.temperature(0.0) - 0.5).abs() < 1e-5);
        assert!((constant.temperature(1.0) - 0.5).abs() < 1e-5);
    }
}

