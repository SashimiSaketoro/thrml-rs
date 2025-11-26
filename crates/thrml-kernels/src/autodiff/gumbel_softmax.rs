//! Gumbel-Softmax for differentiable categorical sampling.
//!
//! Reference: "Categorical Reparameterization with Gumbel-Softmax"
//! Jang et al., ICLR 2017

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor};

/// Gumbel-Softmax differentiable categorical sampling.
///
/// # Arguments
/// * `logits` - Unnormalized log probabilities [batch_size, n_categories]
/// * `temperature` - Temperature parameter (τ). Start at 1.0, anneal to 0.1 during training.
/// * `hard` - If true, use Straight-Through Estimator for hard samples.
/// * `device` - Compute device
///
/// # Returns
/// * If hard=false: Soft samples [batch_size, n_categories] (sums to 1 per row)
/// * If hard=true: One-hot samples [batch_size, n_categories] (exactly one 1 per row)
///
/// # Example
/// ```ignore
/// let logits = Tensor::from_data([[1.0, 2.0, 0.5]], &device);
/// let soft_sample = gumbel_softmax(logits.clone(), 1.0, false, &device);
/// let hard_sample = gumbel_softmax(logits, 0.5, true, &device);
/// ```
pub fn gumbel_softmax<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f32,
    hard: bool,
    device: &B::Device,
) -> Tensor<B, 2> {
    let [batch_size, n_categories] = logits.dims();

    // Sample Gumbel noise: -log(-log(U)) where U ~ Uniform(0,1)
    let uniform: Tensor<B, 2> = Tensor::random(
        [batch_size, n_categories],
        Distribution::Uniform(1e-10, 1.0 - 1e-10),
        device,
    );
    let gumbel_noise = -(-(uniform.log())).log();

    // y = softmax((logits + gumbel_noise) / temperature)
    let y_soft = burn::tensor::activation::softmax(
        (logits + gumbel_noise) / temperature,
        1, // softmax over categories dimension
    );

    if hard {
        // Straight-Through Estimator:
        // Forward pass: use hard one-hot
        // Backward pass: gradients flow through y_soft
        let indices = y_soft.clone().argmax(1); // [batch_size, 1]
        let y_hard = one_hot::<B>(indices.squeeze::<1>(), n_categories, device);

        // The trick: y_hard - y_soft.detach() + y_soft
        // Forward: returns y_hard
        // Backward: gradient of y_soft
        y_hard - y_soft.clone().detach() + y_soft
    } else {
        y_soft
    }
}

/// Create one-hot encoding from indices.
fn one_hot<B: Backend>(
    indices: Tensor<B, 1, Int>,
    n_categories: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = indices.dims()[0];

    // Create range tensor [0, 1, 2, ..., n_categories-1]
    let range: Tensor<B, 1, Int> = Tensor::arange(0..n_categories as i64, device);

    // Expand indices: [batch] -> [batch, 1]
    // Expand range: [cats] -> [1, cats]
    let indices_expanded = indices.unsqueeze_dim::<2>(1); // [batch, 1]
    let range_expanded = range.unsqueeze_dim::<2>(0); // [1, cats]

    // Broadcast and compare to create one-hot
    // Need to expand both to [batch, cats] for comparison
    let indices_broadcast = indices_expanded.repeat_dim(1, n_categories); // [batch, cats]
    let range_broadcast = range_expanded.repeat_dim(0, batch_size); // [batch, cats]

    // Compare: indices_broadcast == range_broadcast
    indices_broadcast.equal(range_broadcast).float()
}

/// Temperature schedule for Gumbel-Softmax annealing.
///
/// Standard schedule: τ(t) = max(τ_min, τ_0 * exp(-r * t))
pub struct TemperatureSchedule {
    pub initial: f32,
    pub minimum: f32,
    pub decay_rate: f32,
}

impl Default for TemperatureSchedule {
    fn default() -> Self {
        TemperatureSchedule {
            initial: 1.0,
            minimum: 0.1,
            decay_rate: 0.001,
        }
    }
}

impl TemperatureSchedule {
    pub fn new(initial: f32, minimum: f32, decay_rate: f32) -> Self {
        TemperatureSchedule {
            initial,
            minimum,
            decay_rate,
        }
    }

    /// Get temperature at training step t.
    pub fn at_step(&self, step: usize) -> f32 {
        (self.initial * (-self.decay_rate * step as f32).exp()).max(self.minimum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gumbel_softmax_soft() {
        use thrml_core::backend::{init_gpu_device, WgpuBackend};

        let device = init_gpu_device();
        let logits: Tensor<WgpuBackend, 2> =
            Tensor::from_data([[1.0f32, 2.0, 0.5], [0.0, 0.0, 0.0]], &device);

        let soft = gumbel_softmax(logits, 1.0, false, &device);
        let soft_data: Vec<f32> = soft.clone().into_data().to_vec().unwrap();

        // Soft samples should sum to ~1 per row
        let row0_sum: f32 = soft_data[0..3].iter().sum();
        let row1_sum: f32 = soft_data[3..6].iter().sum();

        assert!((row0_sum - 1.0).abs() < 0.01, "Row 0 should sum to 1");
        assert!((row1_sum - 1.0).abs() < 0.01, "Row 1 should sum to 1");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gumbel_softmax_hard() {
        use thrml_core::backend::{init_gpu_device, WgpuBackend};

        let device = init_gpu_device();
        let logits: Tensor<WgpuBackend, 2> = Tensor::from_data([[10.0f32, 0.0, 0.0]], &device); // Strong preference for category 0

        let hard = gumbel_softmax(logits, 0.1, true, &device);
        let hard_data: Vec<f32> = hard.into_data().to_vec().unwrap();

        // Hard samples should be one-hot
        let sum: f32 = hard_data.iter().sum();
        let max: f32 = hard_data.iter().cloned().fold(0.0, f32::max);

        assert!((sum - 1.0).abs() < 0.01, "Should sum to 1");
        assert!((max - 1.0).abs() < 0.01, "Max should be 1");
    }

    #[test]
    fn test_temperature_schedule() {
        let schedule = TemperatureSchedule::new(1.0, 0.1, 0.001);

        // At step 0, should be initial
        assert!((schedule.at_step(0) - 1.0).abs() < 0.001);

        // Temperature should decrease over time
        assert!(schedule.at_step(1000) < schedule.at_step(0));

        // Should not go below minimum
        assert!(schedule.at_step(100000) >= 0.1);
    }
}
