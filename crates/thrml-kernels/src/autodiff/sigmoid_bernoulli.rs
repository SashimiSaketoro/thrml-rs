//! Autodiff implementation for sigmoid-Bernoulli kernel using STE.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Compute sigmoid-Bernoulli backward gradient using Straight-Through Estimator.
///
/// Forward: y = Bernoulli(sigmoid(2 * gamma))
/// Backward: dy/dgamma = 2 * sigmoid(2*gamma) * (1 - sigmoid(2*gamma)) * grad_output
///
/// The Bernoulli sampling step uses STE (gradient passes through unchanged).
pub fn sigmoid_bernoulli_backward<B: Backend, const D: usize>(
    gamma: Tensor<B, D>,
    grad_output: Tensor<B, D>,
) -> Tensor<B, D> {
    use burn::tensor::activation::sigmoid;

    // Compute sigmoid(2 * gamma)
    let sig = sigmoid(gamma * 2.0);

    // Sigmoid derivative: sig * (1 - sig)
    let one_minus_sig = sig.clone().neg() + 1.0;

    // Full gradient: 2 * sig * (1 - sig) * grad_output
    sig * one_minus_sig * 2.0 * grad_output
}

/// For operations where we want zero gradient (truly non-differentiable).
pub fn sigmoid_bernoulli_backward_zero<B: Backend>(
    shape: [usize; 1],
    device: &B::Device,
) -> Tensor<B, 1> {
    Tensor::zeros(shape, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_sigmoid_bernoulli_backward() {
        use thrml_core::backend::{init_gpu_device, WgpuBackend};

        let device = init_gpu_device();

        // Test at gamma = 0 where sigmoid(0) = 0.5
        // Gradient should be: 2 * 0.5 * 0.5 * 1.0 = 0.5
        let gamma: Tensor<WgpuBackend, 1> = Tensor::zeros([4], &device);
        let grad_output: Tensor<WgpuBackend, 1> = Tensor::ones([4], &device);

        let grad = sigmoid_bernoulli_backward(gamma, grad_output);
        let grad_data: Vec<f32> = grad.into_data().to_vec().unwrap();

        for &g in &grad_data {
            assert!(
                (g - 0.5).abs() < 0.01,
                "Gradient at gamma=0 should be ~0.5, got {}",
                g
            );
        }
    }
}

