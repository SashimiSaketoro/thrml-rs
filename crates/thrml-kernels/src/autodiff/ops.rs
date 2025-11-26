//! Common autodiff operation utilities.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Zero gradient helper for non-differentiable operations.
pub fn zeros_like<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    device: &B::Device,
) -> Tensor<B, D> {
    Tensor::zeros(tensor.dims(), device)
}
