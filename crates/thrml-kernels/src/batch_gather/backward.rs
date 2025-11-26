//! Backward pass (autodiff) implementation for batch gather kernel.
//!
//! # Gradient Computation
//!
//! The gather operation selects elements from a weight tensor based on indices:
//! ```ignore
//! output[i] = weights[linear_index(i, indices[i])]
//! ```
//!
//! The gradient is a **scatter-add** operation:
//! - For each output position `i`, the gradient flows back to the weight position
//!   that was read from
//! - If the same weight position is read multiple times (duplicate indices),
//!   the gradients are summed
//!
//! ```ignore
//! grad_weights = zeros_like(weights)
//! for i in 0..batch_size:
//!     grad_weights[linear_index(i, indices[i])] += grad_output[i]
//! ```
//!
//! # Properties
//!
//! - **Sparse gradient**: Only positions that were gathered receive non-zero gradients
//! - **Summation on conflicts**: Duplicate indices result in gradient accumulation
//! - **No gradient for indices**: Indices are integers and have no gradient
//!
//! # Implementation Notes
//!
//! The scatter-add pattern can be implemented using Burn's `scatter` operation
//! or a custom kernel for better performance.

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

/// Compute the gradient for batch gather using scatter-add.
///
/// The gradient with respect to weights is sparse: only the positions
/// that were gathered from receive non-zero gradients.
///
/// # Arguments
/// * `weights_shape` - Shape of the original weights tensor
/// * `indices` - The indices used in the forward pass [batch_size, n_indices]
/// * `grad_output` - Gradient of the loss w.r.t. the output `[batch_size]`
/// * `strides` - Stride values for each index dimension
/// * `batch_stride` - Stride for the batch dimension
/// * `device` - Device to create tensors on
///
/// # Returns
/// Gradient w.r.t. weights (same shape as weights, but sparse)
#[allow(dead_code)]
pub fn batch_gather_backward<B: Backend>(
    weights_shape: [usize; 3],
    indices: Tensor<B, 2, Int>,
    grad_output: Tensor<B, 1>,
    strides: &[usize],
    batch_stride: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let [batch_size, n_k, n_dim] = weights_shape;
    let total_elements = batch_size * n_k * n_dim;

    // Create zero tensor for gradient accumulation
    let grad_weights_flat: Tensor<B, 1> = Tensor::zeros([total_elements], device);

    // Compute linear indices (same as forward pass)
    let batch_indices: Tensor<B, 1, Int> =
        Tensor::arange(0..batch_size as i64, device).reshape([batch_size]);

    // Extract indices
    let k_indices: Tensor<B, 1, Int> = indices.clone().slice([0..batch_size, 0..1]).squeeze::<1>();
    let dim_indices: Tensor<B, 1, Int> = indices.slice([0..batch_size, 1..2]).squeeze::<1>();

    // Compute linear indices: batch * batch_stride + k * stride0 + dim * stride1
    let linear_indices = batch_indices * batch_stride as i64
        + k_indices * strides[0] as i64
        + dim_indices * strides.get(1).copied().unwrap_or(1) as i64;

    // Scatter-add: accumulate gradients at the gathered positions
    let grad_weights_flat = grad_weights_flat.scatter(0, linear_indices, grad_output);

    // Reshape back to original weight shape
    grad_weights_flat.reshape([batch_size, n_k, n_dim])
}

// Note: For the full scatter-add gradient implementation, use the
// high-level Tensor API (batch_gather_backward function above)
// which uses Burn's scatter operation.

// ============================================================================
// Autodiff Integration (for future use)
// ============================================================================
//
// To fully integrate with Burn's autodiff system:
//
// ```rust
// use burn_autodiff::{
//     checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
//     grads::Gradients,
//     ops::{Backward, Ops, OpsKind},
//     Autodiff,
// };
//
// #[derive(Debug)]
// struct BatchGatherBackward;
//
// #[derive(Debug)]
// struct BatchGatherState<B: Backend> {
//     weights_shape: [usize; 3],
//     indices: IntTensor<B>,
//     strides: Vec<usize>,
//     batch_stride: usize,
//     device: B::Device,
// }
//
// impl<B: FusedKernelBackend> Backward<B, 1> for BatchGatherBackward {
//     type State = BatchGatherState<B>;
//
//     fn backward(
//         self,
//         ops: Ops<Self::State, 1>,
//         grads: &mut Gradients,
//         _checkpointer: &mut Checkpointer,
//     ) {
//         let [node_weights] = ops.parents;
//         let grad_output = grads.consume::<B>(&ops.node);
//         let state = ops.state;
//
//         if let Some(node) = node_weights {
//             // Convert IntTensor to high-level Tensor for scatter operation
//             let indices: Tensor<B, 2, Int> = Tensor::from_primitive(state.indices);
//             let grad_out: Tensor<B, 1> = Tensor::from_primitive(
//                 TensorPrimitive::Float(grad_output)
//             );
//
//             let grad_weights = batch_gather_backward::<B>(
//                 state.weights_shape,
//                 indices,
//                 grad_out,
//                 &state.strides,
//                 state.batch_stride,
//                 &state.device,
//             );
//
//             grads.register::<B>(
//                 node.id,
//                 grad_weights.into_primitive().tensor(),
//             );
//         }
//
//         // No gradient for indices (integers have no gradient)
//     }
// }
// ```
