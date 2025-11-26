//! Backward pass (autodiff) implementation for Gumbel-max kernel.
//!
//! # Gradient Computation
//!
//! The Gumbel-max argmax operation is **non-differentiable**:
//! - The argmax function is piecewise constant
//! - Its derivative is zero almost everywhere
//! - At discontinuities (ties), the derivative is undefined
//!
//! # Gradient Estimation Strategies
//!
//! For training with discrete categorical sampling, consider:
//!
//! 1. **Gumbel-Softmax (Concrete Distribution)**:
//!    - Use `softmax((logits + gumbel) / temperature)` instead of argmax
//!    - Produces differentiable "soft" samples
//!    - Anneal temperature toward 0 during training
//!
//! 2. **REINFORCE (Policy Gradient)**:
//!    - Use the score function estimator
//!    - `∇θ E[f(x)] ≈ E[f(x) * ∇θ log p(x|θ)]`
//!    - High variance, but unbiased
//!
//! 3. **Straight-Through Estimator (STE)**:
//!    - Forward: use hard argmax
//!    - Backward: pretend argmax is identity (or softmax)
//!    - Biased but low variance
//!
//! # Implementation Notes
//!
//! Since THRML uses Gibbs sampling for inference (not gradient-based optimization),
//! the backward pass registers zero gradients. For training Discrete EBMs,
//! use contrastive divergence or score matching instead of backprop through samples.

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// The Gumbel-max operation has zero gradient.
///
/// The argmax operation is piecewise constant, so its gradient is zero
/// almost everywhere. For differentiable alternatives, use:
/// - Gumbel-Softmax with temperature annealing
/// - REINFORCE/policy gradient methods
/// - Straight-through estimator
/// Create zero gradient for Gumbel-max backward pass.
///
/// Uses the high-level Tensor API for stability.
#[allow(dead_code)]
pub fn gumbel_argmax_backward_zeros<B: Backend>(
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::zeros(shape, device)
}

// ============================================================================
// Autodiff Integration (for future use)
// ============================================================================
//
// To fully integrate with Burn's autodiff system with STE:
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
// struct GumbelArgmaxBackward;
//
// #[derive(Debug)]
// struct GumbelArgmaxState<B: Backend> {
//     logits_shape: burn::tensor::Shape,
//     device: B::Device,
// }
//
// impl<B: FusedKernelBackend> Backward<B, 2> for GumbelArgmaxBackward {
//     type State = GumbelArgmaxState<B>;
//
//     fn backward(
//         self,
//         ops: Ops<Self::State, 2>,
//         grads: &mut Gradients,
//         _checkpointer: &mut Checkpointer,
//     ) {
//         let [node_logits, node_uniform] = ops.parents;
//         let state = ops.state;
//
//         // Argmax has zero gradient
//         let grad_logits = gumbel_argmax_backward::<B>(
//             state.logits_shape.clone(),
//             &state.device,
//         );
//
//         // Gradient w.r.t. uniform is also zero (and semantically meaningless)
//         if let Some(node) = node_logits {
//             grads.register::<B>(node.id, grad_logits.clone());
//         }
//
//         if let Some(node) = node_uniform {
//             grads.register::<B>(node.id, grad_logits);
//         }
//     }
// }
// ```
//
// For Gumbel-Softmax (differentiable approximation):
//
// ```rust
// pub fn gumbel_softmax<B: Backend>(
//     logits: Tensor<B, 2>,
//     uniform: Tensor<B, 2>,
//     temperature: f32,
// ) -> Tensor<B, 2> {
//     let gumbel = -(-(uniform.log())).log();
//     let perturbed = (logits + gumbel) / temperature;
//     burn::tensor::activation::softmax(perturbed, 1)
// }
// ```
