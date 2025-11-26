//! Backward pass (autodiff) implementation for sigmoid-Bernoulli kernel.
//!
//! # Gradient Computation
//!
//! The sigmoid-Bernoulli operation has two components:
//! 1. **Sigmoid**: Differentiable with gradient `sigmoid(x) * (1 - sigmoid(x))`
//! 2. **Bernoulli sampling**: Non-differentiable, uses Straight-Through Estimator (STE)
//!
//! For `y = bernoulli(sigmoid(2 * gamma))`:
//!
//! **Forward:**
//! - `p = sigmoid(2 * gamma)`
//! - `y = 1.0 if uniform < p else 0.0`
//!
//! **Backward (using STE):**
//! - `d_p/d_gamma = 2 * sigmoid(2*gamma) * (1 - sigmoid(2*gamma))`
//! - Bernoulli step: gradient passes through unchanged (STE)
//! - `d_loss/d_gamma = d_loss/d_y * d_p/d_gamma`
//!
//! # Note
//!
//! For discrete sampling operations in THRML, gradients are typically not needed
//! since we use Gibbs sampling for inference rather than gradient-based optimization.
//! If gradient-based training is required, consider using:
//! 1. Gumbel-Softmax with temperature annealing
//! 2. REINFORCE/policy gradient methods
//! 3. The STE implementation pattern below

use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Compute the sigmoid backward gradient.
///
/// For `sigmoid(2 * gamma)`, the gradient is:
/// `d/d_gamma[sigmoid(2*gamma)] = 2 * sigmoid(2*gamma) * (1 - sigmoid(2*gamma))`
///
/// This is then multiplied by the upstream gradient using chain rule.
#[allow(dead_code)]
pub fn sigmoid_bernoulli_backward<B: Backend, const D: usize>(
    gamma: Tensor<B, D>,
    grad_output: Tensor<B, D>,
) -> Tensor<B, D> {
    use burn::tensor::activation::sigmoid;

    // Compute sigmoid(2 * gamma)
    let sig = sigmoid(gamma * 2.0);

    // Compute sigmoid gradient: 2 * sig * (1 - sig)
    let one_minus_sig = sig.clone().neg() + 1.0;
    let grad_sigmoid = sig * one_minus_sig * 2.0;

    // Chain rule: multiply by upstream gradient
    grad_output * grad_sigmoid
}

// Note: For the full FloatTensor primitive implementation, use the
// high-level Tensor API (sigmoid_bernoulli_backward function above)
// and convert to/from primitives as needed. The low-level Backend
// ops API changes between Burn versions, so the high-level API is
// preferred for stability.

// ============================================================================
// Autodiff Integration (for future use)
// ============================================================================
//
// To fully integrate with Burn's autodiff system, implement:
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
// struct SigmoidBernoulliBackward;
//
// #[derive(Debug)]
// struct SigmoidBernoulliState<B: Backend> {
//     gamma: FloatTensor<B>,
// }
//
// impl<B: FusedKernelBackend> Backward<B, 1> for SigmoidBernoulliBackward {
//     type State = SigmoidBernoulliState<B>;
//
//     fn backward(
//         self,
//         ops: Ops<Self::State, 1>,
//         grads: &mut Gradients,
//         _checkpointer: &mut Checkpointer,
//     ) {
//         let [node_gamma] = ops.parents;
//         let grad_output = grads.consume::<B>(&ops.node);
//         let state = ops.state;
//
//         // Compute gradient using STE
//         let grad_gamma = sigmoid_bernoulli_backward_primitive::<B>(
//             state.gamma,
//             grad_output,
//         );
//
//         if let Some(node) = node_gamma {
//             grads.register::<B>(node.id, grad_gamma);
//         }
//     }
// }
// ```
