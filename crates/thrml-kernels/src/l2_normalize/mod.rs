//! L2 normalization fused kernel.

mod forward;
mod kernel;

pub use forward::{l2_normalize_fused, launch_l2_normalize, launch_l2_normalize_with_norms};
