//! L2 normalization fused kernel.

mod forward;
mod kernel;

pub use forward::{launch_l2_normalize, launch_l2_normalize_with_norms, l2_normalize_fused};
