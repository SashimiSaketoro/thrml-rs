use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SamplingSchedule {
    /// The number of warmup steps to run before collecting samples.
    pub n_warmup: usize,
    /// The number of samples to collect.
    pub n_samples: usize,
    /// The number of steps to run between each sample.
    pub steps_per_sample: usize,
}

impl SamplingSchedule {
    pub fn new(n_warmup: usize, n_samples: usize, steps_per_sample: usize) -> Self {
        SamplingSchedule {
            n_warmup,
            n_samples,
            steps_per_sample,
        }
    }
}
