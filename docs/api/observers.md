# Observers

The `thrml-observers` crate provides observation utilities for collecting statistics during sampling.

## AbstractObserver

### Trait Definition

Base trait for sampling observers.

```rust
pub trait AbstractObserver {
    type Output;
    
    /// Initialize observer state before sampling
    fn init(&mut self, blocks: &[Block], device: &WgpuDevice);
    
    /// Called after each sampling step
    fn observe(
        &mut self,
        state: &[Tensor<WgpuBackend, 1>],
        blocks: &[Block],
        device: &WgpuDevice,
    );
    
    /// Finalize and return results
    fn finalize(self) -> Self::Output;
}
```

**Usage with `sample_with_observation`:**

```rust
use thrml_samplers::sample_with_observation;
use thrml_observers::StateObserver;

let observer = StateObserver::new(observe_blocks.clone());
let samples = sample_with_observation(
    key,
    &program,
    &schedule,
    init_state,
    &[],
    observer,
    &device,
);
```

---

## StateObserver

Collects raw state samples at each observation step.

```rust
pub struct StateObserver {
    observe_blocks: Vec<Block>,
    samples: Vec<Vec<Tensor<WgpuBackend, 1>>>,
}

impl StateObserver {
    pub fn new(observe_blocks: Vec<Block>) -> Self;
}

impl AbstractObserver for StateObserver {
    type Output = Vec<Vec<Tensor<WgpuBackend, 1>>>;
    // ...
}
```

**Example:**

```rust
use thrml_observers::StateObserver;
use thrml_samplers::{sample_with_observation, SamplingSchedule};

let observer = StateObserver::new(vec![block.clone()]);

let samples = sample_with_observation(
    RngKey::new(42),
    &program,
    &SamplingSchedule {
        n_warmup: 100,
        n_samples: 1000,
        steps_per_sample: 1,
    },
    init_state,
    &[],
    observer,
    &device,
);

// samples: Vec<Vec<Tensor>> - 1000 samples, each containing state for observe_blocks
println!("Collected {} samples", samples.len());
```

---

## MomentAccumulatorObserver

Accumulates first and second moment statistics for training.

```rust
pub struct MomentAccumulatorObserver {
    first_moment_blocks: Vec<Block>,
    second_moment_blocks: Vec<(Block, Block)>,
    first_moments: Vec<Tensor<WgpuBackend, 1>>,
    second_moments: Vec<Tensor<WgpuBackend, 1>>,
    count: usize,
}

impl MomentAccumulatorObserver {
    pub fn new(
        first_moment_blocks: Vec<Block>,
        second_moment_blocks: Vec<(Block, Block)>,
    ) -> Self;
}

impl AbstractObserver for MomentAccumulatorObserver {
    type Output = MomentSpec;
    // ...
}
```

### MomentSpec

Output type containing accumulated statistics.

```rust
pub struct MomentSpec {
    pub first_moments: Vec<Tensor<WgpuBackend, 1>>,
    pub second_moments: Vec<Tensor<WgpuBackend, 1>>,
}
```

**Example:**

```rust
use thrml_observers::MomentAccumulatorObserver;
use thrml_samplers::sample_with_observation;

// Track first moments for node blocks, second moments for edge pairs
let observer = MomentAccumulatorObserver::new(
    vec![node_block.clone()],           // First moments
    vec![(block_a.clone(), block_b.clone())],  // Second moments (correlations)
);

let moments = sample_with_observation(
    RngKey::new(42),
    &program,
    &schedule,
    init_state,
    &[],
    observer,
    &device,
);

// Access accumulated statistics
let mean_magnetization = &moments.first_moments[0];
let correlations = &moments.second_moments[0];
```

### Spin Transformation

For spin nodes, the observer automatically applies the spin transformation:

```
observed_value = 2 * raw_value - 1
```

This converts {0, 1} boolean representation to {-1, +1} spin values for correct moment computation. Categorical nodes pass through unchanged.

---

## Custom Observers

You can implement custom observers by implementing the `AbstractObserver` trait:

```rust
use thrml_observers::AbstractObserver;

pub struct EntropyObserver {
    blocks: Vec<Block>,
    log_sum: f32,
    count: usize,
}

impl AbstractObserver for EntropyObserver {
    type Output = f32;
    
    fn init(&mut self, blocks: &[Block], _device: &WgpuDevice) {
        self.blocks = blocks.to_vec();
        self.log_sum = 0.0;
        self.count = 0;
    }
    
    fn observe(
        &mut self,
        state: &[Tensor<WgpuBackend, 1>],
        _blocks: &[Block],
        _device: &WgpuDevice,
    ) {
        // Compute and accumulate entropy estimate
        for s in state {
            let values = s.clone().into_data().to_vec::<f32>().unwrap();
            for v in values {
                if v > 0.0 && v < 1.0 {
                    self.log_sum -= v * v.ln() + (1.0 - v) * (1.0 - v).ln();
                }
            }
        }
        self.count += 1;
    }
    
    fn finalize(self) -> f32 {
        self.log_sum / self.count as f32
    }
}
```

---

## Integration with Training

The `MomentAccumulatorObserver` is used internally by `estimate_moments` and `estimate_kl_grad`:

```rust
use thrml_models::ising::estimate_kl_grad;

// Compute gradients for training
let gradients = estimate_kl_grad(
    key,
    &training_spec,
    &schedule,
    init_state_model,
    init_state_data,
    &clamp_data,
    &device,
);

// Use gradients to update model parameters
let new_biases = ebm.biases - learning_rate * gradients.bias_grad;
let new_weights = ebm.edge_weights - learning_rate * gradients.weight_grad;
```

