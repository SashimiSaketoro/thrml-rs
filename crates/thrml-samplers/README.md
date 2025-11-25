# thrml-samplers

Sampling algorithms for the THRML probabilistic computing library.

## Overview

This crate provides GPU-accelerated sampling algorithms for probabilistic graphical models:

- **Block Gibbs Sampling**: Parallel sampling of independent blocks
- **Bernoulli Sampler**: For binary/spin variables
- **Softmax Sampler**: For categorical variables (using Gumbel-max trick)
- **Spin Gibbs Sampler**: Specialized for Ising-type models

## Key Components

### RngKey

Deterministic RNG key system (similar to JAX):

```rust
use thrml_samplers::RngKey;

let key = RngKey::new(42);
let (key1, key2) = key.split_two();
```

### BlockSamplingProgram

The main sampling engine:

```rust
use thrml_samplers::{BlockSamplingProgram, SamplingSchedule};

let schedule = SamplingSchedule::new(100, 1000, 5);
// 100 warmup steps, 1000 samples, 5 steps between samples
```

### High-Level Functions

- `sample_blocks`: Run sampling iterations
- `sample_states`: Sample and collect states for specific blocks
- `sample_with_observation`: Sample with custom observers

## License

MIT OR Apache-2.0

