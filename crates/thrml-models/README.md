# thrml-models

Model implementations for the THRML probabilistic computing library.

## Overview

This crate provides implementations of various probabilistic graphical models:

### Ising Model

The classic Ising model for spin systems:

```rust
use thrml_models::ising::{IsingEBM, IsingSamplingProgram, hinton_init};

// Create an Ising model
let model = IsingEBM::new(nodes, edges, biases, weights, beta);

// Initialize with Hinton's method
let init_state = hinton_init(key, &model, &blocks, &[], &device);

// Compute energy
let energy = model.energy(&state, &blocks, &device);
```

### Discrete EBM Factors

Flexible factor types for building custom energy-based models:

- `SpinEBMFactor`: Binary/spin interactions
- `CategoricalEBMFactor`: Categorical variable interactions
- `DiscreteEBMFactor`: Mixed spin + categorical
- `SquareDiscreteEBMFactor`: Symmetric interactions

### Training Utilities

- `estimate_moments`: Estimate first/second moments via sampling
- `estimate_kl_grad`: Estimate KL divergence gradients

## Key Traits

- `AbstractFactor`: Base trait for factor types
- `AbstractEBM`: Energy-based model interface
- `EBMFactor`: Factors that contribute to energy

## License

MIT OR Apache-2.0

