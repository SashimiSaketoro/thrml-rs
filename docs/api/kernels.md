# Kernels Module

The `thrml-kernels` crate provides GPU-accelerated kernels for differentiable operations, primarily used for training energy-based models.

## Overview

```
thrml-kernels/
├── autodiff/
│   ├── gumbel_softmax.rs     # Gumbel-Softmax for differentiable discrete sampling
│   ├── sigmoid_bernoulli.rs  # Differentiable Bernoulli sampling
│   └── ops.rs                # Common autodiff operations
├── batch_gather/             # Batched gather with gradients
├── gumbel_argmax/            # Gumbel-argmax kernel
└── sigmoid_bernoulli/        # Sigmoid-Bernoulli kernel
```

---

## Gumbel-Softmax

Differentiable approximation to categorical sampling using the Gumbel-Softmax trick.

### `gumbel_softmax`

```rust
use thrml_kernels::autodiff::gumbel_softmax::gumbel_softmax;

// Logits: [batch_size, n_categories]
let logits: Tensor<WgpuBackend, 2> = /* ... */;

// Sample with temperature
let samples = gumbel_softmax(
    logits,
    0.5,      // temperature (lower = harder)
    false,    // hard (use straight-through estimator)
    &device,
);

// Hard selection with gradient flow
let hard_samples = gumbel_softmax(logits, 0.1, true, &device);
```

**Parameters:**
- `temperature`: Controls sharpness (lower = closer to one-hot)
- `hard`: If true, uses straight-through estimator for one-hot output with gradient flow

**Use Cases:**
- Differentiable target selection in navigation
- Training discrete choices end-to-end
- Soft attention mechanisms

---

## Batch Gather

Efficient batched gather operation with backward pass support.

### `batch_gather`

```rust
use thrml_kernels::batch_gather::batch_gather;

// Gather elements from a tensor along a dimension
let data: Tensor<WgpuBackend, 3> = /* [batch, seq, dim] */;
let indices: Tensor<WgpuBackend, 2> = /* [batch, n_indices] */;

let gathered = batch_gather(&data, &indices, 1);  // Gather along seq dimension
// Result: [batch, n_indices, dim]
```

**Use Cases:**
- Selecting neighbor states by index
- Gathering embeddings for interaction computation
- Batched indexing in EBM sampling

---

## Sigmoid-Bernoulli

Differentiable Bernoulli sampling for binary variables.

### `sigmoid_bernoulli_sample`

```rust
use thrml_kernels::autodiff::sigmoid_bernoulli::sigmoid_bernoulli_sample;

// Log-odds (logits) for Bernoulli distribution
let logits: Tensor<WgpuBackend, 2> = /* ... */;

// Sample with gradient flow
let samples = sigmoid_bernoulli_sample(logits, &device);
```

**Use Cases:**
- Spin variable sampling in discrete EBMs
- Binary masking with differentiability
- Stochastic regularization

---

## Gumbel-Argmax

Hard argmax with Gumbel noise for exploration.

### `gumbel_argmax`

```rust
use thrml_kernels::gumbel_argmax::gumbel_argmax;

let logits: Tensor<WgpuBackend, 2> = /* [batch, n_options] */;

// Get argmax with Gumbel noise
let indices = gumbel_argmax(&logits, &device);  // [batch]
```

**Use Cases:**
- Exploration in discrete action spaces
- Stochastic selection
- Temperature-scaled sampling

---

## Autodiff Utilities

### Common Operations

```rust
use thrml_kernels::autodiff::ops;

// Log-sum-exp with numerical stability
let lse = ops::log_sum_exp(&logits, dim);

// Softmax
let probs = ops::softmax(&logits, dim);

// Gumbel noise generation
let gumbel = ops::sample_gumbel(shape, &device);
```

---

## Integration with Navigation

The kernels are used internally by `thrml-sphere` for differentiable navigation:

```rust
// In NavigatorEBM::gumbel_select_targets
use thrml_kernels::autodiff::gumbel_softmax::gumbel_softmax;

let logits = -energies;  // Lower energy = higher probability
let weights = gumbel_softmax(logits, temperature, hard, &device);
```

This enables end-to-end training of navigation paths through discrete choices.

---

## Performance

The kernels are optimized for GPU execution:

- **Fused operations**: Minimize memory bandwidth
- **Warp-level primitives**: Efficient reductions
- **Kernel fusion**: Combine Gumbel sampling with softmax

Benchmarks available in `benches/kernel_fusion.rs`.
