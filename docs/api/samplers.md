# Samplers

The `thrml-samplers` crate provides block Gibbs sampling infrastructure.

## Sampling Programs

### `BlockGibbsSpec`

Specification for block Gibbs sampling with free and clamped blocks.

```rust
pub struct BlockGibbsSpec {
    pub spec: BlockSpec,
    pub free_blocks: Vec<Block>,
    pub sampling_order: Vec<Vec<usize>>,
    pub clamped_blocks: Vec<Block>,
    pub superblocks: Vec<Vec<Block>>,
}

impl BlockGibbsSpec {
    pub fn new(
        free_super_blocks: Vec<SuperBlock>,
        clamped_blocks: Vec<Block>,
        node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
    ) -> Result<Self, String>;
}
```

**Parameters:**

- `free_super_blocks`: Groups of blocks to sample (blocks in same superblock are sampled together)
- `clamped_blocks`: Blocks with fixed values (not sampled)
- `node_shape_dtypes`: Tensor specifications per node type

### `BlockSamplingProgram`

The core sampling program that manages state and performs Gibbs updates.

```rust
pub struct BlockSamplingProgram {
    pub gibbs_spec: BlockGibbsSpec,
    pub samplers: Vec<Box<dyn DynConditionalSampler>>,
    // Pre-computed index tensors for efficient GPU operations
    pub per_block_interactions: Vec<Vec<Tensor<WgpuBackend, 3>>>,
    pub per_block_interaction_active: Vec<Vec<Tensor<WgpuBackend, 2>>>,
    pub per_block_n_spin: Vec<Vec<usize>>,
    // ...
}

impl BlockSamplingProgram {
    pub fn new(
        gibbs_spec: BlockGibbsSpec,
        samplers: Vec<Box<dyn DynConditionalSampler>>,
        interaction_groups: Vec<InteractionGroup>,
    ) -> Result<Self, String>;
    
    pub fn sample_single_block(
        &self,
        block_idx: usize,
        key: RngKey,
        state_free: &[Tensor<WgpuBackend, 1>],
        clamp_state: &[Tensor<WgpuBackend, 1>],
        device: &WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;
}
```

---

## Sampling Schedule

### `SamplingSchedule`

Controls the sampling process timing.

```rust
pub struct SamplingSchedule {
    /// Warmup iterations before collecting samples
    pub n_warmup: usize,
    /// Total number of samples to collect
    pub n_samples: usize,
    /// Steps between collected samples
    pub steps_per_sample: usize,
}
```

**Example:**

```rust
use thrml_samplers::SamplingSchedule;

let schedule = SamplingSchedule {
    n_warmup: 100,
    n_samples: 1000,
    steps_per_sample: 5,
};
// Total iterations: 100 + 1000 * 5 = 5100
```

---

## Sampling Functions

### `sample_blocks`

Run one iteration of block Gibbs sampling over all blocks.

```rust
pub fn sample_blocks(
    key: RngKey,
    program: &BlockSamplingProgram,
    state: Vec<Tensor<WgpuBackend, 1>>,
    clamp_state: &[Tensor<WgpuBackend, 1>],
    device: &WgpuDevice,
) -> (RngKey, Vec<Tensor<WgpuBackend, 1>>);
```

### `run_blocks`

Run multiple iterations of block Gibbs sampling.

```rust
pub fn run_blocks(
    key: RngKey,
    program: &BlockSamplingProgram,
    init_state: Vec<Tensor<WgpuBackend, 1>>,
    clamp_state: &[Tensor<WgpuBackend, 1>],
    n_iters: usize,
    device: &WgpuDevice,
) -> (RngKey, Vec<Tensor<WgpuBackend, 1>>);
```

### `sample_states`

Main user-facing function: run sampling and collect states.

```rust
pub fn sample_states(
    key: RngKey,
    program: &BlockSamplingProgram,
    schedule: &SamplingSchedule,
    init_state: Vec<Tensor<WgpuBackend, 1>>,
    clamp_state: &[Tensor<WgpuBackend, 1>],
    observe_blocks: &[Block],
    device: &WgpuDevice,
) -> Vec<Vec<Tensor<WgpuBackend, 1>>>;
```

**Returns:** Vector of samples, each containing state tensors for observed blocks.

### `sample_with_observation`

Run sampling with a custom observer for statistics collection.

```rust
pub fn sample_with_observation<O: AbstractObserver>(
    key: RngKey,
    program: &BlockSamplingProgram,
    schedule: &SamplingSchedule,
    init_state: Vec<Tensor<WgpuBackend, 1>>,
    clamp_state: &[Tensor<WgpuBackend, 1>],
    observer: O,
    device: &WgpuDevice,
) -> O::Output;
```

---

## RNG Key Management

### `RngKey`

Deterministic RNG key for reproducible sampling.

```rust
pub struct RngKey(pub u64);

impl RngKey {
    pub fn new(seed: u64) -> Self;
    pub fn split(&self) -> (RngKey, RngKey);
    pub fn split_n(&self, n: usize) -> Vec<RngKey>;
}
```

**Example:**

```rust
use thrml_samplers::RngKey;

let key = RngKey::new(42);
let (key1, key2) = key.split();
let keys = key.split_n(10);  // 10 independent keys
```

---

## Conditional Samplers

### `AbstractConditionalSampler`

Trait for sampling from conditional distributions. Supports stateful samplers that carry state
between iterations (e.g., for Metropolis-within-Gibbs).

```rust
pub trait AbstractConditionalSampler {
    /// Type of state carried between iterations. Use `()` for stateless samplers.
    type SamplerState: Clone + Default;
    
    /// Initialize sampler state before sampling begins.
    fn init_state(&self) -> Self::SamplerState {
        Self::SamplerState::default()
    }
    
    /// Sample from the conditional distribution.
    fn sample(
        &self,
        key: RngKey,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        sampler_state: Self::SamplerState,
        output_spec: &TensorSpec,
        device: &WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, Self::SamplerState);
}
```

### `DynConditionalSampler`

Type-erased wrapper trait for heterogeneous sampler collections. Use this when storing
samplers in `BlockSamplingProgram`.

```rust
pub trait DynConditionalSampler: Send + Sync {
    fn sample_stateless(
        &self,
        key: RngKey,
        interactions: &[Tensor<WgpuBackend, 3>],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        output_spec: &TensorSpec,
        device: &WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;
}

// Automatically implemented for all stateless samplers (SamplerState = ())
```

### `BernoulliConditional`

Base sampler for spin-valued Bernoulli distributions.

```rust
pub struct BernoulliConditional;
```

Samples from: P(S=1) = σ(2γ)

### `SpinGibbsConditional`

Gibbs updates for spin variables in discrete EBMs.

```rust
pub struct SpinGibbsConditional;

impl SpinGibbsConditional {
    pub fn new() -> Self;
}
```

Computes: γ = Σᵢ s₁ⁱ...sₖⁱ × W^i[x₁ⁱ, ..., xₘⁱ]

### `SoftmaxConditional`

Base sampler for categorical softmax distributions.

```rust
pub struct SoftmaxConditional {
    pub n_categories: usize,
}

impl SoftmaxConditional {
    pub fn new(n_categories: usize) -> Self;
}
```

Uses Gumbel-max trick for numerically stable GPU sampling.

### `CategoricalGibbsConditional`

Gibbs updates for categorical variables in discrete EBMs.

```rust
pub struct CategoricalGibbsConditional {
    pub n_categories: usize,
    pub n_spin: usize,
}

impl CategoricalGibbsConditional {
    pub fn new(n_categories: usize, n_spin: usize) -> Self;
}
```

Computes: θ = Σᵢ s₁ⁱ...sₖⁱ × W^i[:, x₁ⁱ, ..., xₘⁱ]

---

## Helper Functions

### `categorical_sample`

Sample from categorical distribution using Gumbel-max trick.

```rust
pub fn categorical_sample(
    logits: Tensor<WgpuBackend, 2>,  // [n_nodes, n_categories]
    device: &WgpuDevice,
) -> Tensor<WgpuBackend, 1>;  // [n_nodes] indices
```

This is equivalent to `jax.random.categorical(key, logits, axis=-1)`.

---

## Max-Cut Graph Partitioning

The `maxcut` module provides CPU-based f64 implementations of max-cut algorithms using Gibbs sampling. Useful for graph partitioning where high precision is required.

### `maxcut_gibbs`

Gibbs/Metropolis-Hastings sampling for max-cut:

```rust
use thrml_samplers::maxcut::{maxcut_gibbs, cut_value};

// 4-node cycle graph
let weights = vec![
    vec![0.0, 1.0, 0.0, 1.0],
    vec![1.0, 0.0, 1.0, 0.0],
    vec![0.0, 1.0, 0.0, 1.0],
    vec![1.0, 0.0, 1.0, 0.0],
];

let partition = maxcut_gibbs(&weights, 100, 2.0, 42);
// partition: Vec<i8> with values +1 or -1

let value = cut_value(&weights, &partition);
// value = 4.0 for optimal alternating partition
```

**Parameters:**

- `weights` - Symmetric weight matrix `J[i][j]` for edge (i,j)
- `n_sweeps` - Number of full sweeps through all nodes
- `beta` - Inverse temperature (higher = greedier)
- `seed` - Random seed for reproducibility

### `maxcut_multistart`

Run multiple random restarts and return the best partition:

```rust
use thrml_samplers::maxcut::maxcut_multistart;

let (best_partition, best_value) = maxcut_multistart(
    &weights, 
    100,    // n_sweeps per restart
    2.0,    // beta
    10,     // n_restarts
    42,     // seed
);

println!("Best cut value: {}", best_value);
```

### `maxcut_greedy`

Fast greedy local search (less optimal but fast):

```rust
use thrml_samplers::maxcut::maxcut_greedy;

let partition = maxcut_greedy(&weights, 100, 42);
```

### Utility Functions

```rust
use thrml_samplers::maxcut::{
    cut_value, 
    ising_energy, 
    partition_to_binary, 
    binary_to_partition
};

// Cut value = Σ_{i<j} J[i][j] * (1 - σ_i * σ_j) / 2
let value = cut_value(&weights, &partition);

// Ising energy = -Σ_{i<j} J[i][j] * σ_i * σ_j
let energy = ising_energy(&weights, &partition);

// Convert between {-1, +1} and {0, 1}
let binary = partition_to_binary(&partition);  // Vec<u8>
let spins = binary_to_partition(&binary);      // Vec<i8>
```

---

## Precision Routing

All samplers support precision routing via `*_routed()` methods that dispatch based on hardware capabilities.

### Routed Sampling

```rust
use thrml_samplers::{SpinGibbsConditional, RuntimePolicy, ComputeBackend, OpType};

// Auto-detect hardware
let policy = RuntimePolicy::detect();
let backend = ComputeBackend::from_policy(&policy);

let sampler = SpinGibbsConditional::new();

// Sample with precision routing
let (samples, _) = sampler.sample_routed(
    &backend, key, &interactions, &active, &states, &n_spin, (), &spec, &device
);

// On Apple Silicon: IsingSampling routes to CPU f64
// On HPC GPUs with CUDA: Uses GPU f64 for speed + precision
```

### Available Routed Methods

| Sampler | Method | Op Type |
|---------|--------|---------|
| `SpinGibbsConditional` | `sample_routed()` | `IsingSampling` |
| `CategoricalGibbsConditional` | `sample_routed()` | `CategoricalSampling` |
| `GaussianSampler` | `sample_routed()` | `GradientCompute` |
| `langevin` | `langevin_step_2d_routed()` | `LangevinStep` |

These use the same routing logic as `ComputeBackend::use_cpu(op_type, size)` to determine CPU vs GPU execution.
