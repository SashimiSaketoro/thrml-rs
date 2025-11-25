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

