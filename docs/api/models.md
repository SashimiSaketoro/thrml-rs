# Models

The `thrml-models` crate provides energy-based model implementations and the factor system.

## Factor System

### `AbstractFactor`

Base trait for factors that can be converted to interaction groups.

```rust
pub trait AbstractFactor {
    fn node_groups(&self) -> &[Block];
    fn to_interaction_groups(&self, device: &WgpuDevice) -> Vec<FactorInteractionGroup>;
}
```

### `WeightedFactor`

Trait for factors parameterized by weights.

```rust
pub trait WeightedFactor: AbstractFactor {
    fn get_weights(&self) -> &Tensor<WgpuBackend, 3>;
}
```

### `FactorInteractionGroup`

Interaction group with DiscreteEBMInteraction weights.

```rust
pub struct FactorInteractionGroup {
    pub interaction: DiscreteEBMInteraction,
    pub head_nodes: Block,
    pub tail_nodes: Vec<Block>,
}
```

### `FactorSamplingProgram`

Convenient wrapper that builds a `BlockSamplingProgram` from factors.

```rust
pub struct FactorSamplingProgram {
    pub program: BlockSamplingProgram,
}

impl FactorSamplingProgram {
    pub fn new(
        gibbs_spec: BlockGibbsSpec,
        samplers: Vec<Box<dyn AbstractConditionalSampler>>,
        factors: Vec<Box<dyn AbstractFactor>>,
        other_interaction_groups: Vec<InteractionGroup>,
        device: &WgpuDevice,
    ) -> Result<Self, String>;
}
```

---

## Energy-Based Models

### `AbstractEBM`

Base trait for energy-based models.

```rust
pub trait AbstractEBM {
    fn energy(
        &self,
        state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;
}
```

### `EBMFactor`

Trait for factors that define energy contributions.

```rust
pub trait EBMFactor: AbstractFactor {
    fn factor_energy(
        &self,
        global_state: &[Tensor<WgpuBackend, 1>],
        block_spec: &BlockSpec,
        device: &WgpuDevice,
    ) -> Tensor<WgpuBackend, 1>;
}
```

### `AbstractFactorizedEBM`

Trait for EBMs composed of independent factors.

```rust
pub trait AbstractFactorizedEBM: AbstractEBM {
    fn get_factors(&self, device: &WgpuDevice) -> Vec<Box<dyn EBMFactor>>;
}
```

### `FactorizedEBM`

Generic factorized EBM implementation.

```rust
pub struct FactorizedEBM {
    pub factors: Vec<Box<dyn EBMFactor>>,
}

impl FactorizedEBM {
    pub fn new(factors: Vec<Box<dyn EBMFactor>>) -> Self;
}
```

---

## Discrete EBM

### `DiscreteEBMInteraction`

Interaction data for discrete EBMs.

```rust
pub struct DiscreteEBMInteraction {
    pub n_spin: usize,
    pub weights: Tensor<WgpuBackend, 3>,
}

impl DiscreteEBMInteraction {
    pub fn new(n_spin: usize, weights: Tensor<WgpuBackend, 3>) -> Self;
}
```

### `DiscreteEBMFactor`

Factor for discrete energy-based models with spin and categorical variables.

Implements energy: E(x) = -β × Σ s₁...sₘ × W[c₁, ..., cₙ]

```rust
pub struct DiscreteEBMFactor {
    pub spin_node_groups: Vec<Block>,
    pub categorical_node_groups: Vec<Block>,
    pub weights: Tensor<WgpuBackend, 3>,
}
```

### Specialized Factor Types

```rust
/// Spin-only EBM factor
pub struct SpinEBMFactor {
    inner: SquareDiscreteEBMFactor,
}

/// Categorical-only EBM factor  
pub struct CategoricalEBMFactor {
    inner: DiscreteEBMFactor,
}

/// Square weight tensor optimization.
/// 
/// When a discrete factor has a square weight tensor (shape [b, x, x, ..., x]),
/// interaction groups can be merged for improved GPU efficiency.
/// The `to_interaction_groups` method automatically merges compatible groups.
pub struct SquareDiscreteEBMFactor {
    inner: DiscreteEBMFactor,
}

/// Square categorical factor
pub struct SquareCategoricalEBMFactor {
    inner: SquareDiscreteEBMFactor,
}
```

---

## Ising Models

### `IsingEBM`

Ising model (quadratic spin model / Boltzmann machine).

Energy: E(s) = -β × (Σᵢ hᵢsᵢ + Σᵢⱼ Jᵢⱼsᵢsⱼ)

```rust
pub struct IsingEBM {
    nodes: Vec<Node>,
    edges: Vec<(Node, Node)>,
    biases: Tensor<WgpuBackend, 1>,
    edge_weights: Tensor<WgpuBackend, 1>,
    beta: f32,
}

impl IsingEBM {
    pub fn new(
        nodes: Vec<Node>,
        edges: Vec<(Node, Node)>,
        biases: Tensor<WgpuBackend, 1>,
        edge_weights: Tensor<WgpuBackend, 1>,
        beta: f32,
    ) -> Self;
}
```

### `IsingSamplingProgram`

Pre-configured sampling program for Ising models.

```rust
pub struct IsingSamplingProgram {
    pub program: BlockSamplingProgram,
}

impl IsingSamplingProgram {
    pub fn new(
        ebm: &IsingEBM,
        free_blocks: Vec<Block>,
        clamped_blocks: Vec<Block>,
        device: &WgpuDevice,
    ) -> Result<Self, String>;
}
```

### `IsingTrainingSpec`

Specification for training Ising models.

```rust
pub struct IsingTrainingSpec {
    pub ebm: IsingEBM,
    pub program: IsingSamplingProgram,
    pub data_blocks: Vec<Block>,
    pub all_blocks: Vec<Block>,
}

impl IsingTrainingSpec {
    pub fn new(
        ebm: IsingEBM,
        free_blocks: Vec<Block>,
        clamped_blocks: Vec<Block>,
        data_blocks: Vec<Block>,
        device: &WgpuDevice,
    ) -> Result<Self, String>;
}
```

---

## Ising Training Functions

### `hinton_init`

Initialize states using Hinton's method (random ±1).

```rust
pub fn hinton_init(
    key: RngKey,
    ebm: &IsingEBM,
    free_blocks: &[Block],
    device: &WgpuDevice,
) -> Vec<Tensor<WgpuBackend, 1>>;
```

### `estimate_moments`

Estimate first and second moments of the Ising distribution.

```rust
pub fn estimate_moments(
    key: RngKey,
    training_spec: &IsingTrainingSpec,
    schedule: &SamplingSchedule,
    init_state: Vec<Tensor<WgpuBackend, 1>>,
    clamp_state: &[Tensor<WgpuBackend, 1>],
    device: &WgpuDevice,
) -> MomentSpec;
```

Returns `MomentSpec` with `first_moments` and `second_moments` tensors.

### `estimate_kl_grad`

Estimate KL-divergence gradients for training.

```rust
pub fn estimate_kl_grad(
    key: RngKey,
    training_spec: &IsingTrainingSpec,
    schedule: &SamplingSchedule,
    init_state_model: Vec<Tensor<WgpuBackend, 1>>,
    init_state_data: Vec<Tensor<WgpuBackend, 1>>,
    clamp_data: &[Tensor<WgpuBackend, 1>],
    device: &WgpuDevice,
) -> KLGradSpec;
```

Returns gradients for biases and edge weights.

---

## Helper Functions

### `batch_gather`

Index tensor by multiple categorical indices.

```rust
pub fn batch_gather(
    weights: &Tensor<WgpuBackend, 3>,
    indices: &[Tensor<WgpuBackend, 2>],
    device: &WgpuDevice,
) -> Tensor<WgpuBackend, 2>;
```

Equivalent to Python's `_batch_gather(x, *idx)`.

### `batch_gather_with_k`

Batch gather preserving an inner dimension.

```rust
pub fn batch_gather_with_k(
    weights: &Tensor<WgpuBackend, 3>,
    indices: &[Tensor<WgpuBackend, 2>],
    device: &WgpuDevice,
) -> Tensor<WgpuBackend, 3>;
```

Equivalent to Python's `_batch_gather_with_k(x, *idx)`.

### `spin_product`

Compute element-wise product of spin values.

```rust
pub fn spin_product(
    spin_vals: &[Tensor<WgpuBackend, 1>],
    device: &WgpuDevice,
) -> Tensor<WgpuBackend, 1>;
```

Converts {0, 1} → {-1, +1} and computes Π(2s - 1).

### `split_states`

Split neighbor states into spin and categorical.

```rust
pub fn split_states(
    states: &[Tensor<WgpuBackend, 2>],
    n_spin: usize,
) -> (Vec<Tensor<WgpuBackend, 2>>, Vec<Tensor<WgpuBackend, 2>>);
```

