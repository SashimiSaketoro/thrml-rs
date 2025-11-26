# Developer Documentation

## What is THRML-RS?

THRML-RS is a pure Rust port of [THRML](https://github.com/extropic-ai/thrml), the JAX-based Python package for efficient [block Gibbs sampling](https://proceedings.mlr.press/v15/gonzalez11a/gonzalez11a.pdf) of graphical models. While the Python version leverages JAX's GPU acceleration and PyTree abstractions, THRML-RS achieves the same functionality using the [Burn](https://burn.dev) deep learning framework with native Rust types.

## How does THRML-RS work?

From a user perspective, there are three main components:

### Blocks

Blocks are fundamental to THRML since it implements block sampling. A `Block` is a collection of nodes of the same type with implicit ordering.

```rust
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};

// Create spin nodes
let nodes: Vec<Node> = (0..10)
    .map(|_| Node::new(NodeType::Spin))
    .collect();

// Group into a block
let block = Block::new(nodes);
```

### Factors

Factors organize interactions between variables into a bipartite graph structure. They synthesize collections of interactions via `InteractionGroup` and must implement `to_interaction_groups()`.

```rust
use thrml_models::factor::AbstractFactor;

// DiscreteEBMFactor implements AbstractFactor
let factor = DiscreteEBMFactor::new(
    spin_node_groups,
    categorical_node_groups,
    weights,
);

// Convert to interaction groups for sampling
let groups = factor.to_interaction_groups(&device);
```

### Programs

Programs are the key orchestrating data structures:

- **`BlockSamplingProgram`**: Handles mapping and bookkeeping for padded block Gibbs sampling
- **`FactorSamplingProgram`**: Convenient wrapper that converts factors to interaction groups

```rust
use thrml_samplers::{BlockGibbsSpec, BlockSamplingProgram};

// Create specification
let gibbs_spec = BlockGibbsSpec::new(
    free_super_blocks,
    clamped_blocks,
    node_shape_dtypes,
)?;

// Create sampling program
let program = BlockSamplingProgram::new(
    gibbs_spec,
    samplers,
    interaction_groups,
)?;
```

## Key Differences from Python

### PyTree → Rust Types

Python THRML uses JAX's PyTree abstraction for heterogeneous state. In Rust, we use:

- `NodeType` enum for spin vs categorical distinction
- `TensorSpec` for shape/dtype metadata
- Strongly typed tensors with const generics: `Tensor<B, D, K>`

### JAX Transforms → Burn Operations

| JAX | Rust/Burn |
|-----|-----------|
| `jnp.take(x, indices, axis=0)` | `tensor.select(0, indices)` |
| `jnp.gather(x, indices)` | `tensor.gather(dim, indices)` |
| `jax.random.key(seed)` | `RngKey::new(seed)` |
| `jax.random.split(key)` | `key.split()` |
| `jax.vmap(fn)` | Loop or future batch support |

### Global State Representation

Like Python THRML, the Rust version represents state as contiguous tensors for efficient GPU operations:

```rust
// Block state: list of tensors per block
type BlockState = Vec<Tensor<WgpuBackend, 1>>;

// Global state: concatenated view for efficient indexing
let global_state = block_state_to_global(&block_state, &block_spec);
```

The `BlockSpec` manages the mapping between block-local and global indices.

## Crate Architecture

```
thrml-rs/
├── thrml-core/           # Core types and utilities
│   ├── node.rs           # Node, NodeType, TensorSpec
│   ├── block.rs          # Block
│   ├── blockspec.rs      # BlockSpec, index management
│   ├── interaction.rs    # InteractionGroup
│   ├── state_tree.rs     # State conversion utilities
│   └── backend.rs        # GPU initialization
│
├── thrml-samplers/       # Sampling infrastructure
│   ├── program.rs        # BlockGibbsSpec, BlockSamplingProgram
│   ├── sampling.rs       # sample_blocks, sample_states, etc.
│   ├── schedule.rs       # SamplingSchedule
│   ├── rng.rs            # RngKey (deterministic RNG)
│   ├── sampler.rs        # AbstractConditionalSampler trait
│   ├── bernoulli.rs      # BernoulliConditional
│   ├── softmax.rs        # SoftmaxConditional, CategoricalGibbsConditional
│   └── spin_gibbs.rs     # SpinGibbsConditional
│
├── thrml-models/         # Model implementations
│   ├── factor.rs         # AbstractFactor, FactorSamplingProgram
│   ├── ebm.rs            # AbstractEBM, FactorizedEBM
│   ├── discrete_ebm.rs   # DiscreteEBMFactor, SpinEBMFactor, etc.
│   └── ising.rs          # IsingEBM, IsingSamplingProgram, training
│
├── thrml-observers/      # Observation utilities
│   ├── observer.rs       # AbstractObserver trait
│   ├── state_observer.rs # StateObserver
│   └── moment.rs         # MomentAccumulatorObserver
│
└── thrml-examples/       # Runnable examples
    ├── lib.rs            # Graph utilities (lattice, coloring)
    └── examples/         # categorical_sampling, spin_models, etc.
```

## Factor Hierarchy

```
AbstractFactor
├── WeightedFactor: Parameterized by weights
└── EBMFactor: Defines energy functions
    └── DiscreteEBMFactor: Spin and categorical states
        ├── SquareDiscreteEBMFactor: Square interaction tensors
        │   ├── SpinEBMFactor: Spin-only ({-1, 1})
        │   └── SquareCategoricalEBMFactor
        └── CategoricalEBMFactor: Categorical-only
```

## Sampler Hierarchy

```
AbstractConditionalSampler
├── BernoulliConditional: Spin-valued Bernoulli
│   └── SpinGibbsConditional: Gibbs for spin EBMs
└── SoftmaxConditional: Categorical softmax
    └── CategoricalGibbsConditional: Gibbs for categorical EBMs
```

## GPU Backend

THRML-RS uses [Burn](https://burn.dev) with the WGPU backend, which provides:

- **Metal** support on Apple Silicon (M1/M2/M3/M4)
- **Vulkan** on Linux/Windows
- **DX12** on Windows
- Automatic device selection

```rust
use thrml_core::backend::{init_gpu_device, ensure_backend};

// Ensure Metal is used on macOS
ensure_backend();

// Initialize default GPU device
let device = init_gpu_device();
```

## Tensor Operations

Key operations used throughout:

```rust
// Select rows by indices
let selected = tensor.select(0, indices);

// Gather elements
let gathered = tensor.gather(dim, indices);

// Reshape with const generics
let reshaped: Tensor<B, 3> = tensor.reshape([a, b, c]);

// Squeeze dimension
let squeezed: Tensor<B, 2> = tensor.squeeze_dim::<2>(dim);

// Random sampling
let uniform = Tensor::random(shape, Distribution::Uniform(0.0, 1.0), &device);
```

## Interaction Processing

The core sampling algorithm processes interactions as follows:

1. **Factor → InteractionGroup**: Factors produce `InteractionGroup`s with weight tensors and node mappings

2. **BlockSamplingProgram**: Pre-computes slices for efficient indexing:
   - `per_block_interactions`: Sliced weight tensors per block
   - `per_block_n_spin`: Number of spin tail blocks per interaction
   - `per_block_interaction_global_slices`: Index tensors for neighbor state lookup

3. **Sampler**: Uses `n_spin` to split neighbor states, computes spin products and categorical indexing:
   ```rust
   // Split states into spin and categorical
   let (states_spin, states_cat) = split_states(states, n_spin);
   
   // Compute spin product: Π(2s - 1)
   let spin_prod = compute_spin_product(&states_spin);
   
   // Index weights by categorical neighbors
   let weights = batch_gather(interaction, &states_cat);
   
   // Compute contribution
   let contribution = weights * active * spin_prod;
   ```

## Performance Considerations

1. **Batch Operations**: All tensor operations are batched for GPU efficiency
2. **Pre-computed Indices**: `BlockSamplingProgram` pre-computes all index tensors at construction
3. **Minimal Allocations**: State tensors are reused across sampling iterations
4. **Parallel Blocks**: Superblocks can be sampled in parallel (same algorithmic time)

## Limitations

Like the Python version, THRML-RS inherits fundamental sampling limitations:

- Gibbs sampling can be slow to mix for certain distributions
- Some problems may benefit from other MCMC methods
- Block size affects convergence (larger blocks = fewer updates but more complex conditionals)

For example, a two-node Ising model with `J=-∞, h=0` will never mix between ground states `{-1, -1}` and `{1, 1}` using Gibbs sampling.

