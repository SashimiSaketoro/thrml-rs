# Core Types

The `thrml-core` crate provides fundamental types for building probabilistic graphical models.

## Nodes

### `NodeType`

Enum representing the type of a node in the graphical model.

```rust
pub enum NodeType {
    /// Binary spin variable with values {0, 1} (representing {-1, +1})
    Spin,
    /// Categorical variable with n possible values {0, 1, ..., n-1}
    Categorical { n_categories: u8 },
}
```

**Example:**

```rust
use thrml_core::node::NodeType;

let spin = NodeType::Spin;
let categorical = NodeType::Categorical { n_categories: 5 };
```

### `Node`

A node in the graphical model with a unique ID and type.

```rust
pub struct Node {
    id: usize,
    node_type: NodeType,
}

impl Node {
    pub fn new(node_type: NodeType) -> Self;
    pub fn id(&self) -> usize;
    pub fn node_type(&self) -> &NodeType;
}
```

**Example:**

```rust
use thrml_core::node::{Node, NodeType};

let spin_node = Node::new(NodeType::Spin);
let cat_node = Node::new(NodeType::Categorical { n_categories: 3 });

println!("Spin node ID: {}", spin_node.id());
```

### `TensorSpec`

Specification for tensor shape and dtype, analogous to JAX's `ShapeDtypeStruct`.

```rust
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: burn::tensor::DType,
}
```

---

## Blocks

### `Block`

A collection of nodes of the same type with implicit ordering.

```rust
pub struct Block {
    nodes: Vec<Node>,
}

impl Block {
    pub fn new(nodes: Vec<Node>) -> Self;
    pub fn nodes(&self) -> &[Node];
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn node_type(&self) -> &NodeType;
}
```

**Example:**

```rust
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};

let nodes: Vec<Node> = (0..10)
    .map(|_| Node::new(NodeType::Spin))
    .collect();

let block = Block::new(nodes);
assert_eq!(block.len(), 10);
```

---

## BlockSpec

### `BlockSpec`

Manages the global state representation and index mappings for blocks.

```rust
pub struct BlockSpec {
    pub blocks: Vec<Block>,
    pub node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
    pub node_global_location_map: IndexMap<Node, (usize, usize)>,
}

impl BlockSpec {
    pub fn new(
        blocks: Vec<Block>,
        node_shape_dtypes: IndexMap<NodeType, TensorSpec>,
    ) -> Result<Self, String>;
    
    pub fn get_node_locations(&self, block: &Block) -> Result<(usize, Vec<usize>), String>;
}
```

The `node_global_location_map` maps each node to `(state_index, position)` for efficient state lookup.

**Example:**

```rust
use thrml_core::blockspec::BlockSpec;
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType, TensorSpec};
use indexmap::IndexMap;

let nodes: Vec<Node> = (0..4).map(|_| Node::new(NodeType::Spin)).collect();
let block1 = Block::new(nodes[0..2].to_vec());
let block2 = Block::new(nodes[2..4].to_vec());

let mut dtypes = IndexMap::new();
dtypes.insert(NodeType::Spin, TensorSpec {
    shape: vec![2],
    dtype: burn::tensor::DType::Bool,
});

let spec = BlockSpec::new(vec![block1, block2], dtypes)?;
```

---

## InteractionGroup

### `InteractionGroup`

Defines computational dependencies for conditional sampling updates.

```rust
pub struct InteractionGroup {
    /// Nodes to update
    pub head_nodes: Block,
    /// Nodes providing neighbor state information
    pub tail_nodes: Vec<Block>,
    /// Weight tensor [n_nodes, dim1, dim2]
    pub interaction: Tensor<WgpuBackend, 3>,
    /// Number of spin tail blocks (first n_spin are spin, rest categorical)
    pub n_spin: usize,
}

impl InteractionGroup {
    pub fn new(
        interaction: Tensor<WgpuBackend, 3>,
        head_nodes: Block,
        tail_nodes: Vec<Block>,
        n_spin: usize,
    ) -> Result<Self, String>;
}
```

**Validation:**

- All tail blocks must have the same length as `head_nodes`
- The interaction tensor's first dimension must equal `head_nodes.len()`
- `n_spin` cannot exceed the number of tail blocks

---

## State Utilities

### `block_state_to_global`

Convert block-arranged state to global representation.

```rust
pub fn block_state_to_global<L: StateLeaf>(
    block_state: &[L],
    spec: &BlockSpec,
) -> Vec<L>;
```

### `from_global_state`

Extract block state from global representation.

```rust
pub fn from_global_state<L: StateLeaf>(
    global_state: &[L],
    spec: &BlockSpec,
    blocks: &[Block],
) -> Vec<L>;
```

### `make_empty_block_state`

Create an empty block state for given blocks.

```rust
pub fn make_empty_block_state(
    blocks: &[Block],
    node_shape_dtypes: &IndexMap<NodeType, TensorSpec>,
    batch_shape: Option<&[usize]>,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Vec<Tensor<WgpuBackend, 1>>;
```

### `verify_block_state`

Validate that a block state matches expected structure.

```rust
pub fn verify_block_state(
    blocks: &[Block],
    states: &[Tensor<WgpuBackend, 1>],
    node_shape_dtypes: &IndexMap<NodeType, TensorSpec>,
    block_axis: Option<usize>,
) -> Result<(), String>;
```

This validates:
- Number of states equals number of blocks
- Each state tensor has correct shape for its block
- State dtype is compatible with node type

**Example:**

```rust
use thrml_core::state_tree::verify_block_state;

// Validate states before sampling
verify_block_state(&blocks, &states, &node_shape_dtypes, None)?;
```

---

## GPU Backend

### `init_gpu_device`

Initialize the default GPU device.

```rust
pub fn init_gpu_device() -> WgpuDevice;
```

### `ensure_backend`

Ensure Metal backend is used on macOS (for debugging).

```rust
pub fn ensure_backend();
```

**Example:**

```rust
use thrml_core::backend::{init_gpu_device, ensure_backend};

ensure_backend();  // Optional: verify Metal on macOS
let device = init_gpu_device();
```

---

## Hybrid Compute Backend

The hybrid compute system enables intelligent routing between CPU (f64 precision) and GPU (f32 performance), optimized for Apple Silicon's unified memory architecture.

### `ComputeBackend`

Backend selection strategy for compute operations.

```rust
pub enum ComputeBackend {
    /// Always use GPU (default for most operations)
    GpuOnly { device: Option<WgpuDevice> },
    /// Always use CPU (for precision-sensitive ops)
    CpuOnly,
    /// Unified memory: share data between CPU/GPU without copies
    UnifiedHybrid { threshold_bytes: usize },
    /// Runtime decision based on operation type and data size
    Adaptive { threshold_ops: usize },
}

impl ComputeBackend {
    /// Auto-detect Apple Silicon unified memory
    pub fn apple_silicon() -> Self;
    
    /// Check if operation should use CPU path
    pub fn use_cpu(&self, op_type: OpType, size_hint: Option<usize>) -> bool;
    
    /// Create GPU-only backend with device
    pub fn gpu_only() -> Self;
}
```

**Example:**

```rust
use thrml_core::{ComputeBackend, OpType};

// Auto-detect Apple Silicon
let backend = ComputeBackend::apple_silicon();

// Route precision-sensitive operations to CPU
if backend.use_cpu(OpType::IsingSampling, Some(1000)) {
    // Use CPU f64 implementation
} else {
    // Use GPU f32 implementation
}
```

### `OpType`

Operation classification for routing decisions.

```rust
pub enum OpType {
    /// Ising partition function / sampling (benefits from f64)
    IsingSampling,
    /// Energy computation (usually fine with f32)
    EnergyCompute,
    /// Small matrix operations (overhead not worth GPU)
    SmallMatmul,
    /// Langevin dynamics (precision helps stability)
    LangevinStep,
    /// Similarity computation (fine with f32)
    SimilarityCompute,
    /// Large batch operations (GPU wins)
    LargeBatch,
}
```

### `PrecisionMode`

Precision mode selection with adaptive thresholds.

```rust
pub enum PrecisionMode {
    /// Fast GPU path (f32)
    GpuFast,
    /// Precise CPU path (f64)
    CpuPrecise,
    /// Adaptive based on operation characteristics
    Adaptive {
        small_angle_threshold: f32,
        langevin_renorm_interval: usize,
    },
}

impl PrecisionMode {
    /// Check if angle is small enough to need precision
    pub fn needs_precision(&self, angle: f32) -> bool;
    
    /// Check if Langevin should renormalize at this step
    pub fn should_renormalize(&self, step: usize) -> bool;
}
```

### `HybridConfig`

Combined backend + precision configuration.

```rust
pub struct HybridConfig {
    pub backend: ComputeBackend,
    pub precision: PrecisionMode,
}

impl HybridConfig {
    /// Default for Apple Silicon (unified memory + adaptive precision)
    pub fn apple_silicon() -> Self;
    
    /// Default for discrete GPU systems
    pub fn discrete_gpu() -> Self;
    
    /// CPU-only configuration
    pub fn cpu_only() -> Self;
}
```

**Example:**

```rust
use thrml_core::HybridConfig;

// Configure for Apple Silicon M1-M4
let config = HybridConfig::apple_silicon();

// Or for NVIDIA/AMD discrete GPUs
let config = HybridConfig::discrete_gpu();
```

### `RuntimePolicy`

Auto-detect hardware and create precision-appropriate configuration:

```rust
use thrml_core::compute::{RuntimePolicy, HardwareTier, PrecisionProfile};

// Auto-detect hardware
let policy = RuntimePolicy::detect();

println!("Tier: {:?}", policy.tier);        // e.g., AppleSilicon
println!("Profile: {:?}", policy.profile);  // e.g., CpuFp64Strict
println!("Use GPU: {}", policy.use_gpu);

// Create backend from policy
let backend = ComputeBackend::from_policy(&policy);

// Tier-specific constructors
let apple = RuntimePolicy::apple_silicon();
let hpc = RuntimePolicy::nvidia_hopper();    // H100/H200
let spark = RuntimePolicy::nvidia_spark();   // DGX Spark / GB10
let consumer = RuntimePolicy::nvidia_consumer();
```

### `HardwareTier`

Hardware classification for runtime configuration:

```rust
pub enum HardwareTier {
    AppleSilicon,      // M1–M4 unified memory
    NvidiaConsumer,    // RTX 3080–5090
    NvidiaHopper,      // H100, H200
    NvidiaBlackwell,   // B200
    NvidiaSpark,       // DGX Spark / GB10
    AmdRdna,           // RX 7900 series
    CpuOnly,           // No GPU
    Unknown,           // Fallback
}
```

### `PrecisionProfile`

Precision strategy per hardware tier:

```rust
pub enum PrecisionProfile {
    /// Route precision-sensitive ops to CPU f64 (Apple Silicon, consumer GPUs)
    CpuFp64Strict,
    /// GPU f32 with CPU f64 fallback for specific ops
    GpuMixed,
    /// Full f64 on GPU (HPC: H100, B200, Spark)
    GpuHpcFp64,
}
```

### Testing Utilities

```rust
use thrml_core::compute::{test_both_backends, recommended_tolerance};

// Test function on both CPU and GPU backends
test_both_backends(|backend| {
    let result = my_computation(backend);
    let tolerance = recommended_tolerance(backend);
    assert!((result - expected).abs() < tolerance);
});
```

---

## Retrieval Metrics

The `metrics` module provides standard information retrieval metrics for evaluating ranking quality.

### Core Functions

```rust
use thrml_core::metrics::{recall_at_k, mrr, ndcg, find_rank};

let retrieved = vec![5, 2, 8, 1, 9];
let target = 8;

// Recall@k: 1.0 if target in top-k, else 0.0
assert_eq!(recall_at_k(&retrieved, target, 3), 1.0);  // Found at position 3
assert_eq!(recall_at_k(&retrieved, target, 2), 0.0);  // Not in top 2

// MRR: 1/rank (1-indexed)
assert!((mrr(&retrieved, target) - 1.0/3.0).abs() < 1e-6);  // Rank 3

// nDCG: Normalized DCG for binary relevance
assert!((ndcg(&retrieved, target, 5) - 0.5).abs() < 1e-6);  // 1/log2(4)

// Find rank (returns Option<usize>)
assert_eq!(find_rank(&retrieved, target), Some(3));
```

### Batch Evaluation

```rust
use thrml_core::metrics::{evaluate_retrieval, RetrievalMetrics};

let results = vec![
    (vec![1, 2, 3, 4, 5], 3),  // Target 3 at rank 3
    (vec![5, 4, 3, 2, 1], 5),  // Target 5 at rank 1
    (vec![1, 2, 3, 4, 5], 9),  // Target 9 not found
];

let metrics = evaluate_retrieval(&results, &[1, 3, 5, 10]);

println!("{}", metrics);  // Pretty-printed summary
// Retrieval Metrics (n=3)
//   MRR:      0.4444
//   Recall@1: 0.3333
//   Recall@3: 0.6667
//   ...
```

### Multi-Relevance nDCG

```rust
use thrml_core::metrics::ndcg_multi;

let retrieved = vec![3, 1, 4, 2, 5];
let relevant = vec![1, 4];  // Multiple relevant items

let score = ndcg_multi(&retrieved, &relevant, 5);
// Accounts for both items' positions
```

---

## Text Similarity

The `text` module provides efficient text/byte similarity primitives based on n-gram hashing.

### Rolling Hash

Efficient O(1) sliding window hash for n-grams:

```rust
use thrml_core::text::RollingHash;

let mut hasher = RollingHash::new(3);  // 3-gram
hasher.init(b"abc");

let hash1 = hasher.value();
hasher.roll(b'a', b'd');  // Now hashing "bcd"
let hash2 = hasher.value();

assert_ne!(hash1, hash2);
```

### N-gram Hashing

```rust
use thrml_core::text::{ngram_hashes, ngram_hashes_with_length};

// Compute all n-gram hashes for a byte sequence
let hashes = ngram_hashes(b"hello world", 3, 5);
// Contains hashes for all 3-grams, 4-grams, and 5-grams

// With length info for multi-scale comparison
let hashes_with_len = ngram_hashes_with_length(b"hello", 2, 3);
// (2, hash("he")), (2, hash("el")), ..., (3, hash("hel")), ...
```

### Jaccard Similarity

```rust
use thrml_core::text::{ngram_hashes, jaccard_similarity};
use std::collections::HashSet;

let a = ngram_hashes(b"hello world", 3, 5);
let b = ngram_hashes(b"hello there", 3, 5);

let similarity = jaccard_similarity(&a, &b);
// |A ∩ B| / |A ∪ B|
```

### Substring Containment

```rust
use thrml_core::text::{contains_subsequence, check_containment};

assert!(contains_subsequence(b"hello world", b"lo wo"));
assert!(!contains_subsequence(b"hello", b"world"));

// Check mutual containment
if let Some((a_in_b, container_len, contained_len)) = check_containment(b"hello", b"ell") {
    println!("Container: {} bytes, contained: {} bytes", container_len, contained_len);
}
```

### High-Level Text Similarity

```rust
use thrml_core::text::{text_similarity, TextSimilarityConfig};

let config = TextSimilarityConfig::default()
    .with_ngram_range(4, 64)
    .with_weights(0.7, 0.3);  // embedding_weight, text_weight

let sim = text_similarity(b"function calculate_total", b"calculate_total_price", &config);
```

