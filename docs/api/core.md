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

