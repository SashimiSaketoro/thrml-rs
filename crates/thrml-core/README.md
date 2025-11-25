# thrml-core

Core types and GPU backend for the THRML probabilistic computing library.

## Overview

This crate provides the foundational types for building probabilistic graphical models:

- **`Node`**: Represents a random variable in the graph
- **`NodeType`**: Spin (binary ±1) or Categorical variables
- **`Block`**: A collection of nodes of the same type
- **`BlockSpec`**: Specification for mapping between local and global state
- **`InteractionGroup`**: Defines interactions between node groups

## GPU Backend

The `gpu` feature (enabled by default) provides GPU acceleration via WGPU:

```rust
use thrml_core::backend::{init_gpu_device, ensure_metal_backend};

// Initialize GPU
ensure_metal_backend();
let device = init_gpu_device();
```

## Usage

```rust
use thrml_core::{Node, NodeType, Block};

// Create spin nodes
let nodes: Vec<Node> = (0..10)
    .map(|_| Node::new(NodeType::Spin))
    .collect();

// Group into a block
let block = Block::new(nodes).unwrap();
```

## License

MIT OR Apache-2.0

