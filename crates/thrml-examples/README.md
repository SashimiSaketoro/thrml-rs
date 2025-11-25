# thrml-examples

Example programs demonstrating THRML-RS capabilities.

## Examples

### `ising_chain`
Simple Ising chain demonstration showing GPU initialization, spin nodes, and Hinton initialization.

```bash
cargo run --release --example ising_chain --features gpu
```

### `spin_models`
Comprehensive Ising model example with:
- 10×10 lattice graph generation
- Graph coloring for block partitioning
- Full Gibbs sampling loop
- Performance benchmarking
- Line chart visualization

```bash
cargo run --release --example spin_models --features gpu
```

### `categorical_sampling`
Categorical variable sampling with:
- Grid graph generation
- Potts model coupling
- Bipartite coloring
- Heatmap visualization

```bash
cargo run --release --example categorical_sampling --features gpu
```

### `full_api_walkthrough`
Comprehensive tutorial covering the entire THRML API:
- Node types (Spin, Categorical)
- Block management
- Graph generation and coloring
- RNG key management
- Observers

```bash
cargo run --release --example full_api_walkthrough --features gpu
```

### `train_mnist`
Full MNIST training example demonstrating:
- Double-grid Ising architecture
- Contrastive Divergence training
- KL-divergence gradient estimation
- Multi-epoch training with progress reporting

```bash
cargo run --release --example train_mnist --features gpu
```

## Output

Visualization examples save PNG files to the `output/` directory in the workspace root.

## Dependencies

The examples use additional visualization libraries:
- `plotters`: Line charts, heatmaps
- `petgraph`: Graph data structures

