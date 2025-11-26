# thrml-examples

Example programs demonstrating THRML-RS capabilities.

## Path Configuration

All examples support configurable paths for cache, data, and output directories.
This is useful when working with external storage (e.g., external SSD/NVMe drives).

### CLI Flags

```bash
# Use a base directory (creates cache/, data/, output/ subdirectories)
cargo run --release --features gpu --example train_mnist -- \
    --base-dir /Volumes/ExternalDisk/thrml

# Or set individual directories
cargo run --release --features gpu --example train_mnist -- \
    --data-dir /Volumes/ExternalDisk/data \
    --output-dir /Volumes/ExternalDisk/output \
    --cache-dir /Volumes/ExternalDisk/cache
```

### Environment Variables

```bash
# Set base directory for all THRML files
export THRML_BASE_DIR=/Volumes/ExternalDisk/thrml

# Or individual directories
export THRML_DATA_DIR=/Volumes/ExternalDisk/data
export THRML_OUTPUT_DIR=/Volumes/ExternalDisk/output
export THRML_CACHE_DIR=/Volumes/ExternalDisk/cache

cargo run --release --features gpu --example train_mnist
```

### Config File

Create `~/.config/thrml/config.toml`:

```toml
base_dir = "/Volumes/ExternalDisk/thrml"
# Or individual paths:
# cache_dir = "/Volumes/ExternalDisk/thrml/cache"
# data_dir = "/Volumes/ExternalDisk/thrml/data"
# output_dir = "/Volumes/ExternalDisk/thrml/output"
```

### Priority Order

1. CLI arguments (highest)
2. Environment variables
3. Config file
4. Default system directories

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
- Full CLI configuration for all hyperparameters

```bash
# Basic run
cargo run --release --example train_mnist --features gpu

# With fused GPU kernels (faster)
cargo run --release --example train_mnist --features gpu,fused-kernels

# Full configuration
cargo run --release --features gpu,fused-kernels --example train_mnist -- \
    --base-dir /Volumes/ExternalDisk/thrml \
    --epochs 2000 \
    --learning-rate 0.0003 \
    --batch-size 16 \
    --warmup-neg 100 --samples-neg 100 \
    --warmup-pos 100 --samples-pos 100 \
    --eval-every 100

# View all options
cargo run --features gpu --example train_mnist -- --help
```

#### Training Parameters

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--epochs` | `-e` | 1000 | Number of training epochs |
| `--batch-size` | `-b` | 32 | Batch size |
| `--learning-rate` | `-l` | 0.001 | Learning rate |
| `--side-len` | | 40 | Hidden grid size (40×40) |
| `--jumps` | | 1,4,15 | Connection distances |
| `--warmup-neg` | | 50 | Negative phase warmup |
| `--samples-neg` | | 50 | Negative phase samples |
| `--steps-neg` | | 5 | Steps per neg sample |
| `--warmup-pos` | | 50 | Positive phase warmup |
| `--samples-pos` | | 50 | Positive phase samples |
| `--steps-pos` | | 5 | Steps per pos sample |
| `--seed` | `-s` | 42 | Random seed |
| `--target-classes` | | 0,3,4 | MNIST digits to train on |
| `--label-spots` | | 10 | Label nodes per class |
| `--eval-every` | | 50 | Evaluation frequency |

## Output

Visualization examples save PNG files to the `output/` directory in the workspace root.

## Dependencies

The examples use additional visualization libraries:
- `plotters`: Line charts, heatmaps
- `petgraph`: Graph data structures

