# thrml-viz

Interactive 3D visualization for THRML sphere embeddings and ROOTS hierarchy.

## Features

- **3D Point Cloud Rendering**: Visualize sphere-optimized embeddings with prominence/radius coloring
- **ROOTS Hierarchy View**: See the H-ROOTS tree structure as connected nodes
- **Dual View Mode**: Compare raw embeddings (PCA) vs optimized sphere positions
- **Live Monitoring**: Watch ingestion progress in real-time via IPC
- **Interactive Camera**: Orbit, zoom, and pan controls

## Usage

### Standalone Viewing

```bash
# View SafeTensors file
cargo run --example visualize -- --input output.safetensors

# View sphere coordinates (NPZ)
cargo run --example visualize -- --input sphere.npz
```

### Live Monitoring

```bash
# Terminal 1: Start visualizer
cargo run --example visualize -- --monitor my-session

# Terminal 2: Run ingestion with viz notifications
blt-burn ingest --viz-session my-session input.txt
```

## Controls

| Action | Input |
|--------|-------|
| Orbit | Left-click drag |
| Pan | Right-click drag |
| Zoom | Scroll wheel |
| Toggle help | H key |

## Architecture

```
┌─────────────────┐                    ┌─────────────────┐
│   INGESTION     │                    │   VISUALIZER    │
│                 │    fire-and-forget │                 │
│  write tensors ─┼───────────────────→│  receive notify │
│  (never waits)  │    (unix socket)   │  lazy mmap load │
└─────────────────┘                    └─────────────────┘
```

The visualizer uses:
- **eframe + egui**: Cross-platform GUI framework
- **wgpu**: GPU-accelerated rendering (same backend as Burn)
- **Unix sockets**: Zero-overhead IPC for live monitoring
- **Memory-mapped files**: Lazy loading for large datasets

## Dependencies

This crate uses the same wgpu backend as Burn, enabling future GPU context sharing.

## License

MIT OR Apache-2.0
