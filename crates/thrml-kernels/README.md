# thrml-kernels

**Fused CubeCL GPU kernels for THRML**

This crate provides optimized, fused GPU kernels for performance-critical operations in probabilistic graphical model sampling. By fusing multiple tensor operations into single GPU kernel launches, these kernels eliminate intermediate memory allocations and reduce kernel launch overhead.

## Features

- **Gumbel-Max Argmax**: Fused categorical sampling using the Gumbel-max trick
- **Sigmoid-Bernoulli**: Fused spin variable sampling (sigmoid + Bernoulli in one kernel)
- **Batch Gather**: Fused multi-index weight gathering without intermediate allocations

## Usage

This crate is an **optional optimization** for `thrml-samplers` and `thrml-models`. Enable via feature flag:

```toml
[dependencies]
thrml-samplers = { version = "0.1", features = ["fused-kernels"] }
```

When enabled, sampling operations automatically use fused kernels where available.

## Architecture

Each kernel follows Burn's custom CubeCL pattern:
- `kernel.rs` - `#[cube(launch)]` GPU kernel definition
- `forward.rs` - `CubeBackend` implementation for forward pass
- `backward.rs` - `Autodiff<B>` implementation for gradient computation

## Benchmarks

Run benchmarks to compare fused vs. reference implementations:

```bash
cargo bench --package thrml-kernels --features gpu
```

## License

Licensed under MIT OR Apache-2.0.

