# Contributing to thrml-rs

Thank you for your interest in contributing to thrml-rs! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Rust 1.75 or later
- A GPU with Metal (macOS) or Vulkan (Linux/Windows) support for GPU tests

### Setting Up the Development Environment

```bash
# Clone the repository
git clone https://github.com/sashimisaketoro/thrml-rs.git
cd thrml-rs

# Build the project
cargo build --workspace --features gpu

# Run tests
cargo test --workspace --features gpu
```

## Development Workflow

### Code Style

We use standard Rust formatting and linting:

```bash
# Format code
cargo fmt --all

# Run clippy
cargo clippy --workspace --features gpu -- -D warnings
```

### Running Tests

```bash
# All tests (requires GPU)
cargo test --workspace --features gpu

# CPU-only tests (CI safe)
cargo test --workspace --no-default-features

# Specific crate
cargo test -p thrml-models --features gpu

# With output
cargo test --features gpu -- --nocapture

# Full MNIST training test (takes ~1min)
cargo test -p thrml-models --test test_train_mnist --features gpu -- --ignored --nocapture
```

### Running Examples

```bash
# Simple Ising chain
cargo run --release --example ising_chain --features gpu

# Spin models with benchmarking
cargo run --release --example spin_models --features gpu

# Categorical sampling with visualization
cargo run --release --example categorical_sampling --features gpu

# Full API walkthrough
cargo run --release --example full_api_walkthrough --features gpu

# MNIST training
cargo run --release --example train_mnist --features gpu
```

### Building Documentation

```bash
# Build rustdoc
cargo doc --workspace --no-deps --features gpu --open
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`.

2. **Make your changes** following the code style guidelines.

3. **Add tests** for any new functionality.

4. **Update documentation** if you've changed APIs or added features.

5. **Run the full test suite** to ensure nothing is broken:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --features gpu -- -D warnings
   cargo test --workspace --features gpu
   ```

6. **Submit a pull request** with a clear description of the changes.

## Reporting Issues

When reporting issues, please include:

- Rust version (`rustc --version`)
- Operating system and GPU
- Minimal reproduction case
- Expected vs actual behavior

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct). Please be respectful and constructive in all interactions.

## License

By contributing to thrml-rs, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).

