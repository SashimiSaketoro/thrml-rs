//! Benchmarks comparing fused kernels vs reference implementations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

#[cfg(feature = "gpu")]
use burn::tensor::{Distribution, Tensor};

#[cfg(feature = "gpu")]
use thrml_core::backend::{init_gpu_device, WgpuBackend};

#[cfg(feature = "gpu")]
use thrml_kernels::{batch_gather_fused, gumbel_argmax_fused, sigmoid_bernoulli_fused};

#[cfg(feature = "gpu")]
fn gumbel_argmax_reference(
    logits: Tensor<WgpuBackend, 2>,
    uniform: Tensor<WgpuBackend, 2>,
) -> Tensor<WgpuBackend, 1> {
    let gumbel = -(-(uniform.log())).log();
    let perturbed = logits + gumbel;
    perturbed.argmax(1).float().squeeze_dim(1)
}

#[cfg(feature = "gpu")]
fn sigmoid_bernoulli_reference(
    gamma: Tensor<WgpuBackend, 1>,
    uniform: Tensor<WgpuBackend, 1>,
) -> Tensor<WgpuBackend, 1> {
    use burn::tensor::activation::sigmoid;
    let probs = sigmoid(gamma * 2.0);
    uniform.lower_equal(probs).float()
}

#[cfg(feature = "gpu")]
fn benchmark_gumbel_argmax(c: &mut Criterion) {
    let device = init_gpu_device();

    let mut group = c.benchmark_group("gumbel_argmax");

    for size in [1000, 10000, 100000].iter() {
        let n_categories = 100;

        group.bench_with_input(BenchmarkId::new("reference", size), size, |b, &size| {
            let logits: Tensor<WgpuBackend, 2> = Tensor::random(
                [size, n_categories],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let uniform: Tensor<WgpuBackend, 2> = Tensor::random(
                [size, n_categories],
                Distribution::Uniform(1e-10, 1.0 - 1e-10),
                &device,
            );

            b.iter(|| {
                gumbel_argmax_reference(black_box(logits.clone()), black_box(uniform.clone()))
            });
        });

        group.bench_with_input(BenchmarkId::new("fused", size), size, |b, &size| {
            let logits: Tensor<WgpuBackend, 2> = Tensor::random(
                [size, n_categories],
                Distribution::Normal(0.0, 1.0),
                &device,
            );
            let uniform: Tensor<WgpuBackend, 2> = Tensor::random(
                [size, n_categories],
                Distribution::Uniform(1e-10, 1.0 - 1e-10),
                &device,
            );

            b.iter(|| gumbel_argmax_fused(black_box(logits.clone()), black_box(uniform.clone())));
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn benchmark_sigmoid_bernoulli(c: &mut Criterion) {
    let device = init_gpu_device();

    let mut group = c.benchmark_group("sigmoid_bernoulli");

    for size in [1000, 10000, 100000].iter() {
        group.bench_with_input(BenchmarkId::new("reference", size), size, |b, &size| {
            let gamma: Tensor<WgpuBackend, 1> =
                Tensor::random([size], Distribution::Normal(0.0, 1.0), &device);
            let uniform: Tensor<WgpuBackend, 1> =
                Tensor::random([size], Distribution::Uniform(0.0, 1.0), &device);

            b.iter(|| {
                sigmoid_bernoulli_reference(black_box(gamma.clone()), black_box(uniform.clone()))
            });
        });

        group.bench_with_input(BenchmarkId::new("fused", size), size, |b, &size| {
            let gamma: Tensor<WgpuBackend, 1> =
                Tensor::random([size], Distribution::Normal(0.0, 1.0), &device);
            let uniform: Tensor<WgpuBackend, 1> =
                Tensor::random([size], Distribution::Uniform(0.0, 1.0), &device);

            b.iter(|| {
                sigmoid_bernoulli_fused(black_box(gamma.clone()), black_box(uniform.clone()))
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
fn batch_gather_reference(
    weights: Tensor<WgpuBackend, 3>,
    indices: Tensor<WgpuBackend, 2, burn::tensor::Int>,
) -> Tensor<WgpuBackend, 1> {
    let [batch_size, n_k, n_dim] = weights.dims();
    let device = weights.device();

    // Create batch indices
    let batch_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        Tensor::arange(0..batch_size as i64, &device).reshape([batch_size]);

    // Extract k and dim indices
    let k_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        indices.clone().slice([0..batch_size, 0..1]).squeeze::<1>();
    let dim_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        indices.slice([0..batch_size, 1..2]).squeeze::<1>();

    // Compute linear indices
    let linear_indices =
        batch_indices * (n_k * n_dim) as i64 + k_indices * n_dim as i64 + dim_indices;

    // Flatten weights and gather
    let weights_flat = weights.reshape([batch_size * n_k * n_dim]);
    weights_flat.select(0, linear_indices)
}

#[cfg(feature = "gpu")]
fn benchmark_batch_gather(c: &mut Criterion) {
    let device = init_gpu_device();

    let mut group = c.benchmark_group("batch_gather");

    for size in [1000, 10000, 100000].iter() {
        let k = 16;
        let dim = 64;

        group.bench_with_input(BenchmarkId::new("reference", size), size, |b, &size| {
            let weights: Tensor<WgpuBackend, 3> =
                Tensor::random([size, k, dim], Distribution::Normal(0.0, 1.0), &device);

            // Create random indices within bounds - create float then convert
            let k_indices_float: Tensor<WgpuBackend, 2> = Tensor::random(
                [size, 1],
                Distribution::Uniform(0.0, (k - 1) as f64),
                &device,
            );
            let dim_indices_float: Tensor<WgpuBackend, 2> = Tensor::random(
                [size, 1],
                Distribution::Uniform(0.0, (dim - 1) as f64),
                &device,
            );
            let k_indices: Tensor<WgpuBackend, 2, burn::tensor::Int> = k_indices_float.int();
            let dim_indices: Tensor<WgpuBackend, 2, burn::tensor::Int> = dim_indices_float.int();
            let indices = Tensor::cat(vec![k_indices, dim_indices], 1);

            b.iter(|| {
                batch_gather_reference(black_box(weights.clone()), black_box(indices.clone()))
            });
        });

        group.bench_with_input(BenchmarkId::new("fused", size), size, |b, &size| {
            let weights: Tensor<WgpuBackend, 3> =
                Tensor::random([size, k, dim], Distribution::Normal(0.0, 1.0), &device);

            let k_indices_float: Tensor<WgpuBackend, 2> = Tensor::random(
                [size, 1],
                Distribution::Uniform(0.0, (k - 1) as f64),
                &device,
            );
            let dim_indices_float: Tensor<WgpuBackend, 2> = Tensor::random(
                [size, 1],
                Distribution::Uniform(0.0, (dim - 1) as f64),
                &device,
            );
            let k_indices: Tensor<WgpuBackend, 2, burn::tensor::Int> = k_indices_float.int();
            let dim_indices: Tensor<WgpuBackend, 2, burn::tensor::Int> = dim_indices_float.int();
            let indices = Tensor::cat(vec![k_indices, dim_indices], 1);

            b.iter(|| {
                batch_gather_fused(
                    black_box(weights.clone()),
                    black_box(indices.clone()),
                    &[dim, 1],
                    k * dim,
                )
            });
        });
    }

    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    benchmark_gumbel_argmax,
    benchmark_sigmoid_bernoulli,
    benchmark_batch_gather
);

#[cfg(feature = "gpu")]
criterion_main!(benches);

// Fallback for when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Benchmarks require the 'gpu' feature. Run with: cargo bench --features gpu");
}
