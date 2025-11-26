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
criterion_group!(
    benches,
    benchmark_gumbel_argmax,
    benchmark_sigmoid_bernoulli
);

#[cfg(feature = "gpu")]
criterion_main!(benches);

// Fallback for when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Benchmarks require the 'gpu' feature. Run with: cargo bench --features gpu");
}
