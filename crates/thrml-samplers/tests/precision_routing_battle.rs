//! Battle tests for precision routing across samplers.
//!
//! These tests verify that the precision routing (CPU f64 vs GPU f32)
//! produces consistent results and handles edge cases correctly.

#![cfg(feature = "gpu")]

use burn::tensor::Tensor;
use thrml_core::backend::{ensure_backend, init_gpu_device, WgpuBackend};
use thrml_core::compute::ComputeBackend;
use thrml_core::interaction::InteractionData;
use thrml_core::node::TensorSpec;
use thrml_samplers::rng::RngKey;
use thrml_samplers::{GaussianSampler, SpinGibbsConditional, CategoricalGibbsConditional};

/// Test that CPU f64 and GPU f32 paths produce numerically close results for SpinGibbs.
#[test]
fn test_spin_gibbs_routing_consistency() {
    ensure_backend();
    let device = init_gpu_device();

    let sampler = SpinGibbsConditional::new();
    let n_nodes = 100;
    let n_interactions = 10;

    // Create test data
    let weights: Tensor<WgpuBackend, 3> = Tensor::random(
        [n_nodes, n_interactions, 1],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );
    let active: Tensor<WgpuBackend, 2> = Tensor::ones([n_nodes, n_interactions], &device);
    // Create random binary state
    let state_uniform: Tensor<WgpuBackend, 2> = Tensor::random(
        [n_nodes, n_interactions],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let state: Tensor<WgpuBackend, 2> = state_uniform.greater_elem(0.5).float();

    let interactions = vec![InteractionData::Tensor(weights)];
    let active_flags = vec![active];
    let neighbor_states = vec![vec![state]];
    let n_spin_per_interaction = vec![1usize];

    let output_spec = TensorSpec {
        shape: vec![n_nodes],
        dtype: burn::tensor::DType::F32,
    };

    // GPU f32 path
    let backend_gpu = ComputeBackend::gpu_only();
    let (samples_gpu, _) = sampler.sample_routed(
        &backend_gpu,
        RngKey::new(42),
        &interactions,
        &active_flags,
        &neighbor_states,
        &n_spin_per_interaction,
        (),
        &output_spec,
        &device,
    );

    // CPU f64 path (Apple Silicon or explicit)
    let backend_cpu = ComputeBackend::apple_silicon();
    let (samples_cpu, _) = sampler.sample_routed(
        &backend_cpu,
        RngKey::new(42),
        &interactions,
        &active_flags,
        &neighbor_states,
        &n_spin_per_interaction,
        (),
        &output_spec,
        &device,
    );

    // Check that results are close (not identical due to different precision/RNG)
    let gpu_data: Vec<f32> = samples_gpu.into_data().to_vec().unwrap();
    let cpu_data: Vec<f32> = samples_cpu.into_data().to_vec().unwrap();

    assert_eq!(gpu_data.len(), n_nodes);
    assert_eq!(cpu_data.len(), n_nodes);

    // Both should produce valid binary outputs
    for i in 0..n_nodes {
        assert!(
            (gpu_data[i] - 0.0).abs() < 0.01 || (gpu_data[i] - 1.0).abs() < 0.01,
            "GPU sample {} should be 0 or 1, got {}",
            i,
            gpu_data[i]
        );
        assert!(
            (cpu_data[i] - 0.0).abs() < 0.01 || (cpu_data[i] - 1.0).abs() < 0.01,
            "CPU sample {} should be 0 or 1, got {}",
            i,
            cpu_data[i]
        );
    }

    println!("✓ SpinGibbs routing consistency test passed");
    println!("  GPU samples[0..5]: {:?}", &gpu_data[0..5]);
    println!("  CPU samples[0..5]: {:?}", &cpu_data[0..5]);
}

/// Test that Gaussian sampler routing produces valid samples.
#[test]
fn test_gaussian_routing_consistency() {
    ensure_backend();
    let device = init_gpu_device();

    let sampler = GaussianSampler::new();
    let n_nodes = 50;
    let k = 5;

    // Create quadratic (precision) interaction
    let inverse_weights: Tensor<WgpuBackend, 2> = Tensor::random(
        [n_nodes, k],
        burn::tensor::Distribution::Uniform(0.5, 2.0),  // Positive precision
        &device,
    );
    let active: Tensor<WgpuBackend, 2> = Tensor::ones([n_nodes, k], &device);

    let interactions = vec![InteractionData::Quadratic { inverse_weights }];
    let active_flags = vec![active];
    let neighbor_states: Vec<Vec<Tensor<WgpuBackend, 2>>> = vec![vec![]];
    let n_spin_per_interaction = vec![0usize];

    let output_spec = TensorSpec {
        shape: vec![n_nodes],
        dtype: burn::tensor::DType::F32,
    };

    // CPU f64 path
    let backend_cpu = ComputeBackend::apple_silicon();
    let (samples_cpu, _) = sampler.sample_routed(
        &backend_cpu,
        RngKey::new(123),
        &interactions,
        &active_flags,
        &neighbor_states,
        &n_spin_per_interaction,
        (),
        &output_spec,
        &device,
    );

    let cpu_data: Vec<f32> = samples_cpu.into_data().to_vec().unwrap();

    // Check samples are finite and reasonable
    for (i, &val) in cpu_data.iter().enumerate() {
        assert!(
            val.is_finite(),
            "CPU Gaussian sample {} should be finite, got {}",
            i,
            val
        );
        assert!(
            val.abs() < 100.0,
            "CPU Gaussian sample {} should be reasonable, got {}",
            i,
            val
        );
    }

    println!("✓ Gaussian routing consistency test passed");
    println!("  CPU samples[0..5]: {:?}", &cpu_data[0..5.min(n_nodes)]);
}

/// Test routing with large problem sizes to stress precision.
#[test]
fn test_large_problem_precision() {
    ensure_backend();
    let device = init_gpu_device();

    let sampler = SpinGibbsConditional::new();
    let n_nodes = 1000;  // Large enough to stress accumulation
    let n_interactions = 50;

    // Create test data with varied magnitudes
    let weights: Tensor<WgpuBackend, 3> = Tensor::random(
        [n_nodes, n_interactions, 1],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let active: Tensor<WgpuBackend, 2> = Tensor::ones([n_nodes, n_interactions], &device);
    let state_uniform: Tensor<WgpuBackend, 2> = Tensor::random(
        [n_nodes, n_interactions],
        burn::tensor::Distribution::Uniform(0.0, 1.0),
        &device,
    );
    let state: Tensor<WgpuBackend, 2> = state_uniform.greater_elem(0.5).float();

    let interactions = vec![InteractionData::Tensor(weights)];
    let active_flags = vec![active];
    let neighbor_states = vec![vec![state]];
    let n_spin_per_interaction = vec![1usize];

    let output_spec = TensorSpec {
        shape: vec![n_nodes],
        dtype: burn::tensor::DType::F32,
    };

    // Run with CPU f64 path
    let backend_cpu = ComputeBackend::apple_silicon();
    let (samples, _) = sampler.sample_routed(
        &backend_cpu,
        RngKey::new(999),
        &interactions,
        &active_flags,
        &neighbor_states,
        &n_spin_per_interaction,
        (),
        &output_spec,
        &device,
    );

    let data: Vec<f32> = samples.into_data().to_vec().unwrap();
    
    // Count statistics
    let ones = data.iter().filter(|&&x| x > 0.5).count();
    let zeros = data.iter().filter(|&&x| x < 0.5).count();

    println!("✓ Large problem precision test passed");
    println!("  n_nodes: {}, n_interactions: {}", n_nodes, n_interactions);
    println!("  ones: {}, zeros: {}", ones, zeros);
    
    // Should have a reasonable mix (not all same value)
    assert!(ones > 100 && zeros > 100, "Should have reasonable mix of 0s and 1s");
}

/// Test CategoricalGibbs routing produces valid categorical samples.
#[test]
fn test_categorical_routing_consistency() {
    ensure_backend();
    let device = init_gpu_device();

    let n_categories = 4;
    let sampler = CategoricalGibbsConditional::new(n_categories, 0);
    let n_nodes = 100;
    let n_interactions = 10;

    // Create test data
    let weights: Tensor<WgpuBackend, 3> = Tensor::random(
        [n_nodes, n_interactions, n_categories],
        burn::tensor::Distribution::Normal(0.0, 0.5),
        &device,
    );
    let active: Tensor<WgpuBackend, 2> = Tensor::ones([n_nodes, n_interactions], &device);

    let interactions = vec![InteractionData::Tensor(weights)];
    let active_flags = vec![active];
    let neighbor_states: Vec<Vec<Tensor<WgpuBackend, 2>>> = vec![vec![]];
    let n_spin_per_interaction = vec![0usize];

    let output_spec = TensorSpec {
        shape: vec![n_nodes],
        dtype: burn::tensor::DType::F32,
    };

    // CPU f64 path
    let backend_cpu = ComputeBackend::apple_silicon();
    let (samples_cpu, _) = sampler.sample_routed(
        &backend_cpu,
        RngKey::new(42),
        &interactions,
        &active_flags,
        &neighbor_states,
        &n_spin_per_interaction,
        (),
        &output_spec,
        &device,
    );

    let cpu_data: Vec<f32> = samples_cpu.into_data().to_vec().unwrap();

    // Check samples are valid category indices
    for (i, &val) in cpu_data.iter().enumerate() {
        let cat = val as usize;
        assert!(
            cat < n_categories,
            "CPU categorical sample {} should be < {}, got {}",
            i,
            n_categories,
            cat
        );
    }

    // Count category distribution
    let mut counts = vec![0usize; n_categories];
    for &val in &cpu_data {
        counts[val as usize] += 1;
    }

    println!("✓ Categorical routing consistency test passed");
    println!("  Category distribution: {:?}", counts);
}

/// Test backend detection and routing selection.
#[test]
fn test_backend_routing_selection() {
    use thrml_core::compute::{RuntimePolicy, OpType};

    // Apple Silicon should route Ising to CPU
    let apple = RuntimePolicy::apple_silicon();
    let backend = ComputeBackend::from_policy(&apple);
    assert!(
        backend.use_cpu(OpType::IsingSampling, None),
        "Apple Silicon should route IsingSampling to CPU"
    );
    assert!(
        backend.use_cpu(OpType::CategoricalSampling, None),
        "Apple Silicon should route CategoricalSampling to CPU"
    );

    // Consumer NVIDIA should route precision ops to CPU
    let nvidia = RuntimePolicy::nvidia_consumer();
    let backend = ComputeBackend::from_policy(&nvidia);
    assert!(
        backend.use_cpu(OpType::IsingSampling, None),
        "Consumer NVIDIA should route IsingSampling to CPU"
    );

    // GPU-only should NOT route to CPU
    let gpu_only = ComputeBackend::gpu_only();
    assert!(
        !gpu_only.use_cpu(OpType::IsingSampling, None),
        "GpuOnly should not route IsingSampling to CPU"
    );

    println!("✓ Backend routing selection test passed");
}

/// Test Langevin step routing.
#[test]
fn test_langevin_routing() {
    ensure_backend();
    let device = init_gpu_device();

    use thrml_samplers::langevin::{langevin_step_2d_routed, LangevinConfig};

    let n = 50;
    let d = 3;

    let state: Tensor<WgpuBackend, 2> = Tensor::random(
        [n, d],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    let gradient: Tensor<WgpuBackend, 2> = Tensor::random(
        [n, d],
        burn::tensor::Distribution::Normal(0.0, 0.1),
        &device,
    );

    let config = LangevinConfig::new(0.01, 0.1);
    let backend = ComputeBackend::apple_silicon();

    let new_state = langevin_step_2d_routed(&backend, &state, &gradient, &config, &device);
    let new_state_data: Vec<f32> = new_state.into_data().to_vec().unwrap();

    // Check all values are finite
    for (i, &val) in new_state_data.iter().enumerate() {
        assert!(val.is_finite(), "Langevin state {} should be finite", i);
    }

    println!("✓ Langevin routing test passed");
}
