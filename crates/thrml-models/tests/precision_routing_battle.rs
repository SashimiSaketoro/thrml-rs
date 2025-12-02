//! Battle tests for precision routing in models.
//!
//! These tests verify that the precision routing (CPU f64 vs GPU f32)
//! is correctly configured and selected based on RuntimePolicy.

#![cfg(feature = "gpu")]

use thrml_core::compute::{ComputeBackend, OpType, RuntimePolicy};

/// Test that routing is correctly selected based on RuntimePolicy.
#[test]
fn test_model_backend_selection() {
    // Apple Silicon should route precision ops to CPU
    let apple = RuntimePolicy::apple_silicon();
    let backend = ComputeBackend::from_policy(&apple);

    let uses_cpu_for_gradient = backend.use_cpu(OpType::GradientCompute, None);
    let uses_cpu_for_energy = backend.use_cpu(OpType::EnergyCompute, None);
    let uses_cpu_for_ising = backend.use_cpu(OpType::IsingSampling, None);

    println!("✓ Apple Silicon routing:");
    println!("  GradientCompute → CPU: {}", uses_cpu_for_gradient);
    println!("  EnergyCompute → CPU: {}", uses_cpu_for_energy);
    println!("  IsingSampling → CPU: {}", uses_cpu_for_ising);

    // Verify key precision ops are routed to CPU
    assert!(
        uses_cpu_for_ising,
        "Apple Silicon should route IsingSampling to CPU"
    );
    assert!(
        uses_cpu_for_gradient,
        "Apple Silicon should route GradientCompute to CPU"
    );
}

/// Test that Consumer NVIDIA routes appropriately.
#[test]
fn test_nvidia_consumer_routing() {
    let nvidia = RuntimePolicy::nvidia_consumer();
    let backend = ComputeBackend::from_policy(&nvidia);

    let uses_cpu_for_gradient = backend.use_cpu(OpType::GradientCompute, None);
    let uses_cpu_for_ising = backend.use_cpu(OpType::IsingSampling, None);

    println!("✓ NVIDIA Consumer routing:");
    println!("  GradientCompute → CPU: {}", uses_cpu_for_gradient);
    println!("  IsingSampling → CPU: {}", uses_cpu_for_ising);

    // Consumer NVIDIA should still route precision ops to CPU (limited f64 support)
    assert!(
        uses_cpu_for_ising,
        "Consumer NVIDIA should route IsingSampling to CPU"
    );
}

/// Test GPU-only backend doesn't route to CPU.
#[test]
fn test_gpu_only_routing() {
    let gpu_only = ComputeBackend::gpu_only();

    let uses_cpu_for_gradient = gpu_only.use_cpu(OpType::GradientCompute, None);
    let uses_cpu_for_energy = gpu_only.use_cpu(OpType::EnergyCompute, None);
    let uses_cpu_for_ising = gpu_only.use_cpu(OpType::IsingSampling, None);

    println!("✓ GPU-Only routing:");
    println!("  GradientCompute → CPU: {}", uses_cpu_for_gradient);
    println!("  EnergyCompute → CPU: {}", uses_cpu_for_energy);
    println!("  IsingSampling → CPU: {}", uses_cpu_for_ising);

    // GPU-only should never route to CPU
    assert!(
        !uses_cpu_for_gradient,
        "GpuOnly should not route GradientCompute to CPU"
    );
    assert!(
        !uses_cpu_for_energy,
        "GpuOnly should not route EnergyCompute to CPU"
    );
    assert!(
        !uses_cpu_for_ising,
        "GpuOnly should not route IsingSampling to CPU"
    );
}

/// Test HPC GPU routing behavior.
#[test]
fn test_hpc_gpu_routing() {
    let hopper = RuntimePolicy::nvidia_hopper();
    let backend = ComputeBackend::from_policy(&hopper);

    // When CUDA feature is NOT enabled, HPC should fall back to UnifiedHybrid
    // which routes precision ops to CPU
    #[cfg(not(feature = "cuda"))]
    {
        let uses_cpu_for_ising = backend.use_cpu(OpType::IsingSampling, None);
        println!("✓ HPC GPU routing (no CUDA):");
        println!("  IsingSampling → CPU: {}", uses_cpu_for_ising);

        // Without CUDA, should use CPU for precision
        assert!(
            uses_cpu_for_ising,
            "HPC without CUDA should route IsingSampling to CPU"
        );
    }

    // When CUDA feature IS enabled, HPC can use GPU f64
    #[cfg(feature = "cuda")]
    {
        let uses_gpu_f64 = backend.uses_gpu_f64();
        println!("✓ HPC GPU routing (with CUDA):");
        println!("  Uses GPU f64: {}", uses_gpu_f64);

        // With CUDA, should use GPU f64
        assert!(uses_gpu_f64, "HPC with CUDA should use GPU f64");
    }
}

/// Test RuntimePolicy tier detection.
#[test]
fn test_runtime_policy_tiers() {
    use thrml_core::compute::HardwareTier;

    let apple = RuntimePolicy::apple_silicon();
    let nvidia_consumer = RuntimePolicy::nvidia_consumer();
    let hopper = RuntimePolicy::nvidia_hopper();
    let blackwell = RuntimePolicy::nvidia_blackwell();

    // Verify tier assignments
    assert!(matches!(apple.tier, HardwareTier::AppleSilicon));
    assert!(matches!(nvidia_consumer.tier, HardwareTier::NvidiaConsumer));
    assert!(matches!(hopper.tier, HardwareTier::NvidiaHopper));
    assert!(matches!(blackwell.tier, HardwareTier::NvidiaBlackwell));

    // Verify HPC tier detection
    assert!(!apple.is_hpc_tier(), "Apple Silicon is not HPC tier");
    assert!(
        !nvidia_consumer.is_hpc_tier(),
        "Consumer NVIDIA is not HPC tier"
    );
    assert!(hopper.is_hpc_tier(), "Hopper is HPC tier");
    assert!(blackwell.is_hpc_tier(), "Blackwell is HPC tier");

    println!("✓ RuntimePolicy tier detection test passed");
}

/// Test precision profile assignments.
#[test]
fn test_precision_profiles() {
    use thrml_core::compute::PrecisionProfile;

    let apple = RuntimePolicy::apple_silicon();
    let nvidia_consumer = RuntimePolicy::nvidia_consumer();
    let hopper = RuntimePolicy::nvidia_hopper();

    assert!(matches!(apple.profile, PrecisionProfile::CpuFp64Strict));
    assert!(matches!(
        nvidia_consumer.profile,
        PrecisionProfile::GpuMixed
    ));
    assert!(matches!(hopper.profile, PrecisionProfile::GpuHpcFp64));

    println!("✓ Precision profile test passed");
    println!("  Apple Silicon: {:?}", apple.profile);
    println!("  NVIDIA Consumer: {:?}", nvidia_consumer.profile);
    println!("  NVIDIA Hopper: {:?}", hopper.profile);
}

/// Test OpType routing for various operations.
#[test]
fn test_op_type_routing() {
    let apple = ComputeBackend::apple_silicon();

    // All precision-sensitive ops should route to CPU on Apple Silicon
    let ops_to_check = [
        (OpType::IsingSampling, "IsingSampling"),
        (OpType::CategoricalSampling, "CategoricalSampling"),
        (OpType::GradientCompute, "GradientCompute"),
        (OpType::SphericalHarmonics, "SphericalHarmonics"),
        (OpType::ArcTrig, "ArcTrig"),
        (OpType::ComplexArithmetic, "ComplexArithmetic"),
    ];

    println!("✓ OpType routing on Apple Silicon:");
    for (op, name) in &ops_to_check {
        let uses_cpu = apple.use_cpu(*op, None);
        println!("  {} → CPU: {}", name, uses_cpu);
        assert!(uses_cpu, "{} should route to CPU on Apple Silicon", name);
    }
}
