//! GPU Backend Smoke Tests
//!
//! These tests verify that GPU backends initialize correctly.
//! They are platform-specific and require appropriate hardware.

// =============================================================================
// WGPU Backend Tests
// =============================================================================

/// WGPU Metal backend test (macOS only)
#[cfg(feature = "gpu")]
#[test]
fn test_wgpu_metal_initialization() {
    use thrml_core::backend::*;
    ensure_backend();
    let device = init_gpu_device();
    let tensor = burn::tensor::Tensor::<WgpuBackend, 1>::zeros([4], &device);
    assert_eq!(tensor.dims(), [4]);
    println!("✓ WGPU Metal backend initialized successfully");
}

/// WGPU Vulkan backend test (Linux only)
/// Note: Requires Vulkan drivers and may not work in headless CI
#[cfg(all(feature = "gpu", target_os = "linux"))]
#[test]
#[ignore] // Vulkan often not available in CI - run manually
fn test_wgpu_vulkan_initialization() {
    use thrml_core::backend::*;
    ensure_vulkan_backend();
    let device = init_gpu_device();
    let tensor = burn::tensor::Tensor::<WgpuBackend, 1>::zeros([4], &device);
    assert_eq!(tensor.dims(), [4]);
    println!("✓ WGPU Vulkan backend initialized successfully");
}

// =============================================================================
// CUDA Backend Tests
// =============================================================================

/// CUDA backend test (requires NVIDIA GPU)
#[cfg(feature = "cuda")]
#[test]
#[ignore] // CUDA not available in standard CI - run manually with GPU
fn test_cuda_initialization() {
    use thrml_core::backend::*;
    let device = init_cuda_device();
    let tensor = burn::tensor::Tensor::<CudaBackend, 1>::zeros([4], &device);
    assert_eq!(tensor.dims(), [4]);
    println!("✓ CUDA backend initialized successfully");
}

/// CUDA multi-GPU test
#[cfg(feature = "cuda")]
#[test]
#[ignore] // Requires multiple GPUs - run manually
fn test_cuda_multi_gpu() {
    use thrml_core::backend::*;

    // Try to initialize device 0 and 1
    let device0 = init_cuda_device_index(0);
    let tensor0 = burn::tensor::Tensor::<CudaBackend, 1>::zeros([4], &device0);
    assert_eq!(tensor0.dims(), [4]);
    println!("✓ CUDA device 0 initialized");

    // Device 1 may not exist - this is expected to fail on single-GPU systems
    let device1 = init_cuda_device_index(1);
    let tensor1 = burn::tensor::Tensor::<CudaBackend, 1>::zeros([4], &device1);
    assert_eq!(tensor1.dims(), [4]);
    println!("✓ CUDA device 1 initialized");
}

// =============================================================================
// CPU Backend Tests
// =============================================================================

/// CPU backend test (always works, no GPU required)
#[cfg(feature = "cpu")]
#[test]
fn test_cpu_initialization() {
    use thrml_core::backend::*;
    let device = init_cpu_device();
    let tensor = burn::tensor::Tensor::<CpuBackend, 1>::zeros([4], &device);
    assert_eq!(tensor.dims(), [4]);
    println!("✓ CPU (ndarray) backend initialized successfully");
}

// =============================================================================
// Backend Detection Tests (always run)
// =============================================================================

#[test]
fn test_available_backends() {
    use thrml_core::backend::*;

    let backends = available_backends();
    println!("Available backends: {:?}", backends);

    // At least one backend should be available if any feature is enabled
    #[cfg(any(feature = "gpu", feature = "cuda", feature = "cpu"))]
    assert!(!backends.is_empty(), "No backends available");

    #[cfg(feature = "gpu")]
    assert!(is_wgpu_available());

    #[cfg(feature = "cuda")]
    assert!(is_cuda_available());

    #[cfg(feature = "cpu")]
    assert!(is_cpu_available());
}

// =============================================================================
// Hardware Detection and Tier Policy Tests
// =============================================================================

/// Test hardware detection (GPU feature required)
#[cfg(feature = "gpu")]
#[test]
fn test_hardware_detection() {
    use thrml_core::{HardwareTier, RuntimePolicy};

    let policy = RuntimePolicy::detect();
    println!("Detected hardware tier: {:?}", policy.tier);
    println!("Precision profile: {:?}", policy.profile);
    println!("Real dtype: {:?}", policy.real_dtype);
    println!("Max rel error: {:e}", policy.max_rel_error);

    // Verify tier is not CpuOnly when GPU is available
    if let Some(gpu) = thrml_core::backend::detect_gpu_info() {
        println!("GPU: {} (vendor: 0x{:04X})", gpu.name, gpu.vendor_id);
        assert_ne!(
            policy.tier,
            HardwareTier::CpuOnly,
            "Should detect GPU when GPU is available"
        );
    }
}

/// Test tier policy constructors (always run)
#[test]
fn test_tier_policies() {
    use burn::tensor::DType;
    use thrml_core::{HardwareTier, PrecisionProfile, RuntimePolicy};

    // Apple Silicon should route precision ops to CPU
    let apple = RuntimePolicy::apple_silicon();
    assert_eq!(apple.tier, HardwareTier::AppleSilicon);
    assert_eq!(apple.profile, PrecisionProfile::CpuFp64Strict);
    assert!(apple.use_gpu);
    assert_eq!(apple.real_dtype, DType::F32);

    // NVIDIA consumer should use mixed precision
    let rtx = RuntimePolicy::nvidia_consumer();
    assert_eq!(rtx.tier, HardwareTier::NvidiaConsumer);
    assert_eq!(rtx.profile, PrecisionProfile::GpuMixed);
    assert_eq!(rtx.real_dtype, DType::F32);

    // AMD RDNA should use mixed precision like NVIDIA consumer
    let amd = RuntimePolicy::amd_rdna();
    assert_eq!(amd.tier, HardwareTier::AmdRdna);
    assert_eq!(amd.profile, PrecisionProfile::GpuMixed);

    // H100 should use full GPU fp64
    let h100 = RuntimePolicy::nvidia_hopper();
    assert_eq!(h100.tier, HardwareTier::NvidiaHopper);
    assert_eq!(h100.profile, PrecisionProfile::GpuHpcFp64);
    assert_eq!(h100.real_dtype, DType::F64);

    // B200 should use full GPU fp64
    let b200 = RuntimePolicy::nvidia_blackwell();
    assert_eq!(b200.tier, HardwareTier::NvidiaBlackwell);
    assert_eq!(b200.profile, PrecisionProfile::GpuHpcFp64);

    // CPU-only should not use GPU
    let cpu = RuntimePolicy::cpu_only();
    assert_eq!(cpu.tier, HardwareTier::CpuOnly);
    assert!(!cpu.use_gpu);
    assert_eq!(cpu.real_dtype, DType::F64);
}

/// Test ComputeBackend::from_policy (always run)
#[test]
fn test_compute_backend_from_policy() {
    use thrml_core::{ComputeBackend, OpType, RuntimePolicy};

    // CpuFp64Strict profile should route precision ops to CPU
    let apple_policy = RuntimePolicy::apple_silicon();
    let backend = ComputeBackend::from_policy(&apple_policy);
    assert!(backend.use_cpu(OpType::IsingSampling, None));
    assert!(backend.use_cpu(OpType::GradientCompute, None));
    assert!(!backend.use_cpu(OpType::Similarity, None)); // Bulk ops go to GPU

    // GpuMixed profile should route fewer ops to CPU
    let rtx_policy = RuntimePolicy::nvidia_consumer();
    let backend = ComputeBackend::from_policy(&rtx_policy);
    assert!(backend.use_cpu(OpType::IsingSampling, None));
    assert!(backend.use_cpu(OpType::GradientCompute, None));
    assert!(!backend.use_cpu(OpType::Similarity, None));

    // GpuHpcFp64 profile behavior depends on CUDA availability
    let h100_policy = RuntimePolicy::nvidia_hopper();
    let backend = ComputeBackend::from_policy(&h100_policy);
    
    #[cfg(feature = "cuda")]
    {
        // With CUDA: GPU f64 for everything
        assert!(!backend.use_cpu(OpType::IsingSampling, None));
        assert!(!backend.use_cpu(OpType::GradientCompute, None));
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        // Without CUDA: falls back to UnifiedHybrid (CPU f64 for precision ops)
        assert!(backend.use_cpu(OpType::IsingSampling, None));
        assert!(backend.use_cpu(OpType::GradientCompute, None));
    }
}

/// Test HybridConfig::from_policy (always run)
#[test]
fn test_hybrid_config_from_policy() {
    use thrml_core::{HybridConfig, PrecisionMode, RuntimePolicy};

    // Apple Silicon policy should use CpuPrecise mode
    let apple_policy = RuntimePolicy::apple_silicon();
    let config = HybridConfig::from_policy(&apple_policy);
    assert!(matches!(config.precision, PrecisionMode::CpuPrecise));
    assert!(config.enable_overlap);

    // H100 policy should use GpuFast mode
    let h100_policy = RuntimePolicy::nvidia_hopper();
    let config = HybridConfig::from_policy(&h100_policy);
    assert!(matches!(config.precision, PrecisionMode::GpuFast));
}

/// Test for_tier mapping (always run)
#[test]
fn test_for_tier_mapping() {
    use thrml_core::{HardwareTier, PrecisionProfile, RuntimePolicy};

    // Verify each tier maps to expected profile
    assert_eq!(
        RuntimePolicy::for_tier(HardwareTier::AppleSilicon).profile,
        PrecisionProfile::CpuFp64Strict
    );
    assert_eq!(
        RuntimePolicy::for_tier(HardwareTier::NvidiaConsumer).profile,
        PrecisionProfile::GpuMixed
    );
    assert_eq!(
        RuntimePolicy::for_tier(HardwareTier::AmdRdna).profile,
        PrecisionProfile::GpuMixed
    );
    assert_eq!(
        RuntimePolicy::for_tier(HardwareTier::NvidiaHopper).profile,
        PrecisionProfile::GpuHpcFp64
    );
    assert_eq!(
        RuntimePolicy::for_tier(HardwareTier::NvidiaBlackwell).profile,
        PrecisionProfile::GpuHpcFp64
    );
    assert_eq!(
        RuntimePolicy::for_tier(HardwareTier::CpuOnly).profile,
        PrecisionProfile::CpuFp64Strict
    );
    assert_eq!(
        RuntimePolicy::for_tier(HardwareTier::Unknown).profile,
        PrecisionProfile::CpuFp64Strict
    );
}
