//! GPU Backend Smoke Tests
//!
//! These tests verify that GPU backends initialize correctly.
//! They are platform-specific and require appropriate hardware.

// =============================================================================
// WGPU Backend Tests
// =============================================================================

/// WGPU Metal backend test (macOS only)
#[cfg(all(feature = "gpu", target_os = "macos"))]
#[test]
fn test_wgpu_metal_initialization() {
    use thrml_core::backend::*;
    ensure_metal_backend();
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
