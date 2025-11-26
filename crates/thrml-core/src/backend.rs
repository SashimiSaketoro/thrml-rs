//! Backend Selection
//!
//! THRML supports multiple compute backends:
//! - **WGPU** (default): Metal (macOS), Vulkan (Linux), DX12 (Windows)
//! - **CUDA**: NVIDIA GPUs via native CUDA (Linux/Windows)
//! - **CPU**: Pure Rust via ndarray (no GPU required)
//!
//! ## Feature Flags
//! - `gpu` (default): Enable WGPU backend
//! - `cuda`: Enable CUDA backend (requires `gpu`)
//! - `cpu`: Enable CPU-only backend (no GPU required)
//!
//! ## Usage
//! ```bash
//! # WGPU backend (default) - Metal on macOS, Vulkan on Linux
//! cargo build --features gpu
//!
//! # CUDA backend for NVIDIA GPUs
//! cargo build --features cuda
//!
//! # CPU-only (no GPU required)
//! cargo build --features cpu --no-default-features
//! ```

// =============================================================================
// WGPU Backend (Metal, Vulkan, DX12)
// =============================================================================

#[cfg(feature = "gpu")]
pub use burn::backend::wgpu::WgpuDevice;

#[cfg(feature = "gpu")]
pub type WgpuBackend = burn::backend::Wgpu;

#[cfg(feature = "gpu")]
pub fn init_gpu_device() -> WgpuDevice {
    // Burn's WGPU backend uses Default::default() for device creation
    // Metal backend is automatically selected on macOS when available
    WgpuDevice::default()
}

/// Force Metal backend selection (macOS only)
#[cfg(all(feature = "gpu", target_os = "macos"))]
pub fn ensure_metal_backend() {
    std::env::set_var("BURN_WGPU_BACKEND", "metal");
}

/// Force Vulkan backend selection (Linux/Windows)
#[cfg(all(feature = "gpu", not(target_os = "macos")))]
pub fn ensure_vulkan_backend() {
    std::env::set_var("BURN_WGPU_BACKEND", "vulkan");
}

// =============================================================================
// CUDA Backend (NVIDIA GPUs)
// =============================================================================

#[cfg(feature = "cuda")]
pub use burn::backend::cuda::CudaDevice;

#[cfg(feature = "cuda")]
pub type CudaBackend = burn::backend::Cuda;

/// Initialize CUDA device
///
/// By default uses device 0. Set `CUDA_VISIBLE_DEVICES` to control which GPU.
#[cfg(feature = "cuda")]
pub fn init_cuda_device() -> CudaDevice {
    CudaDevice::default()
}

/// Initialize CUDA device with specific GPU index
#[cfg(feature = "cuda")]
pub fn init_cuda_device_index(index: usize) -> CudaDevice {
    CudaDevice::new(index)
}

// =============================================================================
// CPU Backend (ndarray - no GPU required)
// =============================================================================

#[cfg(feature = "cpu")]
pub use burn::backend::ndarray::NdArrayDevice;

#[cfg(feature = "cpu")]
pub type CpuBackend = burn::backend::NdArray;

/// Initialize CPU device
///
/// This backend requires no GPU and works on any system.
/// Useful for development, testing, or systems without GPU support.
#[cfg(feature = "cpu")]
pub fn init_cpu_device() -> NdArrayDevice {
    NdArrayDevice::default()
}

// =============================================================================
// Backend Detection Utilities
// =============================================================================

/// Check if WGPU backend is available
#[cfg(feature = "gpu")]
pub fn is_wgpu_available() -> bool {
    true
}

#[cfg(not(feature = "gpu"))]
pub fn is_wgpu_available() -> bool {
    false
}

/// Check if CUDA backend is available
#[cfg(feature = "cuda")]
pub fn is_cuda_available() -> bool {
    true
}

#[cfg(not(feature = "cuda"))]
pub fn is_cuda_available() -> bool {
    false
}

/// Check if CPU backend is available
#[cfg(feature = "cpu")]
pub fn is_cpu_available() -> bool {
    true
}

#[cfg(not(feature = "cpu"))]
pub fn is_cpu_available() -> bool {
    false
}

/// Get available backend names
pub fn available_backends() -> Vec<&'static str> {
    let mut backends = Vec::new();
    
    #[cfg(feature = "gpu")]
    {
        #[cfg(target_os = "macos")]
        backends.push("wgpu-metal");
        
        #[cfg(target_os = "linux")]
        backends.push("wgpu-vulkan");
        
        #[cfg(target_os = "windows")]
        backends.push("wgpu-dx12");
    }
    
    #[cfg(feature = "cuda")]
    backends.push("cuda");
    
    #[cfg(feature = "cpu")]
    backends.push("cpu-ndarray");
    
    backends
}
