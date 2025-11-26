//! GPU Backend Selection
//!
//! THRML supports multiple GPU backends:
//! - **WGPU** (default): Metal (macOS), Vulkan (Linux), DX12 (Windows)
//! - **CUDA**: NVIDIA GPUs via native CUDA (Linux/Windows)
//!
//! ## Feature Flags
//! - `gpu` (default): Enable WGPU backend
//! - `cuda`: Enable CUDA backend
//!
//! ## Usage
//! ```ignore
//! // WGPU backend (default)
//! cargo build --features gpu
//!
//! // CUDA backend
//! cargo build --features cuda --no-default-features
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
    
    backends
}
