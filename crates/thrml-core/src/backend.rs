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

/// Auto-select the best WGPU backend for the current platform
/// - macOS: Metal
/// - Linux/Windows: Vulkan
#[cfg(feature = "gpu")]
pub fn ensure_backend() {
    #[cfg(target_os = "macos")]
    std::env::set_var("BURN_WGPU_BACKEND", "metal");

    #[cfg(not(target_os = "macos"))]
    std::env::set_var("BURN_WGPU_BACKEND", "vulkan");
}

/// Force Metal backend selection (macOS only)
#[cfg(feature = "gpu")]
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
    #[allow(unused_mut)]
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

// =============================================================================
// Hardware Detection (GPU feature required)
// =============================================================================

/// Detected GPU information from WGPU adapter.
///
/// Contains hardware identification information used by `RuntimePolicy::detect()`
/// to classify hardware and select appropriate precision profiles.
///
/// Only available with `gpu` feature enabled.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Human-readable GPU name (e.g., "Apple M3 Pro", "NVIDIA GeForce RTX 4090")
    pub name: String,
    /// PCI vendor ID (e.g., 0x10DE for NVIDIA, 0x106B for Apple)
    pub vendor_id: u32,
    /// PCI device ID
    pub device_id: u32,
    /// Device type (Discrete, Integrated, etc.)
    pub device_type: wgpu::DeviceType,
    /// Graphics API backend (Metal, Vulkan, DX12, etc.)
    pub backend: wgpu::Backend,
    /// Driver name/version
    pub driver: String,
}

/// Detect GPU hardware information using WGPU.
///
/// This queries the system for available GPUs and returns info about
/// the best available high-performance GPU.
///
/// Returns `None` if:
/// - No GPU is available
/// - GPU detection fails or panics
/// - Running in headless environment without GPU
///
/// # Example
///
/// ```rust,ignore
/// if let Some(gpu) = detect_gpu_info() {
///     println!("GPU: {} (vendor: 0x{:04X})", gpu.name, gpu.vendor_id);
/// }
/// ```
#[cfg(feature = "gpu")]
pub fn detect_gpu_info() -> Option<GpuInfo> {
    use wgpu::{Backends, Instance, InstanceDescriptor, RequestAdapterOptions};

    // Use std::panic::catch_unwind to handle potential panics from WGPU
    // This can happen in certain CI environments or when drivers are broken
    let result = std::panic::catch_unwind(|| {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        // Request high-performance adapter (prefers discrete GPU)
        // wgpu 26.x returns Result<Result<Adapter, RequestAdapterError>, ...>
        pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
    });

    match result {
        Ok(Ok(adapter)) => {
            let info = adapter.get_info();
            Some(GpuInfo {
                name: info.name,
                vendor_id: info.vendor,
                device_id: info.device,
                device_type: info.device_type,
                backend: info.backend,
                driver: info.driver,
            })
        }
        Ok(Err(_)) => {
            // Adapter request failed (no suitable adapter found)
            None
        }
        Err(_) => {
            // Panic occurred during detection - return None gracefully
            None
        }
    }
}

/// Stub for non-GPU builds - always returns None.
///
/// This allows code to call `detect_gpu_info()` unconditionally
/// without feature-flag checks at the call site.
#[cfg(not(feature = "gpu"))]
pub fn detect_gpu_info() -> Option<()> {
    None
}
