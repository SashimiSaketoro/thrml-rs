use burn::backend::wgpu::WgpuDevice;

pub type WgpuBackend = burn::backend::Wgpu;

pub fn init_gpu_device() -> WgpuDevice {
    // Burn's WGPU backend uses Default::default() for device creation
    // Metal backend is automatically selected on macOS when available
    WgpuDevice::default()
}

// Helper to ensure Metal is selected
pub fn ensure_metal_backend() {
    std::env::set_var("BURN_WGPU_BACKEND", "metal");
}
