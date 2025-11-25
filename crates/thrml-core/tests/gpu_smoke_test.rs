#[cfg(feature = "gpu")]
#[test]
fn test_gpu_initialization() {
    use thrml_core::backend::*;
    ensure_metal_backend();
    let device = init_gpu_device();
    let tensor = burn::tensor::Tensor::<WgpuBackend, 1>::zeros([4], &device);
    assert_eq!(tensor.dims(), [4]);
}
