//! Tests for extended batch_gather kernels (3+ indices)

#[cfg(feature = "gpu")]
mod tests {
    use burn::tensor::{Int, Tensor};
    use thrml_core::backend::{init_gpu_device, WgpuBackend};
    use thrml_kernels::batch_gather_fused;

    #[test]
    fn test_batch_gather_3_indices() {
        let device = init_gpu_device();

        // weights: [2, 3, 20] = 120 elements
        let weights_data: Vec<f32> = (0..120).map(|i| i as f32).collect();
        let weights: Tensor<WgpuBackend, 1> = Tensor::from_data(weights_data.as_slice(), &device);
        let weights: Tensor<WgpuBackend, 3> = weights.reshape([2, 3, 20]);

        // Create indices for 3-index gather
        let indices_data: Vec<i32> = vec![0, 1, 2, 1, 0, 3];
        let indices_1d: Tensor<WgpuBackend, 1, Int> =
            Tensor::from_data(indices_data.as_slice(), &device);
        let indices: Tensor<WgpuBackend, 2, Int> = indices_1d.reshape([2, 3]);

        let strides = [60, 20, 1]; // For [2, 3, 20] shape
        let batch_stride = 60;

        let result = batch_gather_fused(weights, indices, &strides, batch_stride);
        let result_data: Vec<f32> = result.into_data().to_vec().unwrap();

        // Verify results
        assert_eq!(result_data.len(), 2, "Output should have 2 elements");
    }

    #[test]
    fn test_batch_gather_4_indices() {
        let device = init_gpu_device();

        let batch_size = 4;
        let weights: Tensor<WgpuBackend, 3> = Tensor::ones([batch_size, 4, 16], &device);
        let indices: Tensor<WgpuBackend, 2, Int> = Tensor::zeros([batch_size, 4], &device);

        let strides = [16, 4, 2, 1];
        let batch_stride = 64;

        let result = batch_gather_fused(weights, indices, &strides, batch_stride);
        assert_eq!(
            result.dims()[0],
            batch_size,
            "Output batch size should match"
        );
    }

    #[test]
    fn test_batch_gather_5_indices() {
        let device = init_gpu_device();

        let batch_size = 3;
        let weights: Tensor<WgpuBackend, 3> = Tensor::ones([batch_size, 5, 32], &device);
        let indices: Tensor<WgpuBackend, 2, Int> = Tensor::zeros([batch_size, 5], &device);

        let strides = [32, 16, 8, 4, 1];
        let batch_stride = 160;

        let result = batch_gather_fused(weights, indices, &strides, batch_stride);
        assert_eq!(result.dims()[0], batch_size);
    }

    #[test]
    fn test_batch_gather_6_indices() {
        let device = init_gpu_device();

        let batch_size = 2;
        let weights: Tensor<WgpuBackend, 3> = Tensor::ones([batch_size, 6, 64], &device);
        let indices: Tensor<WgpuBackend, 2, Int> = Tensor::zeros([batch_size, 6], &device);

        let strides = [64, 32, 16, 8, 4, 1];
        let batch_stride = 384;

        let result = batch_gather_fused(weights, indices, &strides, batch_stride);
        assert_eq!(result.dims()[0], batch_size);
    }

    #[test]
    fn test_batch_gather_dynamic_7_indices() {
        let device = init_gpu_device();

        // Create a tensor that requires 7 indices
        // This tests the dynamic fallback path
        let batch_size = 4;
        let weights: Tensor<WgpuBackend, 3> = Tensor::ones([batch_size, 2, 128], &device);
        let indices: Tensor<WgpuBackend, 2, Int> = Tensor::zeros([batch_size, 7], &device);

        let strides = [64, 32, 16, 8, 4, 2, 1];
        let batch_stride = 256;

        // Should not panic
        let result = batch_gather_fused(weights, indices, &strides, batch_stride);
        assert_eq!(result.dims()[0], batch_size);
    }

    #[test]
    fn test_batch_gather_dynamic_8_indices() {
        let device = init_gpu_device();

        let batch_size = 2;
        let weights: Tensor<WgpuBackend, 3> = Tensor::ones([batch_size, 2, 256], &device);
        let indices: Tensor<WgpuBackend, 2, Int> = Tensor::zeros([batch_size, 8], &device);

        let strides = [128, 64, 32, 16, 8, 4, 2, 1];
        let batch_stride = 512;

        let result = batch_gather_fused(weights, indices, &strides, batch_stride);
        assert_eq!(result.dims()[0], batch_size);
    }
}
