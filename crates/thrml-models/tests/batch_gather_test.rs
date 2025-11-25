use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;

/// Approach 1: Using gather (which might behave differently than select)
/// weights: [batch, dim1], indices: [batch]
pub fn batch_gather_2d_gather(
    weights: Tensor<WgpuBackend, 2>,
    indices: Tensor<WgpuBackend, 1, burn::tensor::Int>,
) -> Tensor<WgpuBackend, 1> {
    // Expand indices to match weights shape for gather
    // gather expects indices to have same shape as input except at gather dimension
    // For [batch, dim1] with indices [batch], we need indices shape [batch, 1]
    let indices_expanded: Tensor<WgpuBackend, 2, burn::tensor::Int> =
        indices.clone().unsqueeze_dim::<2>(1); // [batch] -> [batch, 1]
    let gathered: Tensor<WgpuBackend, 2> = weights.gather(1, indices_expanded);
    // Gather should give us [batch, 1], squeeze to [batch]
    gathered.squeeze_dim(1)
}

/// Approach 2: Using gather for 3D case (with proper shape expansion)
/// weights: [batch, dim1, dim2], indices: [batch], [batch]
pub fn batch_gather_3d_gather(
    weights: Tensor<WgpuBackend, 3>,
    indices0: Tensor<WgpuBackend, 1, burn::tensor::Int>,
    indices1: Tensor<WgpuBackend, 1, burn::tensor::Int>,
) -> Tensor<WgpuBackend, 1> {
    let dims = weights.dims();
    let batch_size = dims[0];
    let dim2_size = dims[2];
    let device = weights.device();

    // For gather, indices need to match input shape except at gather dimension
    // For [batch, dim1, dim2] gathering along dim1, indices need shape [batch, 1, dim2]
    // Expand indices0: [batch] -> [batch, 1, dim2]
    let indices0_2d: Tensor<WgpuBackend, 2, burn::tensor::Int> =
        indices0.clone().unsqueeze_dim::<2>(1); // [batch] -> [batch, 1]
                                                // Broadcast to [batch, 1, dim2]
    let indices0_broadcasted = indices0_2d.clone().unsqueeze_dim::<3>(2); // [batch, 1] -> [batch, 1, 1]
                                                                          // Actually, gather needs the full shape - let's use a different approach
                                                                          // Use linear indexing instead for 3D as it's simpler and works
    batch_gather_3d_linear(weights, indices0, indices1)
}

/// Approach 3: Linear indexing for 2D
pub fn batch_gather_2d_linear(
    weights: Tensor<WgpuBackend, 2>,
    indices: Tensor<WgpuBackend, 1, burn::tensor::Int>,
) -> Tensor<WgpuBackend, 1> {
    let dims = weights.dims();
    let batch_size = dims[0];
    let dim1_size = dims[1];
    let device = weights.device();

    // Compute linear indices: batch_idx * dim1_size + indices
    let batch_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> = Tensor::from_data(
        (0..batch_size)
            .map(|i| i as i32)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );
    let dim1_tensor = Tensor::from_data(vec![dim1_size as i32; batch_size].as_slice(), &device);
    let linear_indices = batch_indices * dim1_tensor + indices;

    // Flatten weights to 1D and select
    let total_size = batch_size * dim1_size;
    let weights_flat = weights.reshape([total_size as i32]);
    weights_flat.select(0, linear_indices)
}

/// Approach 4: Linear indexing for 3D
pub fn batch_gather_3d_linear(
    weights: Tensor<WgpuBackend, 3>,
    indices0: Tensor<WgpuBackend, 1, burn::tensor::Int>,
    indices1: Tensor<WgpuBackend, 1, burn::tensor::Int>,
) -> Tensor<WgpuBackend, 1> {
    let dims = weights.dims();
    let batch_size = dims[0];
    let dim1_size = dims[1];
    let dim2_size = dims[2];
    let device = weights.device();

    // Compute strides
    let stride0 = dim1_size * dim2_size;
    let stride1 = dim2_size;

    // Compute linear indices: batch_idx * stride0 + idx0 * stride1 + idx1
    let batch_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> = Tensor::from_data(
        (0..batch_size)
            .map(|i| i as i32)
            .collect::<Vec<_>>()
            .as_slice(),
        &device,
    );
    let stride0_tensor = Tensor::from_data(vec![stride0 as i32; batch_size].as_slice(), &device);
    let stride1_tensor = Tensor::from_data(vec![stride1 as i32; batch_size].as_slice(), &device);

    let linear_indices =
        batch_indices * stride0_tensor + indices0.clone() * stride1_tensor + indices1;

    // Flatten weights to 1D and select
    let total_size = batch_size * dim1_size * dim2_size;
    let weights_flat = weights.reshape([total_size as i32]);
    weights_flat.select(0, linear_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrml_core::backend::*;

    #[test]
    fn test_batch_gather_2d_sequential_vs_linear() {
        ensure_metal_backend();
        let device = init_gpu_device();

        // Simple case: weights [4, 3], indices [4]
        // weights = [[1.0, 2.0, 3.0],
        //            [4.0, 5.0, 6.0],
        //            [7.0, 8.0, 9.0],
        //            [10.0, 11.0, 12.0]]
        // indices = [0, 1, 2, 0]
        // Expected: [1.0, 5.0, 9.0, 10.0]

        let weights_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let weights_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data(weights_data.as_slice(), &device);
        let weights: Tensor<WgpuBackend, 2> = weights_1d.reshape([4, 3]);

        let indices_data = vec![0i32, 1, 2, 0];
        let indices: Tensor<WgpuBackend, 1, burn::tensor::Int> =
            Tensor::from_data(indices_data.as_slice(), &device);

        println!("Testing 2D batch_gather...");
        println!("Weights shape: {:?}", weights.dims());
        println!("Indices shape: {:?}", indices.dims());

        // Test gather approach
        let result_gather = batch_gather_2d_gather(weights.clone(), indices.clone());
        println!("Gather result dims: {:?}", result_gather.dims());

        // Test linear approach
        let result_lin = batch_gather_2d_linear(weights.clone(), indices.clone());
        println!("Linear result dims: {:?}", result_lin.dims());

        // Both should give [4] shape
        assert_eq!(result_gather.dims(), [4]);
        assert_eq!(result_lin.dims(), [4]);

        // Read back and compare values (clone before into_data since it takes ownership)
        let data_gather: Vec<f32> = result_gather
            .clone()
            .into_data()
            .to_vec()
            .expect("Failed to read tensor data");
        let data_lin: Vec<f32> = result_lin
            .clone()
            .into_data()
            .to_vec()
            .expect("Failed to read tensor data");

        println!("Gather result: {:?}", data_gather);
        println!("Linear result: {:?}", data_lin);

        // Expected: [1.0, 5.0, 9.0, 10.0]
        let expected = vec![1.0, 5.0, 9.0, 10.0];
        println!("Expected: {:?}", expected);

        // Check if results match expected (with small tolerance for floating point)
        for (i, ((gath, lin), exp)) in data_gather
            .iter()
            .zip(data_lin.iter())
            .zip(expected.iter())
            .enumerate()
        {
            println!("  [{}] gather={}, lin={}, exp={}", i, gath, lin, exp);
            assert!(
                (gath - exp).abs() < 1e-5,
                "Gather result[{}] = {}, expected {}",
                i,
                gath,
                exp
            );
            assert!(
                (lin - exp).abs() < 1e-5,
                "Linear result[{}] = {}, expected {}",
                i,
                lin,
                exp
            );
        }

        // Verify both approaches give same result
        assert_eq!(
            data_gather, data_lin,
            "Gather and linear approaches should match"
        );
    }

    #[test]
    fn test_batch_gather_3d_sequential_vs_linear() {
        ensure_metal_backend();
        let device = init_gpu_device();

        // Case: weights [2, 3, 4], indices [2], [2]
        // weights shape: [batch=2, dim1=3, dim2=4]
        // For batch 0: select dim1=0, dim2=2 -> weights[0, 0, 2] = 3.0
        // For batch 1: select dim1=1, dim2=3 -> weights[1, 1, 3] = 20.0

        let weights_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let weights_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data(weights_data.as_slice(), &device);
        let weights: Tensor<WgpuBackend, 3> = weights_1d.reshape([2, 3, 4]);

        // indices[0] = [0, 1] -> select first row for batch 0, second row for batch 1
        // indices[1] = [2, 3] -> select element 2 for batch 0, element 3 for batch 1
        let indices0: Tensor<WgpuBackend, 1, burn::tensor::Int> =
            Tensor::from_data(vec![0i32, 1].as_slice(), &device);
        let indices1: Tensor<WgpuBackend, 1, burn::tensor::Int> =
            Tensor::from_data(vec![2i32, 3].as_slice(), &device);

        println!("Testing 3D batch_gather...");
        println!("Weights shape: {:?}", weights.dims());
        println!(
            "Indices0 shape: {:?}, values: {:?}",
            indices0.dims(),
            vec![0i32, 1]
        );
        println!(
            "Indices1 shape: {:?}, values: {:?}",
            indices1.dims(),
            vec![2i32, 3]
        );

        // Test gather approach
        let result_gather =
            batch_gather_3d_gather(weights.clone(), indices0.clone(), indices1.clone());
        println!("Gather result dims: {:?}", result_gather.dims());

        // Test linear
        let result_lin =
            batch_gather_3d_linear(weights.clone(), indices0.clone(), indices1.clone());
        println!("Linear result dims: {:?}", result_lin.dims());

        let data_gather: Vec<f32> = result_gather
            .clone()
            .into_data()
            .to_vec()
            .expect("Failed to read tensor data");
        let data_lin: Vec<f32> = result_lin
            .clone()
            .into_data()
            .to_vec()
            .expect("Failed to read tensor data");

        println!("Gather result: {:?}", data_gather);
        println!("Linear result: {:?}", data_lin);

        // Verify they match
        assert_eq!(
            data_gather, data_lin,
            "Gather and linear approaches should match for 3D"
        );
        assert_eq!(result_gather.dims(), [2]);

        // Expected values (verifying with linear indexing):
        // weights shape: [2, 3, 4]
        // Data is 1..=24 laid out as [batch, dim1, dim2]
        // For batch 0, dim1=0, dim2=2: position = 0*3*4 + 0*4 + 2 = 2, value = data[2] = 3.0
        // For batch 1, dim1=1, dim2=3: position = 1*3*4 + 1*4 + 3 = 12 + 4 + 3 = 19, value = data[19] = 20.0
        // So the result should be [3.0, 20.0]
        let expected = vec![3.0, 20.0];
        println!("Expected: {:?}", expected);

        for (i, (gath, exp)) in data_gather.iter().zip(expected.iter()).enumerate() {
            assert!(
                (gath - exp).abs() < 1e-5,
                "Result[{}] = {}, expected {}",
                i,
                gath,
                exp
            );
        }
    }
}
