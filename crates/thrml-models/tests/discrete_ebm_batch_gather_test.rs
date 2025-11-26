use burn::tensor::Tensor;
use thrml_core::backend::*;
use thrml_models::discrete_ebm::batch_gather;

#[test]
fn test_batch_gather_3d() {
    ensure_backend();
    let device = init_gpu_device();

    // Test case: weights [2, 3, 4], indices [2], [2]
    // This matches the Python _batch_gather behavior
    let weights_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let weights_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(weights_data.as_slice(), &device);
    let weights: Tensor<WgpuBackend, 3> = weights_1d.reshape([2, 3, 4]);

    // indices[0] = [0, 1] -> select first row for batch 0, second row for batch 1
    // indices[1] = [2, 3] -> select element 2 for batch 0, element 3 for batch 1
    let indices0: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        Tensor::from_data(vec![0i32, 1].as_slice(), &device);
    let indices1: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        Tensor::from_data(vec![2i32, 3].as_slice(), &device);

    println!("Testing batch_gather from discrete_ebm module...");
    println!("Weights shape: {:?}", weights.dims());

    let result = batch_gather(&weights, &[indices0, indices1]);
    println!("Result dims: {:?}", result.dims());

    let data: Vec<f32> = result
        .clone()
        .into_data()
        .to_vec()
        .expect("Failed to read tensor data");
    println!("Result: {:?}", data);

    // Expected values:
    // For batch 0, dim1=0, dim2=2: position = 0*3*4 + 0*4 + 2 = 2, value = data[2] = 3.0
    // For batch 1, dim1=1, dim2=3: position = 1*3*4 + 1*4 + 3 = 12 + 4 + 3 = 19, value = data[19] = 20.0
    let expected = vec![3.0, 20.0];
    println!("Expected: {:?}", expected);

    assert_eq!(result.dims(), [2]);
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "Result[{}] = {}, expected {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_batch_gather_2d() {
    ensure_backend();
    let device = init_gpu_device();

    // Test case: weights [4, 3, 1], indices [4], [4] (need 2 indices for 3D tensor)
    let weights_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let weights_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(weights_data.as_slice(), &device);
    let weights: Tensor<WgpuBackend, 3> = weights_1d.reshape([4, 3, 1]); // 3D tensor

    let indices0: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        Tensor::from_data(vec![0i32, 1, 2, 0].as_slice(), &device);
    let indices1: Tensor<WgpuBackend, 1, burn::tensor::Int> = Tensor::from_data(
        vec![0i32, 0, 0, 0].as_slice(), // All select index 0 from the last dimension
        &device,
    );

    let result = batch_gather(&weights, &[indices0, indices1]);
    let data: Vec<f32> = result
        .clone()
        .into_data()
        .to_vec()
        .expect("Failed to read tensor data");

    // Expected: [1.0, 5.0, 9.0, 10.0]
    // batch 0: weights[0, 0, 0] = 1.0
    // batch 1: weights[1, 1, 0] = 5.0
    // batch 2: weights[2, 2, 0] = 9.0
    // batch 3: weights[3, 0, 0] = 10.0
    let expected = vec![1.0, 5.0, 9.0, 10.0];

    assert_eq!(result.dims(), [4]);
    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "Result[{}] = {}, expected {}",
            i,
            got,
            exp
        );
    }
}
