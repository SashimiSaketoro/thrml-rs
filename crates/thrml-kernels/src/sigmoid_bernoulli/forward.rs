//! Forward pass implementation for sigmoid-Bernoulli kernel.
//!
//! This module provides the kernel launch logic for the fused sigmoid-Bernoulli
//! operation. The implementation is generic over CubeBackend parameters.

use super::kernel::sigmoid_bernoulli_kernel;
use burn::tensor::ops::FloatTensor;
use burn::tensor::Tensor;
use burn_cubecl::{
    element::BoolElement, kernel::into_contiguous, tensor::CubeTensor, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::{CubeCount, CubeDim};
use thrml_core::backend::CubeWgpuBackend;

/// Launch the sigmoid-Bernoulli kernel on any CubeBackend.
///
/// This is the internal implementation that directly launches the GPU kernel.
/// It is generic over the CubeBackend parameters (Runtime, Float, Int, Bool types).
///
/// # Arguments
/// * `gamma` - Gibbs parameters as FloatTensor
/// * `uniform` - Pre-generated uniform samples as FloatTensor
///
/// # Returns
/// Bernoulli samples (0.0 or 1.0) as FloatTensor
pub fn launch_sigmoid_bernoulli<R, F, I, BT>(
    gamma: FloatTensor<CubeBackend<R, F, I, BT>>,
    uniform: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    // Define cube dimensions - 256 threads per workgroup for element-wise ops
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };

    // Ensure tensors are on the same device
    gamma.assert_is_on_same_device(&uniform);

    // Make tensors contiguous for kernel access
    let gamma = into_contiguous(gamma);
    let uniform = into_contiguous(uniform);

    // Get output shape (same as input)
    let n_elements = gamma.shape.num_elements();
    let shape_out = gamma.shape.clone();

    // Create output buffer
    let buffer = gamma.client.empty(n_elements * core::mem::size_of::<F>());

    // Create output tensor
    let output = CubeTensor::new_contiguous(
        gamma.client.clone(),
        gamma.device.clone(),
        shape_out,
        buffer,
        F::dtype(),
    );

    // Calculate number of cubes needed
    let cubes_needed = f32::ceil(n_elements as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    // Launch the kernel
    sigmoid_bernoulli_kernel::launch::<F, R>(
        &gamma.client,
        cube_count,
        cube_dim,
        gamma.as_tensor_arg::<F>(1),
        uniform.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
    );

    output
}

/// Execute the fused sigmoid-Bernoulli sampling kernel.
///
/// This is a convenience function that works with high-level Tensor types
/// on the raw CubeWgpuBackend (not Fusion).
///
/// # Arguments
/// * `gamma` - Gibbs parameter `[n_nodes]`
/// * `uniform` - Pre-generated uniform samples `[n_nodes]`
///
/// # Returns
/// Spin samples `[n_nodes]` as floats (0.0 or 1.0)
pub fn sigmoid_bernoulli_fused(
    gamma: Tensor<CubeWgpuBackend, 1>,
    uniform: Tensor<CubeWgpuBackend, 1>,
) -> Tensor<CubeWgpuBackend, 1> {
    use burn::tensor::TensorPrimitive;
    use cubecl::wgpu::WgpuRuntime;

    let gamma_prim = gamma.into_primitive().tensor();
    let uniform_prim = uniform.into_primitive().tensor();

    // WgpuBackend is CubeBackend<WgpuRuntime, f32, i32, u32>
    let output = launch_sigmoid_bernoulli::<WgpuRuntime, f32, i32, u32>(gamma_prim, uniform_prim);

    Tensor::from_primitive(TensorPrimitive::Float(output))
}
