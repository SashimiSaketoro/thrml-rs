//! Forward pass implementation for Gumbel-max kernel.
//!
//! This module provides the kernel launch logic for the fused Gumbel-max
//! categorical sampling operation. The implementation is generic over
//! CubeBackend parameters.

use super::kernel::gumbel_argmax_kernel;
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Int, Shape, Tensor};
use burn_cubecl::{
    element::BoolElement, kernel::into_contiguous, tensor::CubeTensor, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::{CubeCount, CubeDim};
use thrml_core::backend::CubeWgpuBackend;

/// Launch the Gumbel-max argmax kernel on any CubeBackend.
///
/// This is the internal implementation that directly launches the GPU kernel.
/// It is generic over the CubeBackend parameters (Runtime, Float, Int, Bool types).
///
/// # Arguments
///
/// * `logits` - Log-probabilities \[n_samples, n_categories\]
/// * `uniform` - Pre-generated uniform samples \[n_samples, n_categories\]
///
/// # Returns
///
/// Category indices \[n_samples\] as IntTensor.
pub fn launch_gumbel_argmax<R, F, I, BT>(
    logits: FloatTensor<CubeBackend<R, F, I, BT>>,
    uniform: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> IntTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    // Define cube dimensions - one thread per sample
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };

    // Ensure tensors are on the same device
    logits.assert_is_on_same_device(&uniform);

    // Make tensors contiguous for kernel access
    let logits = into_contiguous(logits);
    let uniform = into_contiguous(uniform);

    // Get dimensions: [n_samples, n_categories]
    let n_dims = logits.shape.num_dims();
    assert!(n_dims == 2, "logits must be 2D [n_samples, n_categories]");

    let n_samples = logits.shape[0];
    let n_categories = logits.shape[1];

    // Output shape: [n_samples]
    let shape_out = Shape::from(vec![n_samples]);

    // Create output buffer for integers
    let buffer = logits.client.empty(n_samples * core::mem::size_of::<I>());

    // Create output tensor as IntTensor
    let output = CubeTensor::new_contiguous(
        logits.client.clone(),
        logits.device.clone(),
        shape_out,
        buffer,
        I::dtype(),
    );

    // Calculate number of cubes needed (one thread per sample)
    let cubes_needed = f32::ceil(n_samples as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    // Launch the kernel with n_categories as comptime parameter
    gumbel_argmax_kernel::launch::<F, I, R>(
        &logits.client,
        cube_count,
        cube_dim,
        logits.as_tensor_arg::<F>(1),
        uniform.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<I>(1),
        n_categories as u32,
    );

    output
}

/// Executes the fused Gumbel-max argmax kernel.
///
/// This is a convenience function that works with high-level Tensor types
/// on the raw CubeWgpuBackend (not Fusion).
///
/// # Arguments
///
/// * `logits` - Log-probabilities \[n_samples, n_categories\]
/// * `uniform` - Pre-generated uniform samples \[n_samples, n_categories\]
///
/// # Returns
///
/// Category indices \[n_samples\] as integers.
pub fn gumbel_argmax_fused(
    logits: Tensor<CubeWgpuBackend, 2>,
    uniform: Tensor<CubeWgpuBackend, 2>,
) -> Tensor<CubeWgpuBackend, 1, Int> {
    use cubecl::wgpu::WgpuRuntime;

    let logits_prim = logits.into_primitive().tensor();
    let uniform_prim = uniform.into_primitive().tensor();

    // WgpuBackend is CubeBackend<WgpuRuntime, f32, i32, u32>
    let output = launch_gumbel_argmax::<WgpuRuntime, f32, i32, u32>(logits_prim, uniform_prim);

    // Convert IntTensor (CubeTensor) to Tensor<B, D, Int>
    Tensor::from_primitive(output)
}
