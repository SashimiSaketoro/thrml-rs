//! Forward pass implementation for batch gather kernel.
//!
//! This module provides the kernel launch logic for the fused batch gather
//! operation. The implementation is generic over CubeBackend parameters.

use super::kernel::{
    batch_gather_kernel_1, batch_gather_kernel_2, batch_gather_kernel_3, batch_gather_kernel_4,
    batch_gather_kernel_5, batch_gather_kernel_6, batch_gather_kernel_dynamic,
};
use burn::tensor::ops::{FloatTensor, IntTensor};
use burn::tensor::{Int, Shape, Tensor};
use burn_cubecl::{
    element::BoolElement, kernel::into_contiguous, tensor::CubeTensor, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::{CubeCount, CubeDim};
use thrml_core::backend::WgpuBackend;

/// Launch the batch gather kernel on any CubeBackend.
///
/// This is the internal implementation that directly launches the GPU kernel.
/// It is generic over the CubeBackend parameters (Runtime, Float, Int, Bool types).
///
/// # Arguments
/// * `weights` - Weight tensor (flattened for linear indexing)
/// * `indices` - Index tensor [batch_size, n_indices]
/// * `strides` - Stride values for each index dimension
/// * `batch_stride` - Stride for the batch dimension
///
/// # Returns
/// Gathered values `[batch_size]`
pub fn launch_batch_gather<R, F, I, BT>(
    weights: FloatTensor<CubeBackend<R, F, I, BT>>,
    indices: IntTensor<CubeBackend<R, F, I, BT>>,
    strides: &[usize],
    batch_stride: usize,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    // Define cube dimensions
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };

    // Ensure tensors are on the same device
    weights.assert_is_on_same_device(&indices);

    // Make tensors contiguous for kernel access
    let weights = into_contiguous(weights);
    let indices = into_contiguous(indices);

    // Get dimensions
    let idx_dims = indices.shape.dims.clone();
    let batch_size = idx_dims[0];
    let n_indices = if idx_dims.len() > 1 { idx_dims[1] } else { 1 };

    // Flatten weights for linear indexing
    let total_weight_elements = weights.shape.num_elements();
    let weights_shape = Shape::from(vec![total_weight_elements]);
    let weights_flat = CubeTensor::new(
        weights.client.clone(),
        weights.handle.clone(),
        weights_shape,
        weights.device.clone(),
        weights.strides.clone(),
        weights.dtype,
    );

    // Output shape: [batch_size]
    let shape_out = Shape::from(vec![batch_size]);

    // Create output buffer
    let buffer = weights.client.empty(batch_size * core::mem::size_of::<F>());

    // Create output tensor
    let output = CubeTensor::new_contiguous(
        weights.client.clone(),
        weights.device.clone(),
        shape_out,
        buffer,
        F::dtype(),
    );

    // Calculate number of cubes needed
    let cubes_needed = f32::ceil(batch_size as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    // Helper macro to get stride at index or default to 0
    let get_stride = |idx: usize| strides.get(idx).copied().unwrap_or(0) as u32;

    // Launch appropriate kernel based on number of indices
    match n_indices {
        1 => {
            batch_gather_kernel_1::launch::<F, I, R>(
                &weights.client,
                cube_count,
                cube_dim,
                weights_flat.as_tensor_arg::<F>(1),
                indices.as_tensor_arg::<I>(1),
                output.as_tensor_arg::<F>(1),
                get_stride(0),
                batch_stride as u32,
            );
        }
        2 => {
            batch_gather_kernel_2::launch::<F, I, R>(
                &weights.client,
                cube_count,
                cube_dim,
                weights_flat.as_tensor_arg::<F>(1),
                indices.as_tensor_arg::<I>(1),
                output.as_tensor_arg::<F>(1),
                get_stride(0),
                get_stride(1),
                batch_stride as u32,
            );
        }
        3 => {
            batch_gather_kernel_3::launch::<F, I, R>(
                &weights.client,
                cube_count,
                cube_dim,
                weights_flat.as_tensor_arg::<F>(1),
                indices.as_tensor_arg::<I>(1),
                output.as_tensor_arg::<F>(1),
                get_stride(0),
                get_stride(1),
                get_stride(2),
                batch_stride as u32,
            );
        }
        4 => {
            batch_gather_kernel_4::launch::<F, I, R>(
                &weights.client,
                cube_count,
                cube_dim,
                weights_flat.as_tensor_arg::<F>(1),
                indices.as_tensor_arg::<I>(1),
                output.as_tensor_arg::<F>(1),
                get_stride(0),
                get_stride(1),
                get_stride(2),
                get_stride(3),
                batch_stride as u32,
            );
        }
        5 => {
            batch_gather_kernel_5::launch::<F, I, R>(
                &weights.client,
                cube_count,
                cube_dim,
                weights_flat.as_tensor_arg::<F>(1),
                indices.as_tensor_arg::<I>(1),
                output.as_tensor_arg::<F>(1),
                get_stride(0),
                get_stride(1),
                get_stride(2),
                get_stride(3),
                get_stride(4),
                batch_stride as u32,
            );
        }
        6 => {
            batch_gather_kernel_6::launch::<F, I, R>(
                &weights.client,
                cube_count,
                cube_dim,
                weights_flat.as_tensor_arg::<F>(1),
                indices.as_tensor_arg::<I>(1),
                output.as_tensor_arg::<F>(1),
                get_stride(0),
                get_stride(1),
                get_stride(2),
                get_stride(3),
                get_stride(4),
                get_stride(5),
                batch_stride as u32,
            );
        }
        _ => {
            // Dynamic fallback for 7+ indices
            let strides_array: Vec<u32> = strides.iter().map(|&s| s as u32).collect();
            let strides_tensor = create_strides_tensor::<R, I>(&strides_array, &weights);
            batch_gather_kernel_dynamic::launch::<F, I, R>(
                &weights.client,
                cube_count,
                cube_dim,
                weights_flat.as_tensor_arg::<F>(1),
                indices.as_tensor_arg::<I>(1),
                output.as_tensor_arg::<F>(1),
                strides_tensor.as_tensor_arg::<I>(1),
                n_indices as u32,
                batch_stride as u32,
            );
        }
    }

    output
}

/// Helper to create strides tensor for dynamic kernel
fn create_strides_tensor<R: CubeRuntime, I: IntElement>(
    strides: &[u32],
    weights: &CubeTensor<R>,
) -> CubeTensor<R> {
    let n = strides.len();
    let shape = Shape::from(vec![n]);

    // Create buffer with stride values as I elements (i32)
    let data_bytes: Vec<u8> = strides
        .iter()
        .flat_map(|&s| {
            let bytes: [u8; 4] = (s as i32).to_ne_bytes();
            bytes.to_vec()
        })
        .collect();

    let buffer = weights.client.create(&data_bytes);
    CubeTensor::new_contiguous(
        weights.client.clone(),
        weights.device.clone(),
        shape,
        buffer,
        I::dtype(),
    )
}

/// Execute the fused batch gather kernel.
///
/// This is a convenience function that works with high-level Tensor types
/// on the default WgpuBackend.
///
/// # Arguments
/// * `weights` - Weight tensor [n_nodes, k, dim]
/// * `indices` - Index tensor [batch_size, n_indices]
/// * `strides` - Stride values for each index dimension
/// * `batch_stride` - Stride for the batch dimension
///
/// # Returns
/// Gathered values `[batch_size]`
pub fn batch_gather_fused(
    weights: Tensor<WgpuBackend, 3>,
    indices: Tensor<WgpuBackend, 2, Int>,
    strides: &[usize],
    batch_stride: usize,
) -> Tensor<WgpuBackend, 1> {
    use burn::tensor::TensorPrimitive;
    use cubecl::wgpu::WgpuRuntime;

    let weights_prim = weights.into_primitive().tensor();
    // For Int tensors, into_primitive() returns the CubeTensor directly
    let indices_prim = indices.into_primitive();

    // WgpuBackend is CubeBackend<WgpuRuntime, f32, i32, u32>
    let output = launch_batch_gather::<WgpuRuntime, f32, i32, u32>(
        weights_prim,
        indices_prim,
        strides,
        batch_stride,
    );

    Tensor::from_primitive(TensorPrimitive::Float(output))
}
