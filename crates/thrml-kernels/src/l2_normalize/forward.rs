//! Forward pass implementation for L2 normalization kernel.

use super::kernel::{l2_normalize_kernel, l2_normalize_with_norms_kernel};
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Shape, Tensor};
use burn_cubecl::{
    element::BoolElement, kernel::into_contiguous, tensor::CubeTensor, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::{CubeCount, CubeDim};
use thrml_core::backend::CubeWgpuBackend;

/// Launch the L2 normalization kernel on any CubeBackend.
///
/// # Arguments
/// * `input` - Input tensor `[n_rows, dim]`
///
/// # Returns
/// Normalized tensor `[n_rows, dim]` where each row has unit L2 norm.
pub fn launch_l2_normalize<R, F, I, BT>(
    input: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
    let input = into_contiguous(input);

    let n_dims = input.shape.num_dims();
    assert!(n_dims == 2, "input must be 2D `[n_rows, dim]`");

    let n_rows = input.shape[0];
    let dim = input.shape[1];

    let shape_out = Shape::from(vec![n_rows, dim]);
    let buffer = input.client.empty(n_rows * dim * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        buffer,
        F::dtype(),
    );

    let cubes_needed = f32::ceil(n_rows as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    l2_normalize_kernel::launch::<F, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
        dim as u32,
    );

    output
}

/// Launch L2 normalization that also returns the norms.
///
/// # Arguments
/// * `input` - Input tensor `[n_rows, dim]`
///
/// # Returns
/// Tuple of (normalized tensor `[n_rows, dim]`, norms `[n_rows]`)
#[allow(clippy::type_complexity)] // Generic return type inherently complex
pub fn launch_l2_normalize_with_norms<R, F, I, BT>(
    input: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> (
    FloatTensor<CubeBackend<R, F, I, BT>>,
    FloatTensor<CubeBackend<R, F, I, BT>>,
)
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };
    let input = into_contiguous(input);

    let n_dims = input.shape.num_dims();
    assert!(n_dims == 2, "input must be 2D `[n_rows, dim]`");

    let n_rows = input.shape[0];
    let dim = input.shape[1];

    let shape_out = Shape::from(vec![n_rows, dim]);
    let shape_norms = Shape::from(vec![n_rows]);

    let buffer_out = input.client.empty(n_rows * dim * core::mem::size_of::<F>());
    let buffer_norms = input.client.empty(n_rows * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape_out,
        buffer_out,
        F::dtype(),
    );

    let norms = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape_norms,
        buffer_norms,
        F::dtype(),
    );

    let cubes_needed = f32::ceil(n_rows as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    l2_normalize_with_norms_kernel::launch::<F, R>(
        &input.client,
        cube_count,
        cube_dim,
        input.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
        norms.as_tensor_arg::<F>(1),
        dim as u32,
    );

    (output, norms)
}

/// High-level fused L2 normalize for WgpuBackend tensors.
///
/// # Arguments
/// * `input` - Input tensor `[n_rows, dim]`
///
/// # Returns
/// Normalized tensor where each row has unit L2 norm.
pub fn l2_normalize_fused(input: Tensor<CubeWgpuBackend, 2>) -> Tensor<CubeWgpuBackend, 2> {
    use cubecl::wgpu::WgpuRuntime;
    let inner = launch_l2_normalize::<WgpuRuntime, f32, i32, u32>(input.into_primitive().tensor());
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(inner))
}
