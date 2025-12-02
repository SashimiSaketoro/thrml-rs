//! Forward pass implementation for cosine similarity kernels.

use super::kernel::{
    cosine_similarity_batched_kernel, cosine_similarity_prenorm_kernel,
    cosine_similarity_single_kernel,
};
use burn::tensor::ops::FloatTensor;
use burn::tensor::{Shape, Tensor};
use burn_cubecl::{
    element::BoolElement, kernel::into_contiguous, tensor::CubeTensor, CubeBackend, CubeRuntime,
    FloatElement, IntElement,
};
use cubecl::{CubeCount, CubeDim};
use thrml_core::backend::CubeWgpuBackend;

/// Launch single-query cosine similarity kernel.
///
/// # Arguments
/// * `query` - Query vector `[dim]`
/// * `vectors` - Matrix of vectors `[n_vectors, dim]`
///
/// # Returns
/// Similarities `[n_vectors]`
pub fn launch_cosine_similarity_single<R, F, I, BT>(
    query: FloatTensor<CubeBackend<R, F, I, BT>>,
    vectors: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };

    let query = into_contiguous(query);
    let vectors = into_contiguous(vectors);

    assert!(query.shape.num_dims() == 1, "query must be 1D `[dim]`");
    assert!(
        vectors.shape.num_dims() == 2,
        "vectors must be 2D `[n_vectors, dim]`"
    );

    let dim = query.shape[0];
    let n_vectors = vectors.shape[0];
    assert_eq!(vectors.shape[1], dim, "dimension mismatch");

    let shape_out = Shape::from(vec![n_vectors]);
    let buffer = query.client.empty(n_vectors * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        shape_out,
        buffer,
        F::dtype(),
    );

    let cubes_needed = f32::ceil(n_vectors as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    cosine_similarity_single_kernel::launch::<F, R>(
        &query.client,
        cube_count,
        cube_dim,
        query.as_tensor_arg::<F>(1),
        vectors.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
        dim as u32,
    );

    output
}

/// Launch batched cosine similarity kernel.
///
/// # Arguments
/// * `queries` - Query matrix `[n_queries, dim]`
/// * `vectors` - Vector matrix `[n_vectors, dim]`
///
/// # Returns
/// Similarities `[n_queries, n_vectors]`
pub fn launch_cosine_similarity_batched<R, F, I, BT>(
    queries: FloatTensor<CubeBackend<R, F, I, BT>>,
    vectors: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };

    let queries = into_contiguous(queries);
    let vectors = into_contiguous(vectors);

    assert!(
        queries.shape.num_dims() == 2,
        "queries must be 2D `[n_queries, dim]`"
    );
    assert!(
        vectors.shape.num_dims() == 2,
        "vectors must be 2D `[n_vectors, dim]`"
    );

    let n_queries = queries.shape[0];
    let dim = queries.shape[1];
    let n_vectors = vectors.shape[0];
    assert_eq!(vectors.shape[1], dim, "dimension mismatch");

    let shape_out = Shape::from(vec![n_queries, n_vectors]);
    let total = n_queries * n_vectors;
    let buffer = queries.client.empty(total * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        queries.client.clone(),
        queries.device.clone(),
        shape_out,
        buffer,
        F::dtype(),
    );

    let cubes_needed = f32::ceil(total as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    cosine_similarity_batched_kernel::launch::<F, R>(
        &queries.client,
        cube_count,
        cube_dim,
        queries.as_tensor_arg::<F>(1),
        vectors.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
        n_queries as u32,
        n_vectors as u32,
        dim as u32,
    );

    output
}

/// Launch pre-normalized cosine similarity kernel.
///
/// # Arguments
/// * `query` - Query vector `[dim]` (will be normalized)
/// * `normalized_vectors` - Pre-normalized vectors `[n_vectors, dim]`
///
/// # Returns
/// Similarities `[n_vectors]`
pub fn launch_cosine_similarity_prenorm<R, F, I, BT>(
    query: FloatTensor<CubeBackend<R, F, I, BT>>,
    normalized_vectors: FloatTensor<CubeBackend<R, F, I, BT>>,
) -> FloatTensor<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let cube_dim = CubeDim { x: 256, y: 1, z: 1 };

    let query = into_contiguous(query);
    let vectors = into_contiguous(normalized_vectors);

    let dim = query.shape[0];
    let n_vectors = vectors.shape[0];

    let shape_out = Shape::from(vec![n_vectors]);
    let buffer = query.client.empty(n_vectors * core::mem::size_of::<F>());

    let output = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        shape_out,
        buffer,
        F::dtype(),
    );

    let cubes_needed = f32::ceil(n_vectors as f32 / cube_dim.x as f32) as u32;
    let cube_count = CubeCount::Static(cubes_needed, 1, 1);

    cosine_similarity_prenorm_kernel::launch::<F, R>(
        &query.client,
        cube_count,
        cube_dim,
        query.as_tensor_arg::<F>(1),
        vectors.as_tensor_arg::<F>(1),
        output.as_tensor_arg::<F>(1),
        dim as u32,
    );

    output
}

// ============================================================================
// High-level API for CubeWgpuBackend
// ============================================================================

/// Fused cosine similarity for single query against multiple vectors.
///
/// # Arguments
/// * `query` - Query vector `[dim]`
/// * `vectors` - Matrix of vectors `[n_vectors, dim]`
///
/// # Returns
/// Similarities `[n_vectors]`
pub fn cosine_similarity_fused(
    query: Tensor<CubeWgpuBackend, 1>,
    vectors: Tensor<CubeWgpuBackend, 2>,
) -> Tensor<CubeWgpuBackend, 1> {
    use cubecl::wgpu::WgpuRuntime;
    let inner = launch_cosine_similarity_single::<WgpuRuntime, f32, i32, u32>(
        query.into_primitive().tensor(),
        vectors.into_primitive().tensor(),
    );
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(inner))
}

/// Fused batched cosine similarity.
///
/// # Arguments
/// * `queries` - Query matrix `[n_queries, dim]`
/// * `vectors` - Vector matrix `[n_vectors, dim]`
///
/// # Returns
/// Similarities `[n_queries, n_vectors]`
pub fn cosine_similarity_fused_batched(
    queries: Tensor<CubeWgpuBackend, 2>,
    vectors: Tensor<CubeWgpuBackend, 2>,
) -> Tensor<CubeWgpuBackend, 2> {
    use cubecl::wgpu::WgpuRuntime;
    let inner = launch_cosine_similarity_batched::<WgpuRuntime, f32, i32, u32>(
        queries.into_primitive().tensor(),
        vectors.into_primitive().tensor(),
    );
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(inner))
}

/// Cosine similarity with pre-normalized vectors.
///
/// # Arguments
/// * `query` - Query vector `[dim]` (will be normalized)
/// * `normalized_vectors` - Pre-normalized vectors `[n_vectors, dim]`
///
/// # Returns
/// Similarities `[n_vectors]`
pub fn cosine_similarity_prenormalized(
    query: Tensor<CubeWgpuBackend, 1>,
    normalized_vectors: Tensor<CubeWgpuBackend, 2>,
) -> Tensor<CubeWgpuBackend, 1> {
    use cubecl::wgpu::WgpuRuntime;
    let inner = launch_cosine_similarity_prenorm::<WgpuRuntime, f32, i32, u32>(
        query.into_primitive().tensor(),
        normalized_vectors.into_primitive().tensor(),
    );
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(inner))
}
