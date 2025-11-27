//! Distance and kernel utilities for tensor computation.
//!
//! This module provides GPU-accelerated distance and kernel computations:
//!
//! - [`pairwise_distances_sq`]: Squared Euclidean distances between points
//! - [`pairwise_distances`]: Euclidean distances between points
//! - [`gaussian_kernel`]: Gaussian (RBF) kernel from distances
//!
//! These are general-purpose utilities used across many models.

use burn::tensor::Tensor;
use crate::backend::WgpuBackend;

/// Compute pairwise squared Euclidean distances.
///
/// Given positions [N, D], computes [N, N] squared distance matrix where:
/// ```text
/// dist_sq[i,j] = ||pos[i] - pos[j]||^2
/// ```
///
/// Uses the identity:
/// ```text
/// ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * dot(a, b)
/// ```
///
/// # Arguments
/// * `positions` - Position matrix [N, D]
///
/// # Returns
/// Squared distance matrix [N, N]
///
/// # Example
/// ```rust,ignore
/// let positions: Tensor<WgpuBackend, 2> = /* [N, D] */;
/// let dist_sq = pairwise_distances_sq(&positions);
/// ```
pub fn pairwise_distances_sq(
    positions: &Tensor<WgpuBackend, 2>,
) -> Tensor<WgpuBackend, 2> {
    // ||pos[i] - pos[j]||^2 = ||pos[i]||^2 + ||pos[j]||^2 - 2 * dot(pos[i], pos[j])
    // sum_dim(1) on [N, D] returns [N, 1] in Burn 0.19
    let sq_norms = positions.clone().powf_scalar(2.0).sum_dim(1); // [N, 1]
    
    // For broadcasting: need [N, 1] + [1, N] -> [N, N]
    // sq_norms is already [N, 1], transpose to get [1, N]
    let sq_norms_row = sq_norms.clone(); // [N, 1]
    let sq_norms_col = sq_norms.transpose(); // [1, N]
    let dots = positions.clone().matmul(positions.clone().transpose()); // [N, N]

    // Clamp to avoid negative values from numerical error
    (sq_norms_row + sq_norms_col - dots.mul_scalar(2.0)).clamp(0.0, f32::MAX)
}

/// Compute pairwise Euclidean distances.
///
/// # Arguments
/// * `positions` - Position matrix [N, D]
///
/// # Returns
/// Distance matrix [N, N]
pub fn pairwise_distances(
    positions: &Tensor<WgpuBackend, 2>,
) -> Tensor<WgpuBackend, 2> {
    pairwise_distances_sq(positions).sqrt()
}

/// Compute Gaussian kernel weights from squared distances.
///
/// ```text
/// kernel[i,j] = exp(-dist_sq[i,j] / (2 * sigma^2))
/// ```
///
/// Also known as the Radial Basis Function (RBF) kernel.
///
/// # Arguments
/// * `dist_sq` - Squared distance matrix [N, N]
/// * `sigma` - Gaussian kernel width (standard deviation)
///
/// # Returns
/// Gaussian kernel matrix [N, N]
///
/// # Example
/// ```rust,ignore
/// let dist_sq = pairwise_distances_sq(&positions);
/// let kernel = gaussian_kernel(&dist_sq, 1.0);
/// ```
pub fn gaussian_kernel(
    dist_sq: &Tensor<WgpuBackend, 2>,
    sigma: f32,
) -> Tensor<WgpuBackend, 2> {
    let sigma2 = sigma * sigma;
    (-dist_sq.clone() / (2.0 * sigma2)).exp()
}

/// Compute Laplacian kernel from distances.
///
/// ```text
/// kernel[i,j] = exp(-dist[i,j] / sigma)
/// ```
///
/// # Arguments
/// * `dist` - Distance matrix [N, N]
/// * `sigma` - Kernel width
///
/// # Returns
/// Laplacian kernel matrix [N, N]
pub fn laplacian_kernel(
    dist: &Tensor<WgpuBackend, 2>,
    sigma: f32,
) -> Tensor<WgpuBackend, 2> {
    (-dist.clone() / sigma).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use crate::backend::init_gpu_device;

    #[test]
    fn test_pairwise_distances_symmetric() {
        let device = init_gpu_device();
        let n = 5;
        let d = 3;
        
        let positions: Tensor<WgpuBackend, 2> = 
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        
        let dist = pairwise_distances(&positions);
        let dist_data: Vec<f32> = dist.into_data().to_vec().expect("dist to vec");
        
        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                let d_ij = dist_data[i * n + j];
                let d_ji = dist_data[j * n + i];
                assert!((d_ij - d_ji).abs() < 1e-5, "Distance matrix should be symmetric");
            }
        }
    }
    
    #[test]
    fn test_pairwise_distances_diagonal_zero() {
        let device = init_gpu_device();
        let n = 5;
        let d = 3;
        
        let positions: Tensor<WgpuBackend, 2> = 
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        
        let dist = pairwise_distances(&positions);
        let dist_data: Vec<f32> = dist.into_data().to_vec().expect("dist to vec");
        
        // Check diagonal is zero
        for i in 0..n {
            let diag = dist_data[i * n + i];
            assert!(diag.abs() < 1e-5, "Diagonal should be zero, got {}", diag);
        }
    }
    
    #[test]
    fn test_gaussian_kernel_diagonal_one() {
        let device = init_gpu_device();
        let n = 5;
        let d = 3;
        
        let positions: Tensor<WgpuBackend, 2> = 
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        
        let dist_sq = pairwise_distances_sq(&positions);
        let kernel = gaussian_kernel(&dist_sq, 1.0);
        let kernel_data: Vec<f32> = kernel.into_data().to_vec().expect("kernel to vec");
        
        // Check diagonal is 1 (exp(0) = 1)
        for i in 0..n {
            let diag = kernel_data[i * n + i];
            assert!((diag - 1.0).abs() < 1e-5, "Diagonal should be 1, got {}", diag);
        }
    }
}

