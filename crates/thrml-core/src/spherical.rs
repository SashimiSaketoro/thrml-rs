//! Spherical coordinate utilities for sphere optimization.
//!
//! This module provides the `SphericalCoords` struct for representing points
//! in spherical coordinates (r, theta, phi) and converting between spherical
//! and Cartesian coordinate systems.
//!
//! ## Performance Notes
//!
//! Some operations require CPU fallbacks because Burn 0.19 lacks certain
//! trigonometric tensor operations:
//!
//! - **`acos()`**: Used in Cartesian→Spherical conversion for theta.
//!   Currently computed on CPU via `into_data().to_vec()` round-trip.
//! - **`atan2()`**: Used in Cartesian→Spherical conversion for phi.
//!   Currently computed on CPU via `into_data().to_vec()` round-trip.
//!
//! For large datasets (>100K points), consider implementing custom CubeCL
//! GPU kernels for these operations. The `fmod_scalar()` method IS available
//! and used for angle normalization.

use burn::tensor::{Distribution, Tensor};
use crate::backend::WgpuBackend;

/// Spherical coordinates (r, theta, phi) for N points.
///
/// The coordinate system uses physics convention:
/// - `r`: Radial distance \[N\], in \[min_radius, max_radius\]
/// - `theta`: Polar angle \[N\], in \[0, π\] (angle from +z axis)
/// - `phi`: Azimuthal angle \[N\], in \[0, 2π\] (angle from +x axis in xy plane)
///
/// Conversion to Cartesian:
/// - x = r * sin(theta) * cos(phi)
/// - y = r * sin(theta) * sin(phi)
/// - z = r * cos(theta)
#[derive(Clone, Debug)]
pub struct SphericalCoords {
    /// Radial distance from origin
    pub r: Tensor<WgpuBackend, 1>,
    /// Polar angle (from +z axis), in [0, π]
    pub theta: Tensor<WgpuBackend, 1>,
    /// Azimuthal angle (from +x axis in xy plane), in [0, 2π]
    pub phi: Tensor<WgpuBackend, 1>,
}

impl SphericalCoords {
    /// Creates new spherical coords from component tensors.
    ///
    /// # Arguments
    ///
    /// * `r` - Radial distances \[N\]
    /// * `theta` - Polar angles \[N\]
    /// * `phi` - Azimuthal angles \[N\]
    pub fn new(
        r: Tensor<WgpuBackend, 1>,
        theta: Tensor<WgpuBackend, 1>,
        phi: Tensor<WgpuBackend, 1>,
    ) -> Self {
        Self { r, theta, phi }
    }

    /// Number of points.
    pub fn len(&self) -> usize {
        self.r.dims()[0]
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the device these tensors are on.
    pub fn device(&self) -> burn::backend::wgpu::WgpuDevice {
        self.r.device()
    }

    /// Initializes with random angles and given radii.
    ///
    /// Generates uniform random theta in \[0, π\] and phi in \[0, 2π\].
    ///
    /// # Arguments
    ///
    /// * `n` - Number of points
    /// * `init_radii` - Initial radii for each point \[N\]
    /// * `device` - GPU device
    pub fn init_random(
        n: usize,
        init_radii: Tensor<WgpuBackend, 1>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let theta = Tensor::random(
            [n],
            Distribution::Uniform(0.0, std::f64::consts::PI),
            device,
        );
        let phi = Tensor::random(
            [n],
            Distribution::Uniform(0.0, 2.0 * std::f64::consts::PI),
            device,
        );
        Self::new(init_radii, theta, phi)
    }

    /// Initializes with uniform random distribution on the sphere.
    ///
    /// Uses proper uniform sampling on the sphere surface:
    /// - `theta = arccos(1 - 2*u)` for uniform u in \[0,1\]
    /// - `phi = 2π * v` for uniform v in \[0,1\]
    ///
    /// # Arguments
    ///
    /// * `n` - Number of points
    /// * `init_radii` - Initial radii for each point \[N\]
    /// * `device` - GPU device
    pub fn init_uniform_sphere(
        n: usize,
        init_radii: Tensor<WgpuBackend, 1>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        // For uniform distribution on sphere: theta = arccos(1 - 2*u)
        // Since Burn doesn't have acos, we compute on CPU
        let u: Tensor<WgpuBackend, 1> = Tensor::random([n], Distribution::Uniform(0.0, 1.0), device);
        let u_data: Vec<f32> = u.into_data().to_vec().expect("u to vec");
        let theta_data: Vec<f32> = u_data.iter()
            .map(|&u_val| (1.0 - 2.0 * u_val).clamp(-1.0, 1.0).acos())
            .collect();
        let theta = Tensor::from_data(theta_data.as_slice(), device);
        
        let phi = Tensor::random(
            [n],
            Distribution::Uniform(0.0, 2.0 * std::f64::consts::PI),
            device,
        );
        Self::new(init_radii, theta, phi)
    }

    /// Convert to Cartesian coordinates [N, 3].
    ///
    /// Returns a 2D tensor with columns [x, y, z]:
    /// - x = r * sin(theta) * cos(phi)
    /// - y = r * sin(theta) * sin(phi)
    /// - z = r * cos(theta)
    pub fn to_cartesian(&self) -> Tensor<WgpuBackend, 2> {
        // Compute sin/cos of angles
        let sin_theta = self.theta.clone().sin();
        let cos_theta = self.theta.clone().cos();
        let sin_phi = self.phi.clone().sin();
        let cos_phi = self.phi.clone().cos();

        // Compute x, y, z components
        let r = self.r.clone();
        let x = r.clone() * sin_theta.clone() * cos_phi;
        let y = r.clone() * sin_theta * sin_phi;
        let z = r * cos_theta;

        // Stack to [N, 3]
        Tensor::stack(vec![x, y, z], 1)
    }

    /// Convert from Cartesian coordinates [N, 3].
    ///
    /// # Arguments
    /// * `cart` - Cartesian coordinates tensor [N, 3] with columns [x, y, z]
    pub fn from_cartesian(cart: Tensor<WgpuBackend, 2>) -> Self {
        let n = cart.dims()[0];
        let device = cart.device();
        
        // Extract x, y, z columns - use reshape instead of squeeze
        let x_2d = cart.clone().slice([0..n, 0..1]);
        let y_2d = cart.clone().slice([0..n, 1..2]);
        let z_2d = cart.clone().slice([0..n, 2..3]);
        
        // Flatten to 1D
        let x: Tensor<WgpuBackend, 1> = x_2d.reshape([n as i32]);
        let y: Tensor<WgpuBackend, 1> = y_2d.reshape([n as i32]);
        let z: Tensor<WgpuBackend, 1> = z_2d.reshape([n as i32]);

        // Compute r = sqrt(x^2 + y^2 + z^2)
        let r = (x.clone().powf_scalar(2.0)
            + y.clone().powf_scalar(2.0)
            + z.clone().powf_scalar(2.0))
        .sqrt();

        // Compute theta = arccos(z / r) on CPU (Burn doesn't have acos)
        let eps = 1e-8;
        let z_data: Vec<f32> = z.into_data().to_vec().expect("z to vec");
        let r_data: Vec<f32> = r.clone().into_data().to_vec().expect("r to vec");
        let theta_data: Vec<f32> = z_data.iter().zip(r_data.iter())
            .map(|(&z_val, &r_val)| {
                let cos_theta = (z_val / (r_val + eps)).clamp(-1.0, 1.0);
                cos_theta.acos()
            })
            .collect();
        let theta = Tensor::from_data(theta_data.as_slice(), &device);
        
        // Compute phi = atan2(y, x) on CPU (Burn doesn't have atan2)
        let x_data: Vec<f32> = x.into_data().to_vec().expect("x to vec");
        let y_data: Vec<f32> = y.into_data().to_vec().expect("y to vec");
        let phi_data: Vec<f32> = y_data.iter().zip(x_data.iter())
            .map(|(&y_val, &x_val)| y_val.atan2(x_val))
            .collect();
        let phi = Tensor::from_data(phi_data.as_slice(), &device);

        Self::new(r, theta, phi)
    }

    /// Compute pairwise geodesic distances (angular separation) on unit sphere.
    ///
    /// Returns an [N, N] matrix where entry [i, j] is the great-circle angular
    /// distance between points i and j. This is independent of radius.
    ///
    /// Uses the spherical law of cosines:
    /// cos(d) = sin(θ₁)sin(θ₂)cos(φ₁-φ₂) + cos(θ₁)cos(θ₂)
    ///
    /// Note: Requires CPU fallback for acos operation.
    pub fn geodesic_distances(&self) -> Tensor<WgpuBackend, 2> {
        let n = self.len();
        let device = self.device();
        
        // Precompute trig values
        let sin_theta = self.theta.clone().sin();
        let cos_theta = self.theta.clone().cos();
        
        // For pairwise: sin(θi)*sin(θj) and cos(θi)*cos(θj)
        // Use outer products: sin_theta @ sin_theta^T
        let sin_t_2d = sin_theta.clone().reshape([n as i32, 1]);
        let cos_t_2d = cos_theta.clone().reshape([n as i32, 1]);
        
        let sin_outer = sin_t_2d.clone().matmul(sin_t_2d.transpose()); // [N, N]
        let cos_outer = cos_t_2d.clone().matmul(cos_t_2d.transpose()); // [N, N]
        
        // Compute cos(φi - φj) = cos(φi)cos(φj) + sin(φi)sin(φj)
        let sin_phi = self.phi.clone().sin();
        let cos_phi = self.phi.clone().cos();
        let sin_p_2d = sin_phi.reshape([n as i32, 1]);
        let cos_p_2d = cos_phi.reshape([n as i32, 1]);
        
        let cos_phi_diff = cos_p_2d.clone().matmul(cos_p_2d.transpose())
            + sin_p_2d.clone().matmul(sin_p_2d.transpose()); // [N, N]
        
        // cos(d) = sin_outer * cos_phi_diff + cos_outer
        let cos_d = sin_outer * cos_phi_diff + cos_outer;
        let cos_d_clamped = cos_d.clamp(-1.0, 1.0);
        
        // acos on CPU (Burn doesn't have tensor acos)
        let cos_d_data: Vec<f32> = cos_d_clamped.into_data().to_vec().expect("cos_d to vec");
        let distances: Vec<f32> = cos_d_data.iter().map(|&c| c.acos()).collect();
        
        // Create 1D tensor and reshape to 2D
        let dist_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(distances.as_slice(), &device);
        dist_1d.reshape([n as i32, n as i32])
    }
    
    /// Computes geodesic distance to a single point.
    ///
    /// # Arguments
    ///
    /// * `other_theta` - Polar angle of target point
    /// * `other_phi` - Azimuthal angle of target point
    ///
    /// # Returns
    ///
    /// Tensor \[N\] of angular distances from each point to the target.
    pub fn geodesic_distance_to(
        &self,
        other_theta: f32,
        other_phi: f32,
    ) -> Tensor<WgpuBackend, 1> {
        let device = self.device();
        
        let sin_theta = self.theta.clone().sin();
        let cos_theta = self.theta.clone().cos();
        
        let other_sin_t = other_theta.sin();
        let other_cos_t = other_theta.cos();
        
        // cos(φi - φ_other)
        let phi_diff = self.phi.clone() - other_phi;
        let cos_phi_diff = phi_diff.cos();
        
        // cos(d) = sin(θi)*sin(θ_other)*cos(Δφ) + cos(θi)*cos(θ_other)
        let cos_d = sin_theta.mul_scalar(other_sin_t) * cos_phi_diff
            + cos_theta.mul_scalar(other_cos_t);
        let cos_d_clamped = cos_d.clamp(-1.0, 1.0);
        
        // acos on CPU
        let cos_d_data: Vec<f32> = cos_d_clamped.into_data().to_vec().expect("cos_d to vec");
        let distances: Vec<f32> = cos_d_data.iter().map(|&c| c.acos()).collect();
        
        Tensor::from_data(distances.as_slice(), &device)
    }

    /// Clamp radii to valid range.
    ///
    /// # Arguments
    /// * `min_radius` - Minimum allowed radius
    /// * `max_radius` - Maximum allowed radius
    pub fn clamp_radii(self, min_radius: f32, max_radius: f32) -> Self {
        Self {
            r: self.r.clamp(min_radius, max_radius),
            theta: self.theta,
            phi: self.phi,
        }
    }

    /// Normalize angles to standard ranges.
    ///
    /// - theta: [0, π]
    /// - phi: [0, 2π]
    pub fn normalize_angles(self) -> Self {
        let two_pi = 2.0 * std::f32::consts::PI;
        let pi = std::f32::consts::PI;
        
        // Clamp theta to [0, π]
        let theta = self.theta.clamp(0.0, pi);
        
        // Wrap phi to [0, 2π] using fmod_scalar (available in Burn 0.19)
        // First ensure phi is positive, then apply modulo
        let phi_positive = self.phi + two_pi; // Shift to ensure positive
        let phi = phi_positive.fmod_scalar(two_pi);
        
        Self {
            r: self.r,
            theta,
            phi,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::init_gpu_device;

    #[test]
    fn test_spherical_to_cartesian_roundtrip() {
        let device = init_gpu_device();
        let n = 10;
        
        // Create random spherical coords
        let r = Tensor::random([n], Distribution::Uniform(1.0, 10.0), &device);
        let theta = Tensor::random([n], Distribution::Uniform(0.1, 3.0), &device);
        let phi = Tensor::random([n], Distribution::Uniform(0.0, 6.28), &device);
        
        let coords = SphericalCoords::new(r.clone(), theta.clone(), phi.clone());
        
        // Convert to Cartesian and back
        let cart = coords.to_cartesian();
        let coords2 = SphericalCoords::from_cartesian(cart);
        
        // Check r matches (theta/phi may have sign differences at poles)
        let r_data: Vec<f32> = r.into_data().to_vec().expect("r to vec");
        let r2_data: Vec<f32> = coords2.r.into_data().to_vec().expect("r2 to vec");
        
        for (r1, r2) in r_data.iter().zip(r2_data.iter()) {
            assert!((r1 - r2).abs() < 1e-5, "Radii should match: {} vs {}", r1, r2);
        }
    }
    
    #[test]
    fn test_geodesic_distances() {
        let device = init_gpu_device();
        
        // Create points at known positions
        // Point 0: north pole (theta=0)
        // Point 1: equator, phi=0 (theta=π/2, phi=0)
        // Point 2: equator, phi=π/2 (theta=π/2, phi=π/2)
        let pi = std::f32::consts::PI;
        let r = Tensor::from_data([1.0f32, 1.0, 1.0].as_slice(), &device);
        let theta = Tensor::from_data([0.01f32, pi / 2.0, pi / 2.0].as_slice(), &device);
        let phi = Tensor::from_data([0.0f32, 0.0, pi / 2.0].as_slice(), &device);
        
        let coords = SphericalCoords::new(r, theta, phi);
        let distances = coords.geodesic_distances();
        
        let dist_data: Vec<f32> = distances.into_data().to_vec().expect("dist to vec");
        
        // Diagonal should be ~0
        assert!(dist_data[0].abs() < 0.1, "Self-distance should be ~0");
        assert!(dist_data[4].abs() < 0.1, "Self-distance should be ~0");
        
        // North pole to equator should be ~π/2
        assert!((dist_data[1] - pi / 2.0).abs() < 0.1, "Pole to equator ~π/2");
        
        // Two equator points 90° apart should be ~π/2
        assert!((dist_data[5] - pi / 2.0).abs() < 0.1, "90° apart ~π/2");
    }
}

