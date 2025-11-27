//! Hamiltonian (energy function) for sphere optimization.
//!
//! This module defines the energy landscape for water-filling dynamics.
//! The Hamiltonian consists of two terms:
//!
//! 1. **Gravity energy**: Pulls points toward their ideal radii based on prominence
//! 2. **Lateral energy**: Attracts similar embeddings to cluster together
//!
//! The total energy is: `E = E_gravity + E_lateral`

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::SphericalCoords;
use thrml_core::distance::pairwise_distances_sq;

/// Trait for sphere Hamiltonians.
///
/// A Hamiltonian defines both the energy function and its gradients (forces)
/// for use in Langevin dynamics.
pub trait SphereHamiltonian {
    /// Computes gravity energy per particle: `E_g[i] = (r_i - r_ideal_i)^2`
    fn gravity_energy(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 1>;

    /// Computes lateral energy per particle: `E_lat[i] = -sum_j sim_ij * exp(-d_ij^2 / sigma^2)`
    fn lateral_energy(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 1>;

    /// Compute total energy per particle.
    fn total_energy(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 1> {
        self.gravity_energy(coords) + self.lateral_energy(coords)
    }

    /// Compute gravity force (radial): F_g = -dE_g/dr = -2 * (r - r_ideal)
    fn gravity_force(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 1>;

    /// Compute lateral force in Cartesian coordinates [N, 3].
    fn lateral_force_cartesian(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 2>;
}

/// Water-filling Hamiltonian for sphere optimization.
///
/// This Hamiltonian implements the water-filling algorithm:
/// - High prominence points are pushed toward the core (small radii)
/// - Similar embeddings attract each other via Gaussian kernel
///
/// The energy is:
/// ```text
/// E = sum_i (r_i - r_ideal_i)^2 - sum_{i,j} sim_ij * exp(-d_ij^2 / sigma^2)
/// ```
pub struct WaterFillingHamiltonian {
    /// Ideal radii from prominence ranking \[N\]
    pub ideal_radii: Tensor<WgpuBackend, 1>,
    /// Cosine similarity matrix \[N, N\]
    pub similarity: Tensor<WgpuBackend, 2>,
    /// Gaussian interaction radius (sigma)
    pub interaction_radius: f32,
}

impl WaterFillingHamiltonian {
    /// Creates a new water-filling Hamiltonian.
    ///
    /// # Arguments
    ///
    /// * `ideal_radii` - Target radii for each point \[N\]
    /// * `similarity` - Pairwise similarity matrix \[N, N\]
    /// * `interaction_radius` - Width of Gaussian lateral interaction
    pub fn new(
        ideal_radii: Tensor<WgpuBackend, 1>,
        similarity: Tensor<WgpuBackend, 2>,
        interaction_radius: f32,
    ) -> Self {
        Self {
            ideal_radii,
            similarity,
            interaction_radius,
        }
    }
}

impl SphereHamiltonian for WaterFillingHamiltonian {
    fn gravity_energy(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 1> {
        // E_g = (r - r_ideal)^2
        let diff = coords.r.clone() - self.ideal_radii.clone();
        diff.powf_scalar(2.0)
    }

    fn lateral_energy(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 1> {
        // Convert to Cartesian for distance computation
        let positions = coords.to_cartesian();
        let n = positions.dims()[0];
        let dist_sq = pairwise_distances_sq(&positions);
        let sigma2 = self.interaction_radius * self.interaction_radius;

        // Gaussian kernel weighted by similarity
        // weights[i,j] = sim[i,j] * exp(-d[i,j]^2 / sigma^2)
        let weights = self.similarity.clone() * (-dist_sq / sigma2).exp();

        // E_lat[i] = -sum_j weights[i,j]
        // Negative because this is an attractive interaction
        // sum_dim(1) returns [N, 1], reshape to [N]
        let sum_2d = weights.sum_dim(1);
        let sum_1d: Tensor<WgpuBackend, 1> = sum_2d.reshape([n as i32]);
        -sum_1d
    }

    fn gravity_force(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 1> {
        // F_g = -dE_g/dr = -2 * (r - r_ideal)
        // This is the force pushing r toward r_ideal
        let diff = coords.r.clone() - self.ideal_radii.clone();
        diff.mul_scalar(-2.0)
    }

    fn lateral_force_cartesian(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 2> {
        // Convert to Cartesian
        let positions = coords.to_cartesian(); // [N, 3]

        // Compute distances and weights
        let dist_sq = pairwise_distances_sq(&positions);
        let sigma2 = self.interaction_radius * self.interaction_radius;

        // Gaussian weights: sim * exp(-d^2 / sigma^2)
        let weights = self.similarity.clone() * (-dist_sq / sigma2).exp(); // [N, N]

        // The lateral force on particle i from particle j is:
        // F_ij = w_ij * (pos_j - pos_i) * (2 / sigma^2)
        //
        // Total force on i: F_i = sum_j F_ij
        //
        // We can compute this efficiently as:
        // F_i = (2/sigma^2) * (sum_j w_ij * pos_j - pos_i * sum_j w_ij)
        //     = (2/sigma^2) * (weights @ positions - positions * weight_sums)

        // sum_j w_ij * pos_j = weights @ positions
        let weighted_sum = weights.clone().matmul(positions.clone()); // [N, 3]

        // sum_j w_ij
        // sum_dim(1) on [N, N] returns [N, 1] in Burn 0.19
        let weight_sums = weights.sum_dim(1); // [N, 1]

        // F_i = weighted_sum[i] - pos[i] * weight_sum[i]
        // Scale by 2/sigma^2
        let scale = 2.0 / sigma2;
        (weighted_sum - positions * weight_sums).mul_scalar(scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_gravity_energy_at_ideal() {
        let device = init_gpu_device();
        let n = 10;

        // Create coords at ideal radii
        let ideal_radii: Tensor<WgpuBackend, 1> = 
            Tensor::random([n], Distribution::Uniform(10.0, 100.0), &device);
        let coords = SphericalCoords::init_random(n, ideal_radii.clone(), &device);

        let similarity: Tensor<WgpuBackend, 2> = Tensor::zeros([n, n], &device);
        let hamiltonian = WaterFillingHamiltonian::new(
            ideal_radii,
            similarity,
            1.0,
        );

        // Energy should be zero at ideal positions
        let energy = hamiltonian.gravity_energy(&coords);
        let energy_data: Vec<f32> = energy.into_data().to_vec().expect("energy to vec");
        
        for e in energy_data {
            assert!(e.abs() < 1e-6, "Energy at ideal radius should be ~0, got {}", e);
        }
    }
}

