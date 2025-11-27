//! SphereEBM - main model for sphere optimization.
//!
//! This module provides the high-level API for sphere optimization,
//! combining similarity computation, Hamiltonian definition, and
//! Langevin sampling.

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::SphericalCoords;
use thrml_samplers::RngKey;

use crate::config::SphereConfig;
use crate::hamiltonian::WaterFillingHamiltonian;
use crate::langevin::SphereLangevinSampler;
use crate::similarity::cosine_similarity_matrix;

/// Main sphere optimization model.
///
/// SphereEBM encapsulates the complete sphere optimization pipeline:
/// 1. Computing cosine similarity from embeddings
/// 2. Computing ideal radii from prominence (and optionally entropy)
/// 3. Running Langevin dynamics to find optimal positions
///
/// # Example
///
/// ```rust,ignore
/// let ebm = SphereEBM::new(embeddings, prominence, entropies, config, &device);
/// let coords = ebm.optimize(RngKey::new(42), &device);
/// let positions = coords.to_cartesian();
/// ```
#[derive(Clone)]
pub struct SphereEBM {
    /// Original embeddings \[N, D\].
    pub embeddings: Tensor<WgpuBackend, 2>,
    /// Prominence scores \[N\].
    pub prominence: Tensor<WgpuBackend, 1>,
    /// Optional entropy scores \[N\].
    pub entropies: Option<Tensor<WgpuBackend, 1>>,
    /// Precomputed similarity matrix \[N, N\].
    pub similarity: Tensor<WgpuBackend, 2>,
    /// Computed ideal radii \[N\].
    pub ideal_radii: Tensor<WgpuBackend, 1>,
    /// Configuration settings.
    pub config: SphereConfig,
}

impl SphereEBM {
    /// Creates a SphereEBM from embeddings and prominence.
    ///
    /// This constructor:
    /// 1. Computes the cosine similarity matrix from embeddings
    /// 2. Computes ideal radii from prominence (with optional entropy weighting)
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Embedding matrix \[N, D\]
    /// * `prominence` - Prominence scores \[N\]
    /// * `entropies` - Optional entropy scores \[N\]
    /// * `config` - Sphere optimization configuration
    /// * `device` - GPU device
    pub fn new(
        embeddings: Tensor<WgpuBackend, 2>,
        prominence: Tensor<WgpuBackend, 1>,
        entropies: Option<Tensor<WgpuBackend, 1>>,
        config: SphereConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        // Compute similarity matrix
        let similarity = cosine_similarity_matrix(&embeddings, device);

        // Compute ideal radii
        let ideal_radii = compute_ideal_radii(
            &prominence,
            entropies.as_ref(),
            config.min_radius,
            config.max_radius,
            config.entropy_weighted,
            device,
        );

        Self {
            embeddings,
            prominence,
            entropies,
            similarity,
            ideal_radii,
            config,
        }
    }

    /// Get the number of points.
    pub fn n_points(&self) -> usize {
        self.embeddings.dims()[0]
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embeddings.dims()[1]
    }

    /// Create a Hamiltonian from this model.
    pub fn hamiltonian(&self) -> WaterFillingHamiltonian {
        WaterFillingHamiltonian::new(
            self.ideal_radii.clone(),
            self.similarity.clone(),
            self.config.interaction_radius,
        )
    }

    /// Initialize spherical coordinates for optimization.
    ///
    /// Initial radii are set to embedding norms, with random angles.
    pub fn init_coords(&self, device: &burn::backend::wgpu::WgpuDevice) -> SphericalCoords {
        let n = self.n_points();

        // Initial radii from embedding L2 norms
        let init_radii_2d = self.embeddings.clone().powf_scalar(2.0).sum_dim(1).sqrt();
        // Reshape from [N, 1] to [N]
        let init_radii: Tensor<WgpuBackend, 1> = init_radii_2d.reshape([n as i32]);

        SphericalCoords::init_uniform_sphere(n, init_radii, device)
    }

    /// Run sphere optimization.
    ///
    /// # Arguments
    /// * `key` - RNG key for reproducibility
    /// * `device` - GPU device
    ///
    /// # Returns
    /// Optimized spherical coordinates
    pub fn optimize(
        &self,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> SphericalCoords {
        let hamiltonian = self.hamiltonian();
        let sampler = SphereLangevinSampler::from_config(&self.config);
        let init_coords = self.init_coords(device);

        sampler.run(&hamiltonian, init_coords, key, device)
    }

    /// Run sphere optimization with progress logging.
    ///
    /// # Arguments
    /// * `key` - RNG key
    /// * `device` - GPU device
    /// * `log_interval` - Steps between log messages
    pub fn optimize_with_logging(
        &self,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
        log_interval: usize,
    ) -> SphericalCoords {
        let hamiltonian = self.hamiltonian();
        let sampler = SphereLangevinSampler::from_config(&self.config);
        let init_coords = self.init_coords(device);

        sampler.run_with_callback(
            &hamiltonian,
            init_coords,
            key,
            device,
            |step, _coords, energy| {
                let mean_energy: f32 = energy.clone().mean().into_scalar();
                println!("Step {}: mean_energy = {:.4}", step, mean_energy);
            },
            log_interval,
        )
    }
}

/// Compute ideal radii using r^1.5 capacity law.
///
/// Points are ranked by prominence (or weighted prominence if entropy_weighted).
/// Higher ranked points get smaller radii (closer to core).
///
/// The capacity law: `radius = min_radius + span * (rank / n)^(1/1.5)`
/// ensures inner shells have more capacity than outer shells.
///
/// # Arguments
///
/// * `prominence` - Prominence scores \[N\]
/// * `entropies` - Optional entropy scores \[N\]
/// * `min_radius` - Minimum radius
/// * `max_radius` - Maximum radius
/// * `entropy_weighted` - Whether to weight by entropy
/// * `device` - GPU device
pub fn compute_ideal_radii(
    prominence: &Tensor<WgpuBackend, 1>,
    entropies: Option<&Tensor<WgpuBackend, 1>>,
    min_radius: f32,
    max_radius: f32,
    entropy_weighted: bool,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 1> {
    let n = prominence.dims()[0];

    // Compute weights
    let weights = if entropy_weighted {
        if let Some(ent) = entropies {
            // Weight = prominence * (1 + entropy)
            // High entropy increases weight, pushing to outer shell
            prominence.clone() * (ent.clone() + 1.0)
        } else {
            prominence.clone()
        }
    } else {
        prominence.clone()
    };

    // Get ranking by argsort (CPU operation since Burn lacks argsort)
    let weights_data: Vec<f32> = weights
        .clone()
        .into_data()
        .to_vec()
        .expect("weights to vec");

    // Create (index, weight) pairs and sort by weight descending
    // Higher weight = smaller radius = lower rank
    let mut indexed: Vec<(usize, f32)> = weights_data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign ranks (0 = highest prominence = smallest radius)
    let mut ranks = vec![0usize; n];
    for (rank, &(idx, _)) in indexed.iter().enumerate() {
        ranks[idx] = rank;
    }

    // Compute radii using r^1.5 capacity law
    // capacity(r) ∝ r^1.5, so we invert: r = (capacity)^(1/1.5)
    let radius_span = (max_radius - min_radius).max(1.0);
    let radii: Vec<f32> = ranks
        .iter()
        .map(|&rank| {
            let normalized = (rank as f32 + 0.5) / n as f32;
            min_radius + radius_span * normalized.powf(1.0 / 1.5)
        })
        .collect();

    Tensor::from_data(radii.as_slice(), device)
}

/// Alternative radii computation using prominence directly (no ranking).
///
/// `radius = max_radius - (max_radius - min_radius) * (prominence - min) / (max - min)`
///
/// Higher prominence = smaller radius.
pub fn compute_ideal_radii_linear(
    prominence: &Tensor<WgpuBackend, 1>,
    min_radius: f32,
    max_radius: f32,
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 1> {
    let prom_data: Vec<f32> = prominence
        .clone()
        .into_data()
        .to_vec()
        .expect("prom to vec");

    let prom_min = prom_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let prom_max = prom_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let prom_range = (prom_max - prom_min).max(1e-8);
    let radius_span = max_radius - min_radius;

    let radii: Vec<f32> = prom_data
        .iter()
        .map(|&p| {
            let normalized = (p - prom_min) / prom_range;
            // Higher prominence = smaller radius
            max_radius - radius_span * normalized
        })
        .collect();

    Tensor::from_data(radii.as_slice(), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_compute_ideal_radii_ordering() {
        let device = init_gpu_device();

        // Create prominence values
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::from_data([0.1, 0.5, 0.9, 0.3, 0.7].as_slice(), &device);

        let radii = compute_ideal_radii(&prominence, None, 10.0, 100.0, false, &device);

        let radii_data: Vec<f32> = radii.into_data().to_vec().expect("radii to vec");

        // Index 2 (prom=0.9) should have smallest radius
        // Index 0 (prom=0.1) should have largest radius
        assert!(
            radii_data[2] < radii_data[0],
            "Higher prominence should have smaller radius"
        );
        assert!(
            radii_data[4] < radii_data[3],
            "prom=0.7 should have smaller radius than prom=0.3"
        );
    }

    #[test]
    fn test_sphere_ebm_creation() {
        let device = init_gpu_device();
        let n = 20;
        let d = 64;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let config = SphereConfig::default();
        let ebm = SphereEBM::new(embeddings, prominence, None, config, &device);

        assert_eq!(ebm.n_points(), n);
        assert_eq!(ebm.embedding_dim(), d);
        assert_eq!(ebm.similarity.dims(), [n, n]);
        assert_eq!(ebm.ideal_radii.dims(), [n]);
    }

    #[test]
    fn test_optimize_produces_valid_coords() {
        let device = init_gpu_device();
        let n = 10;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let config = SphereConfig::from(crate::config::ScaleProfile::Dev).with_steps(10); // Fast for testing
        let ebm = SphereEBM::new(embeddings, prominence, None, config, &device);

        let key = RngKey::new(42);
        let coords = ebm.optimize(key, &device);

        assert_eq!(coords.len(), n);

        // Check radii are positive
        let r_data: Vec<f32> = coords.r.into_data().to_vec().expect("r to vec");
        for r in r_data {
            assert!(r > 0.0, "Radius should be positive");
        }
    }
}
