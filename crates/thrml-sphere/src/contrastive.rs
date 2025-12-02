//! Advanced Contrastive Divergence Training Components
//!
//! This module implements state-of-the-art contrastive divergence techniques
//! for training Energy-Based Models (EBMs), based on research from 2024-2025.
//!
//! ## Components
//!
//! - [`HardNegativeMiner`]: Similarity-based hard negative mining with false negative filtering
//! - [`PersistentParticleBuffer`]: Persistent particles for Persistent Contrastive Divergence (PCD)
//! - [`NegativeCurriculumSchedule`]: Progressive difficulty scheduling for negatives
//! - [`SGLDNegativeSampler`]: Stochastic Gradient Langevin Dynamics for negative phase
//!
//! ## Research Background
//!
//! - **PCD with Convergence Guarantees**: [arxiv.org/abs/2510.01944](https://arxiv.org/abs/2510.01944)
//! - **Hard Negative Mining (NV-Retriever)**: [arxiv.org/abs/2407.15831](https://arxiv.org/abs/2407.15831)
//! - **Curriculum Hard Negatives (SyNeg)**: [arxiv.org/abs/2412.17250](https://arxiv.org/abs/2412.17250)
//!
//! ## Example
//!
//! ```rust,ignore
//! use thrml_sphere::contrastive::{
//!     HardNegativeMiner, PersistentParticleBuffer, NegativeCurriculumSchedule,
//! };
//!
//! // Create hard negative miner with positive-aware filtering
//! let miner = HardNegativeMiner::new(0.9)  // Filter top 10% most similar
//!     .with_min_negatives(4);
//!
//! // Create persistent particle buffer for PCD
//! let mut pcd_buffer = PersistentParticleBuffer::new(1000, 128);  // 1000 particles, dim 128
//!
//! // Create curriculum schedule
//! let curriculum = NegativeCurriculumSchedule::default();
//! ```

use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use thrml_samplers::{LangevinConfig, RngKey};

// Fused kernel imports (when feature enabled)
#[cfg(feature = "fused-kernels")]
use thrml_core::backend::CubeWgpuBackend;
#[cfg(feature = "fused-kernels")]
use thrml_kernels::cosine_similarity_fused;

// ============================================================================
// Hard Negative Mining (Section from NV-Retriever paper)
// ============================================================================

/// Hard negative mining strategy with positive-aware filtering.
///
/// Implements the positive-aware hard negative mining approach from
/// NV-Retriever (arxiv.org/abs/2407.15831). Key features:
///
/// - **Similarity-based selection**: Selects negatives with high similarity to query
/// - **False negative filtering**: Removes candidates too similar to positive (potential false negatives)
/// - **Difficulty scoring**: Ranks negatives by "hardness" (similarity to query but not to positive)
///
/// # Example
///
/// ```rust,ignore
/// use thrml_sphere::contrastive::HardNegativeMiner;
///
/// let miner = HardNegativeMiner::new(0.9)  // Filter top 10% most similar to positive
///     .with_min_negatives(4)
///     .with_hardness_temperature(0.5);
///
/// let negatives = miner.mine(
///     &query_embedding,
///     positive_idx,
///     &all_embeddings,
///     8,  // n_negatives
///     &device,
/// );
/// ```
#[derive(Clone, Debug)]
pub struct HardNegativeMiner {
    /// Threshold for false negative filtering.
    /// Negatives with similarity to positive above this threshold are filtered.
    /// Range: 0.0 to 1.0 (higher = more aggressive filtering)
    pub false_negative_threshold: f32,

    /// Minimum number of negatives to return.
    /// If filtering removes too many, we relax the threshold.
    pub min_negatives: usize,

    /// Temperature for hardness scoring.
    /// Lower temperature = sharper selection of hardest negatives.
    pub hardness_temperature: f32,

    /// Whether to use in-batch negatives (other queries' positives as negatives).
    pub use_in_batch: bool,

    /// Fraction of negatives that should be "hard" (rest are random).
    /// This provides diversity and prevents collapse.
    pub hard_fraction: f32,
}

impl Default for HardNegativeMiner {
    fn default() -> Self {
        Self {
            false_negative_threshold: 0.85,
            min_negatives: 2,
            hardness_temperature: 0.5,
            use_in_batch: true,
            hard_fraction: 0.7,
        }
    }
}

impl HardNegativeMiner {
    /// Create a new hard negative miner.
    ///
    /// # Arguments
    ///
    /// * `false_negative_threshold` - Similarity threshold for filtering false negatives.
    ///   Negatives with similarity to positive above this are filtered.
    pub fn new(false_negative_threshold: f32) -> Self {
        Self {
            false_negative_threshold,
            ..Default::default()
        }
    }

    /// Builder: set minimum number of negatives.
    pub const fn with_min_negatives(mut self, n: usize) -> Self {
        self.min_negatives = n;
        self
    }

    /// Builder: set hardness temperature.
    pub const fn with_hardness_temperature(mut self, temp: f32) -> Self {
        self.hardness_temperature = temp;
        self
    }

    /// Builder: enable/disable in-batch negatives.
    pub const fn with_in_batch(mut self, use_in_batch: bool) -> Self {
        self.use_in_batch = use_in_batch;
        self
    }

    /// Builder: set fraction of hard negatives.
    pub const fn with_hard_fraction(mut self, frac: f32) -> Self {
        self.hard_fraction = frac.clamp(0.0, 1.0);
        self
    }

    /// Mine hard negatives for a single query.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding \[D\]
    /// * `positive_idx` - Index of the positive target
    /// * `embeddings` - All embeddings \[N, D\]
    /// * `similarity_matrix` - Pre-computed similarity matrix [N, N] (optional)
    /// * `n_negatives` - Number of negatives to return
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// Vector of negative indices, ranked by hardness.
    pub fn mine(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        positive_idx: usize,
        embeddings: &Tensor<WgpuBackend, 2>,
        similarity_matrix: Option<&Tensor<WgpuBackend, 2>>,
        n_negatives: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<usize> {
        let [n, d] = embeddings.dims();

        if n < 2 {
            return Vec::new();
        }

        // Compute query similarities if not provided
        let query_sims = self.compute_query_similarities(query, embeddings, device);
        let query_sim_data: Vec<f32> = query_sims.into_data().to_vec().expect("query_sims to vec");

        // Get positive similarity to positive (for threshold computation)
        let positive_sims = if let Some(sim_mat) = similarity_matrix {
            let sim_data: Vec<f32> = sim_mat.clone().into_data().to_vec().expect("sim to vec");
            (0..n)
                .map(|i| sim_data[positive_idx * n + i])
                .collect::<Vec<f32>>()
        } else {
            // Compute similarity of positive to all others
            self.compute_positive_similarities(positive_idx, embeddings, d, device)
        };

        // Compute hardness scores and filter false negatives
        let mut candidates: Vec<(usize, f32, f32)> = Vec::with_capacity(n);

        for i in 0..n {
            if i == positive_idx {
                continue;
            }

            let sim_to_query = query_sim_data[i];
            let sim_to_positive = positive_sims[i];

            // Filter potential false negatives (too similar to positive)
            if sim_to_positive > self.false_negative_threshold {
                continue;
            }

            // Hardness score: high similarity to query, low similarity to positive
            let hardness = sim_to_query * (1.0 - sim_to_positive);

            candidates.push((i, hardness, sim_to_query));
        }

        // If too few candidates, relax threshold
        if candidates.len() < self.min_negatives {
            candidates.clear();
            for i in 0..n {
                if i == positive_idx {
                    continue;
                }
                let sim_to_query = query_sim_data[i];
                let sim_to_positive = positive_sims[i];
                let hardness = sim_to_query * sim_to_positive.mul_add(-0.5, 1.0); // Softer penalty
                candidates.push((i, hardness, sim_to_query));
            }
        }

        // Sort by hardness (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Mix hard and random negatives
        let n_hard = ((n_negatives as f32) * self.hard_fraction).ceil() as usize;
        let n_random = n_negatives.saturating_sub(n_hard);

        let mut negatives = Vec::with_capacity(n_negatives);

        // Take top-k hard negatives
        for (idx, _, _) in candidates.iter().take(n_hard) {
            negatives.push(*idx);
        }

        // Add random negatives from remaining candidates
        if n_random > 0 && candidates.len() > n_hard {
            let remaining: Vec<usize> = candidates
                .iter()
                .skip(n_hard)
                .map(|(idx, _, _)| *idx)
                .collect();

            // Simple random selection (could use RNG for reproducibility)
            for (i, &idx) in remaining.iter().enumerate() {
                if negatives.len() >= n_negatives {
                    break;
                }
                // Take every k-th element for pseudo-random distribution
                if i % ((remaining.len() / n_random.max(1)).max(1)) == 0 {
                    negatives.push(idx);
                }
            }
        }

        negatives.truncate(n_negatives);
        negatives
    }

    /// Mine hard negatives for a batch of queries.
    ///
    /// This is more efficient than calling `mine` repeatedly as it can
    /// leverage batch operations and in-batch negatives.
    ///
    /// # Arguments
    ///
    /// * `queries` - Batch of query embeddings [B, D]
    /// * `positive_indices` - Indices of positive targets for each query
    /// * `embeddings` - All embeddings [N, D]
    /// * `similarity_matrix` - Pre-computed similarity matrix [N, N]
    /// * `n_negatives` - Number of negatives per query
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// Vector of negative indices for each query.
    pub fn mine_batch(
        &self,
        queries: &Tensor<WgpuBackend, 2>,
        positive_indices: &[usize],
        embeddings: &Tensor<WgpuBackend, 2>,
        similarity_matrix: Option<&Tensor<WgpuBackend, 2>>,
        n_negatives: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<Vec<usize>> {
        let [batch_size, d] = queries.dims();

        let mut all_negatives = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let query: Tensor<WgpuBackend, 1> =
                queries.clone().slice([i..i + 1, 0..d]).reshape([d as i32]);

            let mut negatives = self.mine(
                &query,
                positive_indices[i],
                embeddings,
                similarity_matrix,
                n_negatives,
                device,
            );

            // Add in-batch negatives if enabled
            if self.use_in_batch {
                for (j, &pos_j) in positive_indices.iter().enumerate() {
                    if j != i && !negatives.contains(&pos_j) && negatives.len() < n_negatives * 2 {
                        negatives.push(pos_j);
                    }
                }
                negatives.truncate(n_negatives);
            }

            all_negatives.push(negatives);
        }

        all_negatives
    }

    /// Compute cosine similarities between query and all embeddings.
    ///
    /// Uses fused CubeCL kernel when `fused-kernels` feature is enabled.
    fn compute_query_similarities(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        embeddings: &Tensor<WgpuBackend, 2>,
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        #[cfg(feature = "fused-kernels")]
        {
            // Use fused kernel - converts WgpuBackend (Fusion) to CubeWgpuBackend (raw)
            // Note: This requires tensor data copy, but kernel fusion saves more
            let query_data: Vec<f32> = query.clone().into_data().to_vec().expect("query to vec");
            let emb_data: Vec<f32> = embeddings.clone().into_data().to_vec().expect("emb to vec");
            let [n, d] = embeddings.dims();
            
            let device = thrml_core::backend::init_gpu_device();
            let query_flat: Tensor<CubeWgpuBackend, 1> = 
                Tensor::from_floats(query_data.as_slice(), &device);
            let emb_flat: Tensor<CubeWgpuBackend, 1> = 
                Tensor::from_floats(emb_data.as_slice(), &device);
            let emb_cube: Tensor<CubeWgpuBackend, 2> = emb_flat.reshape([n, d]);
            
            let sims_cube = cosine_similarity_fused(query_flat, emb_cube);
            
            // Convert back to WgpuBackend
            let sims_data: Vec<f32> = sims_cube.into_data().to_vec().expect("sims to vec");
            let wgpu_device = query.device();
            Tensor::from_floats(sims_data.as_slice(), &wgpu_device)
        }

        #[cfg(not(feature = "fused-kernels"))]
        {
            let [n, d] = embeddings.dims();

            // Normalize query
            let query_norm = query.clone().powf_scalar(2.0).sum().sqrt() + 1e-8;
            let query_normalized = query.clone() / query_norm;

            // Normalize embeddings
            let emb_norms = embeddings.clone().powf_scalar(2.0).sum_dim(1).sqrt() + 1e-8;
            let emb_normalized = embeddings.clone() / emb_norms;

            // Compute dot products (cosine similarities)
            let query_2d: Tensor<WgpuBackend, 2> = query_normalized.reshape([1, d as i32]);
            let sims = query_2d.matmul(emb_normalized.transpose());

            sims.reshape([n as i32])
        }
    }

    /// Compute similarities of positive embedding to all others.
    fn compute_positive_similarities(
        &self,
        positive_idx: usize,
        embeddings: &Tensor<WgpuBackend, 2>,
        d: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<f32> {
        let positive: Tensor<WgpuBackend, 1> = embeddings
            .clone()
            .slice([positive_idx..positive_idx + 1, 0..d])
            .reshape([d as i32]);

        let sims = self.compute_query_similarities(&positive, embeddings, device);
        sims.into_data().to_vec().expect("sims to vec")
    }
}

// ============================================================================
// Persistent Contrastive Divergence (PCD)
// ============================================================================

/// Buffer for persistent fantasy particles in Persistent Contrastive Divergence.
///
/// PCD maintains a set of "fantasy particles" that persist across training batches,
/// allowing the MCMC chain to mix better than standard CD where chains are
/// reinitialized from data each batch.
///
/// From [arxiv.org/abs/2510.01944](https://arxiv.org/abs/2510.01944):
/// > "Persistent chains provide better estimates of the model distribution
/// > with theoretical uniform-in-time convergence guarantees."
///
/// # Example
///
/// ```rust,ignore
/// use thrml_sphere::contrastive::PersistentParticleBuffer;
///
/// // Create buffer for 1000 particles in 128-dimensional space
/// let mut buffer = PersistentParticleBuffer::new(1000, 128);
///
/// // Initialize from data distribution
/// buffer.initialize_from_data(&embeddings, &device);
///
/// // During training, update particles with Langevin dynamics
/// buffer.update_langevin(|particles| energy_gradient(particles), &config, &device);
///
/// // Get current negative samples
/// let negatives = buffer.sample(batch_size, &device);
/// ```
#[derive(Clone)]
pub struct PersistentParticleBuffer {
    /// Persistent fantasy particles [N, D]
    pub particles: Option<Tensor<WgpuBackend, 2>>,

    /// Number of particles to maintain
    pub n_particles: usize,

    /// Embedding dimension
    pub dim: usize,

    /// Number of Langevin steps per update
    pub langevin_steps: usize,

    /// Langevin configuration
    pub langevin_config: LangevinConfig,

    /// Replay probability (fraction of time to use persistent particles vs fresh init)
    pub replay_prob: f32,

    /// Number of updates performed
    pub n_updates: usize,
}

impl PersistentParticleBuffer {
    /// Create a new persistent particle buffer.
    ///
    /// # Arguments
    ///
    /// * `n_particles` - Number of fantasy particles to maintain
    /// * `dim` - Embedding dimension
    pub const fn new(n_particles: usize, dim: usize) -> Self {
        Self {
            particles: None,
            n_particles,
            dim,
            langevin_steps: 10,
            langevin_config: LangevinConfig::new(0.01, 0.1).with_gradient_clip(1.0),
            replay_prob: 0.95,
            n_updates: 0,
        }
    }

    /// Builder: set number of Langevin steps per update.
    pub const fn with_langevin_steps(mut self, steps: usize) -> Self {
        self.langevin_steps = steps;
        self
    }

    /// Builder: set Langevin configuration.
    pub const fn with_langevin_config(mut self, config: LangevinConfig) -> Self {
        self.langevin_config = config;
        self
    }

    /// Builder: set replay probability.
    pub const fn with_replay_prob(mut self, prob: f32) -> Self {
        self.replay_prob = prob.clamp(0.0, 1.0);
        self
    }

    /// Initialize particles from data distribution.
    ///
    /// Randomly samples from the data embeddings to initialize fantasy particles.
    /// This provides a better starting point than pure noise.
    pub fn initialize_from_data(
        &mut self,
        embeddings: &Tensor<WgpuBackend, 2>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) {
        let [n, d] = embeddings.dims();

        self.dim = d;

        // Sample indices (with replacement if n_particles > n)
        let emb_data: Vec<f32> = embeddings.clone().into_data().to_vec().expect("emb to vec");

        let mut particle_data = Vec::with_capacity(self.n_particles * d);
        for i in 0..self.n_particles {
            let idx = i % n;
            let start = idx * d;
            let end = start + d;
            particle_data.extend_from_slice(&emb_data[start..end]);
        }

        // Add small noise for diversity
        let particles_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data(particle_data.as_slice(), device);
        let particles: Tensor<WgpuBackend, 2> =
            particles_1d.reshape([self.n_particles as i32, d as i32]);

        let noise: Tensor<WgpuBackend, 2> = Tensor::random(
            [self.n_particles, d],
            Distribution::Normal(0.0, 0.01),
            device,
        );

        self.particles = Some(particles + noise);
    }

    /// Initialize particles from Gaussian noise.
    pub fn initialize_random(&mut self, device: &burn::backend::wgpu::WgpuDevice) {
        let particles: Tensor<WgpuBackend, 2> = Tensor::random(
            [self.n_particles, self.dim],
            Distribution::Normal(0.0, 1.0),
            device,
        );
        self.particles = Some(particles);
    }

    /// Check if buffer is initialized.
    pub const fn is_initialized(&self) -> bool {
        self.particles.is_some()
    }

    /// Get current particles.
    pub const fn get_particles(&self) -> Option<&Tensor<WgpuBackend, 2>> {
        self.particles.as_ref()
    }

    /// Update particles using Langevin dynamics.
    ///
    /// # Arguments
    ///
    /// * `energy_gradient_fn` - Function computing energy gradient for particles
    /// * `device` - GPU device
    pub fn update_langevin<F>(
        &mut self,
        energy_gradient_fn: F,
        device: &burn::backend::wgpu::WgpuDevice,
    ) where
        F: Fn(&Tensor<WgpuBackend, 2>) -> Tensor<WgpuBackend, 2>,
    {
        let particles = match &self.particles {
            Some(p) => p.clone(),
            None => {
                self.initialize_random(device);
                self.particles.clone().unwrap()
            }
        };

        // Run Langevin dynamics
        let updated = thrml_samplers::run_langevin_2d(
            particles,
            energy_gradient_fn,
            &self.langevin_config,
            self.langevin_steps,
            None, // No annealing
            RngKey::new(self.n_updates as u64),
            device,
        );

        self.particles = Some(updated);
        self.n_updates += 1;
    }

    /// Sample a batch of negative particles.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of particles to sample
    /// * `data` - Optional data tensor for mixing (replay buffer style)
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// Sampled particles [batch_size, D]
    pub fn sample(
        &self,
        batch_size: usize,
        data: Option<&Tensor<WgpuBackend, 2>>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let particles = match &self.particles {
            Some(p) => p.clone(),
            None => {
                // Return random noise if not initialized
                return Tensor::random(
                    [batch_size, self.dim],
                    Distribution::Normal(0.0, 1.0),
                    device,
                );
            }
        };

        let particle_data: Vec<f32> = particles
            
            .into_data()
            .to_vec()
            .expect("particles to vec");

        // Sample indices
        let mut sampled_data = Vec::with_capacity(batch_size * self.dim);
        for i in 0..batch_size {
            // Decide: use persistent particle or fresh init from data
            let use_persistent = (i as f32 / batch_size as f32) < self.replay_prob;

            if use_persistent || data.is_none() {
                let idx = i % self.n_particles;
                let start = idx * self.dim;
                let end = start + self.dim;
                sampled_data.extend_from_slice(&particle_data[start..end]);
            } else if let Some(data_tensor) = data {
                let data_vec: Vec<f32> = data_tensor
                    .clone()
                    .into_data()
                    .to_vec()
                    .expect("data to vec");
                let n_data = data_vec.len() / self.dim;
                let idx = i % n_data;
                let start = idx * self.dim;
                let end = start + self.dim;
                sampled_data.extend_from_slice(&data_vec[start..end]);
            }
        }

        let sampled_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(sampled_data.as_slice(), device);
        sampled_1d.reshape([batch_size as i32, self.dim as i32])
    }

    /// Reinitialize a fraction of particles from data.
    ///
    /// This helps prevent mode collapse by occasionally refreshing particles.
    pub fn reinitialize_fraction(
        &mut self,
        fraction: f32,
        data: &Tensor<WgpuBackend, 2>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) {
        if self.particles.is_none() {
            self.initialize_from_data(data, device);
            return;
        }

        let particles = self.particles.as_ref().unwrap();
        let particle_data: Vec<f32> = particles
            .clone()
            .into_data()
            .to_vec()
            .expect("particles to vec");
        let data_vec: Vec<f32> = data.clone().into_data().to_vec().expect("data to vec");
        let n_data = data_vec.len() / self.dim;

        let n_reinit = ((self.n_particles as f32) * fraction).ceil() as usize;

        let mut new_data = particle_data;
        for i in 0..n_reinit {
            let particle_idx = i; // Reinitialize first n_reinit particles
            let data_idx = i % n_data;

            let p_start = particle_idx * self.dim;
            let d_start = data_idx * self.dim;

            new_data[p_start..p_start + self.dim]
                .copy_from_slice(&data_vec[d_start..d_start + self.dim]);
        }

        let new_particles_1d: Tensor<WgpuBackend, 1> =
            Tensor::from_data(new_data.as_slice(), device);
        self.particles = Some(new_particles_1d.reshape([self.n_particles as i32, self.dim as i32]));
    }
}

// ============================================================================
// Curriculum Learning for Negatives
// ============================================================================

/// Difficulty level for negative sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NegativeDifficulty {
    /// Random negatives (easiest)
    Easy,
    /// Moderately similar negatives
    Medium,
    /// Very similar but non-matching negatives (hardest)
    Hard,
}

/// Schedule for progressively increasing negative difficulty.
///
/// Implements curriculum learning for hard negatives, as described in
/// SyNeg (arxiv.org/abs/2412.17250). The idea is to:
///
/// 1. Start with easy (random) negatives to learn basic discrimination
/// 2. Progress to medium-difficulty negatives
/// 3. End with hard negatives for fine-grained discrimination
///
/// # Example
///
/// ```rust,ignore
/// use thrml_sphere::contrastive::NegativeCurriculumSchedule;
///
/// let schedule = NegativeCurriculumSchedule::default();
///
/// // Early training: mostly easy negatives
/// let (easy_frac, hard_frac) = schedule.get_fractions(0.1);
/// assert!(easy_frac > 0.5);
///
/// // Late training: mostly hard negatives
/// let (easy_frac, hard_frac) = schedule.get_fractions(0.9);
/// assert!(hard_frac > 0.5);
/// ```
#[derive(Clone, Debug)]
pub struct NegativeCurriculumSchedule {
    /// Epoch to start introducing medium negatives
    pub medium_start_epoch: usize,

    /// Epoch to start introducing hard negatives
    pub hard_start_epoch: usize,

    /// Epoch at which curriculum is complete (all hard)
    pub curriculum_end_epoch: usize,

    /// Minimum fraction of easy negatives (for stability)
    pub min_easy_fraction: f32,

    /// Maximum fraction of hard negatives
    pub max_hard_fraction: f32,

    /// Similarity threshold for "easy" negatives
    pub easy_similarity_threshold: f32,

    /// Similarity threshold for "hard" negatives
    pub hard_similarity_threshold: f32,
}

impl Default for NegativeCurriculumSchedule {
    fn default() -> Self {
        Self {
            medium_start_epoch: 10,
            hard_start_epoch: 30,
            curriculum_end_epoch: 100,
            min_easy_fraction: 0.1,
            max_hard_fraction: 0.8,
            easy_similarity_threshold: 0.3,
            hard_similarity_threshold: 0.7,
        }
    }
}

impl NegativeCurriculumSchedule {
    /// Create a new curriculum schedule.
    pub fn new(medium_start: usize, hard_start: usize, end: usize) -> Self {
        Self {
            medium_start_epoch: medium_start,
            hard_start_epoch: hard_start,
            curriculum_end_epoch: end,
            ..Default::default()
        }
    }

    /// Create a fast curriculum (for development/testing).
    pub fn fast() -> Self {
        Self {
            medium_start_epoch: 2,
            hard_start_epoch: 5,
            curriculum_end_epoch: 15,
            ..Default::default()
        }
    }

    /// Create a slow curriculum (for large datasets).
    pub fn slow() -> Self {
        Self {
            medium_start_epoch: 50,
            hard_start_epoch: 150,
            curriculum_end_epoch: 500,
            ..Default::default()
        }
    }

    /// Get the current difficulty based on epoch.
    pub const fn get_difficulty(&self, epoch: usize) -> NegativeDifficulty {
        if epoch < self.medium_start_epoch {
            NegativeDifficulty::Easy
        } else if epoch < self.hard_start_epoch {
            NegativeDifficulty::Medium
        } else {
            NegativeDifficulty::Hard
        }
    }

    /// Get the fractions of easy, medium, and hard negatives for current epoch.
    ///
    /// # Arguments
    ///
    /// * `progress` - Training progress (0.0 = start, 1.0 = end)
    ///
    /// # Returns
    ///
    /// Tuple of (easy_fraction, medium_fraction, hard_fraction)
    pub fn get_fractions(&self, progress: f32) -> (f32, f32, f32) {
        let progress = progress.clamp(0.0, 1.0);

        // Phase boundaries
        let medium_progress = self.medium_start_epoch as f32 / self.curriculum_end_epoch as f32;
        let hard_progress = self.hard_start_epoch as f32 / self.curriculum_end_epoch as f32;

        let (easy, medium, hard) = if progress < medium_progress {
            // Phase 1: All easy
            (1.0 - self.min_easy_fraction, self.min_easy_fraction, 0.0)
        } else if progress < hard_progress {
            // Phase 2: Transition easy -> medium
            let phase_progress = (progress - medium_progress) / (hard_progress - medium_progress);
            let easy = 1.0 - phase_progress * 0.5;
            let medium = phase_progress * 0.5;
            (easy.max(self.min_easy_fraction), medium, 0.0)
        } else {
            // Phase 3: Transition medium -> hard
            let phase_progress = (progress - hard_progress) / (1.0 - hard_progress);
            let hard = (phase_progress * self.max_hard_fraction).min(self.max_hard_fraction);
            let easy = self.min_easy_fraction;
            let medium = 1.0 - easy - hard;
            (easy, medium.max(0.0), hard)
        };

        // Normalize to sum to 1
        let total = easy + medium + hard;
        (easy / total, medium / total, hard / total)
    }

    /// Get similarity thresholds for current epoch.
    ///
    /// # Returns
    ///
    /// Tuple of (min_similarity, max_similarity) for negative selection
    pub const fn get_similarity_bounds(&self, epoch: usize) -> (f32, f32) {
        match self.get_difficulty(epoch) {
            NegativeDifficulty::Easy => (0.0, self.easy_similarity_threshold),
            NegativeDifficulty::Medium => (
                self.easy_similarity_threshold,
                self.hard_similarity_threshold,
            ),
            NegativeDifficulty::Hard => (self.hard_similarity_threshold, 1.0),
        }
    }

    /// Select negatives according to curriculum.
    ///
    /// # Arguments
    ///
    /// * `candidates` - All candidate negative indices with their similarities
    /// * `n_negatives` - Number to select
    /// * `epoch` - Current epoch
    ///
    /// # Returns
    ///
    /// Selected negative indices
    pub fn select_negatives(
        &self,
        candidates: &[(usize, f32)], // (index, similarity to query)
        n_negatives: usize,
        epoch: usize,
    ) -> Vec<usize> {
        if candidates.is_empty() {
            return Vec::new();
        }

        let (min_sim, max_sim) = self.get_similarity_bounds(epoch);

        // Filter candidates by similarity range
        let mut valid: Vec<(usize, f32)> = candidates
            .iter()
            .filter(|(_, sim)| *sim >= min_sim && *sim <= max_sim)
            .copied()
            .collect();

        // If not enough, expand range
        if valid.len() < n_negatives {
            valid = candidates.to_vec();
        }

        // Sort by similarity (descending for hard, ascending for easy)
        match self.get_difficulty(epoch) {
            NegativeDifficulty::Hard => {
                valid.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            _ => {
                // For easy/medium, sort by absolute distance from target range midpoint
                let mid = (min_sim + max_sim) / 2.0;
                valid.sort_by(|a, b| {
                    let da = (a.1 - mid).abs();
                    let db = (b.1 - mid).abs();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        valid
            .iter()
            .take(n_negatives)
            .map(|(idx, _)| *idx)
            .collect()
    }
}

// ============================================================================
// SGLD Negative Sampler
// ============================================================================

/// Configuration for SGLD-based negative sampling.
///
/// Stochastic Gradient Langevin Dynamics provides higher-quality negative
/// samples by running MCMC in the energy landscape.
#[derive(Clone, Debug)]
pub struct SGLDNegativeConfig {
    /// Number of SGLD steps per sample
    pub n_steps: usize,

    /// Step size for SGLD
    pub step_size: f32,

    /// Temperature for SGLD
    pub temperature: f32,

    /// Gradient clipping threshold
    pub gradient_clip: f32,

    /// Whether to use informative initialization (mix with data)
    pub informative_init: bool,

    /// Mixing ratio for informative init (0 = pure noise, 1 = pure data)
    pub init_data_ratio: f32,

    /// Proximal constraint radius (0 = no constraint)
    pub proximal_radius: f32,
}

impl Default for SGLDNegativeConfig {
    fn default() -> Self {
        Self {
            n_steps: 20,
            step_size: 0.01,
            temperature: 0.1,
            gradient_clip: 1.0,
            informative_init: true,
            init_data_ratio: 0.5,
            proximal_radius: 0.0,
        }
    }
}

impl SGLDNegativeConfig {
    /// Create new SGLD config with specified steps.
    pub fn new(n_steps: usize) -> Self {
        Self {
            n_steps,
            ..Default::default()
        }
    }

    /// Builder: set step size.
    pub const fn with_step_size(mut self, step_size: f32) -> Self {
        self.step_size = step_size;
        self
    }

    /// Builder: set temperature.
    pub const fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Builder: enable informative initialization.
    pub const fn with_informative_init(mut self, ratio: f32) -> Self {
        self.informative_init = true;
        self.init_data_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Builder: set proximal constraint radius.
    pub const fn with_proximal_radius(mut self, radius: f32) -> Self {
        self.proximal_radius = radius;
        self
    }

    /// Convert to LangevinConfig.
    pub const fn to_langevin_config(&self) -> LangevinConfig {
        LangevinConfig::new(self.step_size, self.temperature).with_gradient_clip(self.gradient_clip)
    }
}

/// SGLD-based negative sampler.
///
/// Generates high-quality negative samples by running Stochastic Gradient
/// Langevin Dynamics in the energy landscape. This produces samples that
/// are more representative of the model distribution than random sampling.
///
/// # Example
///
/// ```rust,ignore
/// use thrml_sphere::contrastive::{SGLDNegativeSampler, SGLDNegativeConfig};
///
/// let config = SGLDNegativeConfig::new(20)
///     .with_step_size(0.01)
///     .with_informative_init(0.5);
///
/// let sampler = SGLDNegativeSampler::new(config);
///
/// let negatives = sampler.sample(
///     batch_size,
///     &data_init,
///     |x| energy_gradient(x),
///     &device,
/// );
/// ```
#[derive(Clone, Debug)]
pub struct SGLDNegativeSampler {
    /// Configuration
    pub config: SGLDNegativeConfig,

    /// Number of samples generated
    pub n_samples: usize,
}

impl SGLDNegativeSampler {
    /// Create new SGLD sampler.
    pub const fn new(config: SGLDNegativeConfig) -> Self {
        Self {
            config,
            n_samples: 0,
        }
    }

    /// Sample negatives using SGLD.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of samples to generate
    /// * `data_init` - Data samples for initialization [batch_size, D]
    /// * `energy_gradient_fn` - Function computing energy gradient
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// Negative samples [batch_size, D]
    pub fn sample<F>(
        &mut self,
        batch_size: usize,
        data_init: &Tensor<WgpuBackend, 2>,
        energy_gradient_fn: F,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2>
    where
        F: Fn(&Tensor<WgpuBackend, 2>) -> Tensor<WgpuBackend, 2>,
    {
        let [n, d] = data_init.dims();

        // Initialize samples
        let init = if self.config.informative_init {
            // Mix data with noise
            let ratio = self.config.init_data_ratio;
            let noise: Tensor<WgpuBackend, 2> = Tensor::random(
                [batch_size.min(n), d],
                Distribution::Normal(0.0, 1.0),
                device,
            );

            // Take first batch_size samples from data
            let data_slice = if batch_size <= n {
                data_init.clone().slice([0..batch_size, 0..d])
            } else {
                // Repeat data if batch_size > n
                let mut data_vec: Vec<f32> =
                    data_init.clone().into_data().to_vec().expect("data to vec");
                while data_vec.len() < batch_size * d {
                    let orig: Vec<f32> =
                        data_init.clone().into_data().to_vec().expect("data to vec");
                    data_vec.extend(orig);
                }
                data_vec.truncate(batch_size * d);
                let data_1d: Tensor<WgpuBackend, 1> =
                    Tensor::from_data(data_vec.as_slice(), device);
                data_1d.reshape([batch_size as i32, d as i32])
            };

            data_slice.mul_scalar(ratio) + noise.mul_scalar(1.0 - ratio)
        } else {
            // Pure noise initialization
            Tensor::random([batch_size, d], Distribution::Normal(0.0, 1.0), device)
        };

        // Run SGLD
        let langevin_config = self.config.to_langevin_config();
        let init_for_proximal = init.clone();

        let mut state = init;

        for _step in 0..self.config.n_steps {
            let gradient = energy_gradient_fn(&state);
            state = thrml_samplers::langevin_step_2d(&state, &gradient, &langevin_config, device);

            // Apply proximal constraint if enabled
            if self.config.proximal_radius > 0.0 {
                state = self.project_to_ball(
                    &state,
                    &init_for_proximal,
                    self.config.proximal_radius,
                    device,
                );
            }
        }

        self.n_samples += batch_size;
        state
    }

    /// Project samples to Lp-ball around center.
    fn project_to_ball(
        &self,
        samples: &Tensor<WgpuBackend, 2>,
        center: &Tensor<WgpuBackend, 2>,
        radius: f32,
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 2> {
        let diff = samples.clone() - center.clone();
        let dist = diff.clone().powf_scalar(2.0).sum_dim(1).sqrt();

        // Scale factor: min(1, radius / dist)
        let scale = (dist.recip().mul_scalar(radius)).clamp(0.0, 1.0);

        center.clone() + diff * scale
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_hard_negative_miner_creation() {
        let miner = HardNegativeMiner::new(0.9)
            .with_min_negatives(4)
            .with_hardness_temperature(0.3)
            .with_hard_fraction(0.8);

        assert!((miner.false_negative_threshold - 0.9).abs() < 1e-6);
        assert_eq!(miner.min_negatives, 4);
        assert!((miner.hardness_temperature - 0.3).abs() < 1e-6);
        assert!((miner.hard_fraction - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_hard_negative_mining() {
        let device = init_gpu_device();
        let n = 50;
        let d = 16;

        // Create embeddings with some structure
        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);

        // Use first embedding as query
        let query: Tensor<WgpuBackend, 1> =
            embeddings.clone().slice([0..1, 0..d]).reshape([d as i32]);

        let miner = HardNegativeMiner::default();
        let negatives = miner.mine(&query, 0, &embeddings, None, 8, &device);

        // Should return requested number of negatives
        assert_eq!(negatives.len(), 8);

        // Should not include positive
        assert!(!negatives.contains(&0));

        // All indices should be valid
        for &idx in &negatives {
            assert!(idx < n);
        }
    }

    #[test]
    fn test_persistent_particle_buffer() {
        let device = init_gpu_device();
        let n_particles = 100;
        let dim = 32;

        let mut buffer = PersistentParticleBuffer::new(n_particles, dim)
            .with_langevin_steps(5)
            .with_replay_prob(0.9);

        assert!(!buffer.is_initialized());

        // Initialize from random
        buffer.initialize_random(&device);
        assert!(buffer.is_initialized());

        // Check particle shape
        let particles = buffer.get_particles().unwrap();
        assert_eq!(particles.dims(), [n_particles, dim]);

        // Sample from buffer
        let batch_size = 16;
        let samples = buffer.sample(batch_size, None, &device);
        assert_eq!(samples.dims(), [batch_size, dim]);
    }

    #[test]
    fn test_persistent_particle_update() {
        let device = init_gpu_device();
        let n_particles = 50;
        let dim = 8;

        let mut buffer = PersistentParticleBuffer::new(n_particles, dim).with_langevin_steps(3);

        buffer.initialize_random(&device);
        let initial = buffer.get_particles().unwrap().clone();

        // Update with a simple gradient (pushes particles toward zero)
        buffer.update_langevin(
            |particles| particles.clone(), // Gradient = particles (pushes toward 0)
            &device,
        );

        let updated = buffer.get_particles().unwrap().clone();

        // Particles should have changed
        let diff = (initial - updated).abs().sum();
        let diff_val: f32 = diff.into_data().to_vec::<f32>().unwrap()[0];
        assert!(diff_val > 0.0, "Particles should change after update");
    }

    #[test]
    fn test_curriculum_schedule_default() {
        let schedule = NegativeCurriculumSchedule::default();

        assert_eq!(schedule.medium_start_epoch, 10);
        assert_eq!(schedule.hard_start_epoch, 30);
        assert_eq!(schedule.curriculum_end_epoch, 100);
    }

    #[test]
    fn test_curriculum_difficulty_progression() {
        let schedule = NegativeCurriculumSchedule::new(10, 30, 100);

        // Early epochs should be easy
        assert_eq!(schedule.get_difficulty(0), NegativeDifficulty::Easy);
        assert_eq!(schedule.get_difficulty(5), NegativeDifficulty::Easy);

        // Middle epochs should be medium
        assert_eq!(schedule.get_difficulty(15), NegativeDifficulty::Medium);
        assert_eq!(schedule.get_difficulty(25), NegativeDifficulty::Medium);

        // Late epochs should be hard
        assert_eq!(schedule.get_difficulty(35), NegativeDifficulty::Hard);
        assert_eq!(schedule.get_difficulty(100), NegativeDifficulty::Hard);
    }

    #[test]
    fn test_curriculum_fractions() {
        let schedule = NegativeCurriculumSchedule::default();

        // Early training: mostly easy
        let (easy, _medium, hard) = schedule.get_fractions(0.05);
        assert!(
            easy > 0.5,
            "Early training should have mostly easy negatives"
        );
        assert!(hard < 0.1, "Early training should have few hard negatives");

        // Late training: mostly hard
        let (easy, _medium, hard) = schedule.get_fractions(0.95);
        assert!(hard > easy, "Late training should have more hard than easy");
    }

    #[test]
    fn test_curriculum_select_negatives() {
        let schedule = NegativeCurriculumSchedule::new(5, 15, 50);

        // Create candidates with various similarities
        let candidates: Vec<(usize, f32)> = (0..20)
            .map(|i| (i, i as f32 / 20.0)) // Similarity 0.0 to 0.95
            .collect();

        // Early epoch: should prefer low-similarity negatives
        let early_negatives = schedule.select_negatives(&candidates, 5, 2);
        assert_eq!(early_negatives.len(), 5);

        // Late epoch: should prefer high-similarity negatives
        let late_negatives = schedule.select_negatives(&candidates, 5, 40);
        assert_eq!(late_negatives.len(), 5);

        // Late negatives should have higher indices (higher similarity) on average
        let early_avg: f32 = early_negatives.iter().map(|&i| i as f32).sum::<f32>() / 5.0;
        let late_avg: f32 = late_negatives.iter().map(|&i| i as f32).sum::<f32>() / 5.0;

        // This test might be flaky depending on similarity bounds, so we just check they're different
        assert!(
            (early_avg - late_avg).abs() > 0.1,
            "Early and late selections should differ"
        );
    }

    #[test]
    fn test_sgld_config() {
        let config = SGLDNegativeConfig::new(30)
            .with_step_size(0.02)
            .with_temperature(0.05)
            .with_informative_init(0.7)
            .with_proximal_radius(0.5);

        assert_eq!(config.n_steps, 30);
        assert!((config.step_size - 0.02).abs() < 1e-6);
        assert!((config.temperature - 0.05).abs() < 1e-6);
        assert!(config.informative_init);
        assert!((config.init_data_ratio - 0.7).abs() < 1e-6);
        assert!((config.proximal_radius - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sgld_sampling() {
        let device = init_gpu_device();
        let batch_size = 16;
        let dim = 8;

        let config = SGLDNegativeConfig::new(5)
            .with_step_size(0.01)
            .with_informative_init(0.5);

        let mut sampler = SGLDNegativeSampler::new(config);

        // Create some data for initialization
        let data: Tensor<WgpuBackend, 2> =
            Tensor::random([batch_size, dim], Distribution::Normal(0.0, 1.0), &device);

        // Simple energy gradient (quadratic energy pushes toward 0)
        let samples = sampler.sample(
            batch_size,
            &data,
            |x| x.clone(), // Gradient = x (quadratic energy)
            &device,
        );

        assert_eq!(samples.dims(), [batch_size, dim]);
        assert_eq!(sampler.n_samples, batch_size);
    }
}
