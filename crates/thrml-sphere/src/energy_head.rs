//! CALM-Inspired Energy Head for Learned Partition Routing
//!
//! This module implements a lightweight energy-based generative head inspired by
//! the CALM (Continuous Autoregressive Language Models) architecture. It replaces
//! the O(n²) pairwise similarity computation in ROOTS with O(n·L) learned routing.
//!
//! # Architecture (from CALM paper arXiv:2510.27688)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │  EnergyHead                                                          │
//! │  ├── Input: hidden state h ∈ ℝ^d (query embedding)                  │
//! │  ├── Input: noise ε ∈ ℝ^d_noise ~ U[-0.5, 0.5]                      │
//! │  ├── L residual MLP blocks (L = depth/4, ~10% of total params)      │
//! │  │   └── Each block: Linear fusion → SwiGLU → Residual              │
//! │  └── Output: partition scores ∈ ℝ^K                                 │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Energy Score Loss (Strictly Proper Scoring Rule)
//!
//! The energy score is likelihood-free and measures alignment via distances:
//!
//! ```text
//! S(P, y) = E[|x' - x''|^α] - 2·E[|x - y|^α]    (α ∈ (0,2), typically α=1)
//!           ─────────────    ───────────────
//!             diversity        fidelity
//! ```
//!
//! Monte Carlo estimation: N=8 model samples, M=100 target samples.
//!
//! # Integration with ROOTS
//!
//! ```text
//! Before (O(n²)):  pairwise_similarity(all_embeddings) → partition_assignment
//! After (O(n·L)):  energy_head(query_embedding, noise) → partition_scores
//! ```
//!
//! The energy head learns to predict which partitions are relevant for a query,
//! avoiding the quadratic similarity computation during inference.

use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};

/// Configuration for the Energy Head.
#[derive(Debug, Clone)]
pub struct EnergyHeadConfig {
    /// Input dimension (embedding dimension).
    pub input_dim: usize,

    /// Hidden dimension for MLP blocks.
    pub hidden_dim: usize,

    /// Noise dimension for stochastic sampling.
    pub noise_dim: usize,

    /// Number of residual MLP blocks (typically depth/4).
    pub num_blocks: usize,

    /// Output dimension (number of partitions K).
    pub output_dim: usize,

    /// Dropout rate for training (0.0 = disabled).
    pub dropout: f32,
}

impl Default for EnergyHeadConfig {
    fn default() -> Self {
        Self {
            input_dim: 2048,
            hidden_dim: 512,
            noise_dim: 128,
            num_blocks: 4,
            output_dim: 256, // Default partition count
            dropout: 0.0,
        }
    }
}

impl EnergyHeadConfig {
    /// Create config for a specific embedding dimension and partition count.
    pub fn for_dims(embed_dim: usize, num_partitions: usize) -> Self {
        // Hidden dim ~1/4 of embed_dim, num_blocks ~4
        let hidden_dim = (embed_dim / 4).max(64);
        Self {
            input_dim: embed_dim,
            hidden_dim,
            noise_dim: hidden_dim / 4,
            num_blocks: 4,
            output_dim: num_partitions,
            dropout: 0.0,
        }
    }

    /// Development config (smaller, faster).
    pub const fn dev() -> Self {
        Self {
            input_dim: 64,
            hidden_dim: 32,
            noise_dim: 16,
            num_blocks: 2,
            output_dim: 16,
            dropout: 0.0,
        }
    }

    /// Config with dropout for training.
    pub const fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }
}

/// A single residual MLP block with SwiGLU activation.
///
/// Architecture:
/// ```text
/// input ──┬──► Linear(d, d) ──► SwiGLU ──► Linear(d, d) ──┬──► output
///         │                                               │
///         └───────────────────────────────────────────────┘ (residual)
/// ```
#[derive(Debug)]
pub struct ResidualMlpBlock<B: Backend> {
    /// Linear layer for input projection.
    linear_in: Linear<B>,
    /// Linear layer for gate (SwiGLU).
    linear_gate: Linear<B>,
    /// Linear layer for output projection.
    linear_out: Linear<B>,
    /// Fusion layer for hidden state conditioning.
    fusion: Linear<B>,
}

impl<B: Backend> ResidualMlpBlock<B> {
    /// Create a new residual MLP block.
    pub fn new(dim: usize, hidden_state_dim: usize, device: &B::Device) -> Self {
        let intermediate_dim = dim * 4; // SwiGLU expansion factor

        Self {
            linear_in: LinearConfig::new(dim, intermediate_dim).init(device),
            linear_gate: LinearConfig::new(dim, intermediate_dim).init(device),
            linear_out: LinearConfig::new(intermediate_dim, dim).init(device),
            fusion: LinearConfig::new(hidden_state_dim, dim).init(device),
        }
    }

    /// Forward pass with hidden state conditioning.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, dim]
    /// * `h` - Hidden state for conditioning [batch, hidden_state_dim]
    pub fn forward(&self, x: Tensor<B, 2>, h: Tensor<B, 2>) -> Tensor<B, 2> {
        // Fuse hidden state into input
        let h_proj = self.fusion.forward(h);
        let x_fused = x.clone() + h_proj;

        // SwiGLU: silu(W1·x) ⊙ (W2·x)
        let gate = silu(self.linear_gate.forward(x_fused.clone()));
        let value = self.linear_in.forward(x_fused);
        let hidden = gate * value;

        // Output projection + residual
        let out = self.linear_out.forward(hidden);
        x + out
    }
}

/// Energy-based generative head for partition prediction.
///
/// This is the core component that enables O(n·L) routing instead of O(n²)
/// pairwise similarity computation.
#[derive(Debug)]
pub struct EnergyHead<B: Backend> {
    /// Configuration.
    config: EnergyHeadConfig,

    /// Projection for hidden state (query embedding).
    hidden_proj: Linear<B>,

    /// Projection for noise input.
    noise_proj: Linear<B>,

    /// Stack of residual MLP blocks.
    blocks: Vec<ResidualMlpBlock<B>>,

    /// Final projection to partition scores.
    output_proj: Linear<B>,
}

impl<B: Backend> EnergyHead<B> {
    /// Create a new energy head.
    pub fn new(config: EnergyHeadConfig, device: &B::Device) -> Self {
        let hidden_dim = config.hidden_dim;

        // Input projections
        let hidden_proj = LinearConfig::new(config.input_dim, hidden_dim).init(device);
        let noise_proj = LinearConfig::new(config.noise_dim, hidden_dim).init(device);

        // Residual blocks
        let blocks: Vec<ResidualMlpBlock<B>> = (0..config.num_blocks)
            .map(|_| ResidualMlpBlock::new(hidden_dim, hidden_dim, device))
            .collect();

        // Output projection
        let output_proj = LinearConfig::new(hidden_dim, config.output_dim).init(device);

        Self {
            config,
            hidden_proj,
            noise_proj,
            blocks,
            output_proj,
        }
    }

    /// Forward pass: predict partition scores from query embedding.
    ///
    /// # Arguments
    /// * `query` - Query embedding [batch, input_dim]
    /// * `noise` - Optional noise tensor [batch, noise_dim]. If None, samples uniformly.
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Partition scores [batch, output_dim] (unnormalized log-probabilities)
    pub fn forward(
        &self,
        query: Tensor<B, 2>,
        noise: Option<Tensor<B, 2>>,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let batch_size = query.dims()[0];

        // Generate noise if not provided: ε ~ U[-0.5, 0.5]
        let noise = noise.unwrap_or_else(|| {
            Tensor::random(
                [batch_size, self.config.noise_dim],
                Distribution::Uniform(-0.5, 0.5),
                device,
            )
        });

        // Project inputs to hidden dimension
        let h = self.hidden_proj.forward(query);
        let mut z = self.noise_proj.forward(noise);

        // Pass through residual blocks with hidden state conditioning
        for block in &self.blocks {
            z = block.forward(z, h.clone());
        }

        // Output projection to partition scores
        self.output_proj.forward(z)
    }

    /// Sample multiple predictions for energy score estimation.
    ///
    /// # Arguments
    /// * `query` - Query embedding [batch, input_dim]
    /// * `n_samples` - Number of samples (default: 8 as per CALM)
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Stacked predictions [n_samples, batch, output_dim]
    pub fn sample_multiple(
        &self,
        query: Tensor<B, 2>,
        n_samples: usize,
        device: &B::Device,
    ) -> Vec<Tensor<B, 2>> {
        (0..n_samples)
            .map(|_| self.forward(query.clone(), None, device))
            .collect()
    }

    /// Predict partition assignment (argmax of scores).
    ///
    /// # Arguments
    /// * `query` - Query embedding `[batch, input_dim]`
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Partition indices `[batch]`
    pub fn predict_partition(&self, query: Tensor<B, 2>, device: &B::Device) -> Vec<usize> {
        let scores = self.forward(query, None, device);
        let indices = scores.argmax(1);
        let data: Vec<i32> = indices.into_data().to_vec().expect("indices to vec");
        data.into_iter().map(|i| i as usize).collect()
    }

    /// Get soft partition assignment (softmax of scores).
    ///
    /// # Arguments
    /// * `query` - Query embedding [batch, input_dim]
    /// * `temperature` - Softmax temperature (default: 1.0)
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Partition probabilities [batch, output_dim]
    pub fn soft_partition(
        &self,
        query: Tensor<B, 2>,
        temperature: f32,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let scores = self.forward(query, None, device);
        burn::tensor::activation::softmax(scores / temperature, 1)
    }

    /// Get configuration.
    pub const fn config(&self) -> &EnergyHeadConfig {
        &self.config
    }
}

/// Energy Score Loss for training the energy head.
///
/// The energy score is a strictly proper scoring rule that measures:
/// - **Diversity**: E[|x' - x''|^α] - encourages diverse predictions
/// - **Fidelity**: E[|x - y|^α] - encourages predictions close to targets
///
/// Total: S(P, y) = diversity - 2·fidelity (higher is better, negate for loss)
#[derive(Debug, Clone)]
pub struct EnergyScoreLoss {
    /// Exponent for distance (typically 1.0).
    pub alpha: f32,
    /// Number of model samples for Monte Carlo estimation.
    pub n_model_samples: usize,
    /// Number of target samples for variance reduction.
    pub n_target_samples: usize,
}

impl Default for EnergyScoreLoss {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            n_model_samples: 8,    // N=8 as per CALM
            n_target_samples: 100, // M=100 as per CALM
        }
    }
}

impl EnergyScoreLoss {
    /// Compute energy score loss.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions [n_samples, batch, dim]
    /// * `targets` - Target values [batch, dim]
    ///
    /// # Returns
    /// Scalar loss value (lower is better)
    pub fn compute<B: Backend>(
        &self,
        predictions: &[Tensor<B, 2>],
        targets: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let n = predictions.len();
        if n == 0 {
            panic!("Need at least one prediction");
        }

        let [batch_size, _dim] = targets.dims();

        // Diversity term: mean pairwise distance between predictions
        let mut diversity_sum: Tensor<B, 1> = Tensor::zeros([batch_size], &targets.device());
        let mut diversity_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let diff = predictions[i].clone() - predictions[j].clone();
                // sum_dim returns [batch, 1], flatten to [batch]
                let dist: Tensor<B, 1> =
                    diff.abs().sum_dim(1).flatten(0, 1).powf_scalar(self.alpha);
                diversity_sum = diversity_sum + dist;
                diversity_count += 1;
            }
        }

        let diversity = if diversity_count > 0 {
            diversity_sum / (diversity_count as f32)
        } else {
            diversity_sum
        };

        // Fidelity term: mean distance from predictions to target
        let mut fidelity_sum: Tensor<B, 1> = Tensor::zeros([batch_size], &targets.device());

        for pred in predictions {
            let diff = pred.clone() - targets.clone();
            // sum_dim returns [batch, 1], flatten to [batch]
            let dist: Tensor<B, 1> = diff.abs().sum_dim(1).flatten(0, 1).powf_scalar(self.alpha);
            fidelity_sum = fidelity_sum + dist;
        }

        let fidelity = fidelity_sum / (n as f32);

        // Energy score: diversity - 2·fidelity (negate for loss)
        let energy_score = diversity - fidelity * 2.0;

        // Return mean loss (negative energy score)
        -energy_score.mean()
    }
}

/// Configuration for training the energy head with ROOTS data.
#[derive(Debug, Clone)]
pub struct EnergyHeadTrainingConfig {
    /// Learning rate.
    pub learning_rate: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Weight decay for regularization.
    pub weight_decay: f32,
    /// Energy score configuration.
    pub energy_score: EnergyScoreLoss,
}

impl Default for EnergyHeadTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            epochs: 100,
            batch_size: 32,
            weight_decay: 0.01,
            energy_score: EnergyScoreLoss::default(),
        }
    }
}

/// Trained energy head wrapper with partition centroids.
///
/// This is the inference-time structure that combines the learned
/// energy head with ROOTS partition data.
#[derive(Debug)]
pub struct LearnedPartitionRouter<B: Backend> {
    /// The trained energy head.
    pub energy_head: EnergyHead<B>,

    /// Partition centroids `[K, D]` for fallback/verification.
    pub centroids: Tensor<B, 2>,

    /// Number of points per partition `[K]`.
    pub partition_sizes: Vec<usize>,
}

impl<B: Backend> LearnedPartitionRouter<B> {
    /// Create a new learned partition router.
    pub const fn new(
        energy_head: EnergyHead<B>,
        centroids: Tensor<B, 2>,
        partition_sizes: Vec<usize>,
    ) -> Self {
        Self {
            energy_head,
            centroids,
            partition_sizes,
        }
    }

    /// Route a query to the best partition.
    ///
    /// # Arguments
    /// * `query` - Query embedding `[D]`
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Best partition index
    pub fn route(&self, query: &Tensor<B, 1>, device: &B::Device) -> usize {
        let query_2d = query.clone().unsqueeze::<2>();
        let indices = self.energy_head.predict_partition(query_2d, device);
        indices[0]
    }

    /// Get top-k partition candidates.
    ///
    /// # Arguments
    /// * `query` - Query embedding `[D]`
    /// * `k` - Number of partitions to return
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Top-k partition indices and their scores
    pub fn route_top_k(
        &self,
        query: &Tensor<B, 1>,
        k: usize,
        device: &B::Device,
    ) -> (Vec<usize>, Vec<f32>) {
        let query_2d = query.clone().unsqueeze::<2>();
        let scores = self.energy_head.forward(query_2d, None, device);

        // Get scores as vec
        let scores_vec: Vec<f32> = scores
            .flatten::<1>(0, 1)
            .into_data()
            .to_vec()
            .expect("scores to vec");

        // Sort by score descending
        let mut indexed: Vec<(usize, f32)> = scores_vec.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
        let indices: Vec<usize> = top_k.iter().map(|(i, _)| *i).collect();
        let scores: Vec<f32> = top_k.iter().map(|(_, s)| *s).collect();

        (indices, scores)
    }

    /// Get soft routing probabilities for all partitions.
    ///
    /// # Arguments
    /// * `query` - Query embedding `[D]`
    /// * `temperature` - Softmax temperature
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Partition probabilities `[K]`
    pub fn soft_route(
        &self,
        query: &Tensor<B, 1>,
        temperature: f32,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let query_2d = query.clone().unsqueeze::<2>();
        let probs = self
            .energy_head
            .soft_partition(query_2d, temperature, device);
        probs.flatten(0, 1)
    }

    /// Number of partitions.
    pub const fn num_partitions(&self) -> usize {
        self.partition_sizes.len()
    }

    /// Get centroid for a partition.
    pub fn centroid(&self, partition_idx: usize) -> Tensor<B, 1> {
        self.centroids
            .clone()
            .slice([
                partition_idx..partition_idx + 1,
                0..self.centroids.dims()[1],
            ])
            .flatten(0, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrml_core::backend::init_gpu_device;
    use thrml_core::backend::WgpuBackend;

    type TestBackend = WgpuBackend;

    #[test]
    fn test_energy_head_forward() {
        let device = init_gpu_device();
        let config = EnergyHeadConfig::dev();
        let head = EnergyHead::<TestBackend>::new(config.clone(), &device);

        let batch_size = 4;
        let query = Tensor::random(
            [batch_size, config.input_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let scores = head.forward(query, None, &device);

        assert_eq!(scores.dims(), [batch_size, config.output_dim]);
    }

    #[test]
    fn test_energy_head_predict_partition() {
        let device = init_gpu_device();
        let config = EnergyHeadConfig::dev();
        let head = EnergyHead::<TestBackend>::new(config.clone(), &device);

        let batch_size = 4;
        let query = Tensor::random(
            [batch_size, config.input_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let indices = head.predict_partition(query, &device);

        assert_eq!(indices.len(), batch_size);
        for idx in &indices {
            assert!(*idx < config.output_dim);
        }
    }

    #[test]
    fn test_energy_head_sample_multiple() {
        let device = init_gpu_device();
        let config = EnergyHeadConfig::dev();
        let head = EnergyHead::<TestBackend>::new(config.clone(), &device);

        let batch_size = 2;
        let query = Tensor::random(
            [batch_size, config.input_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let samples = head.sample_multiple(query, 8, &device);

        assert_eq!(samples.len(), 8);
        for sample in &samples {
            assert_eq!(sample.dims(), [batch_size, config.output_dim]);
        }
    }

    #[test]
    fn test_energy_score_loss() {
        let device = init_gpu_device();
        let loss_fn = EnergyScoreLoss::default();

        let batch_size = 4;
        let dim = 16;

        // Create predictions (8 samples)
        let predictions: Vec<Tensor<TestBackend, 2>> = (0..8)
            .map(|_| Tensor::random([batch_size, dim], Distribution::Normal(0.0, 1.0), &device))
            .collect();

        // Create targets
        let targets = Tensor::random([batch_size, dim], Distribution::Normal(0.0, 1.0), &device);

        let loss = loss_fn.compute(&predictions, targets);

        // Loss should be a scalar
        assert_eq!(loss.dims(), [1]);
    }

    #[test]
    fn test_residual_block() {
        let device = init_gpu_device();
        let dim = 32;
        let block = ResidualMlpBlock::<TestBackend>::new(dim, dim, &device);

        let batch_size = 4;
        let x = Tensor::random([batch_size, dim], Distribution::Normal(0.0, 1.0), &device);
        let h = Tensor::random([batch_size, dim], Distribution::Normal(0.0, 1.0), &device);

        let out = block.forward(x, h);

        assert_eq!(out.dims(), [batch_size, dim]);
    }

    #[test]
    fn test_learned_partition_router() {
        let device = init_gpu_device();
        let config = EnergyHeadConfig::dev();
        let head = EnergyHead::<TestBackend>::new(config.clone(), &device);

        let num_partitions = config.output_dim;
        let embed_dim = config.input_dim;

        let centroids = Tensor::random(
            [num_partitions, embed_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let partition_sizes = vec![10; num_partitions];

        let router = LearnedPartitionRouter::new(head, centroids, partition_sizes);

        let query = Tensor::random([embed_dim], Distribution::Normal(0.0, 1.0), &device);

        let partition = router.route(&query, &device);
        assert!(partition < num_partitions);

        let (top_k_idx, top_k_scores) = router.route_top_k(&query, 3, &device);
        assert_eq!(top_k_idx.len(), 3);
        assert_eq!(top_k_scores.len(), 3);
    }
}
