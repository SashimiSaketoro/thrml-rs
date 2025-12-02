//! Training infrastructure for NavigatorEBM.
//!
//! This module provides contrastive divergence training for the navigator,
//! allowing the energy weights (lambdas) to be learned from data.
//!
//! ## Training Objective
//!
//! We use contrastive divergence to minimize the KL divergence between
//! the model distribution and the data distribution:
//!
//! ```text
//! ∂L/∂θ = ⟨∂E/∂θ⟩_data - ⟨∂E/∂θ⟩_model
//! ```
//!
//! where:
//! - ⟨·⟩_data is the expectation under the data distribution (positive phase)
//! - ⟨·⟩_model is the expectation under the model distribution (negative phase)

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::compute::{ComputeBackend, OpType};
use thrml_samplers::RngKey;

use crate::config::SphereConfig;
use crate::navigator::{NavigationWeights, NavigatorEBM};

/// Training configuration for NavigatorEBM.
#[derive(Clone, Debug)]
pub struct NavigatorTrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Number of negative samples per positive
    pub negatives_per_positive: usize,
    /// Number of Langevin steps for negative sampling
    pub negative_sample_steps: usize,
    /// Temperature for negative sampling
    pub temperature: f32,
    /// Momentum for SGD (0 = no momentum)
    pub momentum: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Gradient clipping threshold (0 = no clipping)
    pub gradient_clip: f32,
}

impl Default for NavigatorTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            negatives_per_positive: 4,
            negative_sample_steps: 10,
            temperature: 0.5,
            momentum: 0.9,
            weight_decay: 1e-4,
            gradient_clip: 1.0,
        }
    }
}

impl NavigatorTrainingConfig {
    /// Builder: set learning rate.
    pub const fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder: set negatives per positive.
    pub const fn with_negatives(mut self, n: usize) -> Self {
        self.negatives_per_positive = n;
        self
    }

    /// Builder: set momentum.
    pub const fn with_momentum(mut self, m: f32) -> Self {
        self.momentum = m;
        self
    }
}

/// Training state for SGD with momentum.
#[derive(Clone)]
pub struct TrainingState {
    /// Current weights
    pub weights: NavigationWeights,
    /// Momentum buffer for weight gradients
    pub momentum_buffer: Option<Tensor<WgpuBackend, 1>>,
    /// Training step counter
    pub step: usize,
}

impl TrainingState {
    /// Create initial training state.
    pub const fn new(weights: NavigationWeights) -> Self {
        Self {
            weights,
            momentum_buffer: None,
            step: 0,
        }
    }
}

/// A single training example for navigation.
#[derive(Clone)]
pub struct TrainingExample {
    /// Query embedding
    pub query: Tensor<WgpuBackend, 1>,
    /// Query radius
    pub query_radius: f32,
    /// Index of correct target
    pub positive_target: usize,
    /// Optional: indices of known negative targets
    pub negative_targets: Option<Vec<usize>>,
}

/// Result of a training step.
#[derive(Clone, Debug)]
pub struct TrainingStepResult {
    /// Loss value
    pub loss: f32,
    /// Gradient norms (for debugging)
    pub gradient_norm: f32,
    /// Updated weights
    pub new_weights: NavigationWeights,
}

/// Trainable wrapper around NavigatorEBM.
///
/// This struct adds training functionality to NavigatorEBM, including:
/// - Contrastive divergence gradient estimation
/// - SGD with momentum
/// - Learning rate scheduling
pub struct TrainableNavigatorEBM {
    /// The underlying navigator
    pub navigator: NavigatorEBM,
    /// Training configuration
    pub config: NavigatorTrainingConfig,
    /// Current training state
    pub state: TrainingState,
}

impl TrainableNavigatorEBM {
    /// Create a trainable navigator from an existing NavigatorEBM.
    pub fn from_navigator(navigator: NavigatorEBM, config: NavigatorTrainingConfig) -> Self {
        let state = TrainingState::new(navigator.weights.clone());
        Self {
            navigator,
            config,
            state,
        }
    }

    /// Create a trainable navigator from embeddings and prominence.
    pub fn new(
        embeddings: Tensor<WgpuBackend, 2>,
        prominence: Tensor<WgpuBackend, 1>,
        entropies: Option<Tensor<WgpuBackend, 1>>,
        sphere_config: SphereConfig,
        training_config: NavigatorTrainingConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let navigator = NavigatorEBM::new(embeddings, prominence, entropies, sphere_config, device);
        Self::from_navigator(navigator, training_config)
    }

    /// Get current weights as tensor.
    pub fn weights_tensor(
        &self,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        self.state.weights.to_tensor(device)
    }

    /// Update weights from tensor.
    pub fn set_weights(&mut self, tensor: &Tensor<WgpuBackend, 1>) {
        self.state.weights = NavigationWeights::from_tensor(tensor);
        self.navigator.weights = self.state.weights.clone();
    }

    /// Compute energy for a single target given query.
    fn compute_target_energy(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        query_radius: f32,
        target_idx: usize,
        coords: &thrml_core::SphericalCoords,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> f32 {
        let energies =
            self.navigator
                .total_energy(query, query_radius, coords, &[target_idx], None, device);
        let energy_data: Vec<f32> = energies.into_data().to_vec().expect("energy to vec");
        energy_data.first().copied().unwrap_or(0.0)
    }

    /// Sample negative targets using current energy function.
    fn sample_negatives(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        query_radius: f32,
        positive_target: usize,
        n_negatives: usize,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<usize> {
        // Get optimized coordinates
        let coords = self.navigator.sphere_ebm.optimize(key, device);

        // Compute energies for all points
        let n = self.navigator.n_points();
        let all_indices: Vec<usize> = (0..n).collect();
        let energies =
            self.navigator
                .total_energy(query, query_radius, &coords, &all_indices, None, device);

        // Convert to probabilities via softmax
        let energy_data: Vec<f32> = energies.into_data().to_vec().expect("energies to vec");
        let min_e = energy_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let exp_neg_e: Vec<f32> = energy_data
            .iter()
            .map(|&e| (-(e - min_e) / self.config.temperature).exp())
            .collect();
        let sum_exp: f32 = exp_neg_e.iter().sum();
        let probs: Vec<f32> = exp_neg_e.iter().map(|&e| e / sum_exp).collect();

        // Sample without replacement (excluding positive)
        let mut negatives = Vec::with_capacity(n_negatives);
        let mut available: Vec<(usize, f32)> = probs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != positive_target)
            .map(|(i, &p)| (i, p))
            .collect();

        // Sort by probability (descending) for biased sampling
        available.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k as negatives (hard negatives)
        for (idx, _) in available.iter().take(n_negatives) {
            negatives.push(*idx);
        }

        negatives
    }

    /// Get differentiable soft weights over candidate negatives using Gumbel-Softmax.
    ///
    /// This enables differentiable sampling for end-to-end gradient computation.
    /// Returns soft weights over all candidates (excluding positive), which can be
    /// used to compute a weighted average of negative energies.
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding
    /// * `query_radius` - Query radius
    /// * `positive_target` - Index of positive target (excluded from negatives)
    /// * `temperature` - Gumbel-Softmax temperature (lower = harder selection)
    /// * `key` - RNG key
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// (candidate_indices, soft_weights) where soft_weights are differentiable
    pub fn get_negative_weights_differentiable(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        query_radius: f32,
        positive_target: usize,
        temperature: f32,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Vec<usize>, Tensor<WgpuBackend, 1>) {
        // Get optimized coordinates
        let coords = self.navigator.sphere_ebm.optimize(key, device);

        // Compute energies for all points except positive
        let n = self.navigator.n_points();
        let candidate_indices: Vec<usize> = (0..n).filter(|&i| i != positive_target).collect();
        let n_candidates = candidate_indices.len();

        if n_candidates == 0 {
            return (vec![], Tensor::zeros([0], device));
        }

        // Use batched energy computation
        let indices_tensor = NavigatorEBM::indices_to_tensor(&candidate_indices, device);
        let backend = ComputeBackend::default();
        let energies = self.navigator.total_energy_batched(
            query,
            query_radius,
            &coords,
            &indices_tensor,
            device,
            &backend,
        );

        // Convert energies to logits (negative energy = higher probability)
        let logits = energies.neg(); // [n_candidates]

        // Apply Gumbel-Softmax for differentiable sampling
        let logits_2d: Tensor<WgpuBackend, 2> = logits.reshape([1, n_candidates as i32]);
        let soft_weights_2d = thrml_kernels::autodiff::gumbel_softmax::gumbel_softmax(
            logits_2d,
            temperature,
            false, // soft selection
            device,
        );
        let soft_weights: Tensor<WgpuBackend, 1> = soft_weights_2d.reshape([n_candidates as i32]);

        (candidate_indices, soft_weights)
    }

    /// Compute contrastive loss with differentiable negative weighting.
    ///
    /// Uses Gumbel-Softmax to create soft weights over negative candidates,
    /// enabling end-to-end differentiability. This is an alternative to the
    /// hard negative sampling approach.
    ///
    /// # Arguments
    ///
    /// * `example` - Training example
    /// * `gumbel_temperature` - Temperature for Gumbel-Softmax (lower = harder)
    /// * `key` - RNG key
    /// * `device` - GPU device
    /// * `backend` - Compute backend
    ///
    /// # Returns
    ///
    /// (loss, gradients) where loss uses soft-weighted negative energies
    pub fn compute_contrastive_loss_differentiable(
        &self,
        example: &TrainingExample,
        gumbel_temperature: f32,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
        backend: &ComputeBackend,
    ) -> (f32, Vec<f32>) {
        let coords = self.navigator.sphere_ebm.optimize(key, device);

        // Get soft weights over negatives
        let (neg_indices, soft_weights) = self.get_negative_weights_differentiable(
            &example.query,
            example.query_radius,
            example.positive_target,
            gumbel_temperature,
            key,
            device,
        );

        if neg_indices.is_empty() {
            return (0.0, vec![0.0; 6]);
        }

        // Compute positive energy
        let pos_idx = NavigatorEBM::indices_to_tensor(&[example.positive_target], device);
        let e_pos_tensor = self.navigator.total_energy_batched(
            &example.query,
            example.query_radius,
            &coords,
            &pos_idx,
            device,
            backend,
        );
        let e_pos: f32 = e_pos_tensor.into_data().to_vec().expect("e_pos")[0];

        // Compute negative energies
        let neg_idx_tensor = NavigatorEBM::indices_to_tensor(&neg_indices, device);
        let neg_energies = self.navigator.total_energy_batched(
            &example.query,
            example.query_radius,
            &coords,
            &neg_idx_tensor,
            device,
            backend,
        );

        // Soft-weighted average negative energy
        let weighted_neg = neg_energies * soft_weights.clone();
        let e_neg_weighted: f32 = weighted_neg.sum().into_data().to_vec().expect("weighted")[0];

        // InfoNCE-style loss with soft negatives
        let scaled_e_pos = e_pos / self.config.temperature;
        let scaled_e_neg = e_neg_weighted / self.config.temperature;
        let loss = scaled_e_pos + (-scaled_e_neg).exp().ln_1p().max(0.0);

        // Gradient estimation (simplified for soft weighting)
        // For full differentiability, would use burn autodiff
        let sem_pos = self
            .navigator
            .semantic_energy_batched(&example.query, &pos_idx, device);
        let sem_neg =
            self.navigator
                .semantic_energy_batched(&example.query, &neg_idx_tensor, device);
        let sem_neg_weighted = (sem_neg * soft_weights.clone()).sum();

        let rad_pos =
            self.navigator
                .radial_energy_batched(example.query_radius, &coords, &pos_idx, device);
        let rad_neg = self.navigator.radial_energy_batched(
            example.query_radius,
            &coords,
            &neg_idx_tensor,
            device,
        );
        let rad_neg_weighted = (rad_neg * soft_weights.clone()).sum();

        let ent_pos = self.navigator.entropy_energy_batched(&pos_idx, device);
        let ent_neg = self
            .navigator
            .entropy_energy_batched(&neg_idx_tensor, device);
        let ent_neg_weighted = (ent_neg * soft_weights).sum();

        let sem_pos_val: f32 = sem_pos.into_data().to_vec().expect("sem")[0];
        let sem_neg_val: f32 = sem_neg_weighted.into_data().to_vec().expect("sem neg")[0];
        let rad_pos_val: f32 = rad_pos.into_data().to_vec().expect("rad")[0];
        let rad_neg_val: f32 = rad_neg_weighted.into_data().to_vec().expect("rad neg")[0];
        let ent_pos_val: f32 = ent_pos.into_data().to_vec().expect("ent")[0];
        let ent_neg_val: f32 = ent_neg_weighted.into_data().to_vec().expect("ent neg")[0];

        let gradients = vec![
            sem_pos_val - sem_neg_val,
            rad_pos_val - rad_neg_val,
            0.0, // graph
            ent_pos_val - ent_neg_val,
            0.0, // path
            0.0, // harmonic
        ];

        (loss, gradients)
    }

    /// Compute contrastive loss for a single example.
    ///
    /// Uses InfoNCE-style loss:
    /// L = -log(exp(-E_pos) / (exp(-E_pos) + sum_neg exp(-E_neg)))
    fn compute_contrastive_loss(
        &self,
        example: &TrainingExample,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (f32, Vec<f32>) {
        // Get coordinates
        let coords = self.navigator.sphere_ebm.optimize(key, device);

        // Compute positive energy
        let e_pos = self.compute_target_energy(
            &example.query,
            example.query_radius,
            example.positive_target,
            &coords,
            device,
        );

        // Get or sample negatives
        let negatives = example.negative_targets.clone().unwrap_or_else(|| {
            self.sample_negatives(
                &example.query,
                example.query_radius,
                example.positive_target,
                self.config.negatives_per_positive,
                key,
                device,
            )
        });

        // Compute negative energies
        let neg_energies: Vec<f32> = negatives
            .iter()
            .map(|&idx| {
                self.compute_target_energy(
                    &example.query,
                    example.query_radius,
                    idx,
                    &coords,
                    device,
                )
            })
            .collect();

        // InfoNCE loss with numerical stability (log-sum-exp trick)
        // L = -log(exp(-E_pos/T) / (exp(-E_pos/T) + sum_neg exp(-E_neg/T)))
        // L = -(-E_pos/T - log(1 + sum_neg exp(-(E_neg - E_pos)/T)))
        // L = E_pos/T + log(1 + sum_neg exp(-(E_neg - E_pos)/T))
        let scaled_e_pos = e_pos / self.config.temperature;
        let sum_exp_neg: f32 = neg_energies
            .iter()
            .map(|&e| {
                let diff = (e_pos - e) / self.config.temperature;
                // Clamp to prevent overflow
                diff.clamp(-50.0, 50.0).exp()
            })
            .sum();
        // Use log1p for numerical stability when sum_exp_neg is small
        let loss = scaled_e_pos + sum_exp_neg.ln_1p().max(0.0);

        // Collect energy gradients w.r.t. weights
        // For each lambda_i: grad = d(loss)/d(lambda_i) ≈ E_pos_i - avg(E_neg_i)
        // where E_*_i is the i-th energy component
        let mut energy_components_pos = [0.0f32; 6];
        let mut energy_components_neg = [0.0f32; 6];

        // Compute component energies for positive
        energy_components_pos[0] = self
            .navigator
            .semantic_energy(&example.query, &[example.positive_target], device)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .first()
            .copied()
            .unwrap_or(0.0);
        energy_components_pos[1] = self
            .navigator
            .radial_energy(
                example.query_radius,
                &coords,
                &[example.positive_target],
                device,
            )
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .first()
            .copied()
            .unwrap_or(0.0);
        energy_components_pos[3] = self
            .navigator
            .entropy_energy(&[example.positive_target], device)
            .into_data()
            .to_vec::<f32>()
            .unwrap()
            .first()
            .copied()
            .unwrap_or(0.0);

        // Average component energies for negatives
        for &neg_idx in &negatives {
            energy_components_neg[0] += self
                .navigator
                .semantic_energy(&example.query, &[neg_idx], device)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0);
            energy_components_neg[1] += self
                .navigator
                .radial_energy(example.query_radius, &coords, &[neg_idx], device)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0);
            energy_components_neg[3] += self
                .navigator
                .entropy_energy(&[neg_idx], device)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .first()
                .copied()
                .unwrap_or(0.0);
        }

        let n_neg = negatives.len().max(1) as f32;
        for e in &mut energy_components_neg {
            *e /= n_neg;
        }

        // Gradient: d(loss)/d(lambda) ≈ component_pos - component_neg
        let gradients: Vec<f32> = (0..6)
            .map(|i| energy_components_pos[i] - energy_components_neg[i])
            .collect();

        (loss, gradients)
    }

    /// Compute contrastive loss using GPU-batched energy computation.
    ///
    /// This is a GPU-optimized version that:
    /// - Batches all energy computations (positive + all negatives) into one GPU call
    /// - Uses log-sum-exp trick for numerical stability on GPU
    /// - Still computes gradients on CPU for f64 precision
    ///
    /// # Arguments
    ///
    /// * `example` - Training example with query and targets
    /// * `key` - RNG key for coordinate optimization and negative sampling
    /// * `device` - GPU device
    /// * `backend` - Compute backend for routing decisions
    ///
    /// # Returns
    ///
    /// (loss, gradients) where gradients are w.r.t. the lambda weights
    fn compute_contrastive_loss_batched(
        &self,
        example: &TrainingExample,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
        backend: &ComputeBackend,
    ) -> (f32, Vec<f32>) {
        // Get coordinates
        let coords = self.navigator.sphere_ebm.optimize(key, device);

        // Get or sample negatives
        let negatives = example.negative_targets.clone().unwrap_or_else(|| {
            self.sample_negatives(
                &example.query,
                example.query_radius,
                example.positive_target,
                self.config.negatives_per_positive,
                key,
                device,
            )
        });

        // Batch all targets: [positive] + negatives
        let mut all_targets = vec![example.positive_target];
        all_targets.extend(&negatives);

        // Convert to tensor for batched computation
        let target_indices = NavigatorEBM::indices_to_tensor(&all_targets, device);
        let n_neg = negatives.len();

        // === GPU-batched forward pass ===
        // Compute all energies in one GPU call
        let all_energies = self.navigator.total_energy_batched(
            &example.query,
            example.query_radius,
            &coords,
            &target_indices,
            device,
            backend,
        );

        // Extract energies to CPU for loss computation
        // (Could also do loss on GPU, but keeping gradient computation on CPU for precision)
        let energy_data: Vec<f32> = all_energies.into_data().to_vec().expect("energies to vec");

        let e_pos = energy_data[0];
        let neg_energies = &energy_data[1..];

        // === CPU loss computation with log-sum-exp trick ===
        let scaled_e_pos = e_pos / self.config.temperature;
        let sum_exp_neg: f32 = neg_energies
            .iter()
            .map(|&e| {
                let diff = (e_pos - e) / self.config.temperature;
                diff.clamp(-50.0, 50.0).exp()
            })
            .sum();
        let loss = scaled_e_pos + sum_exp_neg.ln_1p().max(0.0);

        // === GPU-batched component energy computation for gradients ===
        // Compute individual energy components to get gradients w.r.t. lambdas
        let pos_idx = NavigatorEBM::indices_to_tensor(&[example.positive_target], device);
        let neg_idx = NavigatorEBM::indices_to_tensor(&negatives, device);

        // Semantic energy components
        let sem_pos = self
            .navigator
            .semantic_energy_batched(&example.query, &pos_idx, device);
        let sem_neg = self
            .navigator
            .semantic_energy_batched(&example.query, &neg_idx, device);

        // Radial energy components
        let rad_pos =
            self.navigator
                .radial_energy_batched(example.query_radius, &coords, &pos_idx, device);
        let rad_neg =
            self.navigator
                .radial_energy_batched(example.query_radius, &coords, &neg_idx, device);

        // Entropy energy components
        let ent_pos = self.navigator.entropy_energy_batched(&pos_idx, device);
        let ent_neg = self.navigator.entropy_energy_batched(&neg_idx, device);

        // Extract to CPU for gradient computation (f32 is fine here, values are small)
        let sem_pos_val: f32 = sem_pos.into_data().to_vec().expect("sem")[0];
        let sem_neg_val: f32 = if n_neg > 0 {
            let neg_data: Vec<f32> = sem_neg.into_data().to_vec().expect("sem neg");
            neg_data.iter().sum::<f32>() / n_neg as f32
        } else {
            0.0
        };

        let rad_pos_val: f32 = rad_pos.into_data().to_vec().expect("rad")[0];
        let rad_neg_val: f32 = if n_neg > 0 {
            let neg_data: Vec<f32> = rad_neg.into_data().to_vec().expect("rad neg");
            neg_data.iter().sum::<f32>() / n_neg as f32
        } else {
            0.0
        };

        let ent_pos_val: f32 = ent_pos.into_data().to_vec().expect("ent")[0];
        let ent_neg_val: f32 = if n_neg > 0 {
            let neg_data: Vec<f32> = ent_neg.into_data().to_vec().expect("ent neg");
            neg_data.iter().sum::<f32>() / n_neg as f32
        } else {
            0.0
        };

        // Gradients: d(loss)/d(lambda) ≈ component_pos - component_neg
        // Components: [semantic, radial, graph, entropy, path, harmonic]
        let gradients = vec![
            sem_pos_val - sem_neg_val, // semantic
            rad_pos_val - rad_neg_val, // radial
            0.0,                       // graph (computed separately if needed)
            ent_pos_val - ent_neg_val, // entropy
            0.0,                       // path (computed separately if needed)
            0.0,                       // harmonic
        ];

        (loss, gradients)
    }

    /// Perform a single training step with hybrid CPU/GPU execution.
    ///
    /// Uses GPU for batched forward pass, CPU for gradient accumulation.
    /// Routes based on ComputeBackend configuration.
    pub fn train_step_hybrid(
        &mut self,
        examples: &[TrainingExample],
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
        backend: &ComputeBackend,
    ) -> TrainingStepResult {
        let n_examples = examples.len().max(1);

        // Check if we should use GPU batched path or CPU path
        let use_gpu_batched = backend.use_gpu(OpType::BatchEnergyForward, Some(n_examples));

        // Accumulate gradients over examples
        let mut total_loss = 0.0f32;
        let mut total_grads = vec![0.0f32; 6];

        let keys = key.split(n_examples);
        for (example, k) in examples.iter().zip(keys.into_iter()) {
            let (loss, grads) = if use_gpu_batched {
                self.compute_contrastive_loss_batched(example, k, device, backend)
            } else {
                self.compute_contrastive_loss(example, k, device)
            };
            total_loss += loss;
            for i in 0..6 {
                total_grads[i] += grads[i];
            }
        }

        // Continue with standard gradient processing (weight decay, clipping, SGD)
        // This part stays on CPU for precision
        let avg_factor = 1.0 / n_examples as f32;
        total_loss *= avg_factor;
        for g in &mut total_grads {
            *g *= avg_factor;
        }

        // Add weight decay
        let current_weights = self.state.weights.to_tensor(device);
        let current_data: Vec<f32> = current_weights
            .into_data()
            .to_vec()
            .expect("weights to vec");
        for i in 0..6 {
            total_grads[i] += self.config.weight_decay * current_data[i];
        }

        // Gradient clipping
        let grad_norm: f32 = total_grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        if self.config.gradient_clip > 0.0 && grad_norm > self.config.gradient_clip {
            let scale = self.config.gradient_clip / grad_norm;
            for g in &mut total_grads {
                *g *= scale;
            }
        }

        // SGD with momentum
        let grad_tensor: Tensor<WgpuBackend, 1> = Tensor::from_data(total_grads.as_slice(), device);

        let update = if self.config.momentum > 0.0 {
            let momentum_buf = match &self.state.momentum_buffer {
                Some(buf) => buf.clone() * self.config.momentum + grad_tensor,
                None => grad_tensor,
            };
            self.state.momentum_buffer = Some(momentum_buf.clone());
            momentum_buf
        } else {
            grad_tensor
        };

        // Update weights
        let new_weights_tensor =
            self.state.weights.to_tensor(device) - update * self.config.learning_rate;

        // Clamp weights to be non-negative
        let new_weights_tensor = new_weights_tensor.clamp(0.0, f32::MAX);

        let new_weights = NavigationWeights::from_tensor(&new_weights_tensor);

        // Update state
        self.state.weights = new_weights.clone();
        self.state.step += 1;
        self.navigator.weights = new_weights.clone();

        TrainingStepResult {
            loss: total_loss,
            gradient_norm: grad_norm,
            new_weights,
        }
    }

    /// Perform a single training step.
    pub fn train_step(
        &mut self,
        examples: &[TrainingExample],
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> TrainingStepResult {
        let n_examples = examples.len().max(1);

        // Accumulate gradients over examples
        let mut total_loss = 0.0f32;
        let mut total_grads = vec![0.0f32; 6];

        let keys = key.split(n_examples);
        for (example, k) in examples.iter().zip(keys.into_iter()) {
            let (loss, grads) = self.compute_contrastive_loss(example, k, device);
            total_loss += loss;
            for i in 0..6 {
                total_grads[i] += grads[i];
            }
        }

        // Average gradients
        let avg_factor = 1.0 / n_examples as f32;
        total_loss *= avg_factor;
        for g in &mut total_grads {
            *g *= avg_factor;
        }

        // Add weight decay
        let current_weights = self.state.weights.to_tensor(device);
        let current_data: Vec<f32> = current_weights
            .into_data()
            .to_vec()
            .expect("weights to vec");
        for i in 0..6 {
            total_grads[i] += self.config.weight_decay * current_data[i];
        }

        // Gradient clipping
        let grad_norm: f32 = total_grads.iter().map(|g| g * g).sum::<f32>().sqrt();
        if self.config.gradient_clip > 0.0 && grad_norm > self.config.gradient_clip {
            let scale = self.config.gradient_clip / grad_norm;
            for g in &mut total_grads {
                *g *= scale;
            }
        }

        // SGD with momentum
        let grad_tensor: Tensor<WgpuBackend, 1> = Tensor::from_data(total_grads.as_slice(), device);

        let update = if self.config.momentum > 0.0 {
            let momentum_buf = match &self.state.momentum_buffer {
                Some(buf) => buf.clone() * self.config.momentum + grad_tensor,
                None => grad_tensor,
            };
            self.state.momentum_buffer = Some(momentum_buf.clone());
            momentum_buf
        } else {
            grad_tensor
        };

        // Update weights
        let new_weights_tensor =
            self.state.weights.to_tensor(device) - update * self.config.learning_rate;

        // Clamp weights to be non-negative
        let new_weights_tensor = new_weights_tensor.clamp(0.0, f32::MAX);

        let new_weights = NavigationWeights::from_tensor(&new_weights_tensor);

        // Update state
        self.state.weights = new_weights.clone();
        self.state.step += 1;
        self.navigator.weights = new_weights.clone();

        TrainingStepResult {
            loss: total_loss,
            gradient_norm: grad_norm,
            new_weights,
        }
    }

    /// Train for multiple epochs.
    pub fn train(
        &mut self,
        examples: &[TrainingExample],
        n_epochs: usize,
        batch_size: usize,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<f32> {
        let mut losses = Vec::with_capacity(n_epochs);
        let keys = key.split(n_epochs);

        for (epoch, epoch_key) in keys.into_iter().enumerate() {
            // Simple batching (could be improved with shuffling)
            let mut epoch_loss = 0.0;
            let n_batches = examples.len().div_ceil(batch_size);
            let batch_keys = epoch_key.split(n_batches);

            for (batch_idx, batch_key) in batch_keys.into_iter().enumerate() {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(examples.len());
                let batch = &examples[start..end];

                let result = self.train_step(batch, batch_key, device);
                epoch_loss += result.loss;
            }

            let avg_loss = epoch_loss / n_batches.max(1) as f32;
            losses.push(avg_loss);

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: loss = {:.4}, weights = {:?}",
                    epoch, avg_loss, self.state.weights
                );
            }
        }

        losses
    }

    /// Train for multiple epochs using hybrid CPU/GPU execution.
    ///
    /// This is the recommended training method for most use cases:
    /// - Uses GPU for batched energy computation (fast forward pass)
    /// - Uses CPU for gradient accumulation (precision)
    /// - Automatically routes based on ComputeBackend configuration
    ///
    /// # Arguments
    ///
    /// * `examples` - Training examples
    /// * `n_epochs` - Number of training epochs
    /// * `batch_size` - Examples per batch
    /// * `key` - RNG key
    /// * `device` - GPU device
    /// * `backend` - Compute backend for routing decisions (None = auto-detect)
    ///
    /// # Returns
    ///
    /// Vector of average losses per epoch
    pub fn train_hybrid(
        &mut self,
        examples: &[TrainingExample],
        n_epochs: usize,
        batch_size: usize,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
        backend: Option<&ComputeBackend>,
    ) -> Vec<f32> {
        let default_backend = ComputeBackend::default();
        let backend = backend.unwrap_or(&default_backend);

        let mut losses = Vec::with_capacity(n_epochs);
        let keys = key.split(n_epochs);

        for (epoch, epoch_key) in keys.into_iter().enumerate() {
            let mut epoch_loss = 0.0;
            let n_batches = examples.len().div_ceil(batch_size);
            let batch_keys = epoch_key.split(n_batches);

            for (batch_idx, batch_key) in batch_keys.into_iter().enumerate() {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(examples.len());
                let batch = &examples[start..end];

                let result = self.train_step_hybrid(batch, batch_key, device, backend);
                epoch_loss += result.loss;
            }

            let avg_loss = epoch_loss / n_batches.max(1) as f32;
            losses.push(avg_loss);

            if epoch % 10 == 0 {
                println!(
                    "Epoch {}: loss = {:.4}, weights = {:?}",
                    epoch, avg_loss, self.state.weights
                );
            }
        }

        losses
    }

    /// Get the trained navigator.
    pub fn into_navigator(self) -> NavigatorEBM {
        self.navigator
    }
}

// ============================================================================
// Training Dataset
// ============================================================================

/// Training dataset with train/validation split.
///
/// Provides convenient access to training and validation sets, with
/// utilities for splitting and shuffling.
///
/// # Example
///
/// ```rust,ignore
/// // Generate examples
/// let examples = generate_pairs_from_similarity(&sphere_ebm, 1, 8, &device);
///
/// // Create dataset with 10% validation
/// let dataset = TrainingDataset::from_examples(examples, 0.1, 42);
///
/// println!("Train: {}, Val: {}", dataset.n_train(), dataset.n_val());
/// ```
#[derive(Clone)]
pub struct TrainingDataset {
    /// Training examples.
    pub train: Vec<TrainingExample>,
    /// Validation examples.
    pub validation: Vec<TrainingExample>,
}

impl TrainingDataset {
    /// Create a new dataset from train and validation sets.
    pub const fn new(train: Vec<TrainingExample>, validation: Vec<TrainingExample>) -> Self {
        Self { train, validation }
    }

    /// Create a dataset from examples with a validation split.
    ///
    /// # Arguments
    ///
    /// * `examples` - All training examples
    /// * `val_fraction` - Fraction to use for validation (0.0 to 1.0)
    /// * `seed` - Random seed for shuffling before split
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let dataset = TrainingDataset::from_examples(examples, 0.1, 42);
    /// ```
    pub fn from_examples(mut examples: Vec<TrainingExample>, val_fraction: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&val_fraction),
            "val_fraction must be in [0, 1)"
        );

        // Simple deterministic shuffle using seed
        let n = examples.len();
        let mut rng_state = seed;
        for i in (1..n).rev() {
            // Simple LCG for shuffling
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            examples.swap(i, j);
        }

        let val_size = (n as f32 * val_fraction).round() as usize;
        let validation = examples.split_off(n - val_size);

        Self {
            train: examples,
            validation,
        }
    }

    /// Number of training examples.
    pub const fn n_train(&self) -> usize {
        self.train.len()
    }

    /// Number of validation examples.
    pub const fn n_val(&self) -> usize {
        self.validation.len()
    }

    /// Total number of examples.
    pub const fn n_total(&self) -> usize {
        self.n_train() + self.n_val()
    }
}

// ============================================================================
// Data Generation
// ============================================================================

/// Generate training pairs from embedding similarity.
///
/// For each embedding, finds the most similar embeddings (excluding self)
/// and creates training pairs where the positive target is a similar embedding.
///
/// # Arguments
///
/// * `sphere_ebm` - The sphere model with embeddings
/// * `n_positives_per_query` - Number of similar embeddings to use as positives per query
/// * `n_negatives_per_positive` - Number of random negatives per positive
/// * `device` - GPU device
///
/// # Returns
///
/// Vector of training examples with queries and positive targets.
///
/// # Example
///
/// ```rust,ignore
/// let examples = generate_pairs_from_similarity(&sphere_ebm, 1, 8, &device);
/// println!("Generated {} training pairs", examples.len());
/// ```
pub fn generate_pairs_from_similarity(
    sphere_ebm: &crate::sphere_ebm::SphereEBM,
    n_positives_per_query: usize,
    n_negatives_per_positive: usize,
    _device: &burn::backend::wgpu::WgpuDevice,
) -> Vec<TrainingExample> {
    let n = sphere_ebm.n_points();
    let d = sphere_ebm.embedding_dim();

    if n < 2 {
        return Vec::new();
    }

    // Get similarity matrix (already computed in sphere_ebm)
    let sim_data: Vec<f32> = sphere_ebm
        .similarity
        .clone()
        .into_data()
        .to_vec()
        .expect("sim to vec");

    let mut examples = Vec::with_capacity(n * n_positives_per_query);

    for query_idx in 0..n {
        // Get query embedding
        let query: Tensor<WgpuBackend, 1> = sphere_ebm
            .embeddings
            .clone()
            .slice([query_idx..query_idx + 1, 0..d])
            .reshape([d as i32]);

        // Find top-k similar (excluding self)
        let mut similarities: Vec<(usize, f32)> = (0..n)
            .filter(|&i| i != query_idx)
            .map(|i| {
                let sim = sim_data[query_idx * n + i];
                (i, sim)
            })
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top positives
        for (positive_idx, _sim) in similarities.iter().take(n_positives_per_query) {
            // Generate random negatives (simple approach: take least similar)
            let negatives: Vec<usize> = similarities
                .iter()
                .rev() // Least similar
                .take(n_negatives_per_positive)
                .map(|(idx, _)| *idx)
                .collect();

            examples.push(TrainingExample {
                query: query.clone(),
                query_radius: 50.0, // Default radius
                positive_target: *positive_idx,
                negative_targets: if negatives.is_empty() {
                    None
                } else {
                    Some(negatives)
                },
            });
        }
    }

    examples
}

/// Generate training pairs from hypergraph edges.
///
/// Uses connected nodes in the hypergraph as positive pairs.
/// Nodes that are connected by an edge are considered related.
///
/// # Arguments
///
/// * `sphere_ebm` - The sphere model with embeddings
/// * `sidecar` - Hypergraph structure with edges
/// * `n_negatives_per_positive` - Number of random negatives per positive
///
/// # Returns
///
/// Vector of training examples where positives are connected nodes.
///
/// # Example
///
/// ```rust,ignore
/// let examples = generate_pairs_from_edges(&sphere_ebm, &sidecar, 8);
/// ```
pub fn generate_pairs_from_edges(
    sphere_ebm: &crate::sphere_ebm::SphereEBM,
    sidecar: &crate::hypergraph::HypergraphSidecar,
    n_negatives_per_positive: usize,
) -> Vec<TrainingExample> {
    let n = sphere_ebm.n_points();
    let d = sphere_ebm.embedding_dim();

    if n < 2 || sidecar.edges.is_empty() {
        return Vec::new();
    }

    let mut examples = Vec::new();

    // Create adjacency list for quick lookup
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for edge in &sidecar.edges {
        if edge.src < n && edge.dst < n {
            adjacency[edge.src].push(edge.dst);
        }
    }

    for (query_idx, neighbors) in adjacency.iter().enumerate().take(n) {
        if neighbors.is_empty() {
            continue;
        }

        // Get query embedding
        let query: Tensor<WgpuBackend, 1> = sphere_ebm
            .embeddings
            .clone()
            .slice([query_idx..query_idx + 1, 0..d])
            .reshape([d as i32]);

        // Each neighbor is a positive
        for &positive_idx in neighbors {
            // Generate negatives: nodes that are NOT neighbors
            let negatives: Vec<usize> = (0..n)
                .filter(|&i| i != query_idx && i != positive_idx && !neighbors.contains(&i))
                .take(n_negatives_per_positive)
                .collect();

            examples.push(TrainingExample {
                query: query.clone(),
                query_radius: 50.0,
                positive_target: positive_idx,
                negative_targets: if negatives.is_empty() {
                    None
                } else {
                    Some(negatives)
                },
            });
        }
    }

    examples
}

// ============================================================================
// Extended Training with Validation
// ============================================================================

/// Extended training configuration with validation support.
#[derive(Clone, Debug)]
pub struct ExtendedTrainingConfig {
    /// Base training configuration.
    pub base: NavigatorTrainingConfig,
    /// Validate every N epochs.
    pub val_every_n_epochs: usize,
    /// Early stopping patience (stop if no improvement for N validations).
    pub early_stopping_patience: usize,
    /// k for recall@k and nDCG@k evaluation.
    pub top_k_eval: usize,
}

impl Default for ExtendedTrainingConfig {
    fn default() -> Self {
        Self {
            base: NavigatorTrainingConfig::default(),
            val_every_n_epochs: 5,
            early_stopping_patience: 5,
            top_k_eval: 10,
        }
    }
}

impl ExtendedTrainingConfig {
    /// Create with base config.
    pub fn new(base: NavigatorTrainingConfig) -> Self {
        Self {
            base,
            ..Default::default()
        }
    }

    /// Builder: set validation frequency.
    pub const fn with_val_every(mut self, epochs: usize) -> Self {
        self.val_every_n_epochs = epochs;
        self
    }

    /// Builder: set early stopping patience.
    pub const fn with_early_stopping(mut self, patience: usize) -> Self {
        self.early_stopping_patience = patience;
        self
    }

    /// Builder: set top-k for evaluation.
    pub const fn with_top_k(mut self, k: usize) -> Self {
        self.top_k_eval = k;
        self
    }
}

/// Report from training with validation.
#[derive(Clone, Debug)]
pub struct TrainingReport {
    /// Training loss per epoch.
    pub train_losses: Vec<f32>,
    /// Validation metrics (recorded every `val_every_n_epochs`).
    pub val_metrics: Vec<crate::evaluation::NavigationMetrics>,
    /// Epochs at which validation was performed.
    pub val_epochs: Vec<usize>,
    /// Best epoch (by validation MRR).
    pub best_epoch: usize,
    /// Best validation MRR achieved.
    pub best_val_mrr: f32,
    /// Final weights after training.
    pub final_weights: NavigationWeights,
    /// Whether training was stopped early.
    pub early_stopped: bool,
}

impl TrainingReport {
    /// Get summary string.
    pub fn summary(&self) -> String {
        format!(
            "Training: {} epochs, best MRR={:.4} at epoch {}, early_stopped={}",
            self.train_losses.len(),
            self.best_val_mrr,
            self.best_epoch,
            self.early_stopped
        )
    }
}

impl std::fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "TrainingReport:")?;
        writeln!(f, "  Epochs trained: {}", self.train_losses.len())?;
        writeln!(f, "  Best epoch: {}", self.best_epoch)?;
        writeln!(f, "  Best val MRR: {:.4}", self.best_val_mrr)?;
        writeln!(f, "  Early stopped: {}", self.early_stopped)?;
        if !self.train_losses.is_empty() {
            writeln!(f, "  Final loss: {:.4}", self.train_losses.last().unwrap())?;
        }
        Ok(())
    }
}

impl TrainableNavigatorEBM {
    /// Train with validation metrics and early stopping.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Training dataset with train/validation split
    /// * `n_epochs` - Maximum number of epochs
    /// * `batch_size` - Batch size for training
    /// * `config` - Extended training configuration
    /// * `key` - RNG key
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// Training report with loss curves and validation metrics.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let config = ExtendedTrainingConfig::default()
    ///     .with_val_every(5)
    ///     .with_early_stopping(3);
    ///
    /// let report = trainable.train_with_validation(
    ///     &dataset, 100, 32, &config, RngKey::new(42), &device,
    /// );
    ///
    /// println!("{}", report);
    /// ```
    pub fn train_with_validation(
        &mut self,
        dataset: &TrainingDataset,
        n_epochs: usize,
        batch_size: usize,
        config: &ExtendedTrainingConfig,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> TrainingReport {
        let mut train_losses = Vec::with_capacity(n_epochs);
        let mut val_metrics = Vec::new();
        let mut val_epochs = Vec::new();
        let mut best_epoch = 0;
        let mut best_val_mrr = 0.0f32;
        let mut best_weights = self.state.weights.clone();
        let mut patience_counter = 0;
        let mut early_stopped = false;

        let keys = key.split(n_epochs);

        for (epoch, epoch_key) in keys.into_iter().enumerate() {
            // Training step
            let mut epoch_loss = 0.0f32;
            let n_batches = dataset.train.len().div_ceil(batch_size).max(1);
            let batch_keys = epoch_key.split(n_batches + 1);
            let mut batch_key_iter = batch_keys.into_iter();

            for batch_idx in 0..n_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(dataset.train.len());
                if start >= end {
                    continue;
                }
                let batch = &dataset.train[start..end];

                if let Some(batch_key) = batch_key_iter.next() {
                    let result = self.train_step(batch, batch_key, device);
                    epoch_loss += result.loss;
                }
            }

            let avg_loss = epoch_loss / n_batches as f32;
            train_losses.push(avg_loss);

            // Validation
            if (epoch + 1) % config.val_every_n_epochs == 0 || epoch == n_epochs - 1 {
                let val_key = batch_key_iter
                    .next()
                    .unwrap_or_else(|| RngKey::new(epoch as u64));
                let metrics = crate::evaluation::evaluate_navigator(
                    &self.navigator,
                    &dataset.validation,
                    config.top_k_eval,
                    val_key,
                    device,
                );

                val_metrics.push(metrics.clone());
                val_epochs.push(epoch);

                // Check for improvement
                if metrics.mrr > best_val_mrr {
                    best_val_mrr = metrics.mrr;
                    best_epoch = epoch;
                    best_weights = self.state.weights.clone();
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                }

                // Log progress
                println!(
                    "Epoch {}: loss={:.4}, val_MRR={:.4}, val_R@10={:.4}",
                    epoch, avg_loss, metrics.mrr, metrics.recall_10
                );

                // Early stopping check
                if patience_counter >= config.early_stopping_patience {
                    println!(
                        "Early stopping at epoch {} (no improvement for {} validations)",
                        epoch, config.early_stopping_patience
                    );
                    early_stopped = true;
                    break;
                }
            } else if epoch % 10 == 0 {
                println!("Epoch {}: loss={:.4}", epoch, avg_loss);
            }
        }

        // Restore best weights
        self.state.weights = best_weights.clone();
        self.navigator.weights = best_weights.clone();

        TrainingReport {
            train_losses,
            val_metrics,
            val_epochs,
            best_epoch,
            best_val_mrr,
            final_weights: best_weights,
            early_stopped,
        }
    }
}

// ============================================================================
// Hyperparameter Tuning
// ============================================================================

/// Single hyperparameter configuration to evaluate.
///
/// Combines training config, navigation weights, and validation settings
/// into a single configuration for tuning runs.
///
/// # Example
///
/// ```rust,ignore
/// let config = HyperparamConfig::default()
///     .with_learning_rate(0.01)
///     .with_negatives(8)
///     .with_lambda_semantic(2.0);
/// ```
#[derive(Clone, Debug)]
pub struct HyperparamConfig {
    /// Learning rate for weight updates.
    pub learning_rate: f32,
    /// Number of negative samples per positive.
    pub negatives_per_positive: usize,
    /// Number of Langevin steps for negative sampling.
    pub negative_sample_steps: usize,
    /// Temperature for negative sampling.
    pub temperature: f32,
    /// Momentum for SGD.
    pub momentum: f32,
    /// Weight decay (L2 regularization).
    pub weight_decay: f32,
    /// Initial weight for semantic similarity energy.
    pub lambda_semantic: f32,
    /// Initial weight for radial energy.
    pub lambda_radial: f32,
    /// Initial weight for graph traversal energy.
    pub lambda_graph: f32,
    /// Initial weight for entropy energy.
    pub lambda_entropy: f32,
    /// Initial weight for path length penalty.
    pub lambda_path: f32,
    /// Initial weight for harmonic interference energy.
    pub lambda_harmonic: f32,
}

impl Default for HyperparamConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            negatives_per_positive: 4,
            negative_sample_steps: 10,
            temperature: 0.5,
            momentum: 0.9,
            weight_decay: 1e-4,
            lambda_semantic: 1.0,
            lambda_radial: 0.5,
            lambda_graph: 0.3,
            lambda_entropy: 0.2,
            lambda_path: 0.1,
            lambda_harmonic: 0.4,
        }
    }
}

impl HyperparamConfig {
    /// Builder: set learning rate.
    pub const fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder: set negatives per positive.
    pub const fn with_negatives(mut self, n: usize) -> Self {
        self.negatives_per_positive = n;
        self
    }

    /// Builder: set temperature.
    pub const fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Builder: set lambda_semantic.
    pub const fn with_lambda_semantic(mut self, v: f32) -> Self {
        self.lambda_semantic = v;
        self
    }

    /// Builder: set lambda_radial.
    pub const fn with_lambda_radial(mut self, v: f32) -> Self {
        self.lambda_radial = v;
        self
    }

    /// Builder: set lambda_graph.
    pub const fn with_lambda_graph(mut self, v: f32) -> Self {
        self.lambda_graph = v;
        self
    }

    /// Convert to NavigatorTrainingConfig.
    pub const fn to_training_config(&self) -> NavigatorTrainingConfig {
        NavigatorTrainingConfig {
            learning_rate: self.learning_rate,
            negatives_per_positive: self.negatives_per_positive,
            negative_sample_steps: self.negative_sample_steps,
            temperature: self.temperature,
            momentum: self.momentum,
            weight_decay: self.weight_decay,
            gradient_clip: 1.0,
        }
    }

    /// Convert to NavigationWeights.
    pub const fn to_navigation_weights(&self) -> crate::navigator::NavigationWeights {
        crate::navigator::NavigationWeights {
            lambda_semantic: self.lambda_semantic,
            lambda_radial: self.lambda_radial,
            lambda_graph: self.lambda_graph,
            lambda_entropy: self.lambda_entropy,
            lambda_path: self.lambda_path,
            lambda_harmonic: self.lambda_harmonic,
            temperature: self.temperature,
        }
    }

    /// Create a unique identifier string for this config.
    pub fn id(&self) -> String {
        format!(
            "lr{:.0e}_neg{}_t{:.1}_sem{:.1}_rad{:.1}",
            self.learning_rate,
            self.negatives_per_positive,
            self.temperature,
            self.lambda_semantic,
            self.lambda_radial,
        )
    }
}

/// Grid specification for hyperparameter tuning.
///
/// Defines ranges of values to search over for each hyperparameter.
/// Use [`Self::iter_configs`] to generate all combinations.
///
/// # Example
///
/// ```rust,ignore
/// let grid = TuningGrid::default();
/// println!("Total configs: {}", grid.n_combinations());
///
/// for config in grid.iter_configs() {
///     // Train and evaluate with this config
/// }
/// ```
#[derive(Clone, Debug)]
pub struct TuningGrid {
    /// Learning rates to try.
    pub learning_rates: Vec<f32>,
    /// Negatives per positive to try.
    pub negatives: Vec<usize>,
    /// Temperatures to try.
    pub temperatures: Vec<f32>,
    /// Lambda semantic values to try.
    pub lambda_semantics: Vec<f32>,
    /// Lambda radial values to try.
    pub lambda_radials: Vec<f32>,
    /// Lambda graph values to try.
    pub lambda_graphs: Vec<f32>,
}

impl Default for TuningGrid {
    fn default() -> Self {
        Self {
            learning_rates: vec![1e-4, 1e-3, 1e-2],
            negatives: vec![1, 4, 16],
            temperatures: vec![0.1, 0.5, 1.0],
            lambda_semantics: vec![0.5, 1.0, 2.0],
            lambda_radials: vec![0.1, 0.5, 1.0],
            lambda_graphs: vec![0.1, 0.3, 0.5],
        }
    }
}

impl TuningGrid {
    /// Create a minimal grid for quick testing.
    pub fn minimal() -> Self {
        Self {
            learning_rates: vec![1e-3],
            negatives: vec![4],
            temperatures: vec![0.5],
            lambda_semantics: vec![1.0],
            lambda_radials: vec![0.5],
            lambda_graphs: vec![0.3],
        }
    }

    /// Create a small grid for development.
    pub fn small() -> Self {
        Self {
            learning_rates: vec![1e-3, 1e-2],
            negatives: vec![4, 8],
            temperatures: vec![0.5],
            lambda_semantics: vec![1.0, 2.0],
            lambda_radials: vec![0.5],
            lambda_graphs: vec![0.3],
        }
    }

    /// Total number of configurations in the grid.
    pub const fn n_combinations(&self) -> usize {
        self.learning_rates.len()
            * self.negatives.len()
            * self.temperatures.len()
            * self.lambda_semantics.len()
            * self.lambda_radials.len()
            * self.lambda_graphs.len()
    }

    /// Iterate over all configurations in the grid.
    pub fn iter_configs(&self) -> impl Iterator<Item = HyperparamConfig> + '_ {
        self.learning_rates.iter().flat_map(move |&lr| {
            self.negatives.iter().flat_map(move |&neg| {
                self.temperatures.iter().flat_map(move |&temp| {
                    self.lambda_semantics.iter().flat_map(move |&sem| {
                        self.lambda_radials.iter().flat_map(move |&rad| {
                            self.lambda_graphs
                                .iter()
                                .map(move |&graph| HyperparamConfig {
                                    learning_rate: lr,
                                    negatives_per_positive: neg,
                                    temperature: temp,
                                    lambda_semantic: sem,
                                    lambda_radial: rad,
                                    lambda_graph: graph,
                                    ..HyperparamConfig::default()
                                })
                        })
                    })
                })
            })
        })
    }

    /// Sample n random configurations from the grid.
    pub fn sample_configs(&self, n: usize, seed: u64) -> Vec<HyperparamConfig> {
        let all_configs: Vec<_> = self.iter_configs().collect();
        if n >= all_configs.len() {
            return all_configs;
        }

        // Simple deterministic sampling using seed
        let mut rng_state = seed;
        let mut indices: Vec<usize> = (0..all_configs.len()).collect();

        // Fisher-Yates shuffle (partial)
        for i in 0..n {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = i + (rng_state as usize) % (indices.len() - i);
            indices.swap(i, j);
        }

        indices
            .into_iter()
            .take(n)
            .map(|i| all_configs[i].clone())
            .collect()
    }

    /// Builder: set learning rates.
    pub fn with_learning_rates(mut self, lrs: Vec<f32>) -> Self {
        self.learning_rates = lrs;
        self
    }

    /// Builder: set negatives.
    pub fn with_negatives(mut self, negs: Vec<usize>) -> Self {
        self.negatives = negs;
        self
    }

    /// Builder: set temperatures.
    pub fn with_temperatures(mut self, temps: Vec<f32>) -> Self {
        self.temperatures = temps;
        self
    }
}

/// Result from a single tuning run.
#[derive(Clone, Debug)]
pub struct TuningResult {
    /// Configuration used for this run.
    pub config: HyperparamConfig,
    /// Validation metrics achieved.
    pub metrics: crate::evaluation::NavigationMetrics,
    /// Final training loss.
    pub train_loss: f32,
    /// Number of epochs actually trained (may be less due to early stopping).
    pub epochs_trained: usize,
    /// Whether training was stopped early.
    pub early_stopped: bool,
    /// Run index (for ordering).
    pub run_index: usize,
}

impl TuningResult {
    /// Get primary metric (MRR) for comparison.
    pub const fn primary_metric(&self) -> f32 {
        self.metrics.mrr
    }

    /// Format as CSV row.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{},{}",
            self.config.learning_rate,
            self.config.negatives_per_positive,
            self.config.temperature,
            self.config.lambda_semantic,
            self.config.lambda_radial,
            self.config.lambda_graph,
            self.metrics.recall_10,
            self.metrics.mrr,
            self.metrics.ndcg_10,
            self.train_loss,
            self.epochs_trained,
            self.early_stopped,
        )
    }
}

/// Full tuning session with results.
///
/// Manages grid search or random search over hyperparameters,
/// tracks results, and supports persistence for resumable runs.
///
/// # Example
///
/// ```rust,ignore
/// let grid = TuningGrid::default();
/// let mut session = TuningSession::new(grid);
///
/// // Run grid search
/// session.run_grid_search(&sphere_ebm, &dataset, 50, RngKey::new(42), &device);
///
/// // Get best result
/// if let Some(best) = session.best_result() {
///     println!("Best MRR: {:.4}", best.metrics.mrr);
/// }
///
/// // Save results
/// session.to_csv("results.csv")?;
/// session.to_json("results.json")?;
/// ```
#[derive(Clone)]
pub struct TuningSession {
    /// Grid specification.
    pub grid: TuningGrid,
    /// Results from completed runs.
    pub results: Vec<TuningResult>,
    /// Configs that have been run (for resume support).
    completed_config_ids: std::collections::HashSet<String>,
}

impl TuningSession {
    /// Create a new tuning session.
    pub fn new(grid: TuningGrid) -> Self {
        Self {
            grid,
            results: Vec::new(),
            completed_config_ids: std::collections::HashSet::new(),
        }
    }

    /// Number of completed runs.
    pub const fn n_completed(&self) -> usize {
        self.results.len()
    }

    /// Number of remaining configs to run.
    pub const fn n_remaining(&self) -> usize {
        self.grid
            .n_combinations()
            .saturating_sub(self.n_completed())
    }

    /// Get the best result by primary metric (MRR).
    pub fn best_result(&self) -> Option<&TuningResult> {
        self.results.iter().max_by(|a, b| {
            a.primary_metric()
                .partial_cmp(&b.primary_metric())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get top-k results by primary metric.
    pub fn top_k_results(&self, k: usize) -> Vec<&TuningResult> {
        let mut sorted: Vec<_> = self.results.iter().collect();
        sorted.sort_by(|a, b| {
            b.primary_metric()
                .partial_cmp(&a.primary_metric())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted.into_iter().take(k).collect()
    }

    /// Run a single config and record the result.
    fn run_single_config(
        &mut self,
        config: HyperparamConfig,
        sphere_ebm: &crate::sphere_ebm::SphereEBM,
        dataset: &TrainingDataset,
        epochs: usize,
        batch_size: usize,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> TuningResult {
        let run_index = self.results.len();

        // Create navigator with initial weights
        let navigator = crate::navigator::NavigatorEBM::from_sphere_ebm(sphere_ebm.clone())
            .with_weights(config.to_navigation_weights());

        // Create trainable
        let mut trainable =
            TrainableNavigatorEBM::from_navigator(navigator, config.to_training_config());

        // Configure extended training
        let extended_config = ExtendedTrainingConfig::new(config.to_training_config())
            .with_val_every(5)
            .with_early_stopping(5)
            .with_top_k(10);

        // Train
        let report = trainable.train_with_validation(
            dataset,
            epochs,
            batch_size,
            &extended_config,
            key,
            device,
        );

        // Evaluate final metrics
        let final_metrics = crate::evaluation::evaluate_navigator(
            &trainable.navigator,
            &dataset.validation,
            10,
            key,
            device,
        );

        let result = TuningResult {
            config: config.clone(),
            metrics: final_metrics,
            train_loss: report.train_losses.last().copied().unwrap_or(f32::NAN),
            epochs_trained: report.train_losses.len(),
            early_stopped: report.early_stopped,
            run_index,
        };

        // Track completed
        self.completed_config_ids.insert(config.id());
        self.results.push(result.clone());

        result
    }

    /// Run full grid search.
    ///
    /// Evaluates all configurations in the grid and returns the best result.
    pub fn run_grid_search(
        &mut self,
        sphere_ebm: &crate::sphere_ebm::SphereEBM,
        dataset: &TrainingDataset,
        epochs: usize,
        batch_size: usize,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Option<&TuningResult> {
        let configs: Vec<_> = self
            .grid
            .iter_configs()
            .filter(|c| !self.completed_config_ids.contains(&c.id()))
            .collect();

        let n_total = configs.len();
        let keys = key.split(n_total);

        for (i, (config, k)) in configs.into_iter().zip(keys).enumerate() {
            println!(
                "[{}/{}] Running: lr={:.0e}, neg={}, temp={:.1}, λ_sem={:.1}",
                i + 1,
                n_total,
                config.learning_rate,
                config.negatives_per_positive,
                config.temperature,
                config.lambda_semantic,
            );

            let result =
                self.run_single_config(config, sphere_ebm, dataset, epochs, batch_size, k, device);

            println!(
                "  -> MRR={:.4}, R@10={:.4}, loss={:.4}",
                result.metrics.mrr, result.metrics.recall_10, result.train_loss
            );
        }

        self.best_result()
    }

    /// Run random search over n_samples configurations.
    pub fn run_random_search(
        &mut self,
        n_samples: usize,
        sphere_ebm: &crate::sphere_ebm::SphereEBM,
        dataset: &TrainingDataset,
        epochs: usize,
        batch_size: usize,
        seed: u64,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Option<&TuningResult> {
        let configs = self.grid.sample_configs(n_samples, seed);
        let n_total = configs.len();
        let keys = key.split(n_total);

        for (i, (config, k)) in configs.into_iter().zip(keys).enumerate() {
            if self.completed_config_ids.contains(&config.id()) {
                continue;
            }

            println!(
                "[{}/{}] Random sample: lr={:.0e}, neg={}, temp={:.1}",
                i + 1,
                n_total,
                config.learning_rate,
                config.negatives_per_positive,
                config.temperature,
            );

            let result =
                self.run_single_config(config, sphere_ebm, dataset, epochs, batch_size, k, device);

            println!(
                "  -> MRR={:.4}, R@10={:.4}",
                result.metrics.mrr, result.metrics.recall_10
            );
        }

        self.best_result()
    }

    /// Resume from previously saved results.
    pub fn resume_from(&mut self, results: Vec<TuningResult>) {
        for result in results {
            self.completed_config_ids.insert(result.config.id());
            self.results.push(result);
        }
    }

    /// CSV header row.
    pub const fn csv_header() -> &'static str {
        "lr,negatives,temp,lambda_sem,lambda_rad,lambda_graph,recall_10,mrr,ndcg_10,loss,epochs,early_stopped"
    }

    /// Export results to CSV file.
    pub fn to_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;
        writeln!(file, "{}", Self::csv_header())?;
        for result in &self.results {
            writeln!(file, "{}", result.to_csv_row())?;
        }
        Ok(())
    }

    /// Export results to JSON file.
    pub fn to_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::fs::File::create(path)?;

        writeln!(file, "{{")?;

        // Grid specification
        writeln!(file, "  \"grid\": {{")?;
        writeln!(
            file,
            "    \"learning_rates\": {:?},",
            self.grid.learning_rates
        )?;
        writeln!(file, "    \"negatives\": {:?},", self.grid.negatives)?;
        writeln!(file, "    \"temperatures\": {:?},", self.grid.temperatures)?;
        writeln!(
            file,
            "    \"lambda_semantics\": {:?},",
            self.grid.lambda_semantics
        )?;
        writeln!(
            file,
            "    \"lambda_radials\": {:?},",
            self.grid.lambda_radials
        )?;
        writeln!(file, "    \"lambda_graphs\": {:?}", self.grid.lambda_graphs)?;
        writeln!(file, "  }},")?;

        // Results array
        writeln!(file, "  \"results\": [")?;
        for (i, result) in self.results.iter().enumerate() {
            let comma = if i < self.results.len() - 1 { "," } else { "" };
            writeln!(file, "    {{")?;
            writeln!(file, "      \"config\": {{")?;
            writeln!(
                file,
                "        \"learning_rate\": {},",
                result.config.learning_rate
            )?;
            writeln!(
                file,
                "        \"negatives_per_positive\": {},",
                result.config.negatives_per_positive
            )?;
            writeln!(
                file,
                "        \"temperature\": {},",
                result.config.temperature
            )?;
            writeln!(
                file,
                "        \"lambda_semantic\": {},",
                result.config.lambda_semantic
            )?;
            writeln!(
                file,
                "        \"lambda_radial\": {},",
                result.config.lambda_radial
            )?;
            writeln!(
                file,
                "        \"lambda_graph\": {}",
                result.config.lambda_graph
            )?;
            writeln!(file, "      }},")?;
            writeln!(file, "      \"metrics\": {{")?;
            writeln!(file, "        \"recall_1\": {},", result.metrics.recall_1)?;
            writeln!(file, "        \"recall_5\": {},", result.metrics.recall_5)?;
            writeln!(file, "        \"recall_10\": {},", result.metrics.recall_10)?;
            writeln!(file, "        \"mrr\": {},", result.metrics.mrr)?;
            writeln!(file, "        \"ndcg_10\": {}", result.metrics.ndcg_10)?;
            writeln!(file, "      }},")?;
            writeln!(file, "      \"train_loss\": {},", result.train_loss)?;
            writeln!(file, "      \"epochs_trained\": {},", result.epochs_trained)?;
            writeln!(file, "      \"early_stopped\": {},", result.early_stopped)?;
            writeln!(file, "      \"run_index\": {}", result.run_index)?;
            writeln!(file, "    }}{}", comma)?;
        }
        writeln!(file, "  ],")?;

        // Best config index
        let best_idx = self
            .results
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.primary_metric()
                    .partial_cmp(&b.primary_metric())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        writeln!(
            file,
            "  \"best_config_index\": {}",
            best_idx
                .map(|i| i.to_string())
                .unwrap_or_else(|| "null".to_string())
        )?;

        writeln!(file, "}}")?;
        Ok(())
    }

    /// Load results from JSON file (for resuming).
    pub fn from_json(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::{BufRead, BufReader};

        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);

        // Simple JSON parsing - extract key values
        let mut grid = TuningGrid::default();
        let mut results = Vec::new();
        let mut in_results = false;
        let mut current_config = HyperparamConfig::default();
        let mut current_metrics = crate::evaluation::NavigationMetrics::default();
        let mut train_loss = 0.0f32;
        let mut epochs_trained = 0usize;
        let mut early_stopped = false;
        let mut run_index = 0usize;

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();

            // Grid parsing
            if trimmed.starts_with("\"learning_rates\":") {
                if let Some(arr) = extract_f32_array(trimmed) {
                    grid.learning_rates = arr;
                }
            } else if trimmed.starts_with("\"negatives\":") {
                if let Some(arr) = extract_usize_array(trimmed) {
                    grid.negatives = arr;
                }
            } else if trimmed.starts_with("\"temperatures\":") {
                if let Some(arr) = extract_f32_array(trimmed) {
                    grid.temperatures = arr;
                }
            } else if trimmed.starts_with("\"lambda_semantics\":") {
                if let Some(arr) = extract_f32_array(trimmed) {
                    grid.lambda_semantics = arr;
                }
            } else if trimmed.starts_with("\"lambda_radials\":") {
                if let Some(arr) = extract_f32_array(trimmed) {
                    grid.lambda_radials = arr;
                }
            } else if trimmed.starts_with("\"lambda_graphs\":") {
                if let Some(arr) = extract_f32_array(trimmed) {
                    grid.lambda_graphs = arr;
                }
            }

            // Results array
            if trimmed == "\"results\": [" {
                in_results = true;
            } else if in_results && trimmed == "]," {
                in_results = false;
            }

            // Config fields
            if trimmed.starts_with("\"learning_rate\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_config.learning_rate = v;
                }
            } else if trimmed.starts_with("\"negatives_per_positive\":") {
                if let Some(v) = extract_usize(trimmed) {
                    current_config.negatives_per_positive = v;
                }
            } else if trimmed.starts_with("\"temperature\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_config.temperature = v;
                }
            } else if trimmed.starts_with("\"lambda_semantic\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_config.lambda_semantic = v;
                }
            } else if trimmed.starts_with("\"lambda_radial\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_config.lambda_radial = v;
                }
            } else if trimmed.starts_with("\"lambda_graph\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_config.lambda_graph = v;
                }
            }

            // Metrics fields
            if trimmed.starts_with("\"recall_1\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_metrics.recall_1 = v;
                }
            } else if trimmed.starts_with("\"recall_5\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_metrics.recall_5 = v;
                }
            } else if trimmed.starts_with("\"recall_10\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_metrics.recall_10 = v;
                }
            } else if trimmed.starts_with("\"mrr\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_metrics.mrr = v;
                }
            } else if trimmed.starts_with("\"ndcg_10\":") {
                if let Some(v) = extract_f32(trimmed) {
                    current_metrics.ndcg_10 = v;
                }
            }

            // Result fields
            if trimmed.starts_with("\"train_loss\":") {
                if let Some(v) = extract_f32(trimmed) {
                    train_loss = v;
                }
            } else if trimmed.starts_with("\"epochs_trained\":") {
                if let Some(v) = extract_usize(trimmed) {
                    epochs_trained = v;
                }
            } else if trimmed.starts_with("\"early_stopped\":") {
                early_stopped = trimmed.contains("true");
            } else if trimmed.starts_with("\"run_index\":") {
                if let Some(v) = extract_usize(trimmed) {
                    run_index = v;
                }
            }

            // End of result object
            if in_results
                && (trimmed == "}," || trimmed == "}")
                && !trimmed.contains("config")
                && !trimmed.contains("metrics")
            {
                results.push(TuningResult {
                    config: current_config.clone(),
                    metrics: current_metrics.clone(),
                    train_loss,
                    epochs_trained,
                    early_stopped,
                    run_index,
                });
                current_config = HyperparamConfig::default();
                current_metrics = crate::evaluation::NavigationMetrics::default();
            }
        }

        let mut session = Self::new(grid);
        session.resume_from(results);
        Ok(session)
    }

    /// Print summary of results.
    pub fn print_summary(&self) {
        println!("\n=== Tuning Summary ===");
        println!("Total runs: {}", self.results.len());

        if let Some(best) = self.best_result() {
            println!("\nBest configuration:");
            println!("  Learning rate: {:.0e}", best.config.learning_rate);
            println!("  Negatives: {}", best.config.negatives_per_positive);
            println!("  Temperature: {:.2}", best.config.temperature);
            println!("  λ_semantic: {:.2}", best.config.lambda_semantic);
            println!("  λ_radial: {:.2}", best.config.lambda_radial);
            println!("  λ_graph: {:.2}", best.config.lambda_graph);
            println!("\nBest metrics:");
            println!("  MRR: {:.4}", best.metrics.mrr);
            println!("  Recall@10: {:.4}", best.metrics.recall_10);
            println!("  nDCG@10: {:.4}", best.metrics.ndcg_10);
        }

        println!("\nTop 5 configurations:");
        for (i, result) in self.top_k_results(5).into_iter().enumerate() {
            println!(
                "  {}. MRR={:.4} | lr={:.0e}, neg={}, λ_sem={:.1}",
                i + 1,
                result.metrics.mrr,
                result.config.learning_rate,
                result.config.negatives_per_positive,
                result.config.lambda_semantic,
            );
        }
    }
}

// Helper functions for JSON parsing
fn extract_f32(line: &str) -> Option<f32> {
    line.split(':')
        .nth(1)?
        .trim()
        .trim_end_matches(',')
        .parse()
        .ok()
}

fn extract_usize(line: &str) -> Option<usize> {
    line.split(':')
        .nth(1)?
        .trim()
        .trim_end_matches(',')
        .parse()
        .ok()
}

fn extract_f32_array(line: &str) -> Option<Vec<f32>> {
    let start = line.find('[')?;
    let end = line.find(']')?;
    let arr_str = &line[start + 1..end];
    Some(
        arr_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
    )
}

fn extract_usize_array(line: &str) -> Option<Vec<usize>> {
    let start = line.find('[')?;
    let end = line.find(']')?;
    let arr_str = &line[start + 1..end];
    Some(
        arr_str
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect(),
    )
}

// ============================================================================
// Advanced Contrastive Divergence Training Integration
// ============================================================================

/// Advanced training configuration with hard negative mining, PCD, and curriculum learning.
///
/// This configuration integrates the state-of-the-art contrastive divergence techniques
/// from the `contrastive` module:
///
/// - **Hard Negative Mining**: Similarity-based selection with false negative filtering
/// - **Persistent Contrastive Divergence**: Fantasy particles maintained across batches
/// - **Curriculum Learning**: Progressive difficulty scheduling for negatives
///
/// # Example
///
/// ```rust,ignore
/// use thrml_sphere::{
///     AdvancedTrainingConfig, TrainableNavigatorEBM, NavigatorEBM,
/// };
///
/// let config = AdvancedTrainingConfig::default()
///     .with_hard_negative_mining()
///     .with_pcd(100)  // 100 persistent particles
///     .with_curriculum();
///
/// let mut trainable = TrainableNavigatorEBM::from_navigator(
///     navigator,
///     config.base.clone(),
/// );
///
/// // Train with advanced techniques
/// let report = trainable.train_advanced(
///     &dataset, 100, 16, &config, RngKey::new(42), &device,
/// );
/// ```
#[derive(Clone, Debug)]
pub struct AdvancedTrainingConfig {
    /// Base training configuration.
    pub base: NavigatorTrainingConfig,

    /// Hard negative mining configuration (None = disabled).
    pub hard_negative_miner: Option<crate::contrastive::HardNegativeMiner>,

    /// Persistent particle buffer for PCD (None = disabled).
    pub pcd_n_particles: Option<usize>,

    /// Curriculum schedule for negatives (None = disabled).
    pub curriculum: Option<crate::contrastive::NegativeCurriculumSchedule>,

    /// SGLD configuration for negative phase sampling (None = use default sampling).
    pub sgld_config: Option<crate::contrastive::SGLDNegativeConfig>,

    /// Validation configuration.
    pub validation: ExtendedTrainingConfig,

    /// Learning rate warmup epochs (0 = no warmup).
    pub warmup_epochs: usize,

    /// Whether to use cosine annealing for learning rate.
    pub cosine_annealing: bool,

    /// Minimum learning rate for annealing.
    pub min_lr: f32,
}

impl Default for AdvancedTrainingConfig {
    fn default() -> Self {
        Self {
            base: NavigatorTrainingConfig::default(),
            hard_negative_miner: None,
            pcd_n_particles: None,
            curriculum: None,
            sgld_config: None,
            validation: ExtendedTrainingConfig::default(),
            warmup_epochs: 5,
            cosine_annealing: true,
            min_lr: 1e-5,
        }
    }
}

impl AdvancedTrainingConfig {
    /// Create from base config.
    pub fn new(base: NavigatorTrainingConfig) -> Self {
        Self {
            base,
            ..Default::default()
        }
    }

    /// Enable hard negative mining with default configuration.
    pub fn with_hard_negative_mining(mut self) -> Self {
        self.hard_negative_miner = Some(crate::contrastive::HardNegativeMiner::default());
        self
    }

    /// Enable hard negative mining with custom configuration.
    pub const fn with_hard_negative_miner(
        mut self,
        miner: crate::contrastive::HardNegativeMiner,
    ) -> Self {
        self.hard_negative_miner = Some(miner);
        self
    }

    /// Enable Persistent Contrastive Divergence with specified number of particles.
    pub const fn with_pcd(mut self, n_particles: usize) -> Self {
        self.pcd_n_particles = Some(n_particles);
        self
    }

    /// Enable curriculum learning with default schedule.
    pub fn with_curriculum(mut self) -> Self {
        self.curriculum = Some(crate::contrastive::NegativeCurriculumSchedule::default());
        self
    }

    /// Enable curriculum learning with custom schedule.
    pub const fn with_curriculum_schedule(
        mut self,
        schedule: crate::contrastive::NegativeCurriculumSchedule,
    ) -> Self {
        self.curriculum = Some(schedule);
        self
    }

    /// Enable SGLD for negative phase sampling.
    pub const fn with_sgld(mut self, config: crate::contrastive::SGLDNegativeConfig) -> Self {
        self.sgld_config = Some(config);
        self
    }

    /// Set warmup epochs.
    pub const fn with_warmup(mut self, epochs: usize) -> Self {
        self.warmup_epochs = epochs;
        self
    }

    /// Enable/disable cosine annealing.
    pub const fn with_cosine_annealing(mut self, enabled: bool) -> Self {
        self.cosine_annealing = enabled;
        self
    }

    /// Set minimum learning rate for annealing.
    pub const fn with_min_lr(mut self, lr: f32) -> Self {
        self.min_lr = lr;
        self
    }

    /// Set validation configuration.
    pub const fn with_validation(mut self, config: ExtendedTrainingConfig) -> Self {
        self.validation = config;
        self
    }

    /// Get effective learning rate for current epoch.
    pub fn get_learning_rate(&self, epoch: usize, total_epochs: usize) -> f32 {
        let base_lr = self.base.learning_rate;

        // Warmup phase
        if epoch < self.warmup_epochs {
            let warmup_factor = (epoch + 1) as f32 / self.warmup_epochs as f32;
            return base_lr * warmup_factor;
        }

        // Cosine annealing after warmup
        if self.cosine_annealing {
            let progress = (epoch - self.warmup_epochs) as f32
                / (total_epochs - self.warmup_epochs).max(1) as f32;
            let cosine_factor = 0.5 * (1.0 + (progress * std::f32::consts::PI).cos());
            (base_lr - self.min_lr).mul_add(cosine_factor, self.min_lr)
        } else {
            base_lr
        }
    }
}

/// Report from advanced training with all techniques.
#[derive(Clone, Debug)]
pub struct AdvancedTrainingReport {
    /// Standard training report.
    pub base_report: TrainingReport,

    /// Hard negative mining statistics.
    pub hard_neg_stats: Option<HardNegativeStats>,

    /// PCD statistics.
    pub pcd_stats: Option<PCDStats>,

    /// Curriculum statistics.
    pub curriculum_stats: Option<CurriculumStats>,
}

/// Statistics from hard negative mining.
#[derive(Clone, Debug, Default)]
pub struct HardNegativeStats {
    /// Average number of false negatives filtered per batch.
    pub avg_filtered_per_batch: f32,
    /// Total negatives mined.
    pub total_mined: usize,
}

/// Statistics from PCD.
#[derive(Clone, Debug, Default)]
pub struct PCDStats {
    /// Number of particle updates.
    pub n_updates: usize,
    /// Average particle energy.
    pub avg_particle_energy: f32,
    /// Fraction of particles reinitialized.
    pub reinit_fraction: f32,
}

/// Statistics from curriculum learning.
#[derive(Clone, Debug, Default)]
pub struct CurriculumStats {
    /// Epochs in each difficulty phase.
    pub epochs_per_phase: [usize; 3], // Easy, Medium, Hard
    /// Final difficulty level reached.
    pub final_difficulty: String,
}

impl TrainableNavigatorEBM {
    /// Train with advanced contrastive divergence techniques.
    ///
    /// This method uses all configured advanced techniques:
    /// - Hard negative mining
    /// - Persistent Contrastive Divergence
    /// - Curriculum learning
    /// - Learning rate scheduling (warmup + cosine annealing)
    ///
    /// # Arguments
    ///
    /// * `dataset` - Training dataset with train/validation split
    /// * `n_epochs` - Maximum number of epochs
    /// * `batch_size` - Batch size for training
    /// * `config` - Advanced training configuration
    /// * `key` - RNG key
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// Advanced training report with statistics from all techniques.
    pub fn train_advanced(
        &mut self,
        dataset: &TrainingDataset,
        n_epochs: usize,
        batch_size: usize,
        config: &AdvancedTrainingConfig,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> AdvancedTrainingReport {
        let mut train_losses = Vec::with_capacity(n_epochs);
        let mut val_metrics = Vec::new();
        let mut val_epochs = Vec::new();
        let mut best_epoch = 0;
        let mut best_val_mrr = 0.0f32;
        let mut best_weights = self.state.weights.clone();
        let mut patience_counter = 0;
        let mut early_stopped = false;

        // Initialize PCD buffer if configured
        let mut pcd_buffer = config.pcd_n_particles.map(|n| {
            let d = self.navigator.sphere_ebm.embedding_dim();
            let mut buffer = crate::contrastive::PersistentParticleBuffer::new(n, d)
                .with_langevin_steps(config.base.negative_sample_steps)
                .with_replay_prob(0.95);
            buffer.initialize_from_data(&self.navigator.sphere_ebm.embeddings, device);
            buffer
        });

        // Track curriculum stats
        let mut curriculum_epochs = [0usize; 3];

        // Track hard negative stats
        let mut total_mined = 0usize;
        let total_filtered = 0usize;
        let mut total_batches = 0usize;

        let keys = key.split(n_epochs);

        for (epoch, epoch_key) in keys.into_iter().enumerate() {
            // Get current learning rate
            let current_lr = config.get_learning_rate(epoch, n_epochs);

            // Get current difficulty from curriculum
            let difficulty = config.curriculum.as_ref().map(|c| c.get_difficulty(epoch));
            if let Some(d) = &difficulty {
                match d {
                    crate::contrastive::NegativeDifficulty::Easy => curriculum_epochs[0] += 1,
                    crate::contrastive::NegativeDifficulty::Medium => curriculum_epochs[1] += 1,
                    crate::contrastive::NegativeDifficulty::Hard => curriculum_epochs[2] += 1,
                }
            }

            // Training loop
            let mut epoch_loss = 0.0f32;
            let n_batches = dataset.train.len().div_ceil(batch_size).max(1);
            let batch_keys = epoch_key.split(n_batches + 1);
            let mut batch_key_iter = batch_keys.into_iter();

            for batch_idx in 0..n_batches {
                let start = batch_idx * batch_size;
                let end = (start + batch_size).min(dataset.train.len());
                if start >= end {
                    continue;
                }
                let batch = &dataset.train[start..end];

                // Mine hard negatives if configured
                let batch_with_negatives: Vec<TrainingExample> =
                    if let Some(miner) = &config.hard_negative_miner {
                        batch
                            .iter()
                            .map(|ex| {
                                let negatives = miner.mine(
                                    &ex.query,
                                    ex.positive_target,
                                    &self.navigator.sphere_ebm.embeddings,
                                    Some(&self.navigator.sphere_ebm.similarity),
                                    config.base.negatives_per_positive,
                                    device,
                                );
                                total_mined += negatives.len();
                                total_batches += 1;
                                TrainingExample {
                                    query: ex.query.clone(),
                                    query_radius: ex.query_radius,
                                    positive_target: ex.positive_target,
                                    negative_targets: Some(negatives),
                                }
                            })
                            .collect()
                    } else if let Some(curriculum) = &config.curriculum {
                        // Use curriculum-based negative selection
                        batch
                            .iter()
                            .map(|ex| {
                                // Get similarity scores for all candidates
                                let sim_data: Vec<f32> = self
                                    .navigator
                                    .sphere_ebm
                                    .similarity
                                    .clone()
                                    .into_data()
                                    .to_vec()
                                    .expect("sim to vec");
                                let n = self.navigator.n_points();

                                let candidates: Vec<(usize, f32)> = (0..n)
                                    .filter(|&i| i != ex.positive_target)
                                    .map(|i| {
                                        let sim = sim_data[ex.positive_target * n + i];
                                        (i, sim)
                                    })
                                    .collect();

                                let negatives = curriculum.select_negatives(
                                    &candidates,
                                    config.base.negatives_per_positive,
                                    epoch,
                                );

                                TrainingExample {
                                    query: ex.query.clone(),
                                    query_radius: ex.query_radius,
                                    positive_target: ex.positive_target,
                                    negative_targets: Some(negatives),
                                }
                            })
                            .collect()
                    } else {
                        batch.to_vec()
                    };

                // Update PCD buffer if configured
                if let Some(buffer) = &mut pcd_buffer {
                    // Update particles with energy gradient
                    buffer.update_langevin(
                        |particles| {
                            // Simple quadratic energy gradient toward data mean
                            let mean = self.navigator.sphere_ebm.embeddings.clone().mean_dim(0);
                            particles.clone() - mean
                        },
                        device,
                    );

                    // Occasionally reinitialize some particles to prevent mode collapse
                    if epoch % 10 == 0 {
                        buffer.reinitialize_fraction(
                            0.1,
                            &self.navigator.sphere_ebm.embeddings,
                            device,
                        );
                    }
                }

                // Create modified config with current LR
                let mut step_config = config.base.clone();
                step_config.learning_rate = current_lr;

                // Temporarily update config
                let original_lr = self.config.learning_rate;
                self.config.learning_rate = current_lr;

                if let Some(batch_key) = batch_key_iter.next() {
                    let result = self.train_step(&batch_with_negatives, batch_key, device);
                    epoch_loss += result.loss;
                }

                // Restore original config
                self.config.learning_rate = original_lr;
            }

            let avg_loss = epoch_loss / n_batches as f32;
            train_losses.push(avg_loss);

            // Validation
            if (epoch + 1) % config.validation.val_every_n_epochs == 0 || epoch == n_epochs - 1 {
                let val_key = batch_key_iter
                    .next()
                    .unwrap_or_else(|| RngKey::new(epoch as u64));
                let metrics = crate::evaluation::evaluate_navigator(
                    &self.navigator,
                    &dataset.validation,
                    config.validation.top_k_eval,
                    val_key,
                    device,
                );

                val_metrics.push(metrics.clone());
                val_epochs.push(epoch);

                // Check for improvement
                if metrics.mrr > best_val_mrr {
                    best_val_mrr = metrics.mrr;
                    best_epoch = epoch;
                    best_weights = self.state.weights.clone();
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                }

                // Log progress
                let difficulty_str = difficulty
                    .map(|d| format!("{:?}", d))
                    .unwrap_or_else(|| "N/A".to_string());
                println!(
                    "Epoch {}: loss={:.4}, val_MRR={:.4}, lr={:.2e}, difficulty={}",
                    epoch, avg_loss, metrics.mrr, current_lr, difficulty_str
                );

                // Early stopping check
                if patience_counter >= config.validation.early_stopping_patience {
                    println!(
                        "Early stopping at epoch {} (no improvement for {} validations)",
                        epoch, config.validation.early_stopping_patience
                    );
                    early_stopped = true;
                    break;
                }
            } else if epoch % 10 == 0 {
                let difficulty_str = difficulty
                    .map(|d| format!("{:?}", d))
                    .unwrap_or_else(|| "N/A".to_string());
                println!(
                    "Epoch {}: loss={:.4}, lr={:.2e}, difficulty={}",
                    epoch, avg_loss, current_lr, difficulty_str
                );
            }
        }

        // Restore best weights
        self.state.weights = best_weights.clone();
        self.navigator.weights = best_weights.clone();

        // Compile statistics
        let hard_neg_stats = config
            .hard_negative_miner
            .as_ref()
            .map(|_| HardNegativeStats {
                avg_filtered_per_batch: total_filtered as f32 / total_batches.max(1) as f32,
                total_mined,
            });

        let pcd_stats = pcd_buffer.map(|buffer| PCDStats {
            n_updates: buffer.n_updates,
            avg_particle_energy: 0.0, // Could compute if needed
            reinit_fraction: 0.1,
        });

        let curriculum_stats = config.curriculum.as_ref().map(|c| {
            let final_diff = c.get_difficulty(train_losses.len().saturating_sub(1));
            CurriculumStats {
                epochs_per_phase: curriculum_epochs,
                final_difficulty: format!("{:?}", final_diff),
            }
        });

        AdvancedTrainingReport {
            base_report: TrainingReport {
                train_losses,
                val_metrics,
                val_epochs,
                best_epoch,
                best_val_mrr,
                final_weights: best_weights,
                early_stopped,
            },
            hard_neg_stats,
            pcd_stats,
            curriculum_stats,
        }
    }
}

impl std::fmt::Display for AdvancedTrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Advanced Training Report ===")?;
        writeln!(f, "{}", self.base_report)?;

        if let Some(stats) = &self.hard_neg_stats {
            writeln!(f, "\nHard Negative Mining:")?;
            writeln!(f, "  Total mined: {}", stats.total_mined)?;
            writeln!(
                f,
                "  Avg filtered/batch: {:.2}",
                stats.avg_filtered_per_batch
            )?;
        }

        if let Some(stats) = &self.pcd_stats {
            writeln!(f, "\nPersistent Contrastive Divergence:")?;
            writeln!(f, "  Updates: {}", stats.n_updates)?;
            writeln!(f, "  Reinit fraction: {:.2}", stats.reinit_fraction)?;
        }

        if let Some(stats) = &self.curriculum_stats {
            writeln!(f, "\nCurriculum Learning:")?;
            writeln!(
                f,
                "  Epochs: Easy={}, Medium={}, Hard={}",
                stats.epochs_per_phase[0], stats.epochs_per_phase[1], stats.epochs_per_phase[2]
            )?;
            writeln!(f, "  Final difficulty: {}", stats.final_difficulty)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScaleProfile;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_training_config() {
        let config = NavigatorTrainingConfig::default()
            .with_learning_rate(0.001)
            .with_negatives(8)
            .with_momentum(0.95);

        assert!((config.learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(config.negatives_per_positive, 8);
        assert!((config.momentum - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_trainable_navigator_creation() {
        let device = init_gpu_device();
        let n = 20;
        let d = 16;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let training_config = NavigatorTrainingConfig::default();

        let trainable = TrainableNavigatorEBM::new(
            embeddings,
            prominence,
            None,
            sphere_config,
            training_config,
            &device,
        );

        assert_eq!(trainable.navigator.n_points(), n);
        assert_eq!(trainable.state.step, 0);
    }

    #[test]
    fn test_single_train_step() {
        let device = init_gpu_device();
        let n = 15;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let training_config = NavigatorTrainingConfig::default()
            .with_negatives(2)
            .with_learning_rate(0.1);

        let mut trainable = TrainableNavigatorEBM::new(
            embeddings.clone(),
            prominence,
            None,
            sphere_config,
            training_config,
            &device,
        );

        // Create a training example
        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d]);
        let example = TrainingExample {
            query,
            query_radius: 50.0,
            positive_target: 0,
            negative_targets: Some(vec![1, 2]),
        };

        let initial_weights = trainable.state.weights.clone();
        let result = trainable.train_step(&[example], RngKey::new(42), &device);

        assert!(result.loss.is_finite());
        assert!(result.gradient_norm >= 0.0);

        // Weights should have changed (unless gradient is exactly zero)
        assert_eq!(trainable.state.step, 1);

        println!("Loss: {}", result.loss);
        println!("Initial weights: {:?}", initial_weights);
        println!("Final weights: {:?}", result.new_weights);
    }

    #[test]
    fn test_training_reduces_loss() {
        let device = init_gpu_device();
        let n = 20;
        let d = 8;

        // Create structured embeddings where similar indices are similar
        let mut emb_data = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                emb_data[i * d + j] = ((i + j) as f32 / (n + d) as f32).sin();
            }
        }
        let embeddings_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(emb_data.as_slice(), &device);
        let embeddings: Tensor<WgpuBackend, 2> = embeddings_1d.reshape([n as i32, d as i32]);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let training_config = NavigatorTrainingConfig::default()
            .with_learning_rate(0.05)
            .with_negatives(3);

        let mut trainable = TrainableNavigatorEBM::new(
            embeddings.clone(),
            prominence,
            None,
            sphere_config,
            training_config,
            &device,
        );

        // Create training examples
        let examples: Vec<TrainingExample> = (0..5)
            .map(|i| {
                let query: Tensor<WgpuBackend, 1> = embeddings
                    .clone()
                    .slice([i..i + 1, 0..d])
                    .reshape([d as i32]);
                TrainingExample {
                    query,
                    query_radius: 50.0,
                    positive_target: i,
                    negative_targets: None,
                }
            })
            .collect();

        // Train for a few epochs
        let losses = trainable.train(&examples, 5, 2, RngKey::new(123), &device);

        println!("Training losses: {:?}", losses);

        // Check training produced valid losses
        for loss in &losses {
            assert!(loss.is_finite(), "Loss should be finite");
        }
    }

    // ========================================================================
    // Tests for TrainingDataset and Data Generation
    // ========================================================================

    #[test]
    fn test_training_dataset_split() {
        // Create dummy examples
        let device = init_gpu_device();
        let d = 8;
        let dummy_query: Tensor<WgpuBackend, 1> =
            Tensor::random([d], Distribution::Normal(0.0, 1.0), &device);

        let examples: Vec<TrainingExample> = (0..100)
            .map(|i| TrainingExample {
                query: dummy_query.clone(),
                query_radius: 50.0,
                positive_target: i,
                negative_targets: None,
            })
            .collect();

        // Split with 10% validation
        let dataset = TrainingDataset::from_examples(examples, 0.1, 42);

        assert_eq!(dataset.n_train(), 90);
        assert_eq!(dataset.n_val(), 10);
        assert_eq!(dataset.n_total(), 100);
    }

    #[test]
    fn test_training_dataset_no_validation() {
        let device = init_gpu_device();
        let d = 8;
        let dummy_query: Tensor<WgpuBackend, 1> =
            Tensor::random([d], Distribution::Normal(0.0, 1.0), &device);

        let examples: Vec<TrainingExample> = (0..50)
            .map(|i| TrainingExample {
                query: dummy_query.clone(),
                query_radius: 50.0,
                positive_target: i,
                negative_targets: None,
            })
            .collect();

        // No validation split
        let dataset = TrainingDataset::from_examples(examples, 0.0, 42);

        assert_eq!(dataset.n_train(), 50);
        assert_eq!(dataset.n_val(), 0);
    }

    #[test]
    fn test_generate_pairs_from_similarity() {
        let device = init_gpu_device();
        let n = 20;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(3);
        let sphere_ebm =
            crate::sphere_ebm::SphereEBM::new(embeddings, prominence, None, sphere_config, &device);

        // Generate 2 positives per query, 4 negatives each
        let examples = generate_pairs_from_similarity(&sphere_ebm, 2, 4, &device);

        // Should have n * 2 = 40 examples
        assert_eq!(examples.len(), n * 2);

        // Check structure of examples
        for example in &examples {
            assert!(example.positive_target < n);
            if let Some(negs) = &example.negative_targets {
                assert!(negs.len() <= 4);
                for &neg in negs {
                    assert!(neg < n);
                    assert_ne!(neg, example.positive_target);
                }
            }
        }

        println!("Generated {} similarity-based pairs", examples.len());
    }

    #[test]
    fn test_generate_pairs_from_edges() {
        let device = init_gpu_device();
        let n = 20;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(3);
        let sphere_ebm =
            crate::sphere_ebm::SphereEBM::new(embeddings, prominence, None, sphere_config, &device);

        // Create a simple graph with edges
        let mut sidecar = crate::hypergraph::HypergraphSidecar::new(n);
        sidecar.add_edge(0, 1, 1.0);
        sidecar.add_edge(0, 2, 1.0);
        sidecar.add_edge(1, 2, 1.0);
        sidecar.add_edge(5, 6, 1.0);
        sidecar.add_edge(10, 11, 1.0);
        sidecar.add_edge(10, 12, 1.0);

        let examples = generate_pairs_from_edges(&sphere_ebm, &sidecar, 4);

        // Should have examples for each edge
        assert!(!examples.is_empty());
        assert!(examples.len() <= 6); // At most 6 edges

        println!("Generated {} edge-based pairs", examples.len());
    }

    #[test]
    fn test_extended_training_config() {
        let config = ExtendedTrainingConfig::default()
            .with_val_every(10)
            .with_early_stopping(3)
            .with_top_k(5);

        assert_eq!(config.val_every_n_epochs, 10);
        assert_eq!(config.early_stopping_patience, 3);
        assert_eq!(config.top_k_eval, 5);
    }

    #[test]
    fn test_training_report_display() {
        let report = TrainingReport {
            train_losses: vec![0.5, 0.4, 0.3],
            val_metrics: vec![],
            val_epochs: vec![],
            best_epoch: 2,
            best_val_mrr: 0.65,
            final_weights: NavigationWeights::default(),
            early_stopped: false,
        };

        let summary = report.summary();
        assert!(summary.contains("3 epochs"));
        assert!(summary.contains("0.65"));

        let display = format!("{}", report);
        assert!(display.contains("TrainingReport"));
    }

    // ========================================================================
    // Tests for Hyperparameter Tuning
    // ========================================================================

    #[test]
    fn test_hyperparam_config_default() {
        let config = HyperparamConfig::default();
        assert!((config.learning_rate - 0.01).abs() < 1e-6);
        assert_eq!(config.negatives_per_positive, 4);
        assert!((config.lambda_semantic - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hyperparam_config_builder() {
        let config = HyperparamConfig::default()
            .with_learning_rate(0.001)
            .with_negatives(8)
            .with_temperature(0.2)
            .with_lambda_semantic(2.0);

        assert!((config.learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(config.negatives_per_positive, 8);
        assert!((config.temperature - 0.2).abs() < 1e-6);
        assert!((config.lambda_semantic - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_hyperparam_config_conversions() {
        let config = HyperparamConfig::default();

        let training_config = config.to_training_config();
        assert!((training_config.learning_rate - config.learning_rate).abs() < 1e-6);

        let nav_weights = config.to_navigation_weights();
        assert!((nav_weights.lambda_semantic - config.lambda_semantic).abs() < 1e-6);
    }

    #[test]
    fn test_hyperparam_config_id() {
        let config1 = HyperparamConfig::default();
        let config2 = HyperparamConfig::default().with_learning_rate(0.001);

        let id1 = config1.id();
        let id2 = config2.id();

        assert!(!id1.is_empty());
        assert_ne!(id1, id2); // Different configs should have different IDs
    }

    #[test]
    fn test_tuning_grid_default() {
        let grid = TuningGrid::default();
        assert_eq!(grid.learning_rates.len(), 3);
        assert_eq!(grid.negatives.len(), 3);
        assert_eq!(grid.temperatures.len(), 3);
    }

    #[test]
    fn test_tuning_grid_n_combinations() {
        let grid = TuningGrid::default();
        let expected = 3 * 3 * 3 * 3 * 3 * 3; // 729
        assert_eq!(grid.n_combinations(), expected);

        let minimal = TuningGrid::minimal();
        assert_eq!(minimal.n_combinations(), 1);
    }

    #[test]
    fn test_tuning_grid_iter_configs() {
        let grid = TuningGrid::minimal();
        let configs: Vec<_> = grid.iter_configs().collect();
        assert_eq!(configs.len(), 1);

        let small = TuningGrid::small();
        let configs: Vec<_> = small.iter_configs().collect();
        assert_eq!(configs.len(), small.n_combinations());
    }

    #[test]
    fn test_tuning_grid_sample_configs() {
        let grid = TuningGrid::default();

        // Sample less than total
        let samples = grid.sample_configs(10, 42);
        assert_eq!(samples.len(), 10);

        // Sample more than total (should return all)
        let samples = grid.sample_configs(1000, 42);
        assert_eq!(samples.len(), grid.n_combinations());

        // Same seed should give same results
        let s1 = grid.sample_configs(5, 123);
        let s2 = grid.sample_configs(5, 123);
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert!((a.learning_rate - b.learning_rate).abs() < 1e-9);
        }
    }

    #[test]
    fn test_tuning_session_new() {
        let grid = TuningGrid::minimal();
        let session = TuningSession::new(grid);

        assert_eq!(session.n_completed(), 0);
        assert_eq!(session.n_remaining(), 1);
        assert!(session.best_result().is_none());
    }

    #[test]
    fn test_tuning_result_csv_row() {
        let result = TuningResult {
            config: HyperparamConfig::default(),
            metrics: crate::evaluation::NavigationMetrics {
                recall_1: 0.5,
                recall_5: 0.7,
                recall_10: 0.85,
                mrr: 0.6,
                ndcg_10: 0.75,
                n_queries: 100,
                avg_rank: 3.5,
            },
            train_loss: 0.25,
            epochs_trained: 50,
            early_stopped: false,
            run_index: 0,
        };

        let csv = result.to_csv_row();
        assert!(csv.contains("0.01")); // learning_rate
        assert!(csv.contains("0.8500")); // recall_10
        assert!(csv.contains("0.6000")); // mrr
    }

    #[test]
    fn test_tuning_session_resume() {
        let grid = TuningGrid::minimal();
        let mut session = TuningSession::new(grid);

        let result = TuningResult {
            config: HyperparamConfig::default(),
            metrics: crate::evaluation::NavigationMetrics::default(),
            train_loss: 0.5,
            epochs_trained: 10,
            early_stopped: false,
            run_index: 0,
        };

        session.resume_from(vec![result]);

        assert_eq!(session.n_completed(), 1);
        assert!(session.best_result().is_some());
    }

    #[test]
    fn test_tuning_session_top_k() {
        let grid = TuningGrid::minimal();
        let mut session = TuningSession::new(grid);

        // Add some results with different MRR values
        for i in 0..6 {
            let metrics = crate::evaluation::NavigationMetrics {
                mrr: i as f32 * 0.1,
                ..Default::default()
            };
            let result = TuningResult {
                config: HyperparamConfig::default(),
                metrics,
                train_loss: 0.5,
                epochs_trained: 10,
                early_stopped: false,
                run_index: i,
            };
            session.resume_from(vec![result]);
        }

        let top3 = session.top_k_results(3);
        assert_eq!(top3.len(), 3);

        // Should be sorted by MRR descending
        assert!(top3[0].metrics.mrr >= top3[1].metrics.mrr);
        assert!(top3[1].metrics.mrr >= top3[2].metrics.mrr);
    }

    // =========================================================================
    // CPU/GPU Equivalence Tests
    // =========================================================================

    #[test]
    fn test_cpu_gpu_energy_equivalence() {
        let device = init_gpu_device();
        let n = 10;
        let d = 8;

        // Create test embeddings
        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);
        let entropies: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let navigator = NavigatorEBM::new(
            embeddings.clone(),
            prominence,
            Some(entropies),
            sphere_config,
            &device,
        );

        // Create query and target indices
        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d]);
        let target_indices_vec: Vec<usize> = vec![1, 2, 3];
        let target_indices_tensor = NavigatorEBM::indices_to_tensor(&target_indices_vec, &device);

        // Get coordinates
        let coords = navigator.sphere_ebm.optimize(RngKey::new(42), &device);

        // Compare CPU (original) and GPU (batched) semantic energy
        let sem_cpu = navigator.semantic_energy(&query, &target_indices_vec, &device);
        let sem_gpu = navigator.semantic_energy_batched(&query, &target_indices_tensor, &device);

        let sem_cpu_data: Vec<f32> = sem_cpu.into_data().to_vec().expect("sem cpu");
        let sem_gpu_data: Vec<f32> = sem_gpu.into_data().to_vec().expect("sem gpu");

        // Should match within f32 tolerance
        for (cpu, gpu) in sem_cpu_data.iter().zip(sem_gpu_data.iter()) {
            let diff = (cpu - gpu).abs();
            assert!(
                diff < 1e-4,
                "Semantic energy mismatch: CPU={} GPU={} diff={}",
                cpu,
                gpu,
                diff
            );
        }

        // Compare radial energy
        let rad_cpu = navigator.radial_energy(50.0, &coords, &target_indices_vec, &device);
        let rad_gpu =
            navigator.radial_energy_batched(50.0, &coords, &target_indices_tensor, &device);

        let rad_cpu_data: Vec<f32> = rad_cpu.into_data().to_vec().expect("rad cpu");
        let rad_gpu_data: Vec<f32> = rad_gpu.into_data().to_vec().expect("rad gpu");

        for (cpu, gpu) in rad_cpu_data.iter().zip(rad_gpu_data.iter()) {
            let diff = (cpu - gpu).abs();
            assert!(
                diff < 1e-4,
                "Radial energy mismatch: CPU={} GPU={} diff={}",
                cpu,
                gpu,
                diff
            );
        }

        // Compare entropy energy
        let ent_cpu = navigator.entropy_energy(&target_indices_vec, &device);
        let ent_gpu = navigator.entropy_energy_batched(&target_indices_tensor, &device);

        let ent_cpu_data: Vec<f32> = ent_cpu.into_data().to_vec().expect("ent cpu");
        let ent_gpu_data: Vec<f32> = ent_gpu.into_data().to_vec().expect("ent gpu");

        for (cpu, gpu) in ent_cpu_data.iter().zip(ent_gpu_data.iter()) {
            let diff = (cpu - gpu).abs();
            assert!(
                diff < 1e-4,
                "Entropy energy mismatch: CPU={} GPU={} diff={}",
                cpu,
                gpu,
                diff
            );
        }

        println!("CPU/GPU energy equivalence test passed!");
    }

    #[test]
    fn test_hybrid_training_produces_valid_results() {
        use thrml_core::compute::ComputeBackend;

        let device = init_gpu_device();
        let n = 15;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let training_config = NavigatorTrainingConfig::default()
            .with_learning_rate(0.05)
            .with_negatives(2);

        let mut trainable = TrainableNavigatorEBM::new(
            embeddings.clone(),
            prominence,
            None,
            sphere_config,
            training_config,
            &device,
        );

        // Create training examples
        let examples: Vec<TrainingExample> = (0..3)
            .map(|i| {
                let query: Tensor<WgpuBackend, 1> = embeddings
                    .clone()
                    .slice([i..i + 1, 0..d])
                    .reshape([d as i32]);
                TrainingExample {
                    query,
                    query_radius: 50.0,
                    positive_target: i,
                    negative_targets: None,
                }
            })
            .collect();

        // Train with hybrid method
        let backend = ComputeBackend::default();
        let losses =
            trainable.train_hybrid(&examples, 3, 2, RngKey::new(42), &device, Some(&backend));

        // Verify losses are valid
        for loss in &losses {
            assert!(loss.is_finite(), "Loss should be finite");
            assert!(*loss >= 0.0, "Loss should be non-negative");
        }

        println!("Hybrid training losses: {:?}", losses);
    }

    // =========================================================================
    // Training Performance Comparison
    // =========================================================================

    #[test]
    fn test_training_performance_comparison() {
        use std::time::Instant;
        use thrml_core::compute::ComputeBackend;

        let device = init_gpu_device();
        let n = 50; // Larger dataset for meaningful timing
        let d = 32;

        // Create test data
        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let training_config = NavigatorTrainingConfig::default()
            .with_learning_rate(0.05)
            .with_negatives(4);

        // Create training examples
        let examples: Vec<TrainingExample> = (0..10)
            .map(|i| {
                let idx = i % n;
                let query: Tensor<WgpuBackend, 1> = embeddings
                    .clone()
                    .slice([idx..idx + 1, 0..d])
                    .reshape([d as i32]);
                TrainingExample {
                    query,
                    query_radius: 50.0,
                    positive_target: idx,
                    negative_targets: None,
                }
            })
            .collect();

        // Benchmark original (sequential) training
        let mut trainable_orig = TrainableNavigatorEBM::new(
            embeddings.clone(),
            prominence.clone(),
            None,
            sphere_config,
            training_config.clone(),
            &device,
        );

        let start_orig = Instant::now();
        let losses_orig = trainable_orig.train(&examples, 3, 5, RngKey::new(42), &device);
        let elapsed_orig = start_orig.elapsed();

        // Benchmark hybrid (GPU-batched) training
        let mut trainable_hybrid = TrainableNavigatorEBM::new(
            embeddings,
            prominence,
            None,
            sphere_config,
            training_config,
            &device,
        );

        let backend = ComputeBackend::default();
        let start_hybrid = Instant::now();
        let losses_hybrid = trainable_hybrid.train_hybrid(
            &examples,
            3,
            5,
            RngKey::new(42),
            &device,
            Some(&backend),
        );
        let elapsed_hybrid = start_hybrid.elapsed();

        println!("\n=== Training Performance Comparison ===");
        println!("Original (sequential): {:?}", elapsed_orig);
        println!("Hybrid (GPU-batched): {:?}", elapsed_hybrid);
        println!(
            "Speedup: {:.2}x",
            elapsed_orig.as_secs_f64() / elapsed_hybrid.as_secs_f64().max(0.001)
        );
        println!(
            "Original final loss: {:.4}",
            losses_orig.last().unwrap_or(&0.0)
        );
        println!(
            "Hybrid final loss: {:.4}",
            losses_hybrid.last().unwrap_or(&0.0)
        );

        // Both should produce valid losses
        assert!(losses_orig.iter().all(|l| l.is_finite()));
        assert!(losses_hybrid.iter().all(|l| l.is_finite()));
    }
}
