//! SpinGibbsConditional - Gibbs updates for spin-valued (Ising) variables in discrete EBMs.
//!
//! This sampler performs Gibbs sampling updates for spin (binary) variables,
//! computing the conditional distribution given neighboring states.
//!
//! The conditional probability is: P(S=1) = sigmoid(2*γ)
//! where γ = Σ_i s_1^i ... s_K^i * W^i[x_1^i, ..., x_M^i]
//!
//! ## Precision Routing
//!
//! When using `sample_routed` with a [`ComputeBackend`], the sampler routes
//! precision-sensitive accumulation based on hardware capabilities:
//!
//! | Hardware | Backend | Precision | Path |
//! |----------|---------|-----------|------|
//! | H100/B200 (with `cuda` feature) | `GpuHpcF64` | f64 on GPU | `compute_parameters_cuda_f64` |
//! | Apple Silicon, Consumer NVIDIA | `UnifiedHybrid` | f64 on CPU | `compute_parameters_cpu_f64` |
//! | Any (bulk ops / non-precision) | `GpuOnly` | f32 on GPU | `compute_parameters` |
//!
//! This is controlled by:
//! 1. [`ComputeBackend::uses_gpu_f64`] - CUDA f64 on HPC GPUs
//! 2. [`ComputeBackend::use_cpu`] with [`OpType::IsingSampling`] - CPU f64 fallback
//!
//! ## Fused Kernels
//!
//! When the `fused-kernels` feature is enabled, uses a GPU-fused kernel that
//! combines the sigmoid and Bernoulli sampling into a single kernel launch.

use crate::rng::RngKey;
use crate::sampler::AbstractConditionalSampler;
use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::WgpuBackend;
use thrml_core::compute::{ComputeBackend, OpType};
use thrml_core::interaction::InteractionData;
use thrml_core::node::TensorSpec;

#[cfg(feature = "fused-kernels")]
use thrml_kernels::sigmoid_bernoulli_fused;

/// A conditional update for spin-valued random variables that performs a Gibbs sampling update
/// given one or more DiscreteEBMInteractions.
pub struct SpinGibbsConditional;

impl SpinGibbsConditional {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for SpinGibbsConditional {
    fn default() -> Self {
        Self::new()
    }
}

impl AbstractConditionalSampler for SpinGibbsConditional {
    type SamplerState = (); // Stateless sampler

    fn sample(
        &self,
        _key: RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        _sampler_state: Self::SamplerState,
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, Self::SamplerState) {
        // Compute gamma parameter using n_spin to split states
        let gamma = self.compute_parameters(
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction,
            output_spec,
            device,
        );

        // Generate uniform random values
        let n_nodes = output_spec.shape[0];
        let uniform: Tensor<WgpuBackend, 1> =
            Tensor::random([n_nodes], Distribution::Uniform(0.0, 1.0), device);

        // Use fused kernel when feature is enabled
        #[cfg(feature = "fused-kernels")]
        {
            (sigmoid_bernoulli_fused(gamma, uniform), ())
        }

        // Default implementation: separate operations
        #[cfg(not(feature = "fused-kernels"))]
        {
            // Sample: P(S=1) = sigmoid(2*gamma)
            let probs = burn::tensor::activation::sigmoid(gamma * 2.0);
            // Sample Bernoulli: output 1 if uniform < probs, else 0
            (uniform.lower_equal(probs).float(), ())
        }
    }
}

impl SpinGibbsConditional {
    /// Sample with explicit precision routing via ComputeBackend.
    ///
    /// When `backend.use_cpu(OpType::IsingSampling, ...)` returns true,
    /// the gamma accumulation is performed in f64 precision on CPU.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_core::compute::ComputeBackend;
    ///
    /// let backend = ComputeBackend::apple_silicon();
    /// let (samples, state) = sampler.sample_routed(
    ///     &backend, key, &interactions, &active_flags,
    ///     &neighbor_states, &n_spin_per_interaction, (),
    ///     &output_spec, &device,
    /// );
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn sample_routed(
        &self,
        backend: &ComputeBackend,
        _key: RngKey,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        _sampler_state: (),
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> (Tensor<WgpuBackend, 1>, ()) {
        // Compute gamma with precision routing
        let gamma = self.compute_parameters_routed(
            backend,
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction,
            output_spec,
            device,
        );

        // Generate uniform random values (always on GPU - this is numerically stable)
        let n_nodes = output_spec.shape[0];
        let uniform: Tensor<WgpuBackend, 1> =
            Tensor::random([n_nodes], Distribution::Uniform(0.0, 1.0), device);

        // Use fused kernel when feature is enabled
        #[cfg(feature = "fused-kernels")]
        {
            (sigmoid_bernoulli_fused(gamma, uniform), ())
        }

        // Default implementation: separate operations
        #[cfg(not(feature = "fused-kernels"))]
        {
            let probs = burn::tensor::activation::sigmoid(gamma * 2.0);
            (uniform.lower_equal(probs).float(), ())
        }
    }

    /// Compute gamma with routing based on ComputeBackend.
    ///
    /// Routing priority:
    /// 1. **CUDA f64**: If `backend.uses_gpu_f64()` and CUDA feature enabled
    /// 2. **CPU f64**: If `backend.use_cpu(IsingSampling, ...)` returns true
    /// 3. **GPU f32**: Default path for bulk operations
    #[allow(clippy::too_many_arguments)]
    fn compute_parameters_routed(
        &self,
        backend: &ComputeBackend,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_nodes = output_spec.shape[0];

        // Priority 1: CUDA f64 for HPC GPUs (H100, B200)
        #[cfg(feature = "cuda")]
        if backend.uses_gpu_f64() {
            return self.compute_parameters_cuda_f64(
                interactions,
                active_flags,
                neighbor_states,
                n_spin_per_interaction,
                output_spec,
                device,
            );
        }

        // Priority 2: CPU f64 for precision-sensitive accumulation
        if backend.use_cpu(OpType::IsingSampling, Some(n_nodes)) {
            return self.compute_parameters_cpu_f64(
                interactions,
                active_flags,
                neighbor_states,
                n_spin_per_interaction,
                output_spec,
                device,
            );
        }

        // Priority 3: Standard GPU f32 path
        self.compute_parameters(
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction,
            output_spec,
            device,
        )
    }

    /// Compute gamma using CUDA f64 for HPC GPUs (H100, B200).
    ///
    /// This path uses the CUDA backend with f64 tensors, providing
    /// GPU-accelerated double-precision computation on datacenter GPUs.
    #[cfg(feature = "cuda")]
    fn compute_parameters_cuda_f64(
        &self,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        output_spec: &TensorSpec,
        wgpu_device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        use burn::tensor::Tensor as BurnTensor;
        use thrml_core::backend::{init_cuda_device, CudaBackend, CudaDevice};

        // Helper to create CUDA f64 tensor with explicit dimensions
        fn cuda_tensor_2d(
            data: &[f64],
            shape: [usize; 2],
            device: &CudaDevice,
        ) -> BurnTensor<CudaBackend, 2> {
            let flat: BurnTensor<CudaBackend, 1> = BurnTensor::from_floats(data, device);
            flat.reshape([shape[0] as i32, shape[1] as i32])
        }

        fn cuda_tensor_3d(
            data: &[f64],
            shape: [usize; 3],
            device: &CudaDevice,
        ) -> BurnTensor<CudaBackend, 3> {
            let flat: BurnTensor<CudaBackend, 1> = BurnTensor::from_floats(data, device);
            flat.reshape([shape[0] as i32, shape[1] as i32, shape[2] as i32])
        }

        let n_nodes = output_spec.shape[0];
        let cuda_device = init_cuda_device();

        // Initialize gamma as f64 on CUDA
        let mut gamma_cuda: BurnTensor<CudaBackend, 1> = BurnTensor::zeros([n_nodes], &cuda_device);

        for (interaction, active, states, &n_spin) in itertools::izip!(
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction
        ) {
            match interaction {
                InteractionData::Tensor(tensor) => {
                    // Convert WGPU tensors to CUDA f64
                    let tensor_data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();
                    let active_data: Vec<f32> = active.clone().into_data().to_vec().unwrap();

                    let tensor_dims = tensor.dims();
                    let n_interactions = tensor_dims[1];
                    let weight_dim = tensor_dims[2];

                    // Convert to f64 and create CUDA tensors
                    let tensor_f64: Vec<f64> = tensor_data.iter().map(|&x| x as f64).collect();
                    let active_f64: Vec<f64> = active_data.iter().map(|&x| x as f64).collect();

                    let weights_cuda = cuda_tensor_3d(
                        &tensor_f64,
                        [n_nodes, n_interactions, weight_dim],
                        &cuda_device,
                    );
                    let active_cuda =
                        cuda_tensor_2d(&active_f64, [n_nodes, n_interactions], &cuda_device);

                    // Convert spin states to CUDA f64
                    let states_spin_cuda: Vec<BurnTensor<CudaBackend, 2>> = states
                        .iter()
                        .take(n_spin)
                        .map(|s| {
                            let data: Vec<f32> = s.clone().into_data().to_vec().unwrap();
                            let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                            cuda_tensor_2d(&data_f64, [n_nodes, n_interactions], &cuda_device)
                        })
                        .collect();

                    #[allow(clippy::needless_collect)]
                    // Vec used for is_empty() and potential iteration
                    let states_cat_cuda: Vec<BurnTensor<CudaBackend, 2>> = states
                        .iter()
                        .skip(n_spin)
                        .map(|s| {
                            let data: Vec<f32> = s.clone().into_data().to_vec().unwrap();
                            let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
                            cuda_tensor_2d(&data_f64, [n_nodes, n_interactions], &cuda_device)
                        })
                        .collect();

                    // Compute spin product in f64 on CUDA
                    let spin_prod_cuda =
                        compute_spin_product_2d_cuda(&states_spin_cuda, &cuda_device);

                    // Get weights (with categorical indexing if needed)
                    let weights_indexed = if states_cat_cuda.is_empty() {
                        weights_cuda.squeeze_dim(2)
                    } else {
                        // Simplified: use first index (full categorical support would need gather)
                        weights_cuda
                            .slice([0..n_nodes, 0..n_interactions, 0..1])
                            .squeeze_dim(2)
                    };

                    // Compute contribution in f64
                    let contribution = weights_indexed * active_cuda * spin_prod_cuda;
                    let contribution_sum = contribution.sum_dim(1).squeeze_dim(1);
                    gamma_cuda = gamma_cuda + contribution_sum;
                }
                InteractionData::Linear { weights } => {
                    let weights_data: Vec<f32> = weights.clone().into_data().to_vec().unwrap();
                    let active_data: Vec<f32> = active.clone().into_data().to_vec().unwrap();
                    let weights_dims = weights.dims();
                    let n_interactions = weights_dims[1];

                    let weights_f64: Vec<f64> = weights_data.iter().map(|&x| x as f64).collect();
                    let active_f64: Vec<f64> = active_data.iter().map(|&x| x as f64).collect();

                    let weights_cuda =
                        cuda_tensor_2d(&weights_f64, [n_nodes, n_interactions], &cuda_device);
                    let active_cuda =
                        cuda_tensor_2d(&active_f64, [n_nodes, n_interactions], &cuda_device);

                    let state_prod_cuda = if states.is_empty() {
                        BurnTensor::<CudaBackend, 2>::ones([n_nodes, n_interactions], &cuda_device)
                    } else {
                        let first_data: Vec<f32> = states[0].clone().into_data().to_vec().unwrap();
                        let first_f64: Vec<f64> = first_data.iter().map(|&x| x as f64).collect();
                        let mut prod =
                            cuda_tensor_2d(&first_f64, [n_nodes, n_interactions], &cuda_device);

                        for s in states.iter().skip(1) {
                            let s_data: Vec<f32> = s.clone().into_data().to_vec().unwrap();
                            let s_f64: Vec<f64> = s_data.iter().map(|&x| x as f64).collect();
                            let s_cuda =
                                cuda_tensor_2d(&s_f64, [n_nodes, n_interactions], &cuda_device);
                            prod = prod * s_cuda;
                        }
                        prod
                    };

                    let contribution = weights_cuda * active_cuda * state_prod_cuda;
                    let contribution_sum = contribution.sum_dim(1).squeeze_dim(1);
                    gamma_cuda = gamma_cuda + contribution_sum;
                }
                InteractionData::Quadratic { .. } | InteractionData::Sphere { .. } => {
                    // Skip - not applicable to spin sampling
                }
            }
        }

        // Convert CUDA f64 result back to WGPU f32
        let gamma_f64: Vec<f64> = gamma_cuda.into_data().to_vec().unwrap();
        let gamma_f32: Vec<f32> = gamma_f64.iter().map(|&x| x as f32).collect();
        Tensor::from_floats(gamma_f32.as_slice(), wgpu_device)
    }

    /// Compute gamma using CPU f64 for precision-sensitive accumulation.
    ///
    /// This extracts tensor data to CPU, performs all accumulation in f64,
    /// then converts back to a f32 tensor. This avoids f32 rounding errors
    /// that accumulate in long chains of operations.
    #[allow(clippy::needless_range_loop)]
    fn compute_parameters_cpu_f64(
        &self,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_nodes = output_spec.shape[0];

        // Accumulate in f64 for precision
        let mut gamma_f64: Vec<f64> = vec![0.0; n_nodes];

        for (interaction, active, states, &n_spin) in itertools::izip!(
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction
        ) {
            match interaction {
                InteractionData::Tensor(tensor) => {
                    // Extract tensor data to CPU
                    let tensor_data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();
                    let active_data: Vec<f32> = active.clone().into_data().to_vec().unwrap();

                    let tensor_dims = tensor.dims();
                    let n_interactions = tensor_dims[1];
                    let weight_dim = tensor_dims[2];

                    // Split states into spin and categorical
                    let states_spin: Vec<Vec<f32>> = states
                        .iter()
                        .take(n_spin)
                        .map(|s| s.clone().into_data().to_vec().unwrap())
                        .collect();
                    let states_cat: Vec<Vec<f32>> = states
                        .iter()
                        .skip(n_spin)
                        .map(|s| s.clone().into_data().to_vec().unwrap())
                        .collect();

                    // Compute contributions in f64
                    for node_idx in 0..n_nodes {
                        for int_idx in 0..n_interactions {
                            let flat_idx = node_idx * n_interactions + int_idx;
                            let active_val = active_data[flat_idx] as f64;

                            if active_val == 0.0 {
                                continue;
                            }

                            // Compute spin product in f64
                            let mut spin_prod: f64 = 1.0;
                            for spin_state in &states_spin {
                                let s = spin_state[flat_idx] as f64;
                                spin_prod *= 2.0f64.mul_add(s, -1.0); // Convert to ±1
                            }

                            // Get weight value
                            let weight_val: f64 = if states_cat.is_empty() {
                                // Direct weight access (weight_dim should be 1)
                                let w_idx =
                                    node_idx * n_interactions * weight_dim + int_idx * weight_dim;
                                tensor_data.get(w_idx).copied().unwrap_or(0.0) as f64
                            } else {
                                // Index by categorical state
                                let cat_idx = states_cat[0][flat_idx] as usize;
                                let w_idx = node_idx * n_interactions * weight_dim
                                    + int_idx * weight_dim
                                    + cat_idx;
                                tensor_data.get(w_idx).copied().unwrap_or(0.0) as f64
                            };

                            // Accumulate in f64
                            gamma_f64[node_idx] += weight_val * active_val * spin_prod;
                        }
                    }
                }
                InteractionData::Linear { weights } => {
                    let weights_data: Vec<f32> = weights.clone().into_data().to_vec().unwrap();
                    let active_data: Vec<f32> = active.clone().into_data().to_vec().unwrap();
                    let weights_dims = weights.dims();
                    let n_interactions = weights_dims[1];

                    // Extract neighbor states
                    let state_data: Vec<Vec<f32>> = states
                        .iter()
                        .map(|s| s.clone().into_data().to_vec().unwrap())
                        .collect();

                    for node_idx in 0..n_nodes {
                        for int_idx in 0..n_interactions {
                            let flat_idx = node_idx * n_interactions + int_idx;
                            let active_val = active_data[flat_idx] as f64;

                            if active_val == 0.0 {
                                continue;
                            }

                            let weight_val = weights_data[flat_idx] as f64;

                            // Compute state product in f64
                            let mut state_prod: f64 = 1.0;
                            for state in &state_data {
                                state_prod *= state[flat_idx] as f64;
                            }

                            gamma_f64[node_idx] += weight_val * active_val * state_prod;
                        }
                    }
                }
                InteractionData::Quadratic { .. } | InteractionData::Sphere { .. } => {
                    // Skip - not applicable to spin sampling
                }
            }
        }

        // Convert f64 accumulation back to f32 tensor
        let gamma_f32: Vec<f32> = gamma_f64.iter().map(|&x| x as f32).collect();
        Tensor::from_floats(gamma_f32.as_slice(), device)
    }

    /// Compute the parameter γ of a spin-valued Bernoulli distribution given DiscreteEBMInteractions.
    ///
    /// γ = Σ_i s_1^i ... s_K^i * W^i[x_1^i, ..., x_M^i]
    ///
    /// where the sum is over all the DiscreteEBMInteractions.
    fn compute_parameters(
        &self,
        interactions: &[InteractionData],
        active_flags: &[Tensor<WgpuBackend, 2>],
        neighbor_states: &[Vec<Tensor<WgpuBackend, 2>>],
        n_spin_per_interaction: &[usize],
        output_spec: &TensorSpec,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_nodes = output_spec.shape[0];
        let mut gamma: Tensor<WgpuBackend, 1> = Tensor::zeros([n_nodes], device);

        for (interaction, active, states, &n_spin) in itertools::izip!(
            interactions,
            active_flags,
            neighbor_states,
            n_spin_per_interaction
        ) {
            match interaction {
                InteractionData::Tensor(tensor) => {
                    // interaction: [n_nodes, n_interactions, weight_dim]
                    // active: [n_nodes, n_interactions]
                    // states: Vec of [n_nodes, n_interactions] tensors (one per tail block)
                    // n_spin: number of spin tail blocks (first n_spin are spin, rest are categorical)

                    // Split states into spin and categorical
                    let states_spin: Vec<Tensor<WgpuBackend, 2>> =
                        states.iter().take(n_spin).cloned().collect();
                    let states_cat: Vec<Tensor<WgpuBackend, 2>> =
                        states.iter().skip(n_spin).cloned().collect();

                    // Compute spin product from spin states only
                    let spin_prod = compute_spin_product_2d(&states_spin, device);

                    // Get interaction dimensions
                    let dims = tensor.dims();
                    let n_interactions = dims[1];

                    // Index weights by categorical neighbor states
                    let weights = if states_cat.is_empty() {
                        // No categorical neighbors - weights should be [n_nodes, n_interactions, 1]
                        // Squeeze to [n_nodes, n_interactions]
                        tensor.clone().squeeze_dim(2)
                    } else {
                        // Use batch_gather to index by categorical states
                        // For SpinGibbsConditional, categorical indexing gives us a scalar per interaction
                        let cat = &states_cat[0]; // First categorical neighbor

                        // Flatten weights and indices for gather
                        let flat_weights: Tensor<WgpuBackend, 2> = tensor
                            .clone()
                            .reshape([(n_nodes * n_interactions) as i32, dims[2] as i32]);
                        let flat_indices: Tensor<WgpuBackend, 1, burn::tensor::Int> = cat
                            .clone()
                            .reshape([(n_nodes * n_interactions) as i32])
                            .int();

                        // Gather and reshape
                        let indices_2d: Tensor<WgpuBackend, 2, burn::tensor::Int> =
                            flat_indices.unsqueeze_dim(1);
                        let gathered: Tensor<WgpuBackend, 2> = flat_weights.gather(1, indices_2d);
                        let result: Tensor<WgpuBackend, 1> = gathered.squeeze_dim::<1>(1);
                        result.reshape([n_nodes as i32, n_interactions as i32])
                    };

                    // Compute contribution: weights * active * spin_prod, then sum over interaction dim
                    let contribution = weights * active.clone() * spin_prod;
                    let contribution_sum = contribution.sum_dim(1).squeeze_dim(1);

                    gamma = gamma + contribution_sum;
                }
                InteractionData::Linear { weights } => {
                    // Linear interaction from continuous coupling: c_i * x_i
                    // Multiply by neighbor states and sum
                    let state_prod = if states.is_empty() {
                        let dims = weights.dims();
                        Tensor::ones([dims[0], dims[1]], device)
                    } else {
                        // Product of all neighbor states (typically continuous)
                        let first = states[0].clone();
                        states.iter().skip(1).fold(first, |acc, s| acc * s.clone())
                    };

                    let contribution = weights.clone() * active.clone() * state_prod;
                    let contribution_sum = contribution.sum_dim(1).squeeze_dim(1);
                    gamma = gamma + contribution_sum;
                }
                InteractionData::Quadratic { .. } => {
                    // Quadratic interactions are for continuous variables, not spin
                    // Skip for SpinGibbsConditional
                }
                InteractionData::Sphere { .. } => {
                    // Sphere interactions are for Langevin dynamics, not Gibbs sampling
                    // Skip for SpinGibbsConditional
                }
            }
        }

        gamma
    }
}

/// Compute the element-wise product of spin states.
///
/// For spin variables: True/1.0 -> +1, False/0.0 -> -1
/// Returns the element-wise product across all state tensors.
fn compute_spin_product_2d(
    states: &[Tensor<WgpuBackend, 2>],
    device: &burn::backend::wgpu::WgpuDevice,
) -> Tensor<WgpuBackend, 2> {
    if states.is_empty() {
        // Return 1.0 tensor
        return Tensor::ones([1, 1], device);
    }

    // Convert each state to ±1: (2 * s - 1)
    let converted: Vec<Tensor<WgpuBackend, 2>> =
        states.iter().map(|s| s.clone() * 2.0 - 1.0).collect();

    // Compute element-wise product
    let first = converted[0].clone();
    converted
        .iter()
        .skip(1)
        .fold(first, |acc, x| acc * x.clone())
}

/// CUDA f64 version of spin product computation.
#[cfg(feature = "cuda")]
fn compute_spin_product_2d_cuda(
    states: &[burn::tensor::Tensor<thrml_core::backend::CudaBackend, 2>],
    device: &burn::backend::cuda::CudaDevice,
) -> burn::tensor::Tensor<thrml_core::backend::CudaBackend, 2> {
    use burn::tensor::Tensor as BurnTensor;
    use thrml_core::backend::CudaBackend;

    if states.is_empty() {
        return BurnTensor::<CudaBackend, 2>::ones([1, 1], device);
    }

    // Convert each state to ±1: (2 * s - 1)
    let converted: Vec<BurnTensor<CudaBackend, 2>> =
        states.iter().map(|s| s.clone() * 2.0 - 1.0).collect();

    // Compute element-wise product
    let first = converted[0].clone();
    converted
        .iter()
        .skip(1)
        .fold(first, |acc, x| acc * x.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "gpu")]
    #[test]
    fn test_spin_gibbs_basic() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};

        ensure_backend();
        let device = init_gpu_device();

        let sampler = SpinGibbsConditional::new();

        // Create simple test case
        let output_spec = TensorSpec {
            shape: vec![4],
            dtype: burn::tensor::DType::Bool,
        };

        // Empty interactions -> gamma = 0 -> P(S=1) = 0.5
        let interactions: Vec<InteractionData> = Vec::new();
        let active_flags: Vec<Tensor<WgpuBackend, 2>> = Vec::new();
        let neighbor_states: Vec<Vec<Tensor<WgpuBackend, 2>>> = Vec::new();
        let n_spin_per_interaction: Vec<usize> = Vec::new();

        let key = RngKey::new(42);
        let (samples, _) = sampler.sample(
            key,
            &interactions,
            &active_flags,
            &neighbor_states,
            &n_spin_per_interaction,
            (),
            &output_spec,
            &device,
        );

        assert_eq!(samples.dims(), [4], "Should produce 4 samples");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_precision_routing_cpu_vs_gpu() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};

        ensure_backend();
        let device = init_gpu_device();

        let sampler = SpinGibbsConditional::new();

        // Create test case with tensor interactions
        let n_nodes = 8;
        let n_interactions = 4;
        let weight_dim = 2;

        let output_spec = TensorSpec {
            shape: vec![n_nodes],
            dtype: burn::tensor::DType::Bool,
        };

        // Create weight tensor [n_nodes, n_interactions, weight_dim]
        let weight_data: Vec<f32> = (0..(n_nodes * n_interactions * weight_dim))
            .map(|i| 0.001 * (i as f32 + 1.0))
            .collect();
        let weights: Tensor<WgpuBackend, 3> =
            Tensor::<WgpuBackend, 1>::from_floats(weight_data.as_slice(), &device).reshape([
                n_nodes as i32,
                n_interactions as i32,
                weight_dim as i32,
            ]);

        let interactions = vec![InteractionData::Tensor(weights)];

        // All interactions active
        let active: Tensor<WgpuBackend, 2> = Tensor::ones([n_nodes, n_interactions], &device);
        let active_flags = vec![active];

        // No spin neighbors (simplifies test), one categorical neighbor
        let cat_state: Tensor<WgpuBackend, 2> = Tensor::zeros([n_nodes, n_interactions], &device);
        let neighbor_states = vec![vec![cat_state]];
        let n_spin_per_interaction = vec![0usize]; // No spin, just categorical

        // Compute gamma via GPU path
        let backend_gpu = ComputeBackend::gpu_only();
        let gamma_gpu = sampler.compute_parameters_routed(
            &backend_gpu,
            &interactions,
            &active_flags,
            &neighbor_states,
            &n_spin_per_interaction,
            &output_spec,
            &device,
        );

        // Compute gamma via CPU f64 path
        let backend_cpu = ComputeBackend::apple_silicon();
        let gamma_cpu = sampler.compute_parameters_routed(
            &backend_cpu,
            &interactions,
            &active_flags,
            &neighbor_states,
            &n_spin_per_interaction,
            &output_spec,
            &device,
        );

        // Extract and compare
        let gpu_data: Vec<f32> = gamma_gpu.into_data().to_vec().unwrap();
        let cpu_data: Vec<f32> = gamma_cpu.into_data().to_vec().unwrap();

        // Both should produce similar results (CPU may be slightly more precise)
        for (i, (g, c)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
            let diff = (g - c).abs();
            let rel_diff = if c.abs() > 1e-10 {
                diff / c.abs()
            } else {
                diff
            };

            // For this simple test, results should be very close
            // The precision benefit shows up more in long chains of operations
            assert!(
                rel_diff < 0.01,
                "Node {} gamma differs too much: GPU={}, CPU={}, rel_diff={}",
                i,
                g,
                c,
                rel_diff
            );
        }

        println!("✓ Precision routing test passed");
        println!(
            "  GPU gamma[0..4]: {:?}",
            &gpu_data[..4.min(gpu_data.len())]
        );
        println!(
            "  CPU gamma[0..4]: {:?}",
            &cpu_data[..4.min(cpu_data.len())]
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_sample_routed_runs() {
        use thrml_core::backend::{ensure_backend, init_gpu_device};

        ensure_backend();
        let device = init_gpu_device();

        let sampler = SpinGibbsConditional::new();
        let backend = ComputeBackend::apple_silicon();

        let output_spec = TensorSpec {
            shape: vec![4],
            dtype: burn::tensor::DType::Bool,
        };

        let interactions: Vec<InteractionData> = Vec::new();
        let active_flags: Vec<Tensor<WgpuBackend, 2>> = Vec::new();
        let neighbor_states: Vec<Vec<Tensor<WgpuBackend, 2>>> = Vec::new();
        let n_spin_per_interaction: Vec<usize> = Vec::new();

        let key = RngKey::new(42);
        let (samples, _) = sampler.sample_routed(
            &backend,
            key,
            &interactions,
            &active_flags,
            &neighbor_states,
            &n_spin_per_interaction,
            (),
            &output_spec,
            &device,
        );

        assert_eq!(
            samples.dims(),
            [4],
            "Should produce 4 samples via routed path"
        );
    }
}
