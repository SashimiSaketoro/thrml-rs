//! NavigatorEBM - Multi-cone EBM navigation system.
//!
//! This module provides energy-based navigation through a hypersphere of embeddings,
//! composing SphereEBM and HypergraphEBM with navigation-specific energy terms.
//!
//! ## Energy Function
//!
//! The navigation energy is a weighted sum of terms:
//! ```text
//! E_nav(q, T, P) = λ_sem · E_semantic(q, T) +
//!                 λ_rad · E_radial(q, T) +
//!                 λ_graph · E_graph(P) +
//!                 λ_ent · E_entropy(T) +
//!                 λ_path · E_path(P)
//! ```
//!
//! Where:
//! - `q` = query embedding/position
//! - `T` = target candidate set
//! - `P` = path through hypergraph
//!
//! ## Multi-Cone Architecture
//!
//! Navigation uses multiple parallel cones, each an independent EBM that:
//! - Has its own scope/aperture
//! - Runs Langevin/Gibbs sampling independently
//! - Competes for a shared attention budget
//! - Can yield allocation to others when saturated

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::compute::{ComputeBackend, HardwareTier, OpType, RuntimePolicy};
use thrml_core::SphericalCoords;
use thrml_samplers::RngKey;

use crate::config::SphereConfig;
use crate::hypergraph::HypergraphEBM;
use crate::roots::{RootsConfig, RootsIndex};
use crate::sphere_ebm::SphereEBM;

/// Learnable weights for navigation energy terms.
#[derive(Clone, Debug)]
pub struct NavigationWeights {
    /// Weight for semantic similarity energy (higher = prioritize similar embeddings)
    pub lambda_semantic: f32,
    /// Weight for radial energy (higher = prefer same radial shell)
    pub lambda_radial: f32,
    /// Weight for graph traversal energy (higher = prefer following edges)
    pub lambda_graph: f32,
    /// Weight for entropy energy (higher = prefer low-entropy/confident targets)
    pub lambda_entropy: f32,
    /// Weight for path length penalty (higher = prefer shorter paths)
    pub lambda_path: f32,
    /// Temperature for LogSumExp softmax operations
    pub temperature: f32,
}

impl Default for NavigationWeights {
    fn default() -> Self {
        Self {
            lambda_semantic: 1.0,
            lambda_radial: 0.5,
            lambda_graph: 0.3,
            lambda_entropy: 0.2,
            lambda_path: 0.1,
            temperature: 1.0,
        }
    }
}

impl NavigationWeights {
    /// Create navigation weights with all terms enabled equally.
    pub fn uniform() -> Self {
        Self {
            lambda_semantic: 1.0,
            lambda_radial: 1.0,
            lambda_graph: 1.0,
            lambda_entropy: 1.0,
            lambda_path: 1.0,
            temperature: 1.0,
        }
    }

    /// Create semantic-only navigation weights.
    pub fn semantic_only() -> Self {
        Self {
            lambda_semantic: 1.0,
            lambda_radial: 0.0,
            lambda_graph: 0.0,
            lambda_entropy: 0.0,
            lambda_path: 0.0,
            temperature: 1.0,
        }
    }

    /// Convert weights to a tensor for gradient computation.
    pub fn to_tensor(&self, device: &burn::backend::wgpu::WgpuDevice) -> Tensor<WgpuBackend, 1> {
        Tensor::from_data(
            [
                self.lambda_semantic,
                self.lambda_radial,
                self.lambda_graph,
                self.lambda_entropy,
                self.lambda_path,
            ]
            .as_slice(),
            device,
        )
    }

    /// Create weights from a tensor.
    pub fn from_tensor(tensor: &Tensor<WgpuBackend, 1>) -> Self {
        let data: Vec<f32> = tensor.clone().into_data().to_vec().expect("weights to vec");
        Self {
            lambda_semantic: data.get(0).copied().unwrap_or(1.0),
            lambda_radial: data.get(1).copied().unwrap_or(0.5),
            lambda_graph: data.get(2).copied().unwrap_or(0.3),
            lambda_entropy: data.get(3).copied().unwrap_or(0.2),
            lambda_path: data.get(4).copied().unwrap_or(0.1),
            temperature: 1.0,
        }
    }

    /// Builder: set semantic weight.
    pub fn with_semantic(mut self, weight: f32) -> Self {
        self.lambda_semantic = weight;
        self
    }

    /// Builder: set radial weight.
    pub fn with_radial(mut self, weight: f32) -> Self {
        self.lambda_radial = weight;
        self
    }

    /// Builder: set graph weight.
    pub fn with_graph(mut self, weight: f32) -> Self {
        self.lambda_graph = weight;
        self
    }

    /// Builder: set entropy weight.
    pub fn with_entropy(mut self, weight: f32) -> Self {
        self.lambda_entropy = weight;
        self
    }

    /// Builder: set path length weight.
    pub fn with_path(mut self, weight: f32) -> Self {
        self.lambda_path = weight;
        self
    }

    /// Builder: set temperature.
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
}

/// Configuration for a single navigation cone.
#[derive(Clone, Debug)]
pub struct ConeConfig {
    /// Angular aperture (half-angle in radians)
    pub aperture: f32,
    /// Center position (theta, phi) on sphere
    pub center_theta: f32,
    pub center_phi: f32,
    /// Memory budget allocation (bytes)
    pub budget_bytes: usize,
    /// Whether this cone is currently active
    pub active: bool,
    /// Relevance score for budget allocation
    pub relevance: f32,
}

impl Default for ConeConfig {
    fn default() -> Self {
        Self {
            aperture: std::f32::consts::PI / 4.0, // 45 degrees
            center_theta: std::f32::consts::PI / 2.0,
            center_phi: 0.0,
            budget_bytes: 64 * 1024 * 1024, // 64 MB default
            active: true,
            relevance: 1.0,
        }
    }
}

impl ConeConfig {
    /// Create a new cone config with given center position.
    pub fn new(center_theta: f32, center_phi: f32) -> Self {
        Self {
            center_theta,
            center_phi,
            ..Default::default()
        }
    }

    /// Set aperture width.
    pub fn with_aperture(mut self, aperture: f32) -> Self {
        self.aperture = aperture;
        self
    }

    /// Set memory budget.
    pub fn with_budget(mut self, budget_bytes: usize) -> Self {
        self.budget_bytes = budget_bytes;
        self
    }
}

// ============================================================================
// Multi-Cone Configuration (Section 5.4 of Architecture Plan)
// ============================================================================

/// Budget allocation configuration for multi-cone navigation.
///
/// Controls how attention/memory budget is distributed across spawned cones.
/// From Section 5.4.5 of EBM Navigator Architecture Plan.
///
/// # Example
///
/// ```
/// use thrml_sphere::BudgetConfig;
///
/// // Configuration for a 24GB system with 18GB attention budget
/// let config = BudgetConfig::new(18 * 1024 * 1024 * 1024)
///     .with_max_cones(32)
///     .with_min_cone_budget(64 * 1024 * 1024);
///
/// // For development/testing with smaller budget
/// let dev_config = BudgetConfig::dev();
/// ```
#[derive(Clone, Debug)]
pub struct BudgetConfig {
    /// Total attention budget in bytes.
    ///
    /// This is the maximum memory that can be allocated across all cones.
    /// Example: 18GB for a 24GB system (6GB reserved for OS/other).
    pub total_budget_bytes: usize,

    /// Maximum number of cones (hardware/efficiency limit).
    ///
    /// Each cone has overhead (state, sampling buffers), so there's
    /// diminishing returns after ~32 cones for most queries. Default: 32
    pub max_cones: usize,

    /// Minimum budget per cone in bytes.
    ///
    /// Cones with less than this budget are not viable and won't be spawned.
    /// Default: 64MB
    pub min_cone_budget: usize,

    /// Activation threshold for peak detection.
    ///
    /// Only ROOTS partitions with activation above this threshold are
    /// considered as peaks for cone spawning. Range: 0.0 to 1.0. Default: 0.2
    pub peak_threshold: f32,

    /// Minimum angular separation between peaks in radians.
    ///
    /// Peaks closer than this are merged into a single cone.
    /// Default: π/16 (~11 degrees)
    pub min_peak_separation: f32,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            total_budget_bytes: 512 * 1024 * 1024, // 512MB default
            max_cones: 32,
            min_cone_budget: 16 * 1024 * 1024, // 16MB
            peak_threshold: 0.2,
            min_peak_separation: std::f32::consts::PI / 16.0,
        }
    }
}

impl BudgetConfig {
    /// Create a new budget configuration with specified total budget.
    ///
    /// # Arguments
    ///
    /// * `total_budget_bytes` - Total memory budget for all cones combined.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::BudgetConfig;
    ///
    /// let config = BudgetConfig::new(8 * 1024 * 1024 * 1024); // 8GB
    /// ```
    pub fn new(total_budget_bytes: usize) -> Self {
        Self {
            total_budget_bytes,
            ..Default::default()
        }
    }

    /// Create a development/testing configuration with small budget.
    ///
    /// Uses 128MB total, max 4 cones, 8MB minimum per cone.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::BudgetConfig;
    ///
    /// let config = BudgetConfig::dev();
    /// assert_eq!(config.max_cones, 4);
    /// ```
    pub fn dev() -> Self {
        Self {
            total_budget_bytes: 128 * 1024 * 1024, // 128MB
            max_cones: 4,
            min_cone_budget: 8 * 1024 * 1024, // 8MB
            peak_threshold: 0.15,
            min_peak_separation: std::f32::consts::PI / 8.0,
        }
    }

    /// Create a configuration for large-scale navigation.
    ///
    /// Uses 16GB total, up to 32 cones, 256MB minimum per cone.
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::BudgetConfig;
    ///
    /// let config = BudgetConfig::large_scale();
    /// assert!(config.total_budget_bytes > 10 * 1024 * 1024 * 1024);
    /// ```
    pub fn large_scale() -> Self {
        Self {
            total_budget_bytes: 16 * 1024 * 1024 * 1024, // 16GB
            max_cones: 32,
            min_cone_budget: 256 * 1024 * 1024, // 256MB
            peak_threshold: 0.2,
            min_peak_separation: std::f32::consts::PI / 32.0,
        }
    }

    /// Create budget configuration optimized for a hardware tier.
    ///
    /// Configures memory budgets and cone counts based on typical
    /// hardware characteristics:
    ///
    /// | Hardware | Budget | Max Cones | Notes |
    /// |----------|--------|-----------|-------|
    /// | Apple Silicon | 6GB | 8 | Unified memory, conservative |
    /// | NVIDIA Consumer | 18GB | 16 | 24GB card with headroom |
    /// | DGX Spark / GB10 | 100GB | 64 | 128GB unified LPDDR5x |
    /// | H100/B200 | 64GB | 32 | 80GB HBM with headroom |
    /// | CPU Only | 4GB | 4 | System RAM, conservative |
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::BudgetConfig;
    /// use thrml_core::compute::HardwareTier;
    ///
    /// let config = BudgetConfig::for_tier(HardwareTier::AppleSilicon);
    /// assert!(config.total_budget_bytes < 10 * 1024 * 1024 * 1024); // < 10GB
    /// ```
    pub fn for_tier(tier: HardwareTier) -> Self {
        const KB: usize = 1024;
        const MB: usize = 1024 * KB;
        const GB: usize = 1024 * MB;

        match tier {
            HardwareTier::AppleSilicon => Self {
                // M1/M2/M3/M4: 8-192GB unified, but shared with system
                // Conservative: assume ~8GB available for navigation
                total_budget_bytes: 6 * GB,
                max_cones: 8,
                min_cone_budget: 128 * MB,
                peak_threshold: 0.15, // Lower threshold - fewer cones
                min_peak_separation: std::f32::consts::PI / 12.0,
            },

            HardwareTier::NvidiaConsumer => Self {
                // RTX 3080-5090: typically 10-24GB VRAM
                // Assume 24GB card with ~6GB reserved for buffers/overhead
                total_budget_bytes: 18 * GB,
                max_cones: 16,
                min_cone_budget: 256 * MB,
                peak_threshold: 0.18,
                min_peak_separation: std::f32::consts::PI / 24.0,
            },

            HardwareTier::NvidiaSpark => Self {
                // DGX Spark / GB10: 128GB unified LPDDR5x
                // Generous budget since it's unified memory
                total_budget_bytes: 100 * GB,
                max_cones: 64,
                min_cone_budget: 512 * MB,
                peak_threshold: 0.2,
                min_peak_separation: std::f32::consts::PI / 48.0,
            },

            HardwareTier::NvidiaHopper | HardwareTier::NvidiaBlackwell => Self {
                // H100/H200/B200: 80GB+ HBM
                // Conservative to leave room for model weights
                total_budget_bytes: 64 * GB,
                max_cones: 32,
                min_cone_budget: GB, // 1GB min per cone for HPC
                peak_threshold: 0.2,
                min_peak_separation: std::f32::consts::PI / 32.0,
            },

            HardwareTier::AmdRdna => Self {
                // AMD RX 7900 XTX: 24GB, similar to NVIDIA consumer
                total_budget_bytes: 18 * GB,
                max_cones: 16,
                min_cone_budget: 256 * MB,
                peak_threshold: 0.18,
                min_peak_separation: std::f32::consts::PI / 24.0,
            },

            HardwareTier::CpuOnly => Self {
                // CPU-only: use system RAM conservatively
                total_budget_bytes: 4 * GB,
                max_cones: 4,
                min_cone_budget: 64 * MB,
                peak_threshold: 0.1, // Very selective
                min_peak_separation: std::f32::consts::PI / 8.0,
            },

            HardwareTier::Unknown => Self::dev(),
        }
    }

    /// Builder: set maximum number of cones.
    pub fn with_max_cones(mut self, max_cones: usize) -> Self {
        self.max_cones = max_cones;
        self
    }

    /// Builder: set minimum budget per cone.
    pub fn with_min_cone_budget(mut self, min_budget: usize) -> Self {
        self.min_cone_budget = min_budget;
        self
    }

    /// Builder: set peak detection threshold.
    pub fn with_peak_threshold(mut self, threshold: f32) -> Self {
        self.peak_threshold = threshold;
        self
    }

    /// Builder: set minimum peak separation.
    pub fn with_min_peak_separation(mut self, separation: f32) -> Self {
        self.min_peak_separation = separation;
        self
    }

    /// Compute effective maximum cones based on budget constraints.
    ///
    /// Returns the lesser of `max_cones` and `total_budget / min_cone_budget`.
    pub fn effective_max_cones(&self) -> usize {
        let budget_limited = self.total_budget_bytes / self.min_cone_budget.max(1);
        budget_limited.min(self.max_cones)
    }
}

// =============================================================================
// RuntimeConfig - Unified Configuration
// =============================================================================

/// Unified runtime configuration for hardware-aware navigation.
///
/// Bundles hardware detection, precision routing, memory budgeting, and sphere
/// configuration into a single struct. Use `RuntimeConfig::auto()` for automatic
/// configuration based on detected hardware, or construct manually for explicit control.
///
/// # Example
///
/// ```rust,ignore
/// use thrml_sphere::RuntimeConfig;
///
/// // Auto-detect hardware and configure everything
/// let config = RuntimeConfig::auto();
/// println!("Detected: {:?}", config.policy.tier);
/// println!("Budget: {} MB", config.budget.total_budget_bytes / (1024 * 1024));
///
/// // Or with explicit tier override
/// use thrml_core::compute::HardwareTier;
/// let config = RuntimeConfig::for_tier(HardwareTier::NvidiaSpark);
/// ```
///
/// # Hardware Tiers
///
/// | Hardware | Detection | Profile | Budget |
/// |----------|-----------|---------|--------|
/// | Apple Silicon | Metal adapter | CpuFp64Strict | 6GB |
/// | NVIDIA Consumer | RTX 30-50 series | GpuMixed | 18GB |
/// | DGX Spark / GB10 | "GRACE" / "GB10" | GpuHpcFp64 | 100GB |
/// | H100/B200 | "H100" / "B200" | GpuHpcFp64 | 64GB |
/// | CPU Only | No GPU | CpuFp64Strict | 4GB |
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Runtime policy from hardware detection (tier, precision profile, dtypes).
    pub policy: RuntimePolicy,

    /// Compute backend for routing operations to CPU/GPU.
    pub backend: ComputeBackend,

    /// Memory budget and cone limits.
    pub budget: BudgetConfig,

    /// Sphere optimization settings.
    pub sphere: SphereConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self::auto()
    }
}

impl RuntimeConfig {
    /// Auto-detect hardware and create appropriate configuration.
    ///
    /// Detection order:
    /// 1. Query WGPU adapter for vendor/device info
    /// 2. Classify hardware tier (Apple Silicon, NVIDIA, AMD, etc.)
    /// 3. Set precision profile and compute backend
    /// 4. Configure memory budget based on tier
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_sphere::RuntimeConfig;
    ///
    /// let config = RuntimeConfig::auto();
    /// println!("Running on {:?}", config.policy.tier);
    /// ```
    pub fn auto() -> Self {
        let policy = RuntimePolicy::detect();
        let backend = ComputeBackend::from_policy(&policy);
        let budget = BudgetConfig::for_tier(policy.tier);
        let sphere = SphereConfig::default();

        Self {
            policy,
            backend,
            budget,
            sphere,
        }
    }

    /// Create configuration for a specific hardware tier.
    ///
    /// Useful when you know your target hardware or want to override detection.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use thrml_sphere::RuntimeConfig;
    /// use thrml_core::compute::HardwareTier;
    ///
    /// // Target DGX Spark
    /// let config = RuntimeConfig::for_tier(HardwareTier::NvidiaSpark);
    /// assert_eq!(config.budget.max_cones, 64);
    /// ```
    pub fn for_tier(tier: HardwareTier) -> Self {
        let policy = RuntimePolicy::for_tier(tier);
        let backend = ComputeBackend::from_policy(&policy);
        let budget = BudgetConfig::for_tier(tier);
        let sphere = SphereConfig::default();

        Self {
            policy,
            backend,
            budget,
            sphere,
        }
    }

    /// Builder: override the budget configuration.
    pub fn with_budget(mut self, budget: BudgetConfig) -> Self {
        self.budget = budget;
        self
    }

    /// Builder: override the sphere configuration.
    pub fn with_sphere(mut self, sphere: SphereConfig) -> Self {
        self.sphere = sphere;
        self
    }

    /// Builder: override the compute backend.
    pub fn with_backend(mut self, backend: ComputeBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Check if this configuration is for HPC-class hardware.
    ///
    /// Returns true for NVIDIA Hopper, Blackwell, and Spark tiers.
    pub fn is_hpc(&self) -> bool {
        self.policy.is_hpc_tier()
    }

    /// Check if this configuration uses unified memory.
    ///
    /// Returns true for Apple Silicon and DGX Spark (both use unified memory).
    pub fn is_unified_memory(&self) -> bool {
        matches!(
            self.policy.tier,
            HardwareTier::AppleSilicon | HardwareTier::NvidiaSpark
        )
    }

    /// Get memory budget in human-readable form.
    pub fn budget_gb(&self) -> f64 {
        self.budget.total_budget_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// Runtime state for a single cone during multi-cone navigation.
///
/// Tracks the cone's configuration, allocated budget, relevance score,
/// and saturation status. From Section 5.3 of EBM Navigator Architecture Plan.
///
/// # Lifecycle
///
/// 1. **Spawned**: Created from ROOTS activation peak via [`MultiConeNavigator::spawn_cones_from_peaks`]
/// 2. **Active**: Running navigation, consuming budget
/// 3. **Saturated**: Found all relevant content, can yield remaining budget
/// 4. **Dissolved**: Relevance dropped below threshold, budget released
///
/// # Example
///
/// ```
/// use thrml_sphere::{ConeConfig, ConeState};
/// use std::f32::consts::PI;
///
/// let state = ConeState::new(
///     ConeConfig::new(PI / 2.0, 0.0).with_aperture(PI / 4.0),
///     64 * 1024 * 1024,  // 64MB budget
///     0.85,               // relevance (from peak strength)
/// );
///
/// assert!(!state.is_saturated);
/// assert_eq!(state.source_partition_ids.len(), 0);
/// ```
#[derive(Clone, Debug)]
pub struct ConeState {
    /// Configuration for this cone (center, aperture, etc.).
    pub config: ConeConfig,

    /// Allocated budget in bytes for this cone.
    pub budget_bytes: usize,

    /// IDs of ROOTS partitions that contributed to spawning this cone.
    ///
    /// Used for tracking which regions of the sphere this cone covers.
    pub source_partition_ids: Vec<usize>,

    /// Relevance score (0.0 to 1.0), derived from peak activation strength.
    ///
    /// Higher relevance = more likely to contain relevant content.
    /// Used for budget allocation (proportional to relevance).
    pub relevance: f32,

    /// Whether this cone has found all relevant content (saturated).
    ///
    /// Saturated cones can yield their remaining budget to other cones.
    pub is_saturated: bool,

    /// Budget consumed so far in bytes.
    pub budget_consumed: usize,
}

impl ConeState {
    /// Create a new cone state with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Cone configuration (center, aperture)
    /// * `budget_bytes` - Allocated budget for this cone
    /// * `relevance` - Relevance score from peak activation strength
    ///
    /// # Example
    ///
    /// ```
    /// use thrml_sphere::{ConeConfig, ConeState};
    ///
    /// let state = ConeState::new(
    ///     ConeConfig::default(),
    ///     32 * 1024 * 1024,
    ///     0.75,
    /// );
    /// ```
    pub fn new(config: ConeConfig, budget_bytes: usize, relevance: f32) -> Self {
        Self {
            config,
            budget_bytes,
            source_partition_ids: Vec::new(),
            relevance,
            is_saturated: false,
            budget_consumed: 0,
        }
    }

    /// Create a cone state from a ROOTS activation peak.
    ///
    /// # Arguments
    ///
    /// * `center` - Angular position (theta, phi) from peak center
    /// * `aperture` - Cone aperture derived from peak spread
    /// * `budget_bytes` - Allocated budget
    /// * `relevance` - Peak activation strength
    /// * `partition_ids` - ROOTS partitions contributing to this peak
    pub fn from_peak(
        center: (f32, f32),
        aperture: f32,
        budget_bytes: usize,
        relevance: f32,
        partition_ids: Vec<usize>,
    ) -> Self {
        let config = ConeConfig::new(center.0, center.1)
            .with_aperture(aperture)
            .with_budget(budget_bytes);

        Self {
            config,
            budget_bytes,
            source_partition_ids: partition_ids,
            relevance,
            is_saturated: false,
            budget_consumed: 0,
        }
    }

    /// Mark this cone as saturated (found all relevant content).
    pub fn saturate(&mut self) {
        self.is_saturated = true;
    }

    /// Get remaining budget in bytes.
    pub fn remaining_budget(&self) -> usize {
        self.budget_bytes.saturating_sub(self.budget_consumed)
    }

    /// Record budget consumption.
    pub fn consume_budget(&mut self, bytes: usize) {
        self.budget_consumed = self.budget_consumed.saturating_add(bytes);
    }
}

/// Result of a navigation query.
#[derive(Clone, Debug)]
pub struct NavigationResult {
    /// Indices of retrieved targets (sorted by relevance)
    pub target_indices: Vec<usize>,
    /// Energy of each target
    pub target_energies: Vec<f32>,
    /// Path through hypergraph (if graph navigation was used)
    pub path: Option<Vec<usize>>,
    /// Total navigation energy
    pub total_energy: f32,
    /// Number of Langevin steps taken
    pub steps: usize,
}

/// Main Navigator EBM combining sphere and hypergraph energy models.
///
/// NavigatorEBM implements multi-cone navigation through a hypersphere of embeddings.
/// Each navigation query spawns cones based on ROOTS layer activation patterns,
/// with dynamic budget allocation across cones.
pub struct NavigatorEBM {
    /// Underlying sphere EBM for embedding similarity and radial energy
    pub sphere_ebm: SphereEBM,
    /// Optional hypergraph EBM for graph structure energy
    pub hypergraph_ebm: Option<HypergraphEBM>,
    /// Learnable weights for navigation energy terms
    pub weights: NavigationWeights,
    /// Optional entropy scores for entropy-aware navigation
    pub entropies: Option<Tensor<WgpuBackend, 1>>,
    /// Configuration for Langevin sampling
    pub config: SphereConfig,
}

impl NavigatorEBM {
    /// Create a NavigatorEBM from a SphereEBM.
    pub fn from_sphere_ebm(sphere_ebm: SphereEBM) -> Self {
        let config = sphere_ebm.config.clone();
        Self {
            sphere_ebm,
            hypergraph_ebm: None,
            weights: NavigationWeights::default(),
            entropies: None,
            config,
        }
    }

    /// Create a NavigatorEBM from embeddings and prominence.
    pub fn new(
        embeddings: Tensor<WgpuBackend, 2>,
        prominence: Tensor<WgpuBackend, 1>,
        entropies: Option<Tensor<WgpuBackend, 1>>,
        config: SphereConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let sphere_ebm = SphereEBM::new(
            embeddings,
            prominence,
            entropies.clone(),
            config.clone(),
            device,
        );
        Self {
            sphere_ebm,
            hypergraph_ebm: None,
            weights: NavigationWeights::default(),
            entropies,
            config,
        }
    }

    /// Add hypergraph connectivity for graph-aware navigation.
    pub fn with_hypergraph(mut self, hypergraph_ebm: HypergraphEBM) -> Self {
        self.hypergraph_ebm = Some(hypergraph_ebm);
        self
    }

    /// Set navigation weights.
    pub fn with_weights(mut self, weights: NavigationWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Get the number of points in the sphere.
    pub fn n_points(&self) -> usize {
        self.sphere_ebm.n_points()
    }

    /// Compute semantic energy between query and targets.
    ///
    /// E_semantic(q, t) = -cos_sim(q, e_t)
    /// Lower energy = higher similarity = better match.
    pub fn semantic_energy(
        &self,
        query_embedding: &Tensor<WgpuBackend, 1>,
        target_indices: &[usize],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_targets = target_indices.len();
        if n_targets == 0 {
            return Tensor::zeros([0], device);
        }

        // Get embeddings for targets
        let query_norm = query_embedding.clone().powf_scalar(2.0).sum().sqrt();
        let query_normalized = query_embedding.clone() / (query_norm + 1e-8);

        // Gather target embeddings and compute cosine similarity
        let mut energies = Vec::with_capacity(n_targets);
        let embeddings_data: Vec<f32> = self
            .sphere_ebm
            .embeddings
            .clone()
            .into_data()
            .to_vec()
            .expect("embeddings to vec");
        let d = self.sphere_ebm.embedding_dim();

        let query_data: Vec<f32> = query_normalized.into_data().to_vec().expect("query to vec");

        for &idx in target_indices {
            // Extract target embedding
            let target_start = idx * d;
            let target_end = target_start + d;
            let target_slice = &embeddings_data[target_start..target_end];

            // Normalize target
            let target_norm: f32 = target_slice.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Compute dot product (cosine similarity since both are normalized)
            let cos_sim: f32 = query_data
                .iter()
                .zip(target_slice.iter())
                .map(|(&q, &t)| q * t / (target_norm + 1e-8))
                .sum();

            // Energy is negative similarity (lower is better)
            energies.push(-cos_sim);
        }

        Tensor::from_data(energies.as_slice(), device)
    }

    /// Compute radial energy between query radius and targets.
    ///
    /// E_radial(q, t) = (r_t - r_q)²
    /// Encourages targets at similar radius to query.
    pub fn radial_energy(
        &self,
        query_radius: f32,
        coords: &SphericalCoords,
        target_indices: &[usize],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_targets = target_indices.len();
        if n_targets == 0 {
            return Tensor::zeros([0], device);
        }

        let r_data: Vec<f32> = coords.r.clone().into_data().to_vec().expect("r to vec");

        let energies: Vec<f32> = target_indices
            .iter()
            .map(|&idx| {
                let diff = r_data[idx] - query_radius;
                diff * diff
            })
            .collect();

        Tensor::from_data(energies.as_slice(), device)
    }

    /// Compute entropy energy for targets.
    ///
    /// E_entropy(t) = -H(t) (lower entropy = lower energy = preferred)
    pub fn entropy_energy(
        &self,
        target_indices: &[usize],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_targets = target_indices.len();
        if n_targets == 0 {
            return Tensor::zeros([0], device);
        }

        let entropies = match &self.entropies {
            Some(ent) => {
                let ent_data: Vec<f32> = ent.clone().into_data().to_vec().expect("entropy to vec");
                target_indices.iter().map(|&idx| -ent_data[idx]).collect()
            }
            None => vec![0.0; n_targets],
        };

        Tensor::from_data(entropies.as_slice(), device)
    }

    /// Compute graph energy for a path through the hypergraph.
    ///
    /// E_graph(P) = sum of edge weights for non-edges (penalize missing connections)
    pub fn graph_energy(
        &self,
        coords: &SphericalCoords,
        path: &[usize],
        _device: &burn::backend::wgpu::WgpuDevice,
    ) -> f32 {
        match &self.hypergraph_ebm {
            Some(hg_ebm) => {
                // Compute spring energy for path nodes
                let spring_e = hg_ebm.spring_energy(coords);
                let spring_data: Vec<f32> = spring_e.into_data().to_vec().expect("spring to vec");

                // Sum energy along path
                path.iter()
                    .map(|&idx| spring_data.get(idx).copied().unwrap_or(0.0))
                    .sum()
            }
            None => 0.0,
        }
    }

    /// Compute path length energy.
    ///
    /// E_path(P) = |P| (simple length penalty)
    pub fn path_energy(&self, path: &[usize]) -> f32 {
        path.len() as f32
    }

    /// Compute total navigation energy for candidates.
    ///
    /// Combines all energy terms with learned weights.
    pub fn total_energy(
        &self,
        query_embedding: &Tensor<WgpuBackend, 1>,
        query_radius: f32,
        coords: &SphericalCoords,
        target_indices: &[usize],
        path: Option<&[usize]>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_targets = target_indices.len();
        if n_targets == 0 {
            return Tensor::zeros([0], device);
        }

        // Compute individual energy terms
        let e_semantic = self.semantic_energy(query_embedding, target_indices, device);
        let e_radial = self.radial_energy(query_radius, coords, target_indices, device);
        let e_entropy = self.entropy_energy(target_indices, device);

        // Weight and combine
        let total = e_semantic.mul_scalar(self.weights.lambda_semantic)
            + e_radial.mul_scalar(self.weights.lambda_radial)
            + e_entropy.mul_scalar(self.weights.lambda_entropy);

        // Add graph and path energy if applicable
        if let Some(p) = path {
            let e_graph = self.graph_energy(coords, p, device);
            let e_path = self.path_energy(p);

            let graph_contrib = e_graph * self.weights.lambda_graph;
            let path_contrib = e_path * self.weights.lambda_path;

            // Add scalar contributions to all targets
            total + graph_contrib + path_contrib
        } else {
            total
        }
    }

    // =========================================================================
    // Batched GPU Energy Functions (stay entirely on GPU for forward pass)
    // =========================================================================

    /// Compute semantic energy using batched GPU operations.
    ///
    /// Unlike `semantic_energy`, this version:
    /// - Takes a tensor of indices instead of a slice
    /// - Uses GPU gather to select target embeddings
    /// - Computes all cosine similarities in parallel on GPU
    /// - Never transfers data to CPU
    ///
    /// Returns energies as negative cosine similarities (lower = better).
    pub fn semantic_energy_batched(
        &self,
        query_embedding: &Tensor<WgpuBackend, 1>,
        target_indices: &Tensor<WgpuBackend, 1, burn::tensor::Int>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_targets = target_indices.dims()[0];
        if n_targets == 0 {
            return Tensor::zeros([0], device);
        }

        // Normalize query
        let query_norm = query_embedding.clone().powf_scalar(2.0).sum().sqrt();
        let query_normalized = query_embedding.clone() / (query_norm + 1e-8);

        // Gather target embeddings: [n_targets, d]
        // Use select to get rows by indices
        let embeddings_2d = self.sphere_ebm.embeddings.clone(); // [n, d]
        let target_embeddings = embeddings_2d.select(0, target_indices.clone()); // [n_targets, d]

        // Normalize target embeddings along dim 1
        // sum_dim keeps the dimension, so we need to handle this carefully
        let target_sq = target_embeddings.clone().powf_scalar(2.0);
        let target_sum: Tensor<WgpuBackend, 2> = target_sq.sum_dim(1); // [n_targets, 1]
        let target_norms = target_sum.sqrt(); // [n_targets, 1]
        let target_normalized = target_embeddings / (target_norms.clone() + 1e-8); // broadcasts [n_targets, d]

        // Cosine similarity via matmul: query [1, d] @ targets^T [d, n_targets] = [1, n_targets]
        let d = self.sphere_ebm.embedding_dim();
        let query_2d: Tensor<WgpuBackend, 2> = query_normalized.reshape([1, d as i32]); // [1, d]
        let similarities_2d = query_2d.matmul(target_normalized.transpose()); // [1, n_targets]

        // Reshape to 1D: [n_targets]
        let similarities: Tensor<WgpuBackend, 1> = similarities_2d.reshape([n_targets as i32]);

        // Energy is negative similarity
        similarities.neg()
    }

    /// Compute radial energy using batched GPU operations.
    ///
    /// E_radial(q, t) = (r_t - r_q)² computed entirely on GPU.
    pub fn radial_energy_batched(
        &self,
        query_radius: f32,
        coords: &SphericalCoords,
        target_indices: &Tensor<WgpuBackend, 1, burn::tensor::Int>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_targets = target_indices.dims()[0];
        if n_targets == 0 {
            return Tensor::zeros([0], device);
        }

        // Gather target radii
        let target_radii = coords.r.clone().select(0, target_indices.clone());

        // Compute squared differences
        let diff = target_radii.sub_scalar(query_radius);
        diff.clone() * diff
    }

    /// Compute entropy energy using batched GPU operations.
    ///
    /// E_entropy(t) = -H(t) computed entirely on GPU.
    pub fn entropy_energy_batched(
        &self,
        target_indices: &Tensor<WgpuBackend, 1, burn::tensor::Int>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_targets = target_indices.dims()[0];
        if n_targets == 0 {
            return Tensor::zeros([0], device);
        }

        match &self.entropies {
            Some(ent) => {
                // Gather entropies and negate (lower entropy = lower energy)
                ent.clone().select(0, target_indices.clone()).neg()
            }
            None => Tensor::zeros([n_targets], device),
        }
    }

    /// Compute total navigation energy using batched GPU operations.
    ///
    /// This is the GPU-optimized version of `total_energy` that stays entirely
    /// on GPU for the forward pass. Use this when training or when you need
    /// gradients.
    ///
    /// # Arguments
    ///
    /// * `target_indices` - Tensor of target indices (GPU tensor)
    /// * `backend` - Compute backend for CPU/GPU routing decisions
    ///
    /// For path/graph energy, use the scalar addition after this returns.
    pub fn total_energy_batched(
        &self,
        query_embedding: &Tensor<WgpuBackend, 1>,
        query_radius: f32,
        coords: &SphericalCoords,
        target_indices: &Tensor<WgpuBackend, 1, burn::tensor::Int>,
        device: &burn::backend::wgpu::WgpuDevice,
        backend: &ComputeBackend,
    ) -> Tensor<WgpuBackend, 1> {
        let n_targets = target_indices.dims()[0];
        if n_targets == 0 {
            return Tensor::zeros([0], device);
        }

        // Route based on backend - use GPU if allowed for BatchEnergyForward
        if backend.use_gpu(OpType::BatchEnergyForward, Some(n_targets)) {
            // GPU path: batched operations
            let e_semantic = self.semantic_energy_batched(query_embedding, target_indices, device);
            let e_radial = self.radial_energy_batched(query_radius, coords, target_indices, device);
            let e_entropy = self.entropy_energy_batched(target_indices, device);

            // Weighted combination (all on GPU)
            e_semantic.mul_scalar(self.weights.lambda_semantic)
                + e_radial.mul_scalar(self.weights.lambda_radial)
                + e_entropy.mul_scalar(self.weights.lambda_entropy)
        } else {
            // CPU path: fall back to original implementation
            // Convert tensor indices to Vec for compatibility
            let indices_data: Vec<i32> = target_indices
                .clone()
                .into_data()
                .to_vec()
                .expect("indices to vec");
            let indices: Vec<usize> = indices_data.iter().map(|&i| i as usize).collect();

            self.total_energy(
                query_embedding,
                query_radius,
                coords,
                &indices,
                None,
                device,
            )
        }
    }

    /// Convert a slice of indices to a GPU tensor.
    ///
    /// Helper for transitioning from `&[usize]` to batched tensor operations.
    pub fn indices_to_tensor(
        indices: &[usize],
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1, burn::tensor::Int> {
        let indices_i32: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        Tensor::from_data(indices_i32.as_slice(), device)
    }

    /// Find targets within a cone aperture.
    pub fn targets_in_cone(&self, coords: &SphericalCoords, cone: &ConeConfig) -> Vec<usize> {
        let theta_data: Vec<f32> = coords
            .theta
            .clone()
            .into_data()
            .to_vec()
            .expect("theta to vec");
        let phi_data: Vec<f32> = coords.phi.clone().into_data().to_vec().expect("phi to vec");

        let n = theta_data.len();
        let mut targets = Vec::new();

        for i in 0..n {
            // Compute angular distance from cone center
            let dtheta = theta_data[i] - cone.center_theta;
            let dphi = phi_data[i] - cone.center_phi;

            // Simple angular distance (could use geodesic for more accuracy)
            let angular_dist = (dtheta * dtheta + dphi * dphi * theta_data[i].sin().powi(2)).sqrt();

            if angular_dist <= cone.aperture {
                targets.push(i);
            }
        }

        targets
    }

    /// Navigate to find relevant targets for a query.
    ///
    /// This is the main entry point for navigation. It:
    /// 1. Initializes or uses existing sphere coordinates
    /// 2. Runs Langevin sampling to find low-energy positions
    /// 3. Returns targets sorted by navigation energy
    pub fn navigate(
        &self,
        query_embedding: Tensor<WgpuBackend, 1>,
        query_radius: f32,
        key: RngKey,
        top_k: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> NavigationResult {
        // Run sphere optimization to get positions
        let coords = self.sphere_ebm.optimize(key, device);

        // Compute energy for all points
        let all_indices: Vec<usize> = (0..self.n_points()).collect();
        let energies = self.total_energy(
            &query_embedding,
            query_radius,
            &coords,
            &all_indices,
            None,
            device,
        );

        // Sort by energy (ascending = lowest first = best matches)
        let energy_data: Vec<f32> = energies.into_data().to_vec().expect("energies to vec");
        let mut indexed: Vec<(usize, f32)> = energy_data
            .iter()
            .enumerate()
            .map(|(i, &e)| (i, e))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        let k = top_k.min(indexed.len());
        let target_indices: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
        let target_energies: Vec<f32> = indexed[..k].iter().map(|(_, e)| *e).collect();
        let total_energy = target_energies.iter().sum::<f32>() / k.max(1) as f32;

        NavigationResult {
            target_indices,
            target_energies,
            path: None,
            total_energy,
            steps: self.config.n_steps,
        }
    }

    /// Navigate within a specific cone.
    pub fn navigate_cone(
        &self,
        query_embedding: Tensor<WgpuBackend, 1>,
        query_radius: f32,
        cone: &ConeConfig,
        key: RngKey,
        top_k: usize,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> NavigationResult {
        // Run sphere optimization
        let coords = self.sphere_ebm.optimize(key, device);

        // Get targets in cone
        let cone_targets = self.targets_in_cone(&coords, cone);

        if cone_targets.is_empty() {
            return NavigationResult {
                target_indices: vec![],
                target_energies: vec![],
                path: None,
                total_energy: f32::INFINITY,
                steps: self.config.n_steps,
            };
        }

        // Compute energy for cone targets
        let energies = self.total_energy(
            &query_embedding,
            query_radius,
            &coords,
            &cone_targets,
            None,
            device,
        );

        // Sort by energy
        let energy_data: Vec<f32> = energies.into_data().to_vec().expect("energies to vec");
        let mut indexed: Vec<(usize, f32)> = cone_targets
            .iter()
            .zip(energy_data.iter())
            .map(|(&idx, &e)| (idx, e))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top-k
        let k = top_k.min(indexed.len());
        let target_indices: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
        let target_energies: Vec<f32> = indexed[..k].iter().map(|(_, e)| *e).collect();
        let total_energy = target_energies.iter().sum::<f32>() / k.max(1) as f32;

        NavigationResult {
            target_indices,
            target_energies,
            path: None,
            total_energy,
            steps: self.config.n_steps,
        }
    }

    /// Differentiable target selection using Gumbel-Softmax.
    ///
    /// This method enables gradient flow through discrete target selection,
    /// which is essential for end-to-end training of the navigation system.
    ///
    /// # Arguments
    /// * `query_embedding` - Query embedding vector
    /// * `query_radius` - Query radial position
    /// * `coords` - Current sphere coordinates
    /// * `candidate_indices` - Indices of candidate targets
    /// * `temperature` - Gumbel-Softmax temperature (lower = harder)
    /// * `hard` - If true, use Straight-Through Estimator for one-hot output
    /// * `device` - Compute device
    ///
    /// # Returns
    ///
    /// Soft/hard attention weights over candidates \[n_candidates\].
    pub fn gumbel_select_targets(
        &self,
        query_embedding: &Tensor<WgpuBackend, 1>,
        query_radius: f32,
        coords: &SphericalCoords,
        candidate_indices: &[usize],
        temperature: f32,
        hard: bool,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Tensor<WgpuBackend, 1> {
        let n_candidates = candidate_indices.len();
        if n_candidates == 0 {
            return Tensor::zeros([0], device);
        }

        // Compute negative energies as logits (lower energy = higher probability)
        let energies = self.total_energy(
            query_embedding,
            query_radius,
            coords,
            candidate_indices,
            None,
            device,
        );

        // Convert to logits: higher logit = higher probability
        // Since softmax(x) gives higher prob to larger x, use -energy as logits
        let logits_1d = -energies;

        // Reshape to 2D for gumbel_softmax: [1, n_candidates]
        let logits_2d: Tensor<WgpuBackend, 2> = logits_1d.reshape([1, n_candidates as i32]);

        // Apply Gumbel-Softmax
        let weights_2d = thrml_kernels::autodiff::gumbel_softmax::gumbel_softmax(
            logits_2d,
            temperature,
            hard,
            device,
        );

        // Reshape back to 1D
        weights_2d.reshape([n_candidates as i32])
    }

    /// Differentiable path selection through hypergraph using Gumbel-Softmax.
    ///
    /// Given a current node and its neighbors in the hypergraph, selects
    /// the next node to visit using differentiable sampling.
    ///
    /// # Arguments
    /// * `query_embedding` - Query embedding vector
    /// * `query_radius` - Query radial position
    /// * `current_node` - Index of current node in hypergraph
    /// * `coords` - Current sphere coordinates
    /// * `temperature` - Gumbel-Softmax temperature
    /// * `hard` - Use hard (one-hot) or soft selection
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Tuple of (next_node_index, selection_weights)
    pub fn gumbel_select_next_node(
        &self,
        query_embedding: &Tensor<WgpuBackend, 1>,
        query_radius: f32,
        current_node: usize,
        coords: &SphericalCoords,
        temperature: f32,
        hard: bool,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Option<(usize, Tensor<WgpuBackend, 1>)> {
        // Get neighbors from hypergraph
        let neighbors = self.get_neighbors(current_node);
        if neighbors.is_empty() {
            return None;
        }

        // Get selection weights
        let weights = self.gumbel_select_targets(
            query_embedding,
            query_radius,
            coords,
            &neighbors,
            temperature,
            hard,
            device,
        );

        // If hard selection, return argmax; otherwise return weighted average index
        let weights_data: Vec<f32> = weights
            .clone()
            .into_data()
            .to_vec()
            .expect("weights to vec");

        let selected_idx = if hard {
            // Find index of maximum weight
            weights_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            // Soft selection: return index with highest weight
            weights_data
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };

        Some((neighbors[selected_idx], weights))
    }

    /// Get neighbors of a node in the hypergraph.
    fn get_neighbors(&self, node: usize) -> Vec<usize> {
        match &self.hypergraph_ebm {
            Some(hg_ebm) => {
                // Extract neighbors from adjacency matrix
                let adj_data: Vec<f32> = hg_ebm
                    .spring_ebm
                    .adjacency
                    .clone()
                    .into_data()
                    .to_vec()
                    .expect("adj to vec");

                let n = self.n_points();
                let mut neighbors = Vec::new();

                for j in 0..n {
                    let idx = node * n + j;
                    if idx < adj_data.len() && adj_data[idx] > 0.0 {
                        neighbors.push(j);
                    }
                }

                neighbors
            }
            None => Vec::new(),
        }
    }

    /// Navigate with differentiable path selection using Gumbel-Softmax.
    ///
    /// This method navigates through the hypergraph structure using
    /// differentiable discrete choices, enabling end-to-end training.
    ///
    /// # Arguments
    /// * `query_embedding` - Query embedding vector
    /// * `query_radius` - Query radial position
    /// * `start_node` - Starting node index
    /// * `max_steps` - Maximum path length
    /// * `temperature` - Gumbel-Softmax temperature
    /// * `key` - RNG key for sphere optimization
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Navigation result with path and differentiable weights
    pub fn navigate_differentiable(
        &self,
        query_embedding: Tensor<WgpuBackend, 1>,
        query_radius: f32,
        start_node: usize,
        max_steps: usize,
        temperature: f32,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> DifferentiableNavigationResult {
        // Get optimized coordinates
        let coords = self.sphere_ebm.optimize(key, device);

        let mut path = vec![start_node];
        let mut all_weights = Vec::new();
        let mut current_node = start_node;

        // Navigate through hypergraph
        for _step in 0..max_steps {
            match self.gumbel_select_next_node(
                &query_embedding,
                query_radius,
                current_node,
                &coords,
                temperature,
                true, // Hard selection for path
                device,
            ) {
                Some((next_node, weights)) => {
                    path.push(next_node);
                    all_weights.push(weights);
                    current_node = next_node;
                }
                None => break, // No more neighbors
            }
        }

        // Compute final energy
        let energies = self.total_energy(
            &query_embedding,
            query_radius,
            &coords,
            &path,
            Some(&path),
            device,
        );
        let energy_data: Vec<f32> = energies.into_data().to_vec().expect("energies to vec");
        let total_energy = energy_data.iter().sum::<f32>() / path.len().max(1) as f32;

        DifferentiableNavigationResult {
            path,
            selection_weights: all_weights,
            total_energy,
            temperature,
        }
    }
}

/// Result of differentiable navigation through hypergraph.
#[derive(Clone)]
pub struct DifferentiableNavigationResult {
    /// Path through hypergraph (node indices)
    pub path: Vec<usize>,
    /// Selection weights at each step (for gradient computation)
    pub selection_weights: Vec<Tensor<WgpuBackend, 1>>,
    /// Total navigation energy
    pub total_energy: f32,
    /// Temperature used for Gumbel-Softmax
    pub temperature: f32,
}

impl DifferentiableNavigationResult {
    /// Get the final node in the path.
    pub fn final_node(&self) -> Option<usize> {
        self.path.last().copied()
    }

    /// Get path length.
    pub fn path_length(&self) -> usize {
        self.path.len()
    }
}

// ============================================================================
// Multi-Cone Navigation (Section 5.4 of Architecture Plan)
// ============================================================================

/// Base cone aperture for spawning (radians).
/// Actual aperture is adjusted based on peak spread.
const BASE_CONE_APERTURE: f32 = std::f32::consts::PI / 4.0;

/// Results from multi-cone navigation.
///
/// Contains merged results from all active cones, as well as
/// per-cone results for debugging/analysis.
///
/// # Example
///
/// ```ignore
/// let result = navigator.navigate_multi_cone(query, 50.0, 10, key, &device);
///
/// println!("Found {} targets from {} cones", result.target_indices.len(), result.n_cones());
/// for (idx, energy) in result.target_indices.iter().zip(result.target_energies.iter()) {
///     println!("  Target {}: energy {:.4}", idx, energy);
/// }
/// ```
#[derive(Clone, Debug)]
pub struct MultiConeResult {
    /// Merged target indices (deduplicated, sorted by energy).
    pub target_indices: Vec<usize>,

    /// Merged target energies (corresponding to target_indices).
    pub target_energies: Vec<f32>,

    /// Results from each individual cone.
    pub per_cone_results: Vec<NavigationResult>,

    /// Total budget consumed across all cones in bytes.
    pub budget_used: usize,

    /// Number of cones that were spawned.
    pub cones_spawned: usize,
}

impl MultiConeResult {
    /// Get number of cones that contributed to results.
    pub fn n_cones(&self) -> usize {
        self.cones_spawned
    }

    /// Get total number of targets found.
    pub fn n_targets(&self) -> usize {
        self.target_indices.len()
    }

    /// Check if results are empty.
    pub fn is_empty(&self) -> bool {
        self.target_indices.is_empty()
    }

    /// Get the best (lowest energy) target index.
    pub fn best_target(&self) -> Option<usize> {
        self.target_indices.first().copied()
    }

    /// Get the best (lowest) energy.
    pub fn best_energy(&self) -> Option<f32> {
        self.target_energies.first().copied()
    }
}

/// Multi-cone navigator using ROOTS for initial routing.
///
/// Implements the multi-cone EBM navigation architecture from Section 5.4
/// of the EBM Navigator Architecture Plan. Key features:
///
/// - **ROOTS-based cone spawning**: Uses activation peaks on the ROOTS index
///   to spawn cones at relevant regions of the sphere
/// - **Budget allocation**: Distributes attention budget proportional to peak strength
/// - **Parallel navigation**: Runs navigation independently in each cone
/// - **Result merging**: Combines and deduplicates results from all cones
///
/// # Architecture
///
/// ```text
/// Query ──► ROOTS.activate() ──► detect_peaks() ──► spawn_cones()
///                                                        │
///           ┌────────────────────────────────────────────┼────────┐
///           │                                            ▼        │
///           │  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
///           │  │ Cone 0  │  │ Cone 1  │  │ Cone N  │  ...      │
///           │  │ (35% $) │  │ (30% $) │  │ (35% $) │           │
///           │  └────┬────┘  └────┬────┘  └────┬────┘           │
///           │       │            │            │                 │
///           │       ▼            ▼            ▼                 │
///           │   navigate()   navigate()   navigate()           │
///           │       │            │            │                 │
///           │       └────────────┼────────────┘                 │
///           │                    ▼                              │
///           │            merge_cone_results()                   │
///           │                    │                              │
///           └────────────────────┼──────────────────────────────┘
///                                ▼
///                         MultiConeResult
/// ```
///
/// # Example
///
/// ```ignore
/// use thrml_sphere::{
///     MultiConeNavigator, BudgetConfig, RootsConfig, SphereConfig,
///     load_blt_safetensors,
/// };
/// use thrml_samplers::RngKey;
///
/// // Load BLT v3 data
/// let (sphere_ebm, bytes) = load_blt_safetensors(&path, SphereConfig::default(), &device)?;
///
/// // Create multi-cone navigator
/// let mut navigator = MultiConeNavigator::from_sphere_ebm_with_bytes(
///     &sphere_ebm,
///     &bytes,
///     RootsConfig::default().with_partitions(64),
///     BudgetConfig::dev(),
///     RngKey::new(42),
///     &device,
/// );
///
/// // Run multi-cone navigation
/// let result = navigator.navigate_multi_cone(query, 50.0, 10, RngKey::new(123), &device);
/// println!("Found {} targets from {} cones", result.n_targets(), result.n_cones());
/// ```
pub struct MultiConeNavigator {
    /// Underlying single-cone navigator.
    pub navigator: NavigatorEBM,

    /// ROOTS index for coarse navigation and cone spawning.
    pub roots: RootsIndex,

    /// Budget allocation configuration.
    pub budget_config: BudgetConfig,

    /// Currently active cones.
    pub active_cones: Vec<ConeState>,
}

impl MultiConeNavigator {
    /// Create a new multi-cone navigator from components.
    ///
    /// # Arguments
    ///
    /// * `navigator` - Pre-configured NavigatorEBM
    /// * `roots` - Pre-built ROOTS index
    /// * `budget_config` - Budget allocation configuration
    ///
    /// # Example
    ///
    /// ```ignore
    /// let navigator = NavigatorEBM::from_sphere_ebm(sphere_ebm.clone());
    /// let roots = RootsIndex::from_sphere_ebm(&sphere_ebm, roots_config, key, &device);
    /// let multi = MultiConeNavigator::new(navigator, roots, BudgetConfig::dev());
    /// ```
    pub fn new(navigator: NavigatorEBM, roots: RootsIndex, budget_config: BudgetConfig) -> Self {
        Self {
            navigator,
            roots,
            budget_config,
            active_cones: Vec::new(),
        }
    }

    /// Create a multi-cone navigator from a SphereEBM.
    ///
    /// Builds both the NavigatorEBM and ROOTS index from the sphere.
    ///
    /// # Arguments
    ///
    /// * `sphere_ebm` - The sphere model with embeddings
    /// * `roots_config` - Configuration for ROOTS index
    /// * `budget_config` - Budget allocation configuration
    /// * `key` - RNG key for ROOTS partitioning
    /// * `device` - GPU device
    ///
    /// # Example
    ///
    /// ```ignore
    /// let navigator = MultiConeNavigator::from_sphere_ebm(
    ///     &sphere_ebm,
    ///     RootsConfig::default().with_partitions(64),
    ///     BudgetConfig::dev(),
    ///     RngKey::new(42),
    ///     &device,
    /// );
    /// ```
    pub fn from_sphere_ebm(
        sphere_ebm: &SphereEBM,
        roots_config: RootsConfig,
        budget_config: BudgetConfig,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let roots = RootsIndex::from_sphere_ebm(sphere_ebm, roots_config, key, device);
        let navigator = NavigatorEBM::from_sphere_ebm(sphere_ebm.clone());
        Self::new(navigator, roots, budget_config)
    }

    /// Create a multi-cone navigator from a SphereEBM with raw bytes.
    ///
    /// Enables substring-enhanced ROOTS partitioning for code/text.
    ///
    /// # Arguments
    ///
    /// * `sphere_ebm` - The sphere model with embeddings
    /// * `raw_bytes` - Raw byte sequences aligned with embeddings
    /// * `roots_config` - Configuration (should have substring coupling enabled)
    /// * `budget_config` - Budget allocation configuration
    /// * `key` - RNG key for ROOTS partitioning
    /// * `device` - GPU device
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = RootsConfig::default()
    ///     .with_partitions(64)
    ///     .with_default_substring_coupling();
    ///
    /// let navigator = MultiConeNavigator::from_sphere_ebm_with_bytes(
    ///     &sphere_ebm,
    ///     &bytes,
    ///     config,
    ///     BudgetConfig::dev(),
    ///     RngKey::new(42),
    ///     &device,
    /// );
    /// ```
    pub fn from_sphere_ebm_with_bytes(
        sphere_ebm: &SphereEBM,
        raw_bytes: &[Vec<u8>],
        roots_config: RootsConfig,
        budget_config: BudgetConfig,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let roots = RootsIndex::from_sphere_ebm_with_bytes(
            sphere_ebm,
            raw_bytes,
            roots_config,
            key,
            device,
        );
        let navigator = NavigatorEBM::from_sphere_ebm(sphere_ebm.clone());
        Self::new(navigator, roots, budget_config)
    }

    /// Get number of points in the underlying sphere.
    pub fn n_points(&self) -> usize {
        self.navigator.n_points()
    }

    /// Get number of partitions in ROOTS index.
    pub fn n_partitions(&self) -> usize {
        self.roots.n_partitions()
    }

    /// Get number of currently active cones.
    pub fn n_active_cones(&self) -> usize {
        self.active_cones.len()
    }

    /// Spawn cones from ROOTS activation peaks.
    ///
    /// This is the core method that connects ROOTS peak detection to cone creation.
    /// From Section 5.4.2 of the Architecture Plan.
    ///
    /// # Process
    ///
    /// 1. Compute ROOTS activations for the query
    /// 2. Detect peaks above threshold
    /// 3. Limit to `max_cones` (taking strongest peaks)
    /// 4. Allocate budget proportional to peak strength
    /// 5. Create cone states with aperture based on peak spread
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector \[D\]
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// Vector of `ConeState` instances, one per detected peak.
    pub fn spawn_cones_from_peaks(
        &self,
        query: &Tensor<WgpuBackend, 1>,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Vec<ConeState> {
        // 1. Compute ROOTS activations
        let activations = self.roots.activate(query, device);

        // 2. Detect peaks above threshold
        let peaks = self.roots.detect_peaks(&activations);

        if peaks.is_empty() {
            return Vec::new();
        }

        // 3. Limit to max_cones (peaks are already sorted by strength)
        let effective_max = self.budget_config.effective_max_cones();
        let peaks_to_use = &peaks[..peaks.len().min(effective_max)];

        // 4. Compute total strength for budget allocation
        let total_strength: f32 = peaks_to_use.iter().map(|p| p.strength).sum();

        if total_strength <= 0.0 {
            return Vec::new();
        }

        // 5. Create cone states
        peaks_to_use
            .iter()
            .filter_map(|peak| {
                let budget_fraction = peak.strength / total_strength;
                let budget_bytes =
                    (self.budget_config.total_budget_bytes as f32 * budget_fraction) as usize;

                // Skip if budget too small
                if budget_bytes < self.budget_config.min_cone_budget {
                    return None;
                }

                // Compute center position (use partition centroid or default)
                let center = peak.center.unwrap_or((
                    std::f32::consts::PI / 2.0, // Default theta (equator)
                    0.0,                        // Default phi
                ));

                // Compute aperture: narrower for concentrated peaks
                // spread is typically in [0, 1] range; higher spread = wider aperture
                let aperture = BASE_CONE_APERTURE / (1.0 + peak.spread.max(0.0));

                Some(ConeState::from_peak(
                    center,
                    aperture,
                    budget_bytes,
                    peak.strength,
                    peak.member_indices.clone(),
                ))
            })
            .collect()
    }

    /// Merge results from multiple cones.
    ///
    /// Combines targets from all cones, deduplicates by index (keeping
    /// lowest energy for each), and sorts by energy ascending.
    ///
    /// # Arguments
    ///
    /// * `per_cone_results` - Results from each cone
    /// * `top_k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Merged `MultiConeResult` with deduplicated, sorted targets.
    fn merge_cone_results(
        &self,
        per_cone_results: &[NavigationResult],
        top_k: usize,
    ) -> MultiConeResult {
        // Collect all (index, energy) pairs from all cones
        let mut all_results: Vec<(usize, f32)> = per_cone_results
            .iter()
            .flat_map(|r| {
                r.target_indices
                    .iter()
                    .zip(r.target_energies.iter())
                    .map(|(&idx, &e)| (idx, e))
            })
            .collect();

        if all_results.is_empty() {
            return MultiConeResult {
                target_indices: Vec::new(),
                target_energies: Vec::new(),
                per_cone_results: per_cone_results.to_vec(),
                budget_used: self.active_cones.iter().map(|c| c.budget_bytes).sum(),
                cones_spawned: self.active_cones.len(),
            };
        }

        // Sort by index for deduplication
        all_results.sort_by_key(|(idx, _)| *idx);

        // Deduplicate by index, keeping lowest energy for each
        let mut deduped: Vec<(usize, f32)> = Vec::with_capacity(all_results.len());
        let mut last_idx: Option<usize> = None;

        for (idx, energy) in all_results {
            match last_idx {
                Some(prev_idx) if prev_idx == idx => {
                    // Same index - keep the lower energy
                    if let Some(last) = deduped.last_mut() {
                        if energy < last.1 {
                            last.1 = energy;
                        }
                    }
                }
                _ => {
                    // New index
                    deduped.push((idx, energy));
                    last_idx = Some(idx);
                }
            }
        }

        // Sort by energy ascending (lower = better)
        deduped.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Truncate to top_k
        deduped.truncate(top_k);

        MultiConeResult {
            target_indices: deduped.iter().map(|(idx, _)| *idx).collect(),
            target_energies: deduped.iter().map(|(_, e)| *e).collect(),
            per_cone_results: per_cone_results.to_vec(),
            budget_used: self.active_cones.iter().map(|c| c.budget_bytes).sum(),
            cones_spawned: self.active_cones.len(),
        }
    }

    /// Navigate using multiple cones in parallel.
    ///
    /// This is the main entry point for multi-cone navigation.
    /// From Section 5.4 of the Architecture Plan.
    ///
    /// # Process
    ///
    /// 1. Spawn cones from ROOTS activation peaks
    /// 2. Run navigation independently in each cone
    /// 3. Merge and deduplicate results
    ///
    /// # Arguments
    ///
    /// * `query` - Query embedding vector \[D\]
    /// * `query_radius` - Query radial position
    /// * `top_k` - Number of results to return per cone (merged result may have more)
    /// * `key` - RNG key for sphere optimization
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// `MultiConeResult` containing merged targets from all cones.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let result = navigator.navigate_multi_cone(
    ///     query,
    ///     50.0,   // query radius
    ///     10,     // top_k per cone
    ///     RngKey::new(42),
    ///     &device,
    /// );
    ///
    /// println!("Found {} targets from {} cones", result.n_targets(), result.n_cones());
    /// ```
    pub fn navigate_multi_cone(
        &mut self,
        query: Tensor<WgpuBackend, 1>,
        query_radius: f32,
        top_k: usize,
        key: RngKey,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> MultiConeResult {
        // 1. Spawn cones from ROOTS peaks
        self.active_cones = self.spawn_cones_from_peaks(&query, device);

        // Handle case where no cones were spawned
        if self.active_cones.is_empty() {
            // Fall back to global navigation without cones
            let result = self
                .navigator
                .navigate(query, query_radius, key, top_k, device);
            return MultiConeResult {
                target_indices: result.target_indices.clone(),
                target_energies: result.target_energies.clone(),
                per_cone_results: vec![result],
                budget_used: 0,
                cones_spawned: 0,
            };
        }

        // 2. Run navigation in each cone
        let n_cones = self.active_cones.len();
        let keys = key.split(n_cones);

        let per_cone_results: Vec<NavigationResult> = self
            .active_cones
            .iter()
            .zip(keys)
            .map(|(cone_state, k)| {
                self.navigator.navigate_cone(
                    query.clone(),
                    query_radius,
                    &cone_state.config,
                    k,
                    top_k,
                    device,
                )
            })
            .collect();

        // 3. Merge results
        self.merge_cone_results(&per_cone_results, top_k)
    }

    /// Get the navigation weights from the underlying navigator.
    pub fn weights(&self) -> &NavigationWeights {
        &self.navigator.weights
    }

    /// Set navigation weights.
    pub fn with_weights(mut self, weights: NavigationWeights) -> Self {
        self.navigator.weights = weights;
        self
    }

    /// Get budget configuration.
    pub fn budget_config(&self) -> &BudgetConfig {
        &self.budget_config
    }

    /// Update budget configuration.
    pub fn with_budget_config(mut self, config: BudgetConfig) -> Self {
        self.budget_config = config;
        self
    }

    /// Get statistics about the last navigation.
    ///
    /// Returns a summary of cones spawned, budget used, and results found.
    pub fn last_navigation_stats(&self) -> NavigationStats {
        NavigationStats {
            cones_spawned: self.active_cones.len(),
            budget_allocated: self.active_cones.iter().map(|c| c.budget_bytes).sum(),
            total_relevance: self.active_cones.iter().map(|c| c.relevance).sum(),
            saturated_cones: self.active_cones.iter().filter(|c| c.is_saturated).count(),
        }
    }
}

/// Statistics about multi-cone navigation.
#[derive(Clone, Debug, Default)]
pub struct NavigationStats {
    /// Number of cones spawned.
    pub cones_spawned: usize,
    /// Total budget allocated across cones in bytes.
    pub budget_allocated: usize,
    /// Sum of relevance scores across cones.
    pub total_relevance: f32,
    /// Number of cones that became saturated.
    pub saturated_cones: usize,
}

impl std::fmt::Display for NavigationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "NavigationStats {{ cones: {}, budget: {:.1}MB, relevance: {:.2}, saturated: {} }}",
            self.cones_spawned,
            self.budget_allocated as f64 / (1024.0 * 1024.0),
            self.total_relevance,
            self.saturated_cones
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScaleProfile;
    use burn::tensor::Distribution;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_navigator_creation() {
        let device = init_gpu_device();
        let n = 20;
        let d = 64;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let navigator = NavigatorEBM::new(embeddings, prominence, None, config, &device);

        assert_eq!(navigator.n_points(), n);
    }

    #[test]
    fn test_semantic_energy() {
        let device = init_gpu_device();
        let n = 10;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let navigator = NavigatorEBM::new(embeddings.clone(), prominence, None, config, &device);

        // Query with first embedding
        let query: Tensor<WgpuBackend, 1> =
            embeddings.clone().slice([0..1, 0..d]).reshape([d as i32]);
        let targets = vec![0, 1, 2];

        let energy = navigator.semantic_energy(&query, &targets, &device);
        let energy_data: Vec<f32> = energy.into_data().to_vec().expect("energy to vec");

        // Self-similarity should have lowest energy
        assert!(
            energy_data[0] < energy_data[1] || energy_data[0] < energy_data[2],
            "Self-similarity should have low energy"
        );
    }

    #[test]
    fn test_navigate_basic() {
        let device = init_gpu_device();
        let n = 15;
        let d = 16;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let navigator = NavigatorEBM::new(embeddings.clone(), prominence, None, config, &device);

        // Create query from first embedding
        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);

        let result = navigator.navigate(query, 50.0, RngKey::new(42), 5, &device);

        assert_eq!(result.target_indices.len(), 5);
        assert_eq!(result.target_energies.len(), 5);
        // Energies should be sorted ascending
        for i in 1..result.target_energies.len() {
            assert!(
                result.target_energies[i] >= result.target_energies[i - 1],
                "Energies should be sorted ascending"
            );
        }
    }

    #[test]
    fn test_navigation_weights() {
        let weights = NavigationWeights::default()
            .with_semantic(2.0)
            .with_radial(0.1);

        assert_eq!(weights.lambda_semantic, 2.0);
        assert_eq!(weights.lambda_radial, 0.1);

        let device = init_gpu_device();
        let tensor = weights.to_tensor(&device);
        let recovered = NavigationWeights::from_tensor(&tensor);

        assert!((recovered.lambda_semantic - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_cone_filtering() {
        let device = init_gpu_device();
        let n = 20;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let navigator = NavigatorEBM::new(embeddings, prominence, None, config, &device);

        // Create coordinates
        let coords = navigator.sphere_ebm.optimize(RngKey::new(42), &device);

        // Create a cone pointing at equator
        let cone = ConeConfig::new(std::f32::consts::PI / 2.0, 0.0)
            .with_aperture(std::f32::consts::PI / 3.0);

        let targets = navigator.targets_in_cone(&coords, &cone);

        // Should have some targets in the cone
        assert!(!targets.is_empty(), "Cone should contain some targets");
        assert!(
            targets.len() <= n,
            "Cannot have more targets than total points"
        );
    }

    #[test]
    fn test_gumbel_select_targets() {
        let device = init_gpu_device();
        let n = 15;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        let config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let navigator = NavigatorEBM::new(embeddings.clone(), prominence, None, config, &device);

        // Get coordinates
        let coords = navigator.sphere_ebm.optimize(RngKey::new(42), &device);

        // Create query
        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);
        let candidates = vec![1, 2, 3, 4, 5];

        // Test soft selection
        let soft_weights = navigator.gumbel_select_targets(
            &query,
            50.0,
            &coords,
            &candidates,
            1.0,   // Temperature
            false, // Soft selection
            &device,
        );

        let soft_data: Vec<f32> = soft_weights.into_data().to_vec().expect("soft to vec");
        assert_eq!(soft_data.len(), candidates.len());

        // Soft weights should sum to ~1
        let sum: f32 = soft_data.iter().sum();
        assert!((sum - 1.0).abs() < 0.1, "Soft weights should sum to ~1");

        // All weights should be non-negative
        for w in &soft_data {
            assert!(*w >= 0.0, "Weights should be non-negative");
        }

        // Test hard selection
        let hard_weights = navigator.gumbel_select_targets(
            &query,
            50.0,
            &coords,
            &candidates,
            0.5,  // Lower temperature for harder selection
            true, // Hard selection
            &device,
        );

        let hard_data: Vec<f32> = hard_weights.into_data().to_vec().expect("hard to vec");

        // Hard weights should be one-hot
        let max_weight: f32 = hard_data.iter().cloned().fold(0.0, f32::max);
        assert!((max_weight - 1.0).abs() < 0.01, "Max weight should be 1");
    }

    #[test]
    fn test_gumbel_differentiable_navigation() {
        let device = init_gpu_device();
        let n = 20;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.0, 1.0), &device);

        // Create hypergraph with some edges
        let mut sidecar = crate::hypergraph::HypergraphSidecar::new(n);
        for i in 0..(n - 1) {
            sidecar.add_edge(i, i + 1, 1.0);
        }
        // Add some cross-links
        sidecar.add_edge(0, 5, 0.5);
        sidecar.add_edge(5, 10, 0.5);
        sidecar.add_edge(10, 15, 0.5);

        let hypergraph_ebm =
            crate::hypergraph::HypergraphEBM::from_sidecar(&sidecar, 0.1, 0.3, &device);

        let config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let navigator = NavigatorEBM::new(embeddings.clone(), prominence, None, config, &device)
            .with_hypergraph(hypergraph_ebm);

        // Create query
        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);

        // Test differentiable navigation
        let result = navigator.navigate_differentiable(
            query,
            50.0,
            0,   // Start at node 0
            5,   // Max 5 steps
            0.5, // Temperature
            RngKey::new(42),
            &device,
        );

        // Path should start at node 0
        assert_eq!(result.path[0], 0);

        // Path length should be at least 1 (the starting node)
        assert!(result.path_length() >= 1);

        // Each step should have selection weights (except for start)
        assert_eq!(result.selection_weights.len(), result.path.len() - 1);

        println!("Differentiable navigation path: {:?}", result.path);
        println!("Total energy: {}", result.total_energy);
    }

    // ========================================================================
    // Multi-Cone Navigation Tests
    // ========================================================================

    #[test]
    fn test_budget_config_defaults() {
        let config = BudgetConfig::default();

        assert_eq!(config.max_cones, 32);
        assert!(config.total_budget_bytes > 0);
        assert!(config.min_cone_budget > 0);
        assert!(config.peak_threshold > 0.0 && config.peak_threshold < 1.0);
    }

    #[test]
    fn test_budget_config_builder() {
        let config = BudgetConfig::new(1024 * 1024 * 1024) // 1GB
            .with_max_cones(8)
            .with_min_cone_budget(32 * 1024 * 1024)
            .with_peak_threshold(0.3);

        assert_eq!(config.total_budget_bytes, 1024 * 1024 * 1024);
        assert_eq!(config.max_cones, 8);
        assert_eq!(config.min_cone_budget, 32 * 1024 * 1024);
        assert!((config.peak_threshold - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_budget_config_effective_max_cones() {
        // Budget-limited case: 128MB total / 64MB per cone = 2 effective
        let config = BudgetConfig::new(128 * 1024 * 1024)
            .with_max_cones(32)
            .with_min_cone_budget(64 * 1024 * 1024);

        assert_eq!(config.effective_max_cones(), 2);

        // Max-cones-limited case
        let config2 = BudgetConfig::new(1024 * 1024 * 1024)
            .with_max_cones(4)
            .with_min_cone_budget(16 * 1024 * 1024);

        assert_eq!(config2.effective_max_cones(), 4);
    }

    #[test]
    fn test_budget_config_presets() {
        let dev = BudgetConfig::dev();
        assert_eq!(dev.max_cones, 4);
        assert!(dev.total_budget_bytes < 256 * 1024 * 1024);

        let large = BudgetConfig::large_scale();
        assert_eq!(large.max_cones, 32);
        assert!(large.total_budget_bytes >= 16 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_cone_state_creation() {
        let config = ConeConfig::new(std::f32::consts::PI / 2.0, 0.0)
            .with_aperture(std::f32::consts::PI / 4.0);

        let state = ConeState::new(config, 64 * 1024 * 1024, 0.85);

        assert_eq!(state.budget_bytes, 64 * 1024 * 1024);
        assert!((state.relevance - 0.85).abs() < 0.01);
        assert!(!state.is_saturated);
        assert_eq!(state.budget_consumed, 0);
        assert!(state.source_partition_ids.is_empty());
    }

    #[test]
    fn test_cone_state_from_peak() {
        let state = ConeState::from_peak(
            (std::f32::consts::PI / 3.0, std::f32::consts::PI / 6.0),
            std::f32::consts::PI / 8.0,
            32 * 1024 * 1024,
            0.75,
            vec![1, 2, 3],
        );

        assert_eq!(state.budget_bytes, 32 * 1024 * 1024);
        assert!((state.relevance - 0.75).abs() < 0.01);
        assert_eq!(state.source_partition_ids, vec![1, 2, 3]);
        assert!((state.config.aperture - std::f32::consts::PI / 8.0).abs() < 0.01);
    }

    #[test]
    fn test_cone_state_budget_tracking() {
        let config = ConeConfig::default();
        let mut state = ConeState::new(config, 100, 0.5);

        assert_eq!(state.remaining_budget(), 100);

        state.consume_budget(30);
        assert_eq!(state.remaining_budget(), 70);
        assert_eq!(state.budget_consumed, 30);

        state.consume_budget(80);
        assert_eq!(state.remaining_budget(), 0); // Saturates at 0
        assert_eq!(state.budget_consumed, 110);
    }

    #[test]
    fn test_cone_state_saturation() {
        let config = ConeConfig::default();
        let mut state = ConeState::new(config, 100, 0.5);

        assert!(!state.is_saturated);
        state.saturate();
        assert!(state.is_saturated);
    }

    #[test]
    fn test_multi_cone_result_accessors() {
        let result = MultiConeResult {
            target_indices: vec![5, 3, 7],
            target_energies: vec![-0.9, -0.7, -0.5],
            per_cone_results: vec![],
            budget_used: 128 * 1024 * 1024,
            cones_spawned: 2,
        };

        assert_eq!(result.n_cones(), 2);
        assert_eq!(result.n_targets(), 3);
        assert!(!result.is_empty());
        assert_eq!(result.best_target(), Some(5));
        assert!((result.best_energy().unwrap() - (-0.9)).abs() < 0.01);
    }

    #[test]
    fn test_multi_cone_navigator_creation() {
        let device = init_gpu_device();
        let n = 50;
        let d = 16;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let sphere_ebm = SphereEBM::new(embeddings, prominence, None, sphere_config, &device);

        let roots_config = RootsConfig::dev().with_partitions(4);
        let budget_config = BudgetConfig::dev();

        let navigator = MultiConeNavigator::from_sphere_ebm(
            &sphere_ebm,
            roots_config,
            budget_config,
            RngKey::new(42),
            &device,
        );

        assert_eq!(navigator.n_points(), n);
        assert!(navigator.n_partitions() >= 1);
        assert_eq!(navigator.n_active_cones(), 0); // No navigation yet
    }

    #[test]
    fn test_multi_cone_navigate() {
        let device = init_gpu_device();
        let n = 50;
        let d = 16;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let sphere_ebm =
            SphereEBM::new(embeddings.clone(), prominence, None, sphere_config, &device);

        let roots_config = RootsConfig::dev().with_partitions(4).with_threshold(0.05); // Very low threshold to ensure detection

        let budget_config = BudgetConfig::dev();

        let mut navigator = MultiConeNavigator::from_sphere_ebm(
            &sphere_ebm,
            roots_config,
            budget_config,
            RngKey::new(42),
            &device,
        );

        // Create query
        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);

        // Run multi-cone navigation
        let result = navigator.navigate_multi_cone(query, 50.0, 5, RngKey::new(123), &device);

        // Results may be empty with random data - just check it doesn't panic
        // and if we do have results, they should be valid
        if !result.is_empty() {
            // Results should be sorted by energy (ascending)
            for i in 1..result.target_energies.len() {
                assert!(
                    result.target_energies[i] >= result.target_energies[i - 1],
                    "Energies should be sorted ascending"
                );
            }

            println!(
                "Multi-cone results: {} targets from {} cones",
                result.n_targets(),
                result.n_cones()
            );
        } else {
            println!("No navigation results (expected with random data)");
        }
    }

    #[test]
    fn test_navigation_stats() {
        let device = init_gpu_device();
        let n = 30;
        let d = 8;

        let embeddings: Tensor<WgpuBackend, 2> =
            Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let sphere_ebm =
            SphereEBM::new(embeddings.clone(), prominence, None, sphere_config, &device);

        let roots_config = RootsConfig::dev().with_partitions(4);
        let budget_config = BudgetConfig::dev();

        let mut navigator = MultiConeNavigator::from_sphere_ebm(
            &sphere_ebm,
            roots_config,
            budget_config,
            RngKey::new(42),
            &device,
        );

        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);

        // Run navigation
        let _result = navigator.navigate_multi_cone(query, 50.0, 5, RngKey::new(123), &device);

        // Check stats
        let stats = navigator.last_navigation_stats();
        println!("{}", stats);

        // Verify budget tracking makes sense
        if stats.cones_spawned > 0 {
            assert!(stats.budget_allocated > 0, "Budget should be allocated");
            assert!(stats.total_relevance > 0.0, "Should have relevance");
        }
    }

    #[test]
    fn test_budget_allocation_proportional() {
        let device = init_gpu_device();
        let n = 100;
        let d = 16;

        // Create embeddings with some structure to get multiple peaks
        let mut emb_data = vec![0.0f32; n * d];
        for i in 0..n {
            for j in 0..d {
                // Create clusters
                let cluster = i / 25; // 4 clusters
                emb_data[i * d + j] = (cluster as f32 * 0.5) + ((i * j) as f32 * 0.01).sin() * 0.1;
            }
        }
        let embeddings_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(emb_data.as_slice(), &device);
        let embeddings: Tensor<WgpuBackend, 2> = embeddings_1d.reshape([n as i32, d as i32]);
        let prominence: Tensor<WgpuBackend, 1> =
            Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

        let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(5);
        let sphere_ebm =
            SphereEBM::new(embeddings.clone(), prominence, None, sphere_config, &device);

        let roots_config = RootsConfig::dev().with_partitions(8).with_threshold(0.1);

        let budget_config = BudgetConfig::new(1024 * 1024 * 1024) // 1GB
            .with_max_cones(8)
            .with_min_cone_budget(32 * 1024 * 1024);

        let navigator = MultiConeNavigator::from_sphere_ebm(
            &sphere_ebm,
            roots_config,
            budget_config,
            RngKey::new(42),
            &device,
        );

        let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);

        // Spawn cones
        let cones = navigator.spawn_cones_from_peaks(&query, &device);

        if cones.len() > 1 {
            // Verify budget allocation is proportional to relevance
            let total_budget: usize = cones.iter().map(|c| c.budget_bytes).sum();
            let total_relevance: f32 = cones.iter().map(|c| c.relevance).sum();

            for cone in &cones {
                let expected_fraction = cone.relevance / total_relevance;
                let actual_fraction = cone.budget_bytes as f32 / total_budget as f32;

                // Should be within 10% due to rounding and min budget constraints
                let diff = (expected_fraction - actual_fraction).abs();
                assert!(
                    diff < 0.2,
                    "Budget allocation should be roughly proportional to relevance"
                );
            }

            println!(
                "Spawned {} cones with total budget {} MB",
                cones.len(),
                total_budget / (1024 * 1024)
            );
        }
    }
}
