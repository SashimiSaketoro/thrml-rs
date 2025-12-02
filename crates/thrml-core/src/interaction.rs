use crate::backend::WgpuBackend;
use crate::block::Block;
use burn::tensor::Tensor;

/// Different types of interaction parameters.
///
/// This enum supports both discrete EBM interactions (standard tensor weights)
/// and continuous variable interactions (linear and quadratic terms).
#[derive(Clone, Debug)]
pub enum InteractionData {
    /// Standard EBM weight tensor [n_nodes, k, dim] for discrete variables.
    /// Used for spin-spin, categorical-categorical, and mixed discrete interactions.
    Tensor(Tensor<WgpuBackend, 3>),

    /// Linear interaction of the form `c_i * x_i` for continuous variables.
    /// The weights tensor has shape [n_nodes, k] where k is the number of interactions per node.
    /// Used for bias terms and coupling with neighbor states.
    Linear { weights: Tensor<WgpuBackend, 2> },

    /// Quadratic interaction of the form `d_i * x_i^2` for continuous variables.
    /// The inverse_weights tensor has shape [n_nodes, k] representing 1/A_{ii} (variance).
    /// Used for the diagonal of the inverse covariance matrix in Gaussian models.
    Quadratic {
        inverse_weights: Tensor<WgpuBackend, 2>,
    },

    /// Sphere-specific interaction for water-filling optimization.
    ///
    /// Used by thrml-sphere for placing embeddings on a hypersphere.
    /// Contains precomputed similarity matrix and ideal radii.
    /// Fields are boxed to reduce enum size variance.
    Sphere {
        /// Ideal radii from prominence ranking \[N\].
        ideal_radii: Box<Tensor<WgpuBackend, 1>>,
        /// Cosine similarity matrix \[N, N\].
        similarity: Box<Tensor<WgpuBackend, 2>>,
        /// Gaussian interaction radius for lateral forces.
        interaction_radius: f32,
    },
}

impl InteractionData {
    /// Get the leading dimension (number of nodes) for this interaction.
    pub fn n_nodes(&self) -> usize {
        match self {
            Self::Tensor(t) => t.dims()[0],
            Self::Linear { weights } => weights.dims()[0],
            Self::Quadratic { inverse_weights } => inverse_weights.dims()[0],
            Self::Sphere { ideal_radii, .. } => ideal_radii.dims()[0],
        }
    }

    /// Check if this is a standard tensor interaction.
    pub const fn is_tensor(&self) -> bool {
        matches!(self, Self::Tensor(_))
    }

    /// Check if this is a linear interaction.
    pub const fn is_linear(&self) -> bool {
        matches!(self, Self::Linear { .. })
    }

    /// Check if this is a quadratic interaction.
    pub const fn is_quadratic(&self) -> bool {
        matches!(self, Self::Quadratic { .. })
    }

    /// Check if this is a sphere interaction.
    pub const fn is_sphere(&self) -> bool {
        matches!(self, Self::Sphere { .. })
    }

    /// Get the underlying tensor if this is a Tensor variant.
    pub const fn as_tensor(&self) -> Option<&Tensor<WgpuBackend, 3>> {
        match self {
            Self::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Get sphere interaction data if this is a Sphere variant.
    pub fn as_sphere(&self) -> Option<(&Tensor<WgpuBackend, 1>, &Tensor<WgpuBackend, 2>, f32)> {
        match self {
            Self::Sphere {
                ideal_radii,
                similarity,
                interaction_radius,
            } => Some((
                ideal_radii.as_ref(),
                similarity.as_ref(),
                *interaction_radius,
            )),
            _ => None,
        }
    }
}

/// Defines computational dependencies for conditional sampling updates.
///
/// An `InteractionGroup` specifies information that is required to update the state of some subset
/// of the nodes of a PGM during a block sampling routine.
///
/// The `interaction` field stores weights/parameters in one of several formats:
/// - `InteractionData::Tensor`: Standard [n_nodes, dim1, dim2] for discrete EBMs
/// - `InteractionData::Linear`: [n_nodes, k] weights for linear terms
/// - `InteractionData::Quadratic`: [n_nodes, k] inverse weights for quadratic terms
pub struct InteractionGroup {
    /// The nodes whose conditional updates should be affected by this InteractionGroup.
    pub head_nodes: Block,
    /// The nodes whose state information is required to update `head_nodes`.
    pub tail_nodes: Vec<Block>,
    /// The static information associated with the interaction.
    /// Can be a tensor, linear weights, or quadratic inverse weights.
    pub interaction: InteractionData,
    /// The number of spin (binary) tail blocks. The first n_spin tail blocks are spin-type,
    /// and the remaining are categorical-type. This is needed for correct state splitting
    /// in samplers.
    pub n_spin: usize,
}

impl InteractionGroup {
    /// Create a new InteractionGroup with a standard tensor interaction.
    ///
    /// - `interaction`: Weight tensor with shape [n_nodes, dim1, dim2]
    /// - `head_nodes`: Block of nodes to update
    /// - `tail_nodes`: List of blocks providing neighbor states
    /// - `n_spin`: How many of the tail blocks are spin-type (first n_spin are spin, rest are categorical)
    pub fn new(
        interaction: Tensor<WgpuBackend, 3>,
        head_nodes: Block,
        tail_nodes: Vec<Block>,
        n_spin: usize,
    ) -> Result<Self, String> {
        Self::with_data(
            InteractionData::Tensor(interaction),
            head_nodes,
            tail_nodes,
            n_spin,
        )
    }

    /// Create a new InteractionGroup with arbitrary interaction data.
    ///
    /// - `interaction`: InteractionData (Tensor, Linear, or Quadratic)
    /// - `head_nodes`: Block of nodes to update
    /// - `tail_nodes`: List of blocks providing neighbor states
    /// - `n_spin`: How many of the tail blocks are spin-type
    pub fn with_data(
        interaction: InteractionData,
        head_nodes: Block,
        tail_nodes: Vec<Block>,
        n_spin: usize,
    ) -> Result<Self, String> {
        let interaction_size = head_nodes.len();

        for block in &tail_nodes {
            if block.len() != interaction_size {
                return Err(
                    "All tail node blocks must have the same length as head_nodes".to_string(),
                );
            }
        }

        // Verify interaction has correct leading dimension
        if interaction.n_nodes() != interaction_size {
            return Err(
                "All arrays in interaction must have leading dimension equal to the length of head_nodes".to_string()
            );
        }

        if n_spin > tail_nodes.len() {
            return Err("n_spin cannot exceed number of tail blocks".to_string());
        }

        Ok(Self {
            head_nodes,
            tail_nodes,
            interaction,
            n_spin,
        })
    }

    /// Create a linear interaction group for continuous variables.
    ///
    /// - `weights`: Weight tensor with shape [n_nodes, k]
    /// - `head_nodes`: Block of nodes to update
    /// - `tail_nodes`: List of blocks providing neighbor states
    pub fn linear(
        weights: Tensor<WgpuBackend, 2>,
        head_nodes: Block,
        tail_nodes: Vec<Block>,
    ) -> Result<Self, String> {
        // Linear interactions don't distinguish spin vs categorical tail nodes
        // since they're for continuous models
        Self::with_data(
            InteractionData::Linear { weights },
            head_nodes,
            tail_nodes,
            0,
        )
    }

    /// Create a quadratic interaction group for continuous variables.
    ///
    /// - `inverse_weights`: Inverse weight tensor with shape [n_nodes, k] (1/variance)
    /// - `head_nodes`: Block of nodes to update
    pub fn quadratic(
        inverse_weights: Tensor<WgpuBackend, 2>,
        head_nodes: Block,
    ) -> Result<Self, String> {
        // Quadratic interactions have no tail nodes (self-interaction only)
        Self::with_data(
            InteractionData::Quadratic { inverse_weights },
            head_nodes,
            vec![],
            0,
        )
    }
}
