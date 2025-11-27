use burn::tensor::DType;
use once_cell::sync::Lazy;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// Binary spin variable with values {0, 1} (representing {-1, +1})
    Spin,
    /// Categorical variable with n possible values {0, 1, ..., n-1}
    Categorical { n_categories: u8 },
    /// Continuous variable with float32 state
    Continuous,
    /// Spherical coordinate node (r, theta, phi) for sphere optimization.
    /// Used by thrml-sphere for placing embeddings on a hypersphere.
    Spherical {
        min_radius: OrderedFloat<f32>,
        max_radius: OrderedFloat<f32>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Node {
    id: usize, // Assigned by IndexSet insertion order
    node_type: NodeType,
}

// Global counter for unique node IDs (replaces Python's _counter)
static NODE_COUNTER: Lazy<Mutex<usize>> = Lazy::new(|| Mutex::new(0));

impl Node {
    pub fn new(node_type: NodeType) -> Self {
        let mut counter = NODE_COUNTER.lock().unwrap();
        let id = *counter;
        *counter += 1;
        Node { id, node_type }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn node_type(&self) -> &NodeType {
        &self.node_type
    }
}

// TensorSpec (replaces jax.ShapeDtypeStruct)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorSpec {
    pub fn for_spin() -> Self {
        Self {
            shape: vec![],
            dtype: DType::Bool,
        }
    }

    pub fn for_categorical(_n_categories: u8) -> Self {
        Self {
            shape: vec![],
            dtype: DType::U8,
        }
    }

    /// TensorSpec for continuous float32 variables
    pub fn for_continuous() -> Self {
        Self {
            shape: vec![],
            dtype: DType::F32,
        }
    }

    /// TensorSpec for spherical coordinate variables (r, theta, phi)
    pub fn for_spherical() -> Self {
        Self {
            shape: vec![3], // r, theta, phi
            dtype: DType::F32,
        }
    }
}
