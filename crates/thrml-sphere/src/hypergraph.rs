//! Re-export of graph utilities from thrml-models, plus sphere-specific extensions.
//!
//! For hypergraph connectivity in sphere optimization, use:
//!
//! - `GraphSidecar`: General graph structure (re-exported as [`HypergraphSidecar`])
//! - `GraphEdge`: Edge definition (re-exported as [`HypergraphEdge`])
//! - [`SpringEBM`]: Spring forces for connected nodes
//!
//! Plus sphere-specific extensions:
//!
//! - [`HypergraphEBM`]: Sphere-specific EBM with coherence support

use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
use thrml_core::SphericalCoords;

// Re-export from thrml-models with sphere-friendly names
pub use thrml_models::graph_ebm::{
    GraphEdge as HypergraphEdge, GraphSidecar as HypergraphSidecar, NodeBiasEBM, SpringEBM,
};

/// Sphere-specific hypergraph EBM with coherence support.
///
/// Extends the general [`SpringEBM`] with:
/// - Coherence energy term (bias high-coherence nodes toward sphere core)
/// - Spherical coordinate support
pub struct HypergraphEBM {
    /// Underlying spring EBM
    pub spring_ebm: SpringEBM,
    /// Coherence scores per node \[N\]
    pub coherence: Option<Tensor<WgpuBackend, 1>>,
    /// Coherence weighting factor
    pub coherence_weight: f32,
}

impl HypergraphEBM {
    /// Create from graph sidecar.
    pub fn from_sidecar(
        sidecar: &HypergraphSidecar,
        spring_constant: f32,
        coherence_weight: f32,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Self {
        let spring_ebm = SpringEBM::from_graph(sidecar, spring_constant, device);
        let coherence = sidecar.node_weights_tensor(device);

        Self {
            spring_ebm,
            coherence,
            coherence_weight,
        }
    }

    /// Create with just adjacency matrix.
    pub fn new(adjacency: Tensor<WgpuBackend, 2>, spring_constant: f32) -> Self {
        Self {
            spring_ebm: SpringEBM::new(adjacency, spring_constant),
            coherence: None,
            coherence_weight: 0.0,
        }
    }

    /// Set coherence scores.
    pub fn with_coherence(mut self, coherence: Tensor<WgpuBackend, 1>, weight: f32) -> Self {
        self.coherence = Some(coherence);
        self.coherence_weight = weight;
        self
    }

    /// Compute spring energy for connected nodes.
    pub fn spring_energy(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 1> {
        let positions = coords.to_cartesian();
        self.spring_ebm.energy(&positions)
    }

    /// Compute coherence energy term.
    ///
    /// Higher coherence nodes should move toward smaller radii.
    /// E_coherence = -coherence * log(r / r_max)
    pub fn coherence_energy(
        &self,
        coords: &SphericalCoords,
        max_radius: f32,
    ) -> Option<Tensor<WgpuBackend, 1>> {
        self.coherence.as_ref().map(|c| {
            let log_ratio = (coords.r.clone() / max_radius).log();
            -c.clone() * log_ratio * self.coherence_weight
        })
    }

    /// Compute spring force in Cartesian coordinates.
    pub fn spring_force(&self, coords: &SphericalCoords) -> Tensor<WgpuBackend, 2> {
        let positions = coords.to_cartesian();
        self.spring_ebm.force(&positions)
    }

    /// Compute coherence force (radial).
    pub fn coherence_force(&self, coords: &SphericalCoords) -> Option<Tensor<WgpuBackend, 1>> {
        self.coherence.as_ref().map(|c| {
            let r_safe = coords.r.clone().clamp(1e-8, f32::MAX);
            c.clone() * self.coherence_weight / r_safe
        })
    }

    /// Total energy contribution from hypergraph terms.
    pub fn total_energy(
        &self,
        coords: &SphericalCoords,
        max_radius: f32,
    ) -> Tensor<WgpuBackend, 1> {
        let spring_e = self.spring_energy(coords);

        if let Some(coherence_e) = self.coherence_energy(coords, max_radius) {
            spring_e + coherence_e
        } else {
            spring_e
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrml_core::backend::init_gpu_device;

    #[test]
    fn test_hypergraph_sidecar_adjacency() {
        let device = init_gpu_device();

        let mut sidecar = HypergraphSidecar::new(4);
        sidecar.add_edge(0, 1, 1.0);
        sidecar.add_edge(1, 2, 0.5);
        sidecar.add_edge(2, 3, 0.8);

        let adj = sidecar.to_symmetric_adjacency(&device);
        let adj_data: Vec<f32> = adj.into_data().to_vec().expect("adj to vec");

        // Check edge 0-1 is symmetric
        assert!((adj_data[0 * 4 + 1] - 1.0).abs() < 1e-6);
        assert!((adj_data[1 * 4 + 0] - 1.0).abs() < 1e-6);
    }
}
