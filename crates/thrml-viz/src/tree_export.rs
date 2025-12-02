//! Tree visualization data export.
//!
//! Converts ROOTS hierarchy to 3D visualization data.

/// Node data for visualization.
#[derive(Clone, Debug)]
pub struct VizNode {
    /// Position in 3D space
    pub position: [f32; 3],

    /// Visual radius (proportional to point count)
    pub radius: f32,

    /// Prominence value (for coloring)
    pub prominence: f32,

    /// Whether this is a leaf node
    pub is_leaf: bool,

    /// Node ID (partition ID for leaves)
    pub id: usize,
}

impl VizNode {
    pub fn new(position: [f32; 3], radius: f32, prominence: f32, is_leaf: bool, id: usize) -> Self {
        Self {
            position,
            radius,
            prominence,
            is_leaf,
            id,
        }
    }
}

/// Tree visualization data.
#[derive(Clone, Debug, Default)]
pub struct TreeVizData {
    /// All nodes (internal + leaves)
    pub nodes: Vec<VizNode>,

    /// Edges as (parent_idx, child_idx) pairs
    pub edges: Vec<(usize, usize)>,
}

impl TreeVizData {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if there's data to render.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of edges.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get leaf nodes only.
    pub fn leaves(&self) -> impl Iterator<Item = &VizNode> {
        self.nodes.iter().filter(|n| n.is_leaf)
    }

    /// Get internal nodes only.
    pub fn internal_nodes(&self) -> impl Iterator<Item = &VizNode> {
        self.nodes.iter().filter(|n| !n.is_leaf)
    }

    /// Compute bounding box center and radius.
    pub fn bounding_sphere(&self) -> ([f32; 3], f32) {
        if self.nodes.is_empty() {
            return ([0.0, 0.0, 0.0], 100.0);
        }

        // Find center
        let mut center = [0.0f32; 3];
        for node in &self.nodes {
            center[0] += node.position[0];
            center[1] += node.position[1];
            center[2] += node.position[2];
        }
        let n = self.nodes.len() as f32;
        center[0] /= n;
        center[1] /= n;
        center[2] /= n;

        // Find max distance from center
        let mut max_dist = 0.0f32;
        for node in &self.nodes {
            let dx = node.position[0] - center[0];
            let dy = node.position[1] - center[1];
            let dz = node.position[2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            max_dist = max_dist.max(dist);
        }

        (center, max_dist.max(100.0))
    }

    /// Generate line vertices for edges (for rendering).
    pub fn edge_vertices(&self) -> Vec<[f32; 3]> {
        let mut vertices = Vec::with_capacity(self.edges.len() * 2);
        for (parent_idx, child_idx) in &self.edges {
            if let (Some(parent), Some(child)) = (
                self.nodes.get(*parent_idx),
                self.nodes.get(*child_idx),
            ) {
                vertices.push(parent.position);
                vertices.push(child.position);
            }
        }
        vertices
    }
}

/// Node data for tree export (simple struct-based approach).
#[derive(Clone, Debug)]
pub struct ExportNode {
    pub centroid: Vec<f32>,
    pub point_count: usize,
    pub prominence_range: (f32, f32),
    pub is_leaf: bool,
    pub node_id: usize,
    pub children: Option<(Box<ExportNode>, Box<ExportNode>)>,
}

impl ExportNode {
    /// Export this node and its descendants to visualization data.
    pub fn export_recursive<F>(
        &self,
        parent_idx: Option<usize>,
        data: &mut TreeVizData,
        projector: &F,
        scale: f32,
    ) where
        F: Fn(&[f32]) -> [f32; 3],
    {
        // Project centroid to 3D
        let position = projector(&self.centroid);
        let position = [
            position[0] * scale,
            position[1] * scale,
            position[2] * scale,
        ];

        // Compute visual radius from point count
        let radius = (self.point_count as f32).ln().max(1.0) * 2.0;

        // Get mean prominence
        let (prom_min, prom_max) = self.prominence_range;
        let prominence = (prom_min + prom_max) / 2.0;

        // Create node
        let node_idx = data.nodes.len();
        data.nodes.push(VizNode::new(
            position,
            radius,
            prominence,
            self.is_leaf,
            self.node_id,
        ));

        // Add edge from parent
        if let Some(parent) = parent_idx {
            data.edges.push((parent, node_idx));
        }

        // Recurse into children
        if let Some((ref left, ref right)) = self.children {
            left.export_recursive(Some(node_idx), data, projector, scale);
            right.export_recursive(Some(node_idx), data, projector, scale);
        }
    }
}

/// Export a tree to visualization data.
pub fn export_tree<F>(tree: &ExportNode, projector: F, scale: f32) -> TreeVizData
where
    F: Fn(&[f32]) -> [f32; 3],
{
    let mut data = TreeVizData::new();
    tree.export_recursive(None, &mut data, &projector, scale);
    data
}

/// Simple 3D projector that uses first 3 dimensions.
pub fn first_3_dims(centroid: &[f32]) -> [f32; 3] {
    let mut pos = [0.0f32; 3];
    for (i, &v) in centroid.iter().enumerate().take(3) {
        pos[i] = v;
    }
    pos
}

/// Spherical projector using mean theta/phi.
pub fn spherical_projector(r: f32) -> impl Fn(&[f32]) -> [f32; 3] {
    move |centroid: &[f32]| {
        // Assume centroid is [theta, phi, ...] or use first 2 dims as angles
        if centroid.len() >= 2 {
            let theta = centroid[0];
            let phi = centroid[1];
            [
                r * theta.sin() * phi.cos(),
                r * theta.sin() * phi.sin(),
                r * theta.cos(),
            ]
        } else {
            [0.0, 0.0, r]
        }
    }
}
