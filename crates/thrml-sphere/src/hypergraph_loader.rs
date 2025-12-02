//! Hypergraph loading from blt-burn SQLite sidecars.
//!
//! This module provides functions for loading hypergraph connectivity data
//! from the SQLite databases produced by `blt-burn` ingestion. The hypergraph
//! encodes structural relationships between patches:
//!
//! - **"next"** edges: Sequential connections between patches (patch_i → patch_{i+1})
//! - **"contains"** edges: Hierarchical containment (Branch → Leaf)
//! - **"same_source"** edges: Cross-view connections for multi-modal data
//! - **"semantic"** edges: High cosine similarity pairs (from ROOTS partitioning)
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use thrml_sphere::{load_hypergraph_from_sqlite, load_blt_with_hypergraph};
//!
//! // Load hypergraph separately
//! let hg_ebm = load_hypergraph_from_sqlite(
//!     Path::new("output.hypergraph.db"),
//!     99,   // n_patches from SafeTensors
//!     0.5,  // spring_constant
//!     0.3,  // coherence_weight
//!     &device,
//! )?;
//!
//! // Or load everything together
//! let (sphere_ebm, bytes, hg_ebm) = load_blt_with_hypergraph(
//!     Path::new("output.safetensors"),
//!     SphereConfig::default(),
//!     0.5,  // spring_constant
//!     0.3,  // coherence_weight
//!     &device,
//! )?;
//! ```
//!
//! # blt-burn SQLite Schema
//!
//! The hypergraph database has three tables:
//!
//! | Table | Contents |
//! |-------|----------|
//! | `meta` | Schema version, sharding info |
//! | `nodes` | Bincode-encoded NodeData (Trunk/Branch/Leaf) |
//! | `hyperedges` | Bincode-encoded edges with vertices and labels |
//!
//! # Edge Types and Spring Physics
//!
//! | Edge Label | Weight | Physics Effect |
//! |------------|--------|----------------|
//! | `"next"` | 1.0 | Strong spring between sequential patches |
//! | `"contains"` | 0.5 | Hierarchical containment (weaker) |
//! | `"same_source"` | 1.0 | Cross-view identity |
//! | `"semantic"` | 0.8 | High-similarity pairs (cosine ≥ 0.7) |

use anyhow::{Context, Result};
use std::path::Path;

// Re-export for convenience
pub use crate::hypergraph::{HypergraphEBM, HypergraphEdge, HypergraphSidecar};

/// Configuration for hypergraph loading.
#[derive(Clone, Debug)]
pub struct HypergraphLoadConfig {
    /// Spring constant for connected patches (higher = stronger attraction).
    /// Typical range: 0.1 to 1.0
    pub spring_constant: f32,

    /// Coherence weighting factor (higher = more bias toward core for coherent patches).
    /// Typical range: 0.1 to 0.5
    pub coherence_weight: f32,

    /// Whether to include "next" (sequential) edges.
    pub include_next_edges: bool,

    /// Whether to include "contains" (hierarchical) edges.
    pub include_contains_edges: bool,

    /// Whether to include "same_source" (cross-view) edges.
    pub include_same_source_edges: bool,

    /// Whether to include "semantic" (high-similarity) edges.
    pub include_semantic_edges: bool,

    /// Weight multiplier for "next" edges.
    pub next_edge_weight: f32,

    /// Weight multiplier for "contains" edges.
    pub contains_edge_weight: f32,

    /// Weight multiplier for "same_source" edges.
    pub same_source_edge_weight: f32,

    /// Weight multiplier for "semantic" edges.
    pub semantic_edge_weight: f32,
}

impl Default for HypergraphLoadConfig {
    fn default() -> Self {
        Self {
            spring_constant: 0.5,
            coherence_weight: 0.3,
            include_next_edges: true,
            include_contains_edges: false, // Usually too many hierarchical edges
            include_same_source_edges: true,
            include_semantic_edges: true, // High-similarity edges from ROOTS
            next_edge_weight: 1.0,
            contains_edge_weight: 0.5,
            same_source_edge_weight: 1.0,
            semantic_edge_weight: 0.8, // Slightly weaker than sequential
        }
    }
}

impl HypergraphLoadConfig {
    /// Create config optimized for code navigation.
    ///
    /// Strong sequential springs to keep related code together.
    pub const fn for_code() -> Self {
        Self {
            spring_constant: 0.7,
            coherence_weight: 0.4,
            include_next_edges: true,
            include_contains_edges: false,
            include_same_source_edges: true,
            include_semantic_edges: true,
            next_edge_weight: 1.0,
            contains_edge_weight: 0.3,
            same_source_edge_weight: 0.8,
            semantic_edge_weight: 0.9, // Strong for code similarity
        }
    }

    /// Create config optimized for text/document navigation.
    ///
    /// Moderate springs with emphasis on cross-view connections.
    pub const fn for_text() -> Self {
        Self {
            spring_constant: 0.4,
            coherence_weight: 0.3,
            include_next_edges: true,
            include_contains_edges: false,
            include_same_source_edges: true,
            include_semantic_edges: true,
            next_edge_weight: 0.8,
            contains_edge_weight: 0.2,
            same_source_edge_weight: 1.0,
            semantic_edge_weight: 0.7, // Moderate for document similarity
        }
    }

    /// Builder: set spring constant.
    pub const fn with_spring_constant(mut self, k: f32) -> Self {
        self.spring_constant = k;
        self
    }

    /// Builder: set coherence weight.
    pub const fn with_coherence_weight(mut self, w: f32) -> Self {
        self.coherence_weight = w;
        self
    }
}

// ============================================================================
// SQLite Loading (feature-gated)
// ============================================================================

#[cfg(feature = "hypergraph")]
mod sqlite_loader {
    use super::*;
    use rusqlite::Connection;
    use serde::{Deserialize, Serialize};

    /// Edge structure matching blt-burn's HypergraphEdge.
    #[derive(Debug, Serialize, Deserialize)]
    pub struct BltHypergraphEdge {
        pub vertices: Vec<u64>,
        pub label: String,
        pub weight: f32,
    }

    /// Node data structure matching blt-burn's NodeData enum.
    ///
    /// We only need to decode enough to identify Leaf nodes and extract
    /// the patch_index from metadata.
    ///
    /// Note: Must NOT use `#[serde(tag = "type")]` because blt-burn uses
    /// bincode's default index-based enum encoding.
    #[derive(Debug, Serialize, Deserialize)]
    pub enum BltNodeData {
        Trunk {
            source_hash: String,
            total_bytes: usize,
        },
        Branch {
            label: String,
            modality: String,
        },
        Leaf(BltByteSegment),
    }

    /// Simplified ByteSegment for deserialization.
    #[derive(Debug, Serialize, Deserialize)]
    pub struct BltByteSegment {
        #[serde(skip)]
        pub bytes: Vec<u8>,
        pub label: Option<String>,
        pub metadata: Option<BltSegmentMetadata>,
    }

    /// Simplified SegmentMetadata for deserialization.
    #[derive(Debug, Serialize, Deserialize)]
    pub struct BltSegmentMetadata {
        pub start_offset: usize,
        pub end_offset: usize,
        pub confidence: f32,
        pub extra: Option<serde_json::Value>,
    }

    /// Load hypergraph edges from a blt-burn SQLite sidecar.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.hypergraph.db` file
    /// * `n_patches` - Number of patches (from SafeTensors `embeddings.shape[0]`)
    /// * `config` - Loading configuration
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// `HypergraphEBM` with springs connecting related patches.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Database cannot be opened
    /// - Required tables are missing
    /// - Bincode deserialization fails
    pub fn load_hypergraph_from_sqlite(
        path: &Path,
        n_patches: usize,
        config: &HypergraphLoadConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Result<HypergraphEBM> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open hypergraph database: {:?}", path))?;

        // Check for schema v3 (sphere-optimized tables)
        let has_sphere_tables = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='sphere_patches'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap_or(0)
            > 0;

        if has_sphere_tables {
            // Fast path: Use sphere-optimized tables (no bincode decode)
            load_from_sphere_tables(&conn, n_patches, config, device)
        } else {
            // Fallback: Use legacy bincode-encoded tables (schema v2)
            load_from_legacy_tables(&conn, n_patches, config, device)
        }
    }

    /// Load from sphere-optimized tables (schema v3+, 7x faster)
    fn load_from_sphere_tables(
        conn: &Connection,
        n_patches: usize,
        config: &HypergraphLoadConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Result<HypergraphEBM> {
        // Load node weights (prominence) directly from sphere_patches
        let mut node_weights = vec![0.0f32; n_patches];
        let mut has_weights = false;

        let mut stmt = conn.prepare("SELECT patch_idx, prominence FROM sphere_patches")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let patch_idx: i64 = row.get(0)?;
            let prominence: f64 = row.get(1)?;
            if (patch_idx as usize) < n_patches {
                node_weights[patch_idx as usize] = prominence as f32;
                has_weights = true;
            }
        }

        // Create graph sidecar
        let mut sidecar = HypergraphSidecar::new(n_patches);
        if has_weights {
            sidecar.node_weights = Some(node_weights);
        }

        // Load edges directly from sphere_edges (already remapped to patch indices)
        let mut stmt =
            conn.prepare("SELECT src_patch, dst_patch, weight, label FROM sphere_edges")?;
        let mut rows = stmt.query([])?;

        let mut _edge_count = 0;
        while let Some(row) = rows.next()? {
            let src_patch: i64 = row.get(0)?;
            let dst_patch: i64 = row.get(1)?;
            let weight: f64 = row.get(2)?;
            let label: String = row.get(3)?;

            // Filter by edge type and apply weight multiplier
            let final_weight = match label.as_str() {
                "next" if config.include_next_edges => {
                    Some(weight as f32 * config.next_edge_weight)
                }
                "contains" if config.include_contains_edges => {
                    Some(weight as f32 * config.contains_edge_weight)
                }
                "same_source" if config.include_same_source_edges => {
                    Some(weight as f32 * config.same_source_edge_weight)
                }
                "semantic" if config.include_semantic_edges => {
                    Some(weight as f32 * config.semantic_edge_weight)
                }
                _ => None,
            };

            if let Some(w) = final_weight {
                let patch_a = src_patch as usize;
                let patch_b = dst_patch as usize;
                if patch_a < n_patches && patch_b < n_patches {
                    sidecar.add_edge(patch_a, patch_b, w);
                    _edge_count += 1;
                }
            }
        }

        // Debug info (controlled by log level in production)
        #[cfg(debug_assertions)]
        eprintln!(
            "Loaded {} patches, {} edges from sphere-optimized tables",
            n_patches, _edge_count
        );

        // Silence unused warning in release builds
        let _ = _edge_count;

        Ok(HypergraphEBM::from_sidecar(
            &sidecar,
            config.spring_constant,
            config.coherence_weight,
            device,
        ))
    }

    /// Load from legacy bincode-encoded tables (schema v2, backward compatible)
    fn load_from_legacy_tables(
        conn: &Connection,
        n_patches: usize,
        config: &HypergraphLoadConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Result<HypergraphEBM> {
        // Build node index -> patch metadata mapping
        let node_to_patch = build_node_to_patch_map(conn)?;

        // Extract prominence/coherence scores
        let mut node_weights = vec![0.0f32; n_patches];
        let mut has_weights = false;
        for meta in node_to_patch.values() {
            if let Some(prom) = meta.prominence {
                if meta.patch_index < n_patches {
                    node_weights[meta.patch_index] = prom;
                    has_weights = true;
                }
            }
        }

        let mut sidecar = HypergraphSidecar::new(n_patches);
        if has_weights {
            sidecar.node_weights = Some(node_weights);
        }

        // Load edges via bincode decode
        let mut stmt = conn.prepare("SELECT data FROM hyperedges ORDER BY id ASC")?;
        let mut rows = stmt.query([])?;

        let mut _edge_count = 0;
        while let Some(row) = rows.next()? {
            let blob: Vec<u8> = row.get(0)?;

            let (edge, _): (BltHypergraphEdge, _) =
                bincode::serde::decode_from_slice(&blob, bincode::config::standard())
                    .context("Failed to decode hyperedge")?;

            let weight = match edge.label.as_str() {
                "next" if config.include_next_edges => Some(edge.weight * config.next_edge_weight),
                "contains" if config.include_contains_edges => {
                    Some(edge.weight * config.contains_edge_weight)
                }
                "same_source" if config.include_same_source_edges => {
                    Some(edge.weight * config.same_source_edge_weight)
                }
                "semantic" if config.include_semantic_edges => {
                    Some(edge.weight * config.semantic_edge_weight)
                }
                _ => None,
            };

            if let Some(w) = weight {
                if edge.vertices.len() == 2 {
                    let node_a = edge.vertices[0] as usize;
                    let node_b = edge.vertices[1] as usize;

                    if let (Some(meta_a), Some(meta_b)) =
                        (node_to_patch.get(&node_a), node_to_patch.get(&node_b))
                    {
                        let patch_a = meta_a.patch_index;
                        let patch_b = meta_b.patch_index;
                        if patch_a < n_patches && patch_b < n_patches {
                            sidecar.add_edge(patch_a, patch_b, w);
                            _edge_count += 1;
                        }
                    }
                }
            }
        }

        Ok(HypergraphEBM::from_sidecar(
            &sidecar,
            config.spring_constant,
            config.coherence_weight,
            device,
        ))
    }

    /// Patch metadata extracted from sidecar nodes.
    #[derive(Debug, Clone)]
    struct PatchMetadata {
        patch_index: usize,
        prominence: Option<f32>,
    }

    /// Build a mapping from SQLite node ID to patch metadata.
    ///
    /// Scans all nodes, identifies Leaf nodes, and extracts their patch_index
    /// and prominence from metadata.extra.
    fn build_node_to_patch_map(
        conn: &Connection,
    ) -> Result<std::collections::HashMap<usize, PatchMetadata>> {
        let mut node_to_patch = std::collections::HashMap::new();

        let mut stmt = conn.prepare("SELECT id, data FROM nodes ORDER BY id ASC")?;
        let mut rows = stmt.query([])?;

        while let Some(row) = rows.next()? {
            let node_id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;

            // Try to decode as NodeData - combine pattern matches
            if let Ok((BltNodeData::Leaf(segment), _)) = bincode::serde::decode_from_slice::<BltNodeData, _>(
                &blob,
                bincode::config::standard(),
            ) {
                // Extract patch_index and prominence from metadata.extra
                if let Some(meta) = segment.metadata {
                    if let Some(extra) = meta.extra {
                        if let Some(patch_idx) =
                            extra.get("patch_index").and_then(|v| v.as_u64())
                        {
                            let prominence = extra
                                .get("prominence")
                                .and_then(|v| v.as_f64())
                                .map(|p| p as f32);

                            node_to_patch.insert(
                                node_id as usize,
                                PatchMetadata {
                                    patch_index: patch_idx as usize,
                                    prominence,
                                },
                            );
                        }
                    }
                }
                // Fallback: parse from label like "patch_0"
                else if let Some(label) = segment.label {
                    if let Some(idx_str) = label.strip_prefix("patch_") {
                        if let Ok(idx) = idx_str.parse::<usize>() {
                            node_to_patch.insert(
                                node_id as usize,
                                PatchMetadata {
                                    patch_index: idx,
                                    prominence: None,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(node_to_patch)
    }

    /// Load both SafeTensors and hypergraph together.
    ///
    /// This is the recommended entry point for loading blt-burn output with
    /// spring physics enabled.
    ///
    /// # Arguments
    ///
    /// * `safetensors_path` - Path to the `.safetensors` file
    /// * `sphere_config` - Sphere optimization configuration
    /// * `hg_config` - Hypergraph loading configuration
    /// * `device` - GPU device
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - `SphereEBM` - Sphere model with embeddings
    /// - `Vec<Vec<u8>>` - Raw patch bytes
    /// - `Option<HypergraphEBM>` - Hypergraph (if sidecar exists)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (sphere, bytes, hg) = load_blt_with_hypergraph(
    ///     Path::new("output.safetensors"),
    ///     SphereConfig::default(),
    ///     HypergraphLoadConfig::for_code(),
    ///     &device,
    /// )?;
    ///
    /// let mut navigator = NavigatorEBM::from_sphere_ebm(sphere);
    /// if let Some(hg_ebm) = hg {
    ///     navigator = navigator.with_hypergraph(hg_ebm);
    /// }
    /// ```
    pub fn load_blt_with_hypergraph(
        safetensors_path: &Path,
        sphere_config: crate::config::SphereConfig,
        hg_config: HypergraphLoadConfig,
        device: &burn::backend::wgpu::WgpuDevice,
    ) -> Result<(
        crate::sphere_ebm::SphereEBM,
        Vec<Vec<u8>>,
        Option<HypergraphEBM>,
    )> {
        // Load SafeTensors
        let (sphere_ebm, patch_bytes) =
            crate::loader::load_blt_safetensors(safetensors_path, sphere_config, device)?;

        // Derive hypergraph path: foo.safetensors -> foo.hypergraph.db
        let hg_path = safetensors_path.with_extension("hypergraph.db");

        // Load hypergraph if sidecar exists
        let hg_ebm = if hg_path.exists() {
            let n_patches = sphere_ebm.n_points();
            match load_hypergraph_from_sqlite(&hg_path, n_patches, &hg_config, device) {
                Ok(hg) => {
                    // Loaded hypergraph from {hg_path:?} ({n_patches} patches)
                    Some(hg)
                }
                Err(e) => {
                    eprintln!("Warning: Failed to load hypergraph: {}", e);
                    None
                }
            }
        } else {
            // No hypergraph sidecar at {hg_path:?}
            None
        };

        Ok((sphere_ebm, patch_bytes, hg_ebm))
    }

    /// Get hypergraph statistics from a SQLite sidecar.
    ///
    /// Useful for debugging and inspection.
    pub fn hypergraph_stats(path: &Path) -> Result<HypergraphStats> {
        let conn = Connection::open(path)?;

        // Count nodes by type
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM nodes")?;
        let total_nodes: i64 = stmt.query_row([], |row| row.get(0))?;

        // Count edges
        let mut stmt = conn.prepare("SELECT COUNT(*) FROM hyperedges")?;
        let total_edges: i64 = stmt.query_row([], |row| row.get(0))?;

        // Count edges by type
        let mut next_edges = 0;
        let mut contains_edges = 0;
        let mut other_edges = 0;

        let mut stmt = conn.prepare("SELECT data FROM hyperedges")?;
        let mut rows = stmt.query([])?;

        while let Some(row) = rows.next()? {
            let blob: Vec<u8> = row.get(0)?;
            if let Ok((edge, _)) = bincode::serde::decode_from_slice::<BltHypergraphEdge, _>(
                &blob,
                bincode::config::standard(),
            ) {
                match edge.label.as_str() {
                    "next" => next_edges += 1,
                    "contains" => contains_edges += 1,
                    _ => other_edges += 1,
                }
            }
        }

        Ok(HypergraphStats {
            total_nodes: total_nodes as usize,
            total_edges: total_edges as usize,
            next_edges,
            contains_edges,
            other_edges,
        })
    }
}

/// Statistics about a hypergraph sidecar.
#[derive(Clone, Debug, Default)]
pub struct HypergraphStats {
    /// Total number of nodes (Trunk + Branch + Leaf).
    pub total_nodes: usize,
    /// Total number of hyperedges.
    pub total_edges: usize,
    /// Number of "next" (sequential) edges.
    pub next_edges: usize,
    /// Number of "contains" (hierarchical) edges.
    pub contains_edges: usize,
    /// Number of other edge types.
    pub other_edges: usize,
}

impl std::fmt::Display for HypergraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HypergraphStats {{ nodes: {}, edges: {} (next: {}, contains: {}, other: {}) }}",
            self.total_nodes,
            self.total_edges,
            self.next_edges,
            self.contains_edges,
            self.other_edges
        )
    }
}

// Re-export feature-gated functions
#[cfg(feature = "hypergraph")]
pub use sqlite_loader::{
    hypergraph_stats, load_blt_with_hypergraph, load_hypergraph_from_sqlite, BltHypergraphEdge,
    BltNodeData,
};
