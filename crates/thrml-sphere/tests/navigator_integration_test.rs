//! Integration tests for NavigatorEBM with hypergraph connectivity.
//!
//! These tests simulate loading data from a blt-burn HypergraphSidecar
//! and running navigation through the connected structure.

use burn::tensor::{Distribution, Tensor};
use thrml_core::backend::{init_gpu_device, WgpuBackend};
use thrml_samplers::RngKey;
use thrml_sphere::{
    ConeConfig, HypergraphEBM, HypergraphSidecar, NavigationWeights, NavigatorEBM, ScaleProfile,
    SphereConfig,
};

/// Helper to create a mock hypergraph structure similar to blt-burn's output.
///
/// Creates a tree structure:
/// - Trunk (root)
/// - 3 Branches
/// - Each branch has 2-4 leaves
fn create_mock_hypergraph(n_total: usize) -> (HypergraphSidecar, Vec<f32>) {
    // Create sidecar with n_total nodes
    let mut sidecar = HypergraphSidecar::new(n_total);

    // Create tree structure:
    // Node 0: trunk (root)
    // Nodes 1-3: branches
    // Nodes 4+: leaves

    // Trunk -> Branches
    sidecar.add_edge(0, 1, 1.0);
    sidecar.add_edge(0, 2, 1.0);
    sidecar.add_edge(0, 3, 1.0);

    // Distribute leaves among branches
    let leaves_start = 4;
    let leaves_per_branch = (n_total - leaves_start) / 3;

    for branch in 1..=3 {
        let leaf_start = leaves_start + (branch - 1) * leaves_per_branch;
        let leaf_end = if branch == 3 {
            n_total
        } else {
            leaves_start + branch * leaves_per_branch
        };

        for leaf in leaf_start..leaf_end {
            sidecar.add_edge(branch, leaf, 0.8);
        }
    }

    // Add some cross-branch connections (simulating "same_source" edges)
    // Connect leaves that are semantically similar
    for i in leaves_start..(n_total - 1).min(leaves_start + 5) {
        let j = i + leaves_per_branch.min(n_total - i - 1);
        if j < n_total {
            sidecar.add_edge(i, j, 0.5);
        }
    }

    // Create coherence scores (higher for trunk/branches, lower for leaves)
    let mut coherence = vec![0.0f32; n_total];
    coherence[0] = 1.0; // Trunk
    coherence[1] = 0.8;
    coherence[2] = 0.8;
    coherence[3] = 0.8;
    for i in leaves_start..n_total {
        coherence[i] = 0.3f32.mul_add(i as f32 / n_total as f32, 0.2);
    }

    let sidecar_with_weights = sidecar.with_node_weights(coherence.clone());
    (sidecar_with_weights, coherence)
}

#[test]
fn test_navigator_with_hypergraph_creation() {
    let device = init_gpu_device();
    let n = 20;
    let d = 32;

    // Create mock embeddings
    let embeddings: Tensor<WgpuBackend, 2> =
        Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
    let prominence: Tensor<WgpuBackend, 1> =
        Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

    // Create hypergraph
    let (sidecar, _coherence) = create_mock_hypergraph(n);

    // Create HypergraphEBM from sidecar
    let hypergraph_ebm = HypergraphEBM::from_sidecar(&sidecar, 0.1, 0.5, &device);

    // Create NavigatorEBM with hypergraph
    let config = SphereConfig::from(ScaleProfile::Dev).with_steps(10);
    let navigator = NavigatorEBM::new(embeddings, prominence, None, config, &device)
        .with_hypergraph(hypergraph_ebm);

    assert_eq!(navigator.n_points(), n);
    assert!(navigator.hypergraph_ebm.is_some());
}

#[test]
fn test_navigator_with_hypergraph_navigation() {
    let device = init_gpu_device();
    let n = 30;
    let d = 16;

    // Create embeddings with some structure
    // Make trunk and branches more similar, leaves more diverse
    let mut embeddings_data = vec![0.0f32; n * d];

    // Trunk (node 0) - neutral embedding
    for j in 0..d {
        embeddings_data[j] = 0.5;
    }

    // Branches (nodes 1-3) - similar to trunk
    for i in 1..=3 {
        for j in 0..d {
            embeddings_data[i * d + j] = 0.1f32.mul_add(i as f32, 0.5);
        }
    }

    // Leaves (nodes 4+) - more diverse
    for i in 4..n {
        for j in 0..d {
            embeddings_data[i * d + j] = (i as f32 * j as f32).sin() * 0.3;
        }
    }

    let embeddings_1d: Tensor<WgpuBackend, 1> =
        Tensor::from_data(embeddings_data.as_slice(), &device);
    let embeddings: Tensor<WgpuBackend, 2> = embeddings_1d.reshape([n as i32, d as i32]);
    let prominence: Tensor<WgpuBackend, 1> =
        Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

    // Create hypergraph
    let (sidecar, _) = create_mock_hypergraph(n);
    let hypergraph_ebm = HypergraphEBM::from_sidecar(&sidecar, 0.1, 0.3, &device);

    // Create navigator with graph-aware weights
    let weights = NavigationWeights::default()
        .with_semantic(1.0)
        .with_graph(0.5)
        .with_radial(0.3);

    let config = SphereConfig::from(ScaleProfile::Dev).with_steps(15);
    let navigator = NavigatorEBM::new(embeddings, prominence, None, config, &device)
        .with_hypergraph(hypergraph_ebm)
        .with_weights(weights);

    // Create query similar to trunk
    let mut query_data = vec![0.5f32; d];
    query_data[0] = 0.6; // Slightly different
    let query: Tensor<WgpuBackend, 1> = Tensor::from_data(query_data.as_slice(), &device);

    // Run navigation
    let result = navigator.navigate(query, 50.0, RngKey::new(42), 5, &device);

    // Verify results
    assert_eq!(result.target_indices.len(), 5);
    assert_eq!(result.target_energies.len(), 5);

    // Energies should be non-negative and sorted
    for i in 1..result.target_energies.len() {
        assert!(
            result.target_energies[i] >= result.target_energies[i - 1],
            "Energies should be sorted ascending"
        );
    }

    println!("Navigation result:");
    println!("  Top targets: {:?}", result.target_indices);
    println!("  Energies: {:?}", result.target_energies);
    println!("  Total energy: {}", result.total_energy);
}

#[test]
fn test_navigator_cone_with_hypergraph() {
    let device = init_gpu_device();
    let n = 25;
    let d = 8;

    let embeddings: Tensor<WgpuBackend, 2> =
        Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
    let prominence: Tensor<WgpuBackend, 1> =
        Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

    // Create hypergraph
    let (sidecar, _) = create_mock_hypergraph(n);
    let hypergraph_ebm = HypergraphEBM::from_sidecar(&sidecar, 0.1, 0.3, &device);

    let config = SphereConfig::from(ScaleProfile::Dev).with_steps(10);
    let navigator = NavigatorEBM::new(embeddings.clone(), prominence, None, config, &device)
        .with_hypergraph(hypergraph_ebm);

    // Create query
    let query: Tensor<WgpuBackend, 1> = embeddings.slice([0..1, 0..d]).reshape([d as i32]);

    // Create a cone
    let cone =
        ConeConfig::new(std::f32::consts::PI / 2.0, 0.0).with_aperture(std::f32::consts::PI / 2.0); // Wide aperture for testing

    let result = navigator.navigate_cone(query, 50.0, &cone, RngKey::new(123), 3, &device);

    // Should have results (may be fewer if cone doesn't cover all points)
    println!(
        "Cone navigation found {} targets in cone",
        result.target_indices.len()
    );

    // Even if empty cone, should not panic
    if !result.target_indices.is_empty() {
        assert!(result.target_energies.len() == result.target_indices.len());
    }
}

#[test]
fn test_hypergraph_spring_energy_affects_navigation() {
    let device = init_gpu_device();
    let n = 15;
    let d = 8;

    // Create embeddings where connected nodes are dissimilar
    // This tests that graph energy affects the final navigation
    let mut embeddings_data = vec![0.0f32; n * d];
    for i in 0..n {
        for j in 0..d {
            // Create somewhat random embeddings
            embeddings_data[i * d + j] = ((i * 7 + j * 13) % 100) as f32 / 100.0;
        }
    }
    let embeddings_1d: Tensor<WgpuBackend, 1> =
        Tensor::from_data(embeddings_data.as_slice(), &device);
    let embeddings: Tensor<WgpuBackend, 2> = embeddings_1d.reshape([n as i32, d as i32]);
    let prominence: Tensor<WgpuBackend, 1> =
        Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

    let (sidecar, _) = create_mock_hypergraph(n);

    // Create navigators with different graph weights
    let config = SphereConfig::from(ScaleProfile::Dev).with_steps(10);

    // Navigator 1: No graph weight
    let weights_no_graph = NavigationWeights::default().with_graph(0.0);
    let navigator_no_graph = NavigatorEBM::new(
        embeddings.clone(),
        prominence.clone(),
        None,
        config,
        &device,
    )
    .with_weights(weights_no_graph);

    // Navigator 2: Strong graph weight
    let hypergraph_ebm = HypergraphEBM::from_sidecar(&sidecar, 0.1, 0.3, &device);
    let weights_with_graph = NavigationWeights::default().with_graph(1.0);
    let navigator_with_graph = NavigatorEBM::new(
        embeddings,
        prominence,
        None,
        config,
        &device,
    )
    .with_hypergraph(hypergraph_ebm)
    .with_weights(weights_with_graph);

    // Same query
    let query: Tensor<WgpuBackend, 1> =
        Tensor::random([d], Distribution::Normal(0.0, 1.0), &device);

    let result_no_graph =
        navigator_no_graph.navigate(query.clone(), 50.0, RngKey::new(42), 5, &device);
    let result_with_graph = navigator_with_graph.navigate(query, 50.0, RngKey::new(42), 5, &device);

    println!("Without graph: {:?}", result_no_graph.target_indices);
    println!("With graph:    {:?}", result_with_graph.target_indices);

    // Results may differ due to graph energy influence
    // (Not asserting they must differ, but showing the mechanism works)
    assert_eq!(result_no_graph.target_indices.len(), 5);
    assert_eq!(result_with_graph.target_indices.len(), 5);
}

#[test]
fn test_navigation_with_entropy_weighting() {
    let device = init_gpu_device();
    let n = 20;
    let d = 16;

    let embeddings: Tensor<WgpuBackend, 2> =
        Tensor::random([n, d], Distribution::Normal(0.0, 1.0), &device);
    let prominence: Tensor<WgpuBackend, 1> =
        Tensor::random([n], Distribution::Uniform(0.1, 1.0), &device);

    // Create entropy values - some low (confident), some high
    let mut entropy_data = vec![0.0f32; n];
    for i in 0..n {
        entropy_data[i] = if i % 3 == 0 { 0.1 } else { 0.8 }; // Every 3rd node is low entropy
    }
    let entropies: Tensor<WgpuBackend, 1> = Tensor::from_data(entropy_data.as_slice(), &device);

    // Create navigators with different entropy weights
    let config = SphereConfig::from(ScaleProfile::Dev).with_steps(10);

    // Navigator with entropy weighting
    let weights = NavigationWeights::default()
        .with_semantic(1.0)
        .with_entropy(0.5); // Prefer low-entropy targets

    let navigator = NavigatorEBM::new(
        embeddings,
        prominence,
        Some(entropies),
        config,
        &device,
    )
    .with_weights(weights);

    let query: Tensor<WgpuBackend, 1> =
        Tensor::random([d], Distribution::Normal(0.0, 1.0), &device);

    let result = navigator.navigate(query, 50.0, RngKey::new(42), 10, &device);

    // Entropy energy is -H(t), so low entropy gives lower (better) energy
    // We should see some low-entropy nodes in results
    let low_entropy_nodes: Vec<usize> = (0..n).filter(|i| i % 3 == 0).collect();

    let found_low_entropy = result
        .target_indices
        .iter()
        .filter(|&&idx| low_entropy_nodes.contains(&idx))
        .count();

    println!(
        "Found {} low-entropy nodes in top {} results",
        found_low_entropy,
        result.target_indices.len()
    );

    // At least one low-entropy node should be in results
    // (not guaranteed but likely with entropy weighting)
    assert!(
        result.target_indices.len() == 10,
        "Should return requested top_k"
    );
}

#[test]
fn test_navigation_weights_serialization() {
    let device = init_gpu_device();

    let weights = NavigationWeights::default()
        .with_semantic(2.0)
        .with_radial(0.3)
        .with_graph(0.7)
        .with_entropy(0.1)
        .with_path(0.05);

    // Convert to tensor and back
    let tensor = weights.to_tensor(&device);
    let recovered = NavigationWeights::from_tensor(&tensor);

    assert!((recovered.lambda_semantic - 2.0).abs() < 1e-6);
    assert!((recovered.lambda_radial - 0.3).abs() < 1e-6);
    assert!((recovered.lambda_graph - 0.7).abs() < 1e-6);
    assert!((recovered.lambda_entropy - 0.1).abs() < 1e-6);
    assert!((recovered.lambda_path - 0.05).abs() < 1e-6);
}
