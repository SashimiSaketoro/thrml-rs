//! Integration test for ProbabilisticGraphEBM with real FineWeb-Edu data.
//!
//! Tests the full flow:
//! 1. Load SafeTensors + hypergraph sidecar
//! 2. Create ProbabilisticGraphEBM
//! 3. Run Gibbs sampling
//! 4. Detect high-variance edges

use std::path::Path;
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_sphere::{
    load_blt_with_hypergraph, HypergraphLoadConfig, ProbabilisticGraphConfig,
    ProbabilisticGraphEBM, SphereConfig,
};

/// Path to test data (update if needed)
const TEST_DATA_PATH: &str = "/Volumes/CrucialX6/sphere_test/corpus.safetensors";

#[test]
#[ignore] // Run with: cargo test -p thrml-sphere graph_ebm_integration -- --ignored --nocapture
fn test_probabilistic_graph_ebm_e2e() {
    ensure_backend();
    let device = init_gpu_device();

    let safetensors_path = Path::new(TEST_DATA_PATH);
    if !safetensors_path.exists() {
        eprintln!("Test data not found at {}. Skipping.", TEST_DATA_PATH);
        return;
    }

    // 1. Load data
    println!("Loading data from {:?}...", safetensors_path);
    let (sphere_ebm, _bytes, hg_ebm) = load_blt_with_hypergraph(
        safetensors_path,
        SphereConfig::default(),
        HypergraphLoadConfig::default(),
        &device,
    )
    .expect("Failed to load data");

    let n_patches = sphere_ebm.n_points();
    let embed_dim = sphere_ebm.embedding_dim();
    println!(
        "✅ Loaded: {} patches, {} dim embeddings",
        n_patches, embed_dim
    );

    if hg_ebm.is_some() {
        println!("✅ Hypergraph loaded (spring EBM)");
    }

    // 2. Create ProbabilisticGraphEBM
    println!("\nCreating ProbabilisticGraphEBM...");
    let config = ProbabilisticGraphConfig::default().with_temperature(0.5);

    let prob_graph = ProbabilisticGraphEBM::new(
        sphere_ebm.embeddings.clone(),
        sphere_ebm.prominence,
        &config,
        &device,
    );

    println!("✅ ProbabilisticGraphEBM created");
    println!("   Sparse mode: {}", prob_graph.is_sparse());
    println!(
        "   Memory: {} MB",
        prob_graph.memory_bytes() / (1024 * 1024)
    );

    // 3. Run Gibbs sampling
    println!("\nRunning Gibbs sampling (4 sweeps)...");
    let edge_probs = prob_graph.gibbs_sample_edges(4, &device);
    let probs_shape = edge_probs.dims();
    println!("✅ Edge probabilities: {:?}", probs_shape);

    // Sample statistics
    let probs_data: Vec<f32> = edge_probs.into_data().to_vec().expect("probs to vec");
    let mean_prob: f32 = probs_data.iter().sum::<f32>() / probs_data.len() as f32;
    let max_prob = probs_data.iter().cloned().fold(0.0f32, f32::max);
    let min_prob = probs_data.iter().cloned().fold(1.0f32, f32::min);
    println!(
        "   Mean prob: {:.4}, Min: {:.4}, Max: {:.4}",
        mean_prob, min_prob, max_prob
    );

    // 4. Detect high-variance edges
    println!("\nDetecting high-variance edges (threshold=0.2)...");
    let high_var_edges = prob_graph.detect_high_variance_edges(0.2, 8, &device);
    println!("✅ Found {} high-variance edges", high_var_edges.len());

    if !high_var_edges.is_empty() {
        println!("   First 5:");
        for (i, j) in high_var_edges.iter().take(5) {
            println!("     ({}, {})", i, j);
        }
    }

    // 5. Sample paths (if enough nodes)
    if n_patches >= 10 {
        println!("\nSampling paths from node 0 to node {}...", n_patches - 1);
        let paths = prob_graph.sample_paths(0, n_patches - 1, 3, 10, &device);
        println!("✅ Sampled {} paths", paths.len());
        for (i, path) in paths.iter().enumerate() {
            println!("   Path {}: {:?}", i, path);
        }
    }

    println!("\n🎉 All tests passed!");
}

#[test]
#[ignore]
fn test_sparse_mode() {
    ensure_backend();
    let device = init_gpu_device();

    let safetensors_path = Path::new(TEST_DATA_PATH);
    if !safetensors_path.exists() {
        return;
    }

    let (sphere_ebm, _, _) = load_blt_with_hypergraph(
        safetensors_path,
        SphereConfig::default(),
        HypergraphLoadConfig::default(),
        &device,
    )
    .expect("Failed to load");

    // Test sparse mode (k=32 neighbors)
    let config = ProbabilisticGraphConfig::sparse(32);
    let prob_graph = ProbabilisticGraphEBM::new(
        sphere_ebm.embeddings.clone(),
        sphere_ebm.prominence.clone(),
        &config,
        &device,
    );

    assert!(prob_graph.is_sparse());
    println!("Sparse memory: {} KB", prob_graph.memory_bytes() / 1024);

    // Should still work
    let edge_probs = prob_graph.gibbs_sample_edges(2, &device);
    assert_eq!(edge_probs.dims()[0], sphere_ebm.n_points());
}
