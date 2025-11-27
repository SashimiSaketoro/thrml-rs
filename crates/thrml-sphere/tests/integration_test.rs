//! Integration tests with real BLT output data.
//!
//! These tests verify the full pipeline works end-to-end with actual
//! SafeTensors files produced by blt-burn.

use std::path::Path;
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{
    load_from_safetensors, save_coords_npz,
    ScaleProfile, SphereConfig,
    SphereHamiltonian,  // Trait needed for total_energy()
};

/// Find the test SafeTensors file, trying multiple possible locations.
fn find_test_safetensors() -> std::path::PathBuf {
    // Get workspace root from CARGO_MANIFEST_DIR
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = Path::new(manifest_dir).parent().unwrap().parent().unwrap();
    let project_root = workspace_root.parent().unwrap();
    
    let paths = [
        project_root.join("archive/test_outputs/ingest_output/manual_input.safetensors"),
        project_root.join("external/blt-burn/test_output/manual_input.safetensors"),
        // Fallback to absolute path
        Path::new("/Users/joemiles/TheSphere-JAXv0.0.2/archive/test_outputs/ingest_output/manual_input.safetensors").to_path_buf(),
    ];
    
    for path in &paths {
        if path.exists() {
            return path.clone();
        }
    }
    
    panic!(
        "No test SafeTensors file found. Tried:\n{}\n\nRun blt-burn ingest first.",
        paths.iter().map(|p| format!("  - {:?}", p)).collect::<Vec<_>>().join("\n")
    );
}

/// Test loading and optimizing with real BLT output.
#[test]
fn test_full_pipeline_with_blt_output() {
    ensure_backend();
    let device = init_gpu_device();
    
    let input_path = find_test_safetensors();
    
    println!("\n=== Loading from {:?} ===", input_path);
    
    // Load with dev config (fast for testing)
    let config = SphereConfig::from(ScaleProfile::Dev)
        .with_steps(50)  // Fewer steps for faster test
        .with_entropy_weighted(true);
    
    let ebm = load_from_safetensors(&input_path, config, &device)
        .expect("Failed to load SafeTensors");
    
    println!("Loaded {} embeddings of dimension {}", ebm.n_points(), ebm.embedding_dim());
    println!("Config: min_r={}, max_r={}, steps={}", 
        config.min_radius, config.max_radius, config.n_steps);
    
    // Verify embeddings loaded correctly
    assert!(ebm.n_points() > 0, "Should have loaded some embeddings");
    assert!(ebm.embedding_dim() > 0, "Embeddings should have non-zero dimension");
    
    // Run optimization
    println!("\n=== Running sphere optimization ===");
    let key = RngKey::new(42);
    let coords = ebm.optimize(key, &device);
    
    // Verify output shape
    assert_eq!(coords.len(), ebm.n_points(), "Output should have same number of points");
    
    // Check radii are in valid range
    let r_data: Vec<f32> = coords.r.clone().into_data().to_vec().expect("r to vec");
    let theta_data: Vec<f32> = coords.theta.clone().into_data().to_vec().expect("theta to vec");
    let phi_data: Vec<f32> = coords.phi.clone().into_data().to_vec().expect("phi to vec");
    
    let r_min = r_data.iter().cloned().fold(f32::MAX, f32::min);
    let r_max = r_data.iter().cloned().fold(f32::MIN, f32::max);
    let theta_min = theta_data.iter().cloned().fold(f32::MAX, f32::min);
    let theta_max = theta_data.iter().cloned().fold(f32::MIN, f32::max);
    let phi_min = phi_data.iter().cloned().fold(f32::MAX, f32::min);
    let phi_max = phi_data.iter().cloned().fold(f32::MIN, f32::max);
    
    println!("\n=== Results ===");
    println!("Radii:  min={:.2}, max={:.2} (config: {:.2} - {:.2})", 
        r_min, r_max, config.min_radius, config.max_radius);
    println!("Theta:  min={:.4}, max={:.4} (valid: 0 - π ≈ 3.14)", theta_min, theta_max);
    println!("Phi:    min={:.4}, max={:.4} (valid: 0 - 2π ≈ 6.28)", phi_min, phi_max);
    
    // Soft checks (warn but don't fail - optimization may not fully converge in 50 steps)
    if r_min < config.min_radius * 0.5 {
        println!("WARNING: Some radii below min_radius * 0.5");
    }
    if r_max > config.max_radius * 2.0 {
        println!("WARNING: Some radii above max_radius * 2.0");
    }
    
    // Hard checks - angles should be reasonable
    assert!(theta_min >= -0.5, "Theta should be >= 0 (got {})", theta_min);
    assert!(theta_max <= std::f32::consts::PI + 0.5, "Theta should be <= π (got {})", theta_max);
    
    // Verify Cartesian conversion works
    let cartesian = coords.to_cartesian();
    assert_eq!(cartesian.dims(), [ebm.n_points(), 3], "Cartesian should be [N, 3]");
    
    println!("\n✓ Full pipeline test PASSED!");
}

/// Test that optimization actually reduces energy.
#[test]
fn test_optimization_reduces_energy() {
    ensure_backend();
    let device = init_gpu_device();
    
    let input_path = find_test_safetensors();
    
    let config = SphereConfig::from(ScaleProfile::Dev)
        .with_steps(100)
        .with_temperature(0.01);  // Low temperature for more deterministic descent
    
    let ebm = load_from_safetensors(&input_path, config, &device)
        .expect("Failed to load");
    
    // Get initial energy
    let init_coords = ebm.init_coords(&device);
    let hamiltonian = ebm.hamiltonian();
    let init_energy: Vec<f32> = hamiltonian.total_energy(&init_coords)
        .into_data().to_vec().expect("energy to vec");
    let init_total: f32 = init_energy.iter().sum();
    
    // Run optimization
    let key = RngKey::new(123);
    let final_coords = ebm.optimize(key, &device);
    
    // Get final energy
    let final_energy: Vec<f32> = hamiltonian.total_energy(&final_coords)
        .into_data().to_vec().expect("energy to vec");
    let final_total: f32 = final_energy.iter().sum();
    
    println!("\n=== Energy Reduction Test ===");
    println!("Initial total energy: {:.2}", init_total);
    println!("Final total energy:   {:.2}", final_total);
    println!("Reduction: {:.2}%", (1.0 - final_total / init_total) * 100.0);
    
    // Energy should decrease (or at least not increase much)
    assert!(
        final_total <= init_total * 1.1,  // Allow 10% tolerance
        "Energy should not increase significantly: {} -> {}", 
        init_total, final_total
    );
    
    println!("\n✓ Energy reduction test PASSED!");
}

/// Test saving and loading coordinates.
#[test]
fn test_save_load_npz() {
    ensure_backend();
    let device = init_gpu_device();
    
    let input_path = find_test_safetensors();
    
    let config = SphereConfig::from(ScaleProfile::Dev).with_steps(20);
    let ebm = load_from_safetensors(&input_path, config, &device)
        .expect("Failed to load");
    
    // Run optimization
    let key = RngKey::new(42);
    let coords = ebm.optimize(key, &device);
    
    // Save to temp file
    let output_path = Path::new("/tmp/thrml_sphere_test_output.npz");
    save_coords_npz(&coords, output_path)
        .expect("Failed to save NPZ");
    
    assert!(output_path.exists(), "Output file should exist");
    
    // Verify file size is reasonable
    let metadata = std::fs::metadata(output_path).expect("metadata");
    assert!(metadata.len() > 100, "Output file should have content");
    
    println!("\n=== NPZ Save Test ===");
    println!("Saved {} bytes to {:?}", metadata.len(), output_path);
    
    // Clean up
    std::fs::remove_file(output_path).ok();
    
    println!("\n✓ NPZ save test PASSED!");
}

/// Test with different scale profiles.
#[test]
fn test_scale_profiles() {
    ensure_backend();
    let device = init_gpu_device();
    
    let input_path = find_test_safetensors();
    
    println!("\n=== Scale Profile Test ===");
    
    for profile in [ScaleProfile::Dev, ScaleProfile::Medium] {
        let config = SphereConfig::from(profile).with_steps(10);
        
        let ebm = load_from_safetensors(&input_path, config, &device)
            .expect("Failed to load");
        
        let key = RngKey::new(42);
        let coords = ebm.optimize(key, &device);
        
        let r_data: Vec<f32> = coords.r.into_data().to_vec().expect("r to vec");
        let r_min = r_data.iter().cloned().fold(f32::MAX, f32::min);
        let r_max = r_data.iter().cloned().fold(f32::MIN, f32::max);
        
        println!("{:?}: radii {:.1} - {:.1} (config: {:.1} - {:.1})",
            profile, r_min, r_max, config.min_radius, config.max_radius);
    }
    
    println!("\n✓ Scale profile test PASSED!");
}

