//! BLT to Sphere optimization example.
//!
//! This example demonstrates how to optimize BLT embeddings on a hypersphere
//! using the water-filling algorithm.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example blt_sphere -- --input embeddings.safetensors --output sphere.npz
//! ```
//!
//! # Input Format
//!
//! The input SafeTensors file should contain:
//! - `embeddings`: [N, D] float32 - The embedding vectors
//! - `prominence`: [N] float32 - Prominence scores (higher = more important)
//! - `entropies`: [N] float32 (optional) - Entropy values
//!
//! # Output Format
//!
//! The output NPZ file will contain:
//! - `r`: [N] - Radii
//! - `theta`: [N] - Polar angles
//! - `phi`: [N] - Azimuthal angles
//! - `cartesian`: [N, 3] - Cartesian coordinates

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{load_from_safetensors, save_coords_npz, ScaleProfile, SphereConfig};

#[derive(Parser)]
#[command(name = "blt_sphere")]
#[command(author, version, about = "Optimize BLT embeddings on a hypersphere")]
struct Args {
    /// Input SafeTensors file containing embeddings and prominence
    #[arg(short, long)]
    input: PathBuf,

    /// Output NPZ file for spherical coordinates
    #[arg(short, long, default_value = "sphere_output.npz")]
    output: PathBuf,

    /// Scale profile: dev, medium, large, planetary
    #[arg(long, default_value = "dev")]
    scale: String,

    /// Enable entropy-weighted radii calculation
    #[arg(long)]
    entropy_weighted: bool,

    /// Override number of optimization steps
    #[arg(long)]
    steps: Option<usize>,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Log progress every N steps (0 to disable)
    #[arg(long, default_value = "20")]
    log_interval: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== BLT Sphere Optimization ===\n");

    // Initialize GPU backend
    ensure_backend();
    let device = init_gpu_device();
    println!("GPU device initialized");

    // Parse scale profile
    let profile = ScaleProfile::from_str(&args.scale).unwrap_or_else(|| {
        eprintln!("Warning: Unknown scale '{}', using 'dev'", args.scale);
        ScaleProfile::Dev
    });

    // Build configuration
    let mut config = SphereConfig::from(profile).with_entropy_weighted(args.entropy_weighted);

    if let Some(steps) = args.steps {
        config = config.with_steps(steps);
    }

    println!("Configuration:");
    println!("  Scale: {:?}", profile);
    println!(
        "  Radius range: [{}, {}]",
        config.min_radius, config.max_radius
    );
    println!("  Steps: {}", config.n_steps);
    println!("  Temperature: {}", config.temperature);
    println!("  Entropy weighted: {}", config.entropy_weighted);
    println!();

    // Load data
    println!("Loading from {:?}...", args.input);
    let ebm = load_from_safetensors(&args.input, config, &device)?;

    println!(
        "Loaded {} points with {} dimensions\n",
        ebm.n_points(),
        ebm.embedding_dim()
    );

    // Run optimization
    println!("Running sphere optimization...");
    let key = RngKey::new(args.seed);

    let coords = if args.log_interval > 0 {
        ebm.optimize_with_logging(key, &device, args.log_interval)
    } else {
        ebm.optimize(key, &device)
    };

    // Print summary statistics
    let r_data: Vec<f32> = coords.r.clone().into_data().to_vec()?;
    let r_min = r_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let r_max = r_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let r_mean = r_data.iter().sum::<f32>() / r_data.len() as f32;

    println!("\nOptimization complete!");
    println!(
        "  Radius stats: min={:.2}, max={:.2}, mean={:.2}",
        r_min, r_max, r_mean
    );

    // Save output
    println!("\nSaving to {:?}...", args.output);
    save_coords_npz(&coords, &args.output)?;

    println!("\nDone! Output saved to {:?}", args.output);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_profile_parsing() {
        assert!(ScaleProfile::from_str("dev").is_some());
        assert!(ScaleProfile::from_str("MEDIUM").is_some());
        assert!(ScaleProfile::from_str("invalid").is_none());
    }
}
