//! Demonstrates multi-cone navigation from ROOTS peaks.
//!
//! This example shows how to use the `MultiConeNavigator` to perform
//! ROOTS-guided navigation through a sphere of embeddings.
//!
//! # Usage
//!
//! ```bash
//! # From BLT v3 SafeTensors (recommended - includes bytes for substring coupling)
//! cargo run --example navigator_demo --release -- \
//!   --blt-safetensors output.safetensors \
//!   --top-k 10
//!
//! # From separate embeddings file
//! cargo run --example navigator_demo --release -- \
//!   --input embeddings.safetensors \
//!   --top-k 10
//!
//! # With custom budget configuration
//! cargo run --example navigator_demo --release -- \
//!   --blt-safetensors output.safetensors \
//!   --max-cones 8 \
//!   --budget-mb 512 \
//!   --partitions 64
//! ```
//!
//! # Architecture
//!
//! The multi-cone navigator works as follows:
//!
//! 1. Build ROOTS index from embeddings (with optional substring coupling)
//! 2. For each query:
//!    a. Compute ROOTS activations
//!    b. Detect activation peaks
//!    c. Spawn cones at peak locations with budget proportional to strength
//!    d. Run navigation in each cone
//!    e. Merge and deduplicate results

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{
    load_blt_safetensors, load_from_safetensors, BudgetConfig, MultiConeNavigator, RootsConfig,
    ScaleProfile, SphereConfig,
};

#[derive(Parser)]
#[command(name = "navigator_demo")]
#[command(
    author,
    version,
    about = "Demonstrates multi-cone navigation from ROOTS peaks"
)]
struct Args {
    /// BLT v3 SafeTensors file (contains embeddings + bytes together)
    /// Preferred format for substring-enhanced partitioning
    #[arg(long)]
    blt_safetensors: Option<PathBuf>,

    /// Legacy: Input SafeTensors file containing embeddings and prominence only
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Number of results to return per cone
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Number of ROOTS partitions
    #[arg(long, default_value = "64")]
    partitions: usize,

    /// Maximum number of cones to spawn
    #[arg(long, default_value = "8")]
    max_cones: usize,

    /// Total attention budget in MB
    #[arg(long, default_value = "256")]
    budget_mb: usize,

    /// Peak detection threshold (0.0-1.0)
    #[arg(long, default_value = "0.2")]
    peak_threshold: f32,

    /// Enable substring-enhanced coupling for ROOTS
    #[arg(long)]
    substring_coupling: bool,

    /// Query index (which embedding to use as query)
    #[arg(long, default_value = "0")]
    query_idx: usize,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Print verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Multi-Cone Navigator Demo ===\n");

    // Initialize GPU backend
    ensure_backend();
    let device = init_gpu_device();
    if args.verbose {
        println!("GPU device initialized");
    }

    // Parse scale profile
    let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(10);

    // Load embeddings and optionally bytes
    let (sphere_ebm, blt_bytes) = if let Some(blt_path) = &args.blt_safetensors {
        println!("Loading BLT v3 SafeTensors from {:?}...", blt_path);
        let (ebm, bytes) = load_blt_safetensors(blt_path, sphere_config, &device)
            .context("Failed to load BLT SafeTensors")?;
        println!(
            "  {} patches, {} dimensions",
            ebm.n_points(),
            ebm.embedding_dim()
        );
        (ebm, Some(bytes))
    } else if let Some(input_path) = &args.input {
        println!("Loading embeddings from {:?}...", input_path);
        let ebm = load_from_safetensors(input_path, sphere_config, &device)
            .context("Failed to load embeddings")?;
        println!(
            "  {} points, {} dimensions",
            ebm.n_points(),
            ebm.embedding_dim()
        );
        (ebm, None)
    } else {
        anyhow::bail!(
            "Must provide either --blt-safetensors or --input\n\
             Use --blt-safetensors for BLT v3 format (recommended)"
        );
    };

    // Build ROOTS config
    let mut roots_config = RootsConfig::dev()
        .with_partitions(args.partitions)
        .with_threshold(args.peak_threshold);

    if args.substring_coupling {
        roots_config = roots_config.with_default_substring_coupling();
        println!("\nSubstring coupling enabled (α=0.7, β=0.3)");
    }

    // Build budget config
    let budget_config = BudgetConfig::new(args.budget_mb * 1024 * 1024)
        .with_max_cones(args.max_cones)
        .with_min_cone_budget(16 * 1024 * 1024) // 16MB min per cone
        .with_peak_threshold(args.peak_threshold);

    println!("\nConfiguration:");
    println!("  ROOTS partitions: {}", args.partitions);
    println!("  Max cones: {}", args.max_cones);
    println!("  Total budget: {} MB", args.budget_mb);
    println!("  Peak threshold: {}", args.peak_threshold);

    // Build multi-cone navigator
    println!("\nBuilding ROOTS index and navigator...");
    let key = RngKey::new(args.seed);

    let mut navigator = if args.substring_coupling {
        if let Some(bytes) = &blt_bytes {
            MultiConeNavigator::from_sphere_ebm_with_bytes(
                &sphere_ebm,
                bytes,
                roots_config,
                budget_config,
                key,
                &device,
            )
        } else {
            eprintln!(
                "Warning: --substring-coupling requires BLT v3 format with bytes.\n\
                 Building without substring coupling."
            );
            MultiConeNavigator::from_sphere_ebm(
                &sphere_ebm,
                roots_config,
                budget_config,
                key,
                &device,
            )
        }
    } else {
        MultiConeNavigator::from_sphere_ebm(&sphere_ebm, roots_config, budget_config, key, &device)
    };

    println!(
        "  Built {} ROOTS partitions for {} points",
        navigator.n_partitions(),
        navigator.n_points()
    );

    // Create query from specified index
    let n_points = navigator.n_points();
    let query_idx = args.query_idx.min(n_points - 1);
    let d = sphere_ebm.embedding_dim();

    use burn::tensor::Tensor;
    use thrml_core::backend::WgpuBackend;

    let query: Tensor<WgpuBackend, 1> = sphere_ebm
        .embeddings
        .clone()
        .slice([query_idx..query_idx + 1, 0..d])
        .reshape([d as i32]);

    println!("\nRunning multi-cone navigation...");
    println!("  Query: embedding {}", query_idx);

    // Run navigation
    let nav_key = RngKey::new(args.seed + 1);
    let result = navigator.navigate_multi_cone(query, 50.0, args.top_k, nav_key, &device);

    // Print results
    println!("\n=== Navigation Results ===\n");
    println!(
        "Spawned {} cones, found {} targets",
        result.n_cones(),
        result.n_targets()
    );

    if result.n_cones() > 0 {
        println!("Budget used: {} MB", result.budget_used / (1024 * 1024));
    }

    println!("\nTop {} results:", result.n_targets().min(args.top_k));
    println!("{:-<50}", "");
    println!("{:>6}  {:>10}  {:>12}", "Rank", "Index", "Energy");
    println!("{:-<50}", "");

    for (i, (idx, energy)) in result
        .target_indices
        .iter()
        .zip(result.target_energies.iter())
        .enumerate()
    {
        let marker = if *idx == query_idx { " (query)" } else { "" };
        println!("{:>6}  {:>10}  {:>12.4}{}", i + 1, idx, energy, marker);
    }
    println!("{:-<50}", "");

    // Print per-cone breakdown if verbose
    if args.verbose && !result.per_cone_results.is_empty() {
        println!("\n=== Per-Cone Breakdown ===\n");
        for (i, cone_result) in result.per_cone_results.iter().enumerate() {
            println!(
                "Cone {}: {} targets, total_energy={:.4}",
                i,
                cone_result.target_indices.len(),
                cone_result.total_energy
            );
            if !cone_result.target_indices.is_empty() {
                println!(
                    "  Top targets: {:?}",
                    &cone_result.target_indices[..cone_result.target_indices.len().min(5)]
                );
            }
        }
    }

    // Print statistics
    let stats = navigator.last_navigation_stats();
    println!("\n{}", stats);

    println!("\nDone!");

    Ok(())
}
