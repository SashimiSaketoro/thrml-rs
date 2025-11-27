//! Hyperparameter tuning for NavigatorEBM.
//!
//! This example runs grid search or random search over hyperparameters
//! and saves results to CSV and/or JSON.
//!
//! # Usage
//!
//! ```bash
//! # Full grid search (default grid has 729 configs)
//! cargo run --example tune_navigator --release -- \
//!   --blt-safetensors data.safetensors \
//!   --mode grid \
//!   --epochs 30 \
//!   --output-csv results.csv
//!
//! # Random search (faster, 20 samples)
//! cargo run --example tune_navigator --release -- \
//!   --blt-safetensors data.safetensors \
//!   --mode random \
//!   --n-samples 20 \
//!   --epochs 30 \
//!   --output-json results.json
//!
//! # Resume from previous run
//! cargo run --example tune_navigator --release -- \
//!   --blt-safetensors data.safetensors \
//!   --resume results.json \
//!   --output-json results.json
//!
//! # Use small grid for development
//! cargo run --example tune_navigator --release -- \
//!   --blt-safetensors data.safetensors \
//!   --grid small \
//!   --epochs 10
//! ```

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{
    generate_pairs_from_similarity, load_blt_safetensors, load_from_safetensors, ScaleProfile,
    SphereConfig, TrainingDataset, TuningGrid, TuningSession,
};

#[derive(Clone, ValueEnum)]
enum TuningMode {
    /// Full grid search over all combinations
    Grid,
    /// Random search over n_samples
    Random,
}

#[derive(Clone, ValueEnum)]
enum GridSize {
    /// Single config (for testing)
    Minimal,
    /// Small grid (8 configs)
    Small,
    /// Default grid (729 configs)
    Default,
}

#[derive(Parser)]
#[command(name = "tune_navigator")]
#[command(author, version, about = "Hyperparameter tuning for NavigatorEBM")]
struct Args {
    /// BLT v3 SafeTensors file (recommended)
    #[arg(long)]
    blt_safetensors: Option<PathBuf>,

    /// Legacy: Input SafeTensors file with embeddings
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Tuning mode: grid or random
    #[arg(short, long, default_value = "grid")]
    mode: TuningMode,

    /// Grid size preset
    #[arg(long, default_value = "default")]
    grid: GridSize,

    /// Number of samples for random search
    #[arg(long, default_value = "20")]
    n_samples: usize,

    /// Number of training epochs per config
    #[arg(short, long, default_value = "30")]
    epochs: usize,

    /// Batch size for training
    #[arg(short, long, default_value = "16")]
    batch_size: usize,

    /// Fraction of data for validation
    #[arg(long, default_value = "0.1")]
    val_split: f32,

    /// Resume from previous JSON results
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Output CSV file (optional)
    #[arg(long)]
    output_csv: Option<PathBuf>,

    /// Output JSON file (optional)
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Print verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Navigator Hyperparameter Tuning ===\n");

    // Initialize GPU
    ensure_backend();
    let device = init_gpu_device();
    if args.verbose {
        println!("GPU device initialized");
    }

    // Load data
    let sphere_config = SphereConfig::from(ScaleProfile::Dev).with_steps(10);

    let sphere_ebm = if let Some(blt_path) = &args.blt_safetensors {
        println!("Loading BLT v3 SafeTensors from {:?}...", blt_path);
        let (ebm, _bytes) = load_blt_safetensors(blt_path, sphere_config, &device)
            .context("Failed to load BLT SafeTensors")?;
        println!(
            "  {} patches, {} dimensions",
            ebm.n_points(),
            ebm.embedding_dim()
        );
        ebm
    } else if let Some(input_path) = &args.input {
        println!("Loading embeddings from {:?}...", input_path);
        let ebm = load_from_safetensors(input_path, sphere_config, &device)
            .context("Failed to load embeddings")?;
        println!(
            "  {} points, {} dimensions",
            ebm.n_points(),
            ebm.embedding_dim()
        );
        ebm
    } else {
        anyhow::bail!(
            "Must provide either --blt-safetensors or --input\n\
             Use --blt-safetensors for BLT v3 format (recommended)"
        );
    };

    // Generate training data
    println!("\nGenerating training pairs...");
    let examples = generate_pairs_from_similarity(&sphere_ebm, 1, 8, &device);
    println!("  Generated {} training examples", examples.len());

    if examples.is_empty() {
        anyhow::bail!("No training examples generated. Need at least 2 embeddings.");
    }

    // Create dataset
    let dataset = TrainingDataset::from_examples(examples, args.val_split, args.seed);
    println!(
        "  Train: {} examples, Validation: {} examples",
        dataset.n_train(),
        dataset.n_val()
    );

    // Create grid
    let grid = match args.grid {
        GridSize::Minimal => TuningGrid::minimal(),
        GridSize::Small => TuningGrid::small(),
        GridSize::Default => TuningGrid::default(),
    };

    println!("\nGrid configuration:");
    println!("  Learning rates: {:?}", grid.learning_rates);
    println!("  Negatives: {:?}", grid.negatives);
    println!("  Temperatures: {:?}", grid.temperatures);
    println!("  Total combinations: {}", grid.n_combinations());

    // Create or resume session
    let mut session = if let Some(resume_path) = &args.resume {
        println!("\nResuming from {:?}...", resume_path);
        TuningSession::from_json(resume_path).context("Failed to load resume file")?
    } else {
        TuningSession::new(grid)
    };

    if session.n_completed() > 0 {
        println!("  Already completed: {} runs", session.n_completed());
        println!("  Remaining: {} runs", session.n_remaining());
    }

    // Run tuning
    let key = RngKey::new(args.seed);

    println!("\n=== Starting Tuning ===\n");

    match args.mode {
        TuningMode::Grid => {
            println!("Mode: Grid search");
            session.run_grid_search(
                &sphere_ebm,
                &dataset,
                args.epochs,
                args.batch_size,
                key,
                &device,
            );
        }
        TuningMode::Random => {
            println!("Mode: Random search ({} samples)", args.n_samples);
            session.run_random_search(
                args.n_samples,
                &sphere_ebm,
                &dataset,
                args.epochs,
                args.batch_size,
                args.seed,
                key,
                &device,
            );
        }
    }

    // Print summary
    session.print_summary();

    // Save results
    if let Some(csv_path) = &args.output_csv {
        session
            .to_csv(csv_path)
            .context("Failed to write CSV")?;
        println!("\nResults saved to {:?}", csv_path);
    }

    if let Some(json_path) = &args.output_json {
        session
            .to_json(json_path)
            .context("Failed to write JSON")?;
        println!("Results saved to {:?}", json_path);
    }

    if args.output_csv.is_none() && args.output_json.is_none() {
        println!("\nTip: Use --output-csv or --output-json to save results");
    }

    println!("\nDone!");
    Ok(())
}
