//! Train a NavigatorEBM with evaluation metrics.
//!
//! This example demonstrates the full training pipeline:
//! 1. Load BLT v3 SafeTensors (or regular embeddings)
//! 2. Generate training pairs from similarity
//! 3. Train with validation and early stopping
//! 4. Evaluate and report metrics
//!
//! # Usage
//!
//! ```bash
//! # From BLT v3 SafeTensors (recommended)
//! cargo run --example train_navigator --release -- \
//!   --blt-safetensors output.safetensors \
//!   --epochs 50 \
//!   --val-split 0.1
//!
//! # From regular embeddings file
//! cargo run --example train_navigator --release -- \
//!   --input embeddings.safetensors \
//!   --epochs 50
//!
//! # With custom training parameters
//! cargo run --example train_navigator --release -- \
//!   --blt-safetensors output.safetensors \
//!   --epochs 100 \
//!   --batch-size 32 \
//!   --lr 0.01 \
//!   --negatives 8 \
//!   --early-stopping 10
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{
    evaluate_navigator, generate_pairs_from_similarity, load_blt_safetensors,
    load_from_safetensors, ExtendedTrainingConfig, NavigatorTrainingConfig, ScaleProfile,
    SphereConfig, TrainableNavigatorEBM, TrainingDataset,
};

#[derive(Parser)]
#[command(name = "train_navigator")]
#[command(
    author,
    version,
    about = "Train a NavigatorEBM with evaluation metrics"
)]
struct Args {
    /// BLT v3 SafeTensors file (contains embeddings + bytes together)
    #[arg(long)]
    blt_safetensors: Option<PathBuf>,

    /// Legacy: Input SafeTensors file containing embeddings and prominence only
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Number of training epochs
    #[arg(short, long, default_value = "50")]
    epochs: usize,

    /// Batch size for training
    #[arg(short, long, default_value = "16")]
    batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.01")]
    lr: f32,

    /// Number of negative samples per positive
    #[arg(long, default_value = "8")]
    negatives: usize,

    /// Fraction of data for validation (0.0 to 1.0)
    #[arg(long, default_value = "0.1")]
    val_split: f32,

    /// Validate every N epochs
    #[arg(long, default_value = "5")]
    val_every: usize,

    /// Early stopping patience (epochs without improvement)
    #[arg(long, default_value = "5")]
    early_stopping: usize,

    /// Top-k for evaluation metrics
    #[arg(long, default_value = "10")]
    top_k: usize,

    /// Number of positives per query for training data generation
    #[arg(long, default_value = "1")]
    positives_per_query: usize,

    /// Random seed
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Print verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Navigator Training ===\n");

    // Initialize GPU backend
    ensure_backend();
    let device = init_gpu_device();
    if args.verbose {
        println!("GPU device initialized");
    }

    // Load embeddings
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
    let examples = generate_pairs_from_similarity(
        &sphere_ebm,
        args.positives_per_query,
        args.negatives,
        &device,
    );
    println!("  Generated {} training examples", examples.len());

    if examples.is_empty() {
        anyhow::bail!("No training examples generated. Need at least 2 embeddings.");
    }

    // Create dataset with train/val split
    let dataset = TrainingDataset::from_examples(examples, args.val_split, args.seed);
    println!(
        "  Train: {} examples, Validation: {} examples",
        dataset.n_train(),
        dataset.n_val()
    );

    // Configure training
    let base_config = NavigatorTrainingConfig::default()
        .with_learning_rate(args.lr)
        .with_negatives(args.negatives)
        .with_momentum(0.9);

    let extended_config = ExtendedTrainingConfig::new(base_config)
        .with_val_every(args.val_every)
        .with_early_stopping(args.early_stopping)
        .with_top_k(args.top_k);

    println!("\nTraining configuration:");
    println!("  Epochs: {}", args.epochs);
    println!("  Batch size: {}", args.batch_size);
    println!("  Learning rate: {}", args.lr);
    println!("  Negatives per positive: {}", args.negatives);
    println!("  Validation every {} epochs", args.val_every);
    println!("  Early stopping patience: {}", args.early_stopping);

    // Create trainable navigator
    let mut trainable = TrainableNavigatorEBM::from_navigator(
        thrml_sphere::NavigatorEBM::from_sphere_ebm(sphere_ebm),
        extended_config.base.clone(),
    );

    // Train
    println!("\n=== Training ===\n");
    let key = RngKey::new(args.seed);
    let report = trainable.train_with_validation(
        &dataset,
        args.epochs,
        args.batch_size,
        &extended_config,
        key,
        &device,
    );

    // Print training report
    println!("\n=== Training Complete ===\n");
    println!("{}", report);

    // Final evaluation
    println!("=== Final Evaluation ===\n");
    let eval_key = RngKey::new(args.seed + 1);
    let final_metrics = evaluate_navigator(
        &trainable.navigator,
        &dataset.validation,
        args.top_k,
        eval_key,
        &device,
    );
    println!("{}", final_metrics);

    // Summary
    println!("\n=== Summary ===");
    println!("Best validation MRR: {:.4}", report.best_val_mrr);
    println!("Best epoch: {}", report.best_epoch);
    if report.early_stopped {
        println!("Training stopped early due to no improvement");
    }
    println!("Final weights: {:?}", report.final_weights);

    println!("\nDone!");

    Ok(())
}
