//! Build ROOTS index from sphere embeddings.
//!
//! This example demonstrates how to build a ROOTS index for coarse navigation
//! with optional substring-enhanced partitioning.
//!
//! # Usage
//!
//! ```bash
//! # From BLT v3 output (recommended - includes bytes automatically)
//! cargo run --example build_roots -- \
//!   --blt-safetensors output.safetensors \
//!   --substring-coupling \
//!   --partitions 64
//!
//! # Legacy: separate embeddings and bytes files
//! cargo run --example build_roots -- \
//!   --input sphere.safetensors \
//!   --bytes patches.txt \
//!   --substring-coupling \
//!   --alpha 0.7 --beta 0.3
//!
//! # Custom partition count
//! cargo run --example build_roots -- \
//!   --blt-safetensors output.safetensors \
//!   --partitions 512 \
//!   --min-partition-size 50
//! ```
//!
//! # Input Formats
//!
//! ## BLT v3 format (--blt-safetensors)
//! Single SafeTensors file containing:
//! - `embeddings`: [N, D] float32 - Patch-level embedding vectors
//! - `prominence`: [N] float32 - Prominence scores
//! - `bytes`: [total_bytes] U8 - Concatenated raw bytes
//! - `patch_lengths`: [N] I32 - Length of each patch
//!
//! ## Legacy format (--input + --bytes)
//! - `--input`: SafeTensors with embeddings/prominence only
//! - `--bytes`: Text file with one byte sequence per line
//!
//! # Output Format
//!
//! The output will contain the serialized RootsIndex.

use anyhow::{Context, Result};
use clap::Parser;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_samplers::RngKey;
use thrml_sphere::{
    load_blt_safetensors, load_from_safetensors, RootsConfig, RootsIndex, ScaleProfile,
    SphereConfig, SubstringConfig,
};

#[derive(Parser)]
#[command(name = "build_roots")]
#[command(author, version, about = "Build ROOTS index from sphere embeddings")]
struct Args {
    /// BLT v3 SafeTensors file (contains embeddings + bytes together)
    /// Preferred format - produces from `blt-burn ingest` with updated output
    #[arg(long)]
    blt_safetensors: Option<PathBuf>,

    /// Legacy: Input SafeTensors file containing embeddings and prominence only
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output file for ROOTS index (binary)
    #[arg(short, long, default_value = "roots_index.bin")]
    output: PathBuf,

    /// Legacy: Optional bytes file for substring-enhanced partitioning
    /// (one byte sequence per line, or length-prefixed binary)
    #[arg(short, long)]
    bytes: Option<PathBuf>,

    /// Number of partitions (power of 2 recommended)
    #[arg(long, default_value = "256")]
    partitions: usize,

    /// Minimum points per partition
    #[arg(long, default_value = "10")]
    min_partition_size: usize,

    /// Enable substring-enhanced coupling
    #[arg(long)]
    substring_coupling: bool,

    /// Weight for embedding similarity in coupling (α)
    #[arg(long, default_value = "0.7")]
    alpha: f64,

    /// Weight for substring similarity in coupling (β)
    #[arg(long, default_value = "0.3")]
    beta: f64,

    /// Minimum substring length for matching
    #[arg(long, default_value = "4")]
    min_substring_len: usize,

    /// Scale profile for sphere config: dev, medium, large
    #[arg(long, default_value = "dev")]
    scale: String,

    /// Inverse temperature for Ising partitioning (higher = sharper cuts)
    #[arg(long, default_value = "1.0")]
    ising_beta: f32,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Print verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== ROOTS Index Builder ===\n");

    // Initialize GPU backend
    ensure_backend();
    let device = init_gpu_device();
    if args.verbose {
        println!("GPU device initialized");
    }

    // Parse scale profile
    let profile = ScaleProfile::from_str(&args.scale).unwrap_or_else(|| {
        eprintln!("Warning: Unknown scale '{}', using 'dev'", args.scale);
        ScaleProfile::Dev
    });
    let sphere_config = SphereConfig::from(profile);

    // Load embeddings and optionally bytes - two modes:
    // 1. BLT v3 format (--blt-safetensors): single file with embeddings + bytes
    // 2. Legacy format (--input + optional --bytes): separate files
    let (sphere_ebm, blt_bytes) = if let Some(blt_path) = &args.blt_safetensors {
        println!("Loading BLT v3 SafeTensors from {:?}...", blt_path);
        let (ebm, bytes) = load_blt_safetensors(blt_path, sphere_config, &device)
            .context("Failed to load BLT SafeTensors (is this v3 format with embeddings?)")?;
        println!(
            "  {} patches, {} dimensions, {} total bytes",
            ebm.n_points(),
            ebm.embedding_dim(),
            bytes.iter().map(|b| b.len()).sum::<usize>()
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
             Use --blt-safetensors for BLT v3 format (recommended)\n\
             Use --input for legacy format"
        );
    };

    // Build ROOTS config
    let mut roots_config = RootsConfig::default()
        .with_partitions(args.partitions)
        .with_min_partition_size(args.min_partition_size)
        .with_beta(args.ising_beta);

    // Configure substring coupling if enabled
    if args.substring_coupling {
        let sub_config = SubstringConfig::with_weights(args.alpha, args.beta)
            .with_min_length(args.min_substring_len);
        roots_config = roots_config.with_substring_coupling(sub_config);
        println!(
            "\nSubstring coupling enabled: α={:.2}, β={:.2}, min_len={}",
            args.alpha, args.beta, args.min_substring_len
        );
    }

    println!("\nROOTS Configuration:");
    println!("  Partitions: {}", args.partitions);
    println!("  Min partition size: {}", args.min_partition_size);
    println!("  Ising β: {}", args.ising_beta);

    // Determine which bytes to use:
    // - BLT v3 mode: use bytes from SafeTensors automatically
    // - Legacy mode: use --bytes file if provided
    let raw_bytes: Option<Vec<Vec<u8>>> = if args.substring_coupling {
        if let Some(bytes) = blt_bytes {
            // BLT v3 mode - bytes already loaded
            println!("\n  Using embedded bytes from BLT SafeTensors");
            Some(bytes)
        } else if let Some(bytes_path) = &args.bytes {
            // Legacy mode - load from separate file
            println!("\nLoading bytes from {:?}...", bytes_path);
            Some(load_bytes_file(bytes_path, sphere_ebm.n_points())?)
        } else {
            eprintln!("Warning: --substring-coupling enabled but no bytes available");
            eprintln!("         Use --blt-safetensors or provide --bytes file");
            None
        }
    } else {
        None
    };

    // Build ROOTS index
    println!("\nBuilding ROOTS index...");
    let key = RngKey::new(args.seed);

    let roots = if let Some(bytes) = &raw_bytes {
        println!("  Using substring-enhanced partitioning");
        RootsIndex::from_sphere_ebm_with_bytes(&sphere_ebm, bytes, roots_config, key, &device)
    } else {
        RootsIndex::from_sphere_ebm(&sphere_ebm, roots_config, key, &device)
    };

    // Print stats
    let stats = roots.stats();
    println!("\n{}", stats);

    // Save output
    println!("\nSaving to {:?}...", args.output);
    save_roots_index(&roots, &args.output)?;

    println!("\nDone! ROOTS index saved.");

    Ok(())
}

/// Load bytes from a file (newline-separated text or binary)
fn load_bytes_file(path: &PathBuf, expected_count: usize) -> Result<Vec<Vec<u8>>> {
    let file = fs::File::open(path).context("Failed to open bytes file")?;
    let reader = BufReader::new(file);

    let mut bytes: Vec<Vec<u8>> = Vec::new();

    for line in reader.lines() {
        let line = line.context("Failed to read line")?;
        bytes.push(line.into_bytes());
    }

    if bytes.len() != expected_count {
        eprintln!(
            "Warning: bytes file has {} entries, expected {}",
            bytes.len(),
            expected_count
        );
        // Pad or truncate
        bytes.resize(expected_count, Vec::new());
    }

    println!("  Loaded {} byte sequences", bytes.len());
    Ok(bytes)
}

/// Save ROOTS index (placeholder - implement serialization as needed)
fn save_roots_index(roots: &RootsIndex, path: &PathBuf) -> Result<()> {
    // For now, just save the stats as JSON
    let stats = roots.stats();
    let json = format!(
        r#"{{
  "n_partitions": {},
  "n_points": {},
  "embedding_dim": {},
  "min_partition_size": {},
  "max_partition_size": {},
  "mean_partition_size": {:.2},
  "has_classifier": {},
  "has_ngrams": {}
}}"#,
        stats.n_partitions,
        stats.n_points,
        stats.embedding_dim,
        stats.min_partition_size,
        stats.max_partition_size,
        stats.mean_partition_size,
        stats.has_classifier,
        stats.has_ngrams
    );

    fs::write(path, json).context("Failed to write ROOTS index")?;

    // TODO: Implement full binary serialization with serde
    // fs::write(path, bincode::serialize(roots)?)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_substring_config_parsing() {
        let config = SubstringConfig::with_weights(0.5, 0.5).with_min_length(8);
        assert!((config.alpha - 0.5).abs() < 0.01);
        assert!((config.beta - 0.5).abs() < 0.01);
        assert_eq!(config.min_length, 8);
    }
}
