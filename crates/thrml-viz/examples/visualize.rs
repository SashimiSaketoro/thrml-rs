//! TheSphere 3D Visualizer
//!
//! Interactive visualization of sphere-optimized embeddings and ROOTS hierarchy.
//!
//! # Usage
//!
//! ## Standalone viewing
//!
//! View a SafeTensors file:
//! ```bash
//! cargo run --example visualize -- --input output.safetensors
//! ```
//!
//! View sphere coordinates (NPZ):
//! ```bash
//! cargo run --example visualize -- --input sphere.npz
//! ```
//!
//! ## Live monitoring
//!
//! Start the visualizer in monitor mode:
//! ```bash
//! cargo run --example visualize -- --monitor my-session
//! ```
//!
//! Then run ingestion with viz notifications:
//! ```bash
//! blt-burn ingest --viz-session my-session input.txt
//! ```

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use thrml_viz::{run, VizConfig};

#[derive(Parser)]
#[command(name = "sphere-viz")]
#[command(author, version, about = "TheSphere 3D Visualizer")]
struct Args {
    /// Input file path (SafeTensors or NPZ)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Monitor live ingestion with this session ID
    #[arg(long)]
    monitor: Option<String>,

    /// Show ROOTS hierarchy tree
    #[arg(long)]
    tree: bool,

    /// Number of ROOTS partitions (if building tree)
    #[arg(long, default_value = "32")]
    partitions: usize,

    /// Window width
    #[arg(long, default_value = "1280")]
    width: u32,

    /// Window height
    #[arg(long, default_value = "800")]
    height: u32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate arguments
    if args.input.is_none() && args.monitor.is_none() {
        eprintln!("Error: Must provide either --input or --monitor");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  View file:    sphere-viz --input output.safetensors");
        eprintln!("  Live monitor: sphere-viz --monitor my-session");
        std::process::exit(1);
    }

    // Build configuration
    let config = VizConfig {
        input: args.input,
        monitor_session: args.monitor,
        show_tree: args.tree,
        partitions: args.partitions,
        window_size: (args.width, args.height),
    };

    // Print startup info
    println!("=== TheSphere Visualizer ===");
    println!();
    if let Some(ref path) = config.input {
        println!("Input: {:?}", path);
    }
    if let Some(ref session) = config.monitor_session {
        println!("Monitoring session: {}", session);
        println!("Waiting for ingestion to connect...");
    }
    println!();
    println!("Controls:");
    println!("  Left-click drag:  Orbit camera");
    println!("  Right-click drag: Pan camera");
    println!("  Scroll:           Zoom in/out");
    println!("  H:                Toggle help");
    println!();

    // Run the application
    run(config)
}
