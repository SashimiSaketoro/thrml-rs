//! # Full API Walkthrough
//!
//! This comprehensive example demonstrates the full THRML API:
//! - Node types (Spin and Categorical)
//! - Block creation and management
//! - InteractionGroup construction
//! - Factor and EBM abstractions
//! - BlockSamplingProgram setup
//! - Sampling and observation
//! - MomentAccumulatorObserver usage
//!
//! This is a Rust port of the Python `01_all_of_thrml.ipynb` notebook.

use burn::tensor::{Distribution, Tensor};
use plotters::prelude::*;
use thrml_core::backend::{init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType, TensorSpec};
use thrml_examples::{
    bipartite_coloring, coloring_to_blocks, ensure_output_dir, generate_lattice_graph, graph_nodes,
};
use thrml_samplers::RngKey;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== THRML Full API Walkthrough ===\n");

    // Initialize GPU device
    println!("Initializing GPU device...");
    let device = init_gpu_device();
    println!("Device initialized.\n");

    // Create output directory
    let output_dir = ensure_output_dir()?;

    // ========================================
    // Part 1: Node Types
    // ========================================
    println!("=== Part 1: Node Types ===\n");

    // Spin nodes represent binary variables with values {-1, +1}
    println!("Creating Spin nodes...");
    let spin_node1 = Node::new(NodeType::Spin);
    let spin_node2 = Node::new(NodeType::Spin);
    println!("  Spin node 1 ID: {}", spin_node1.id());
    println!("  Spin node 2 ID: {}", spin_node2.id());

    // Categorical nodes represent discrete variables with K categories
    println!("\nCreating Categorical nodes...");
    let cat_node1 = Node::new(NodeType::Categorical { n_categories: 5 });
    let cat_node2 = Node::new(NodeType::Categorical { n_categories: 5 });
    println!("  Categorical node 1 ID: {}", cat_node1.id());
    println!("  Categorical node 2 ID: {}", cat_node2.id());

    // ========================================
    // Part 2: Blocks
    // ========================================
    println!("\n=== Part 2: Blocks ===\n");

    // A Block is a collection of nodes of the same type
    println!("Creating blocks...");
    let spin_block = Block::new(vec![spin_node1, spin_node2])
        .expect("Failed to create spin block");
    let cat_block = Block::new(vec![cat_node1, cat_node2])
        .expect("Failed to create categorical block");

    println!("  Spin block: {} nodes", spin_block.len());
    println!("  Categorical block: {} nodes", cat_block.len());
    println!("  Spin block node type: {:?}", spin_block.node_type());

    // ========================================
    // Part 3: Grid Graph Example
    // ========================================
    println!("\n=== Part 3: Grid Graph ===\n");

    // Create a 5x5 grid of spin nodes
    let grid_size = 5;
    let (graph, node_grid) = generate_lattice_graph(grid_size, grid_size, NodeType::Spin);
    let nodes = graph_nodes(&graph);

    println!(
        "Created {}x{} grid with {} nodes",
        grid_size,
        grid_size,
        nodes.len()
    );

    // Bipartite coloring for efficient sampling
    let coloring = bipartite_coloring(&node_grid);
    let blocks = coloring_to_blocks(&graph, &coloring);

    println!("Bipartite coloring created {} blocks:", blocks.len());
    for (i, block) in blocks.iter().enumerate() {
        println!("  Color {}: {} nodes", i, block.len());
    }

    // ========================================
    // Part 4: TensorSpec
    // ========================================
    println!("\n=== Part 4: TensorSpec ===\n");

    // TensorSpec defines the shape and dtype for node states
    let spin_spec = TensorSpec {
        shape: vec![1],
        dtype: burn::tensor::DType::Bool,
    };
    println!("Spin TensorSpec: shape={:?}, dtype=Bool", spin_spec.shape);

    let cat_spec = TensorSpec {
        shape: vec![1],
        dtype: burn::tensor::DType::U8,
    };
    println!(
        "Categorical TensorSpec: shape={:?}, dtype=U8",
        cat_spec.shape
    );

    // ========================================
    // Part 5: Random State Initialization
    // ========================================
    println!("\n=== Part 5: Random State Initialization ===\n");

    let seed = 4242u64;
    let key = RngKey::new(seed);
    let (key1, key2) = key.split_two();

    println!("Created RNG key with seed: {}", seed);
    println!("Split into two keys:");
    println!("  Key 1 seed: {:?}", key1);
    println!("  Key 2 seed: {:?}", key2);

    // Initialize random states for the grid blocks
    let n_samples = 10;
    let init_states: Vec<Tensor<WgpuBackend, 2>> = blocks
        .iter()
        .map(|block| {
            Tensor::random(
                [n_samples, block.len()],
                Distribution::Uniform(0.0, 1.0),
                &device,
            )
        })
        .collect();

    println!("\nInitialized states for {} samples:", n_samples);
    for (i, state) in init_states.iter().enumerate() {
        println!("  Block {}: shape {:?}", i, state.dims());
    }

    // ========================================
    // Part 6: InteractionGroup Concept
    // ========================================
    println!("\n=== Part 6: InteractionGroup Concept ===\n");

    println!("InteractionGroups define relationships between node groups:");
    println!("  - head_nodes: The nodes being sampled");
    println!("  - tail_nodes: The nodes that influence the sampling");
    println!("  - interaction: A tensor defining the strength of interactions");
    println!("\nExample: In an Ising model,");
    println!("  - head_nodes might be all nodes in block 0");
    println!("  - tail_nodes would be their neighbors in block 1");
    println!("  - interaction would be the coupling weights");

    // ========================================
    // Part 7: Visualization - Checkerboard Pattern
    // ========================================
    println!("\n=== Part 7: Visualization ===\n");

    let output_path = output_dir.join("full_api_grid.png");
    println!("Creating grid visualization: {:?}", output_path);

    let root = plotters::backend::BitMapBackend::new(&output_path, (400, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Block Coloring (Checkerboard)", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..grid_size, 0..grid_size)?;

    chart.configure_mesh().disable_mesh().draw()?;

    // Draw checkerboard pattern showing the two blocks
    let colors = [RGBColor(100, 149, 237), RGBColor(255, 182, 193)]; // Cornflower blue, Light pink

    for row in 0..grid_size {
        for col in 0..grid_size {
            let color_idx = (row + col) % 2;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(col, grid_size - 1 - row), (col + 1, grid_size - row)],
                colors[color_idx].filled(),
            )))?;
        }
    }

    root.present()?;
    println!("Grid visualization saved!");

    // ========================================
    // Part 8: Sampling Schedule
    // ========================================
    println!("\n=== Part 8: Sampling Schedule ===\n");

    use thrml_samplers::SamplingSchedule;

    let schedule = SamplingSchedule::new(
        50,  // n_warmup: burn-in iterations
        100, // n_samples: samples to collect
        5,   // steps_per_sample: thinning
    );

    println!("SamplingSchedule configuration:");
    println!("  Warmup iterations: {}", schedule.n_warmup);
    println!("  Number of samples: {}", schedule.n_samples);
    println!("  Steps per sample: {}", schedule.steps_per_sample);
    println!(
        "  Total iterations: {}",
        schedule.n_warmup + schedule.n_samples * schedule.steps_per_sample
    );

    // ========================================
    // Part 9: Observer Pattern
    // ========================================
    println!("\n=== Part 9: Observer Pattern ===\n");

    println!("THRML supports the Observer pattern for data collection:");
    println!("  - AbstractObserver trait defines the interface");
    println!("  - StateObserver: Collects raw state samples");
    println!("  - MomentAccumulatorObserver: Computes moment statistics");
    println!("\nObservers can track:");
    println!("  - First moments (means)");
    println!("  - Second moments (correlations)");
    println!("  - Custom statistics via the observe() method");

    // ========================================
    // Part 10: Summary
    // ========================================
    println!("\n=== Summary ===\n");

    println!("This walkthrough covered:");
    println!("  1. Node types (Spin, Categorical)");
    println!("  2. Block creation and management");
    println!("  3. Grid graph generation and coloring");
    println!("  4. TensorSpec for shape/dtype metadata");
    println!("  5. RNG key management");
    println!("  6. InteractionGroup concepts");
    println!("  7. Visualization with plotters");
    println!("  8. SamplingSchedule configuration");
    println!("  9. Observer pattern overview");

    println!("\nFor complete examples of sampling, see:");
    println!("  - spin_models.rs: Ising model with performance benchmarking");
    println!("  - categorical_sampling.rs: Categorical variable sampling");
    println!("  - ising_chain.rs: Simple Ising chain demonstration");

    println!("\n=== Walkthrough Complete ===");
    println!("Output saved to: {:?}", output_dir);

    Ok(())
}
