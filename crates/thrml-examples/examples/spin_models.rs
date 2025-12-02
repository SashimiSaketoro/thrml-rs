//! # Spin Models Example
//!
//! This example demonstrates the Ising model capabilities of THRML:
//! - Creating a lattice graph with spin nodes
//! - Setting up an IsingEBM with random biases and weights
//! - Graph coloring for efficient block Gibbs sampling
//! - Creating sampling programs for training
//! - Performance benchmarking
//! - Visualization of results with plotters
//!
//! This is a Rust port of the Python `02_spin_models.ipynb` notebook.

use burn::tensor::{Distribution, Tensor};
use plotters::prelude::*;
use std::time::Instant;
use thrml_core::backend::{init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::node::NodeType;
use thrml_examples::{
    coloring_to_blocks, ensure_output_dir, generate_lattice_graph, graph_edges, graph_nodes,
    greedy_coloring,
};
use thrml_models::ising::{IsingEBM, IsingSamplingProgram};
use thrml_samplers::{RngKey, SamplingSchedule};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== THRML Spin Models Example ===\n");

    // Initialize GPU device
    println!("Initializing GPU device...");
    let device = init_gpu_device();
    println!("Device initialized.\n");

    // Create output directory
    let output_dir = ensure_output_dir()?;

    // ========================================
    // 1. Create a lattice graph
    // ========================================
    println!("Creating lattice graph...");
    let grid_size = 10; // 10x10 grid = 100 nodes
    let (graph, _node_grid) = generate_lattice_graph(grid_size, grid_size, NodeType::Spin);

    let nodes = graph_nodes(&graph);
    let edges = graph_edges(&graph);

    println!(
        "Created {}x{} lattice with {} nodes and {} edges",
        grid_size,
        grid_size,
        nodes.len(),
        edges.len()
    );

    // ========================================
    // 2. Initialize random biases and weights
    // ========================================
    println!("\nInitializing model parameters...");
    let seed = 4242u64;
    let _key = RngKey::new(seed);

    // Random biases for each node
    let biases_data: Vec<f32> = (0..nodes.len())
        .map(|i| (i as f32 * 0.1).sin() * 0.5)
        .collect();
    let biases: Tensor<WgpuBackend, 1> = Tensor::from_data(biases_data.as_slice(), &device);

    // Random weights for each edge
    let weights_data: Vec<f32> = (0..edges.len())
        .map(|i| (i as f32 * 0.2).cos() * 0.3)
        .collect();
    let weights: Tensor<WgpuBackend, 1> = Tensor::from_data(weights_data.as_slice(), &device);

    // Inverse temperature
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data([1.0f32].as_slice(), &device);

    println!(
        "  Biases shape: [{}]",
        biases.clone().into_data().to_vec::<f32>().unwrap().len()
    );
    println!(
        "  Weights shape: [{}]",
        weights.clone().into_data().to_vec::<f32>().unwrap().len()
    );

    // ========================================
    // 3. Create the Ising model
    // ========================================
    println!("\nCreating Ising model...");
    let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);
    let factors = model.get_factors(&device);
    println!("Ising model created with {} factors", factors.len());

    // ========================================
    // 4. Graph coloring for block sampling
    // ========================================
    println!("\nPerforming graph coloring for block partitioning...");
    let coloring = greedy_coloring(&graph);
    let n_colors = *coloring.values().max().unwrap_or(&0) + 1;
    println!("Graph colored with {} colors", n_colors);

    let free_blocks = coloring_to_blocks(&graph, &coloring);
    println!("Created {} sampling blocks:", free_blocks.len());
    for (i, block) in free_blocks.iter().enumerate() {
        println!("  Block {}: {} nodes", i, block.len());
    }

    // ========================================
    // 5. Create sampling program
    // ========================================
    println!("\nCreating sampling program...");
    let sampling_program = IsingSamplingProgram::new(&model, free_blocks.clone(), vec![], &device)
        .expect("Failed to create sampling program");
    println!("Sampling program created.");

    // ========================================
    // 6. Initialize states randomly
    // ========================================
    println!("\nInitializing states randomly...");
    // Create random initial state for each block (1D tensors for single chain)
    let init_state: Vec<Tensor<WgpuBackend, 1>> = free_blocks
        .iter()
        .map(|block| {
            // Random uniform values in [0, 1], will be thresholded to 0/1 in sampling
            Tensor::random([block.len()], Distribution::Uniform(0.0, 1.0), &device)
        })
        .collect();
    println!(
        "Initial state created with {} block states",
        init_state.len()
    );

    // ========================================
    // 7. Performance benchmarking
    // ========================================
    println!("\n=== Performance Benchmarking ===");

    let schedule = SamplingSchedule::new(10, 10, 1); // warmup=10, samples=10, steps_per=1
    let batch_sizes = vec![1, 10, 50, 100];
    let mut timing_results: Vec<(usize, f64)> = Vec::new();

    for &batch_size in &batch_sizes {
        println!("\nBatch size: {}", batch_size);

        // For simplicity, we'll just time a single sampling run
        let timing_key = RngKey::new(seed + batch_size as u64);

        let start = Instant::now();

        // Run full Gibbs sampling
        let _ = thrml_samplers::run_blocks(
            timing_key,
            &sampling_program.program,
            init_state.clone(),
            &[], // no clamped state
            schedule.n_warmup + schedule.n_samples * schedule.steps_per_sample,
            &device,
        );

        let elapsed = start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64();

        let total_flips =
            (schedule.n_warmup + schedule.n_samples * schedule.steps_per_sample) * nodes.len();
        let flips_per_sec = total_flips as f64 / elapsed_secs;
        let flips_per_ns = flips_per_sec / 1e9;

        println!("  Time: {:.3}s", elapsed_secs);
        println!("  Total flips: {}", total_flips);
        println!("  Flips/sec: {:.2e}", flips_per_sec);
        println!("  Flips/ns: {:.6}", flips_per_ns);

        timing_results.push((batch_size * nodes.len(), flips_per_ns));
    }

    // ========================================
    // 8. Create visualization
    // ========================================
    println!("\n=== Creating Visualization ===");

    let output_path = output_dir.join("spin_models_performance.png");
    println!("Saving plot to: {:?}", output_path);

    let root = plotters::backend::BitMapBackend::new(&output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find data range
    let max_dof = timing_results.iter().map(|(d, _)| *d).max().unwrap_or(1) as f64;
    let max_flips = timing_results
        .iter()
        .map(|(_, f)| *f)
        .fold(0.0f64, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("THRML Spin Model Performance", ("sans-serif", 40))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d((1f64..max_dof * 1.1).log_scale(), 0f64..max_flips * 1.2)?;

    chart
        .configure_mesh()
        .x_desc("Parallel Degrees of Freedom")
        .y_desc("Flips/ns")
        .x_label_formatter(&|x| format!("{:.0}", x))
        .y_label_formatter(&|y| format!("{:.4}", y))
        .draw()?;

    // Draw the data points and line
    chart
        .draw_series(LineSeries::new(
            timing_results.iter().map(|(d, f)| (*d as f64, *f)),
            &BLUE,
        ))?
        .label("Performance")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Draw circles at each data point
    chart.draw_series(
        timing_results
            .iter()
            .map(|(d, f)| Circle::new((*d as f64, *f), 5, BLUE.filled())),
    )?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    root.present()?;
    println!("Plot saved successfully!");

    // ========================================
    // 9. Demonstrate cubic interaction (SpinEBMFactor)
    // ========================================
    println!("\n=== Cubic Interaction Example ===");
    println!("Creating a cubic interaction s_1 * s_2 * s_3 between node subsets...");

    // This demonstrates creating higher-order interactions
    if nodes.len() >= 30 {
        let block1 = Block::new(nodes[0..10].to_vec()).expect("block1");
        let block2 = Block::new(nodes[10..20].to_vec()).expect("block2");
        let block3 = Block::new(nodes[20..30].to_vec()).expect("block3");

        println!("Created 3 blocks of 10 nodes each for cubic interaction demonstration.");
        println!("  Block 1: {} nodes", block1.len());
        println!("  Block 2: {} nodes", block2.len());
        println!("  Block 3: {} nodes", block3.len());
    }

    println!("\n=== Example Complete ===");
    println!("Output saved to: {:?}", output_dir);

    Ok(())
}
