//! # Categorical Sampling Example
//!
//! This example demonstrates categorical variable sampling in THRML:
//! - Creating a grid graph with categorical nodes
//! - Setting up a CategoricalEBMFactor for nearest-neighbor interactions
//! - Bipartite graph coloring for efficient block sampling
//! - Running **full** Gibbs sampling using FactorSamplingProgram
//! - Visualizing the sampled states
//!
//! This is a Rust port of the Python `00_probabilistic_computing.ipynb` notebook.

use burn::tensor::{Distribution, Tensor};
use indexmap::IndexMap;
use plotters::prelude::*;
use thrml_core::backend::{init_gpu_device, WgpuBackend};
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};
use thrml_examples::{bipartite_coloring, ensure_output_dir, generate_lattice_graph};
use thrml_models::{AbstractFactor, CategoricalEBMFactor, FactorSamplingProgram};
use thrml_samplers::{
    sample_states, BlockGibbsSpec, CategoricalGibbsConditional, DynConditionalSampler, RngKey,
    SamplingSchedule,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== THRML Categorical Sampling Example ===\n");

    // Initialize GPU device
    println!("Initializing GPU device...");
    let device = init_gpu_device();
    println!("Device initialized.\n");

    // Create output directory
    let output_dir = ensure_output_dir()?;

    // ========================================
    // 1. Create a grid graph with categorical nodes
    // ========================================
    let side_length = 10; // Smaller for faster demo
    let n_categories = 5u8;

    println!(
        "Creating {}x{} grid with {} categories per node...",
        side_length, side_length, n_categories
    );

    let node_type = NodeType::Categorical { n_categories };
    let (graph, node_grid) = generate_lattice_graph(side_length, side_length, node_type.clone());

    // Get edge endpoints as separate lists (like Python's u, v)
    let mut u_nodes: Vec<Node> = Vec::new();
    let mut v_nodes: Vec<Node> = Vec::new();

    for edge_idx in graph.edge_indices() {
        if let Some((a_idx, b_idx)) = graph.edge_endpoints(edge_idx) {
            if let (Some(a), Some(b)) = (graph.node_weight(a_idx), graph.node_weight(b_idx)) {
                u_nodes.push(a.clone());
                v_nodes.push(b.clone());
            }
        }
    }

    println!(
        "Created grid with {} nodes and {} edges",
        graph.node_count(),
        u_nodes.len()
    );

    // ========================================
    // 2. Bipartite coloring for block sampling
    // ========================================
    println!("\nPerforming bipartite coloring...");
    let coloring = bipartite_coloring(&node_grid);

    // Collect nodes by color
    let mut color0_nodes: Vec<Node> = Vec::new();
    let mut color1_nodes: Vec<Node> = Vec::new();

    for (&node_idx, &color) in &coloring {
        if let Some(node) = graph.node_weight(node_idx) {
            if color == 0 {
                color0_nodes.push(node.clone());
            } else {
                color1_nodes.push(node.clone());
            }
        }
    }

    let block0 = Block::new(color0_nodes.clone()).expect("Failed to create block 0");
    let block1 = Block::new(color1_nodes.clone()).expect("Failed to create block 1");
    let blocks = [block0.clone(), block1.clone()];

    println!("Created 2 sampling blocks:");
    println!("  Block 0: {} nodes", blocks[0].len());
    println!("  Block 1: {} nodes", blocks[1].len());

    // ========================================
    // 3. Define coupling interaction (Potts model)
    // ========================================
    println!("\nSetting up Potts model coupling interaction...");

    // Temperature parameter
    let beta = 1.0f32;

    // Identity matrix for each edge - promotes same-category neighbors
    // This implements W^{ij} = β * δ(x_i, x_j)
    let n_edges = u_nodes.len();
    let mut weights_data: Vec<f32> =
        Vec::with_capacity(n_edges * n_categories as usize * n_categories as usize);
    for _edge_idx in 0..n_edges {
        for i in 0..n_categories {
            for j in 0..n_categories {
                if i == j {
                    weights_data.push(beta);
                } else {
                    weights_data.push(0.0);
                }
            }
        }
    }

    let weights_1d: Tensor<WgpuBackend, 1> = Tensor::from_data(weights_data.as_slice(), &device);
    let weights: Tensor<WgpuBackend, 3> =
        weights_1d.reshape([n_edges as i32, n_categories as i32, n_categories as i32]);

    // Create the CategoricalEBMFactor
    let u_block = Block::new(u_nodes).expect("Failed to create u block");
    let v_block = Block::new(v_nodes).expect("Failed to create v block");
    let coupling_factor = CategoricalEBMFactor::new(vec![u_block, v_block], weights)
        .expect("Failed to create coupling factor");

    println!("Created coupling factor with {} interactions", n_edges);

    // ========================================
    // 4. Create BlockGibbsSpec and samplers
    // ========================================
    println!("\nSetting up sampling program...");

    // Node shape/dtype spec
    let mut node_shape_dtypes = IndexMap::new();
    node_shape_dtypes.insert(
        node_type,
        thrml_core::node::TensorSpec {
            shape: vec![1],
            dtype: burn::tensor::DType::U8,
        },
    );

    // Create BlockGibbsSpec - each block is a superblock
    let gibbs_spec = BlockGibbsSpec::new(
        vec![vec![block0], vec![block1]], // Two superblocks, each with one block
        vec![],                           // No clamped blocks
        node_shape_dtypes,
    )
    .expect("Failed to create gibbs spec");

    // Create samplers - one for each free block
    let samplers: Vec<Box<dyn DynConditionalSampler>> = vec![
        Box::new(CategoricalGibbsConditional::new(n_categories as usize, 0)),
        Box::new(CategoricalGibbsConditional::new(n_categories as usize, 0)),
    ];

    // Create FactorSamplingProgram
    let factors: Vec<&dyn AbstractFactor> = vec![coupling_factor.inner()];
    let program = FactorSamplingProgram::new(
        gibbs_spec,
        samplers,
        &factors,
        vec![], // No additional interaction groups
        &device,
    )
    .expect("Failed to create sampling program");

    println!("Sampling program created.");

    // ========================================
    // 5. Initialize states and run ACTUAL sampling
    // ========================================
    println!("\nInitializing states...");

    let seed = 4242u64;
    let key = RngKey::new(seed);

    // Initialize random categorical values for each block (1D for single chain)
    let init_state: Vec<Tensor<WgpuBackend, 1>> = blocks
        .iter()
        .map(|block| {
            // Random values in [0, n_categories), floored to integers
            let rand_vals: Tensor<WgpuBackend, 1> = Tensor::random(
                [block.len()],
                Distribution::Uniform(0.0, n_categories as f64),
                &device,
            );
            rand_vals.floor()
        })
        .collect();

    println!("Created initial state");

    // Define sampling schedule
    let schedule = SamplingSchedule::new(
        50, // n_warmup: warmup iterations
        20, // n_samples: number of samples to collect
        5,  // steps_per_sample: steps between samples
    );

    println!(
        "Sampling schedule: {} warmup, {} samples, {} steps/sample",
        schedule.n_warmup, schedule.n_samples, schedule.steps_per_sample
    );

    // Create block to sample from (all nodes)
    let all_nodes: Vec<Node> = color0_nodes
        .iter()
        .chain(color1_nodes.iter())
        .cloned()
        .collect();
    let all_nodes_block = Block::new(all_nodes).expect("Failed to create all nodes block");

    // RUN ACTUAL GIBBS SAMPLING
    println!("\nRunning Gibbs sampling...");
    let samples = sample_states(
        key,
        &program.program,
        &schedule,
        init_state,
        &[], // No clamped state
        &[all_nodes_block],
        &device,
    )
    .expect("Sampling failed");

    println!(
        "Sampling complete! Collected {} samples",
        samples[0].dims()[0]
    );

    // ========================================
    // 6. Visualize final sample as heatmap
    // ========================================
    println!("\n=== Creating Visualization ===");

    // Extract the last sample for visualization
    let last_sample_idx = samples[0].dims()[0] - 1;
    let indices: Tensor<WgpuBackend, 1, burn::tensor::Int> =
        Tensor::from_data([last_sample_idx as i32].as_slice(), &device);
    let final_sample: Tensor<WgpuBackend, 2> = samples[0].clone().select(0, indices);
    let n = final_sample.dims()[1] as i32;
    let final_state: Tensor<WgpuBackend, 1> = final_sample.reshape([n]);
    let state_data: Vec<f32> = final_state.into_data().to_vec().expect("read state data");

    let output_path = output_dir.join("categorical_sampling_state.png");
    println!("Saving heatmap to: {:?}", output_path);

    let root = plotters::backend::BitMapBackend::new(&output_path, (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Gibbs Sampled Categorical States", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..side_length, 0..side_length)?;

    chart.configure_mesh().draw()?;

    // Draw colored rectangles for each cell
    let color_palette = [
        RGBColor(31, 119, 180),  // Blue
        RGBColor(255, 127, 14),  // Orange
        RGBColor(44, 160, 44),   // Green
        RGBColor(214, 39, 40),   // Red
        RGBColor(148, 103, 189), // Purple
    ];

    // Map node indices to grid positions
    // The samples are ordered by [color0_nodes..., color1_nodes...]
    let n_color0 = color0_nodes.len();

    for row in 0..side_length {
        for col in 0..side_length {
            // Determine which color group this cell belongs to
            let is_color0 = (row + col) % 2 == 0;

            // Find the index within that color group
            let idx_in_color = if is_color0 {
                // Count how many color0 cells come before this one
                let mut count = 0;
                for r in 0..side_length {
                    for c in 0..side_length {
                        if (r + c) % 2 == 0 {
                            if r == row && c == col {
                                break;
                            }
                            count += 1;
                        }
                    }
                    if r == row {
                        break;
                    }
                }
                count
            } else {
                // Count how many color1 cells come before this one
                let mut count = 0;
                for r in 0..side_length {
                    for c in 0..side_length {
                        if (r + c) % 2 == 1 {
                            if r == row && c == col {
                                break;
                            }
                            count += 1;
                        }
                    }
                    if r == row {
                        break;
                    }
                }
                count + n_color0
            };

            if idx_in_color < state_data.len() {
                let category = state_data[idx_in_color] as usize % n_categories as usize;
                let color = color_palette[category];

                chart.draw_series(std::iter::once(Rectangle::new(
                    [(col, side_length - 1 - row), (col + 1, side_length - row)],
                    color.filled(),
                )))?;
            }
        }
    }

    root.present()?;
    println!("Heatmap saved successfully!");

    // ========================================
    // 7. Display sample statistics
    // ========================================
    println!("\n=== Sample Statistics ===");

    // Count category frequencies
    let mut category_counts = vec![0usize; n_categories as usize];
    for &val in &state_data {
        let cat = val as usize % n_categories as usize;
        category_counts[cat] += 1;
    }

    println!("Category distribution in final sample:");
    for (cat, count) in category_counts.iter().enumerate() {
        let pct = (*count as f64 / state_data.len() as f64) * 100.0;
        println!("  Category {}: {} nodes ({:.1}%)", cat, count, pct);
    }

    println!("\nCategory colors:");
    for (i, color) in color_palette.iter().enumerate() {
        println!(
            "  Category {}: RGB({}, {}, {})",
            i, color.0, color.1, color.2
        );
    }

    println!("\n=== Example Complete ===");
    println!("Output saved to: {:?}", output_dir);
    println!("\nThis example ran FULL Gibbs sampling with:");
    println!("  - {} warmup iterations", schedule.n_warmup);
    println!("  - {} collected samples", schedule.n_samples);
    println!("  - {} steps between samples", schedule.steps_per_sample);

    Ok(())
}
