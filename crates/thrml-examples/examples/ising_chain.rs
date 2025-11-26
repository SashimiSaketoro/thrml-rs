use burn::tensor::Tensor;
use thrml_core::backend::WgpuBackend;
/// Ising Chain Example
///
/// This example demonstrates the complete workflow for sampling from an Ising model:
/// 1. Create spin nodes
/// 2. Define biases and edge weights
/// 3. Build the IsingEBM
/// 4. Create a sampling program
/// 5. Run sampling
///
/// The Ising model energy is: E(s) = -β * (Σ b_i * s_i + Σ J_ij * s_i * s_j)
use thrml_core::backend::{ensure_backend, init_gpu_device};
use thrml_core::block::Block;
use thrml_core::node::{Node, NodeType};
use thrml_models::ising::{hinton_init, IsingEBM};
use thrml_samplers::rng::RngKey;
use thrml_samplers::SamplingSchedule;

fn main() {
    println!("THRML Ising Chain Example");
    println!("=========================\n");

    // Initialize GPU (Metal on macOS)
    ensure_backend();
    let device = init_gpu_device();
    println!("✓ GPU device initialized\n");

    // Create a chain of 5 spin nodes
    let n_nodes = 5;
    let nodes: Vec<Node> = (0..n_nodes).map(|_| Node::new(NodeType::Spin)).collect();
    println!("✓ Created {} spin nodes\n", n_nodes);

    // Define edges (linear chain: 0-1-2-3-4)
    let edges: Vec<(Node, Node)> = (0..n_nodes - 1)
        .map(|i| (nodes[i].clone(), nodes[i + 1].clone()))
        .collect();
    println!("✓ Created {} edges (linear chain)\n", edges.len());

    // Define biases (all zero for simplicity)
    let biases: Tensor<WgpuBackend, 1> = Tensor::zeros([n_nodes], &device);

    // Define edge weights (all equal coupling strength)
    let coupling_strength = 1.0f32;
    let weights: Tensor<WgpuBackend, 1> =
        Tensor::from_data(vec![coupling_strength; edges.len()].as_slice(), &device);

    // Temperature parameter (inverse temperature)
    let beta: Tensor<WgpuBackend, 1> = Tensor::from_data(vec![1.0f32].as_slice(), &device);

    // Create the Ising EBM
    let model = IsingEBM::new(nodes.clone(), edges, biases, weights, beta);
    println!("✓ Created IsingEBM\n");

    // Create a single block with all nodes (for simplicity)
    let all_nodes_block = Block::new(nodes.clone()).expect("Failed to create block");

    println!("✓ Created block with {} nodes\n", all_nodes_block.len());

    // Define sampling schedule
    let schedule = SamplingSchedule::new(
        100, // n_warmup: warmup iterations
        10,  // n_samples: number of samples to collect
        5,   // steps_per_sample: MCMC steps between samples
    );
    println!(
        "Sampling schedule: {} warmup, {} samples, {} steps/sample\n",
        schedule.n_warmup, schedule.n_samples, schedule.steps_per_sample
    );

    // Initialize state using Hinton initialization
    let key = RngKey::new(42);
    let batch_shape = vec![1usize];
    let init_states = hinton_init(
        key,
        &model,
        &[all_nodes_block.clone()],
        &batch_shape,
        &device,
    );

    println!("✓ Initial states generated using Hinton initialization\n");

    // Print initial state
    if let Some(first_state) = init_states.first() {
        let state_data: Vec<f32> = first_state
            .clone()
            .into_data()
            .to_vec()
            .expect("read state");
        println!(
            "Initial state: {}\n",
            state_data
                .iter()
                .map(|&x| if x > 0.5 { "↑" } else { "↓" })
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    // Get factors for the model
    let factors = model.get_factors(&device);
    println!(
        "✓ Model has {} factors (bias + edge terms)\n",
        factors.len()
    );

    // Demonstrate state representation
    let sample_state: Tensor<WgpuBackend, 1> = Tensor::from_data(
        vec![1.0f32, 1.0, 1.0, 1.0, 1.0].as_slice(), // All spins up
        &device,
    );

    let state_data: Vec<f32> = sample_state
        .clone()
        .into_data()
        .to_vec()
        .expect("read state");
    println!(
        "All-up state representation: {:?}\n",
        state_data
            .iter()
            .map(|&x| if x > 0.5 { "↑" } else { "↓" })
            .collect::<Vec<_>>()
            .join(" ")
    );

    // Create alternating state
    let alt_state: Tensor<WgpuBackend, 1> = Tensor::from_data(
        vec![1.0f32, 0.0, 1.0, 0.0, 1.0].as_slice(), // Alternating
        &device,
    );

    let alt_state_data: Vec<f32> = alt_state.clone().into_data().to_vec().expect("read state");
    println!(
        "Alternating state representation: {:?}\n",
        alt_state_data
            .iter()
            .map(|&x| if x > 0.5 { "↑" } else { "↓" })
            .collect::<Vec<_>>()
            .join(" ")
    );

    println!("Example completed successfully!");
    println!("\nThis example demonstrates:");
    println!("  - GPU initialization (Metal backend on macOS)");
    println!("  - Spin node creation");
    println!("  - Ising model setup with biases and edge weights");
    println!("  - Hinton initialization for state vectors");
    println!("\nThe full sampling loop requires the BlockSamplingProgram,");
    println!("which is now implemented in thrml-samplers.");
}
