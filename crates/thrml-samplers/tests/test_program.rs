//! Tests for BlockSamplingProgram
//!
//! Port of Python tests/test_block_sampling.py

use burn::tensor::Tensor;
use indexmap::IndexMap;
use thrml_core::backend::WgpuBackend;
use thrml_core::block::Block;
use thrml_core::interaction::InteractionGroup;
use thrml_core::node::{Node, NodeType, TensorSpec};
use thrml_samplers::program::{BlockGibbsSpec, BlockSamplingProgram};
use thrml_samplers::rng::RngKey;
use thrml_samplers::sampling::run_blocks;
use thrml_samplers::schedule::SamplingSchedule;

/// Create a simple test setup with continuous scalar nodes
fn create_simple_setup(
    device: &burn::backend::wgpu::WgpuDevice,
) -> (
    Vec<Block>,
    Vec<Block>,
    Vec<InteractionGroup>,
    IndexMap<NodeType, TensorSpec>,
) {
    // Create nodes (all spin type for simplicity)
    let free_nodes: Vec<Node> = (0..3).map(|_| Node::new(NodeType::Spin)).collect();
    let clamped_nodes: Vec<Node> = (0..4).map(|_| Node::new(NodeType::Spin)).collect();

    let free_blocks = vec![
        Block::new(vec![free_nodes[0].clone()]).unwrap(),
        Block::new(vec![free_nodes[1].clone(), free_nodes[2].clone()]).unwrap(),
    ];

    let clamped_blocks = vec![Block::new(clamped_nodes.clone()).unwrap()];

    // Create a simple interaction group
    let interaction_weights: Tensor<WgpuBackend, 3> = Tensor::ones([3, 1, 1], device);

    let interaction_group = InteractionGroup::new(
        interaction_weights,
        Block::new(free_nodes.clone()).unwrap(),
        vec![Block::new(free_nodes.clone()).unwrap()],
        1, // n_spin
    )
    .unwrap();

    let mut node_shape_dtypes = IndexMap::new();
    node_shape_dtypes.insert(NodeType::Spin, TensorSpec::for_spin());

    (
        free_blocks,
        clamped_blocks,
        vec![interaction_group],
        node_shape_dtypes,
    )
}

#[cfg(feature = "gpu")]
#[test]
fn test_block_gibbs_spec_creation() {
    use thrml_core::backend::{ensure_metal_backend, init_gpu_device};

    ensure_metal_backend();
    let device = init_gpu_device();

    let (free_blocks, clamped_blocks, _, node_shape_dtypes) = create_simple_setup(&device);

    // Wrap each free block in a vec (each block is its own superblock)
    let free_super_blocks: Vec<Vec<Block>> = free_blocks.iter().map(|b| vec![b.clone()]).collect();

    let spec = BlockGibbsSpec::new(free_super_blocks, clamped_blocks.clone(), node_shape_dtypes);

    assert!(spec.is_ok(), "BlockGibbsSpec creation should succeed");

    let spec = spec.unwrap();
    assert_eq!(spec.free_blocks.len(), 2, "Should have 2 free blocks");
    assert_eq!(spec.clamped_blocks.len(), 1, "Should have 1 clamped block");
    assert_eq!(
        spec.sampling_order.len(),
        2,
        "Should have 2 sampling groups"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_block_sampling_program_creation() {
    use thrml_core::backend::{ensure_metal_backend, init_gpu_device};
    use thrml_samplers::bernoulli::BernoulliConditional;
    use thrml_samplers::sampler::DynConditionalSampler;

    ensure_metal_backend();
    let device = init_gpu_device();

    let (free_blocks, clamped_blocks, interaction_groups, node_shape_dtypes) =
        create_simple_setup(&device);

    let free_super_blocks: Vec<Vec<Block>> = free_blocks.iter().map(|b| vec![b.clone()]).collect();
    let spec =
        BlockGibbsSpec::new(free_super_blocks, clamped_blocks.clone(), node_shape_dtypes).unwrap();

    // Create samplers (one per free block)
    let samplers: Vec<Box<dyn DynConditionalSampler>> = vec![
        Box::new(BernoulliConditional),
        Box::new(BernoulliConditional),
    ];

    let result = BlockSamplingProgram::new(spec, samplers, interaction_groups);

    assert!(
        result.is_ok(),
        "BlockSamplingProgram creation should succeed: {:?}",
        result.err()
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_sampler_count_validation() {
    use thrml_core::backend::{ensure_metal_backend, init_gpu_device};
    use thrml_samplers::bernoulli::BernoulliConditional;
    use thrml_samplers::sampler::DynConditionalSampler;

    ensure_metal_backend();
    let device = init_gpu_device();

    let (free_blocks, clamped_blocks, interaction_groups, node_shape_dtypes) =
        create_simple_setup(&device);

    let free_super_blocks: Vec<Vec<Block>> = free_blocks.iter().map(|b| vec![b.clone()]).collect();
    let spec =
        BlockGibbsSpec::new(free_super_blocks, clamped_blocks.clone(), node_shape_dtypes).unwrap();

    // Wrong number of samplers (only 1, need 2)
    let samplers: Vec<Box<dyn DynConditionalSampler>> = vec![Box::new(BernoulliConditional)];

    let result = BlockSamplingProgram::new(spec, samplers, interaction_groups);

    assert!(result.is_err(), "Should fail with wrong sampler count");

    let err = result.err().unwrap();
    assert!(
        err.contains("sampler") || err.contains("Sampler") || err.contains("number"),
        "Error should mention sampler count, got: {}",
        err
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_run_blocks() {
    use thrml_core::backend::{ensure_metal_backend, init_gpu_device};
    use thrml_samplers::bernoulli::BernoulliConditional;
    use thrml_samplers::sampler::DynConditionalSampler;

    ensure_metal_backend();
    let device = init_gpu_device();

    let (free_blocks, clamped_blocks, interaction_groups, node_shape_dtypes) =
        create_simple_setup(&device);

    let free_super_blocks: Vec<Vec<Block>> = free_blocks.iter().map(|b| vec![b.clone()]).collect();
    let spec =
        BlockGibbsSpec::new(free_super_blocks, clamped_blocks.clone(), node_shape_dtypes).unwrap();

    let samplers: Vec<Box<dyn DynConditionalSampler>> = vec![
        Box::new(BernoulliConditional),
        Box::new(BernoulliConditional),
    ];

    let program = BlockSamplingProgram::new(spec, samplers, interaction_groups).unwrap();

    // Create initial states
    let init_state: Vec<Tensor<WgpuBackend, 1>> =
        vec![Tensor::zeros([1], &device), Tensor::zeros([2], &device)];
    let clamp_state: Vec<Tensor<WgpuBackend, 1>> = vec![Tensor::zeros([4], &device)];

    let key = RngKey::new(42);

    // Run a few iterations
    let result = run_blocks(key, &program, init_state, &clamp_state, 5, &device);

    assert!(
        result.is_ok(),
        "run_blocks should succeed: {:?}",
        result.err()
    );

    let final_state = result.unwrap();
    assert_eq!(final_state.len(), 2, "Should have 2 free block states");
    assert_eq!(
        final_state[0].dims()[0],
        1,
        "First block should have 1 node"
    );
    assert_eq!(
        final_state[1].dims()[0],
        2,
        "Second block should have 2 nodes"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn test_sampling_schedule() {
    let schedule = SamplingSchedule::new(100, 50, 10);
    assert_eq!(schedule.n_warmup, 100);
    assert_eq!(schedule.n_samples, 50);
    assert_eq!(schedule.steps_per_sample, 10);
}
