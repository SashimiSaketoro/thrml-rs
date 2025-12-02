//! Point cloud rendering for sphere visualization.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Vertex data for a point in the cloud.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PointVertex {
    pub position: [f32; 3],
    pub color_value: f32, // prominence or radius for coloring
}

impl PointVertex {
    pub fn new(position: [f32; 3], color_value: f32) -> Self {
        Self {
            position,
            color_value,
        }
    }
}

/// Renderer for point cloud visualization.
pub struct PointCloudRenderer {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    n_points: u32,
}

impl PointCloudRenderer {
    /// Create a new point cloud renderer.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        uniform_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Self {
        // Load shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Point Cloud Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/point_cloud.wgsl").into()),
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Point Cloud Pipeline Layout"),
            bind_group_layouts: &[uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Point Cloud Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PointVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // position
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x3,
                        },
                        // color_value
                        wgpu::VertexAttribute {
                            offset: 12,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            vertex_buffer: None,
            n_points: 0,
        }
    }

    /// Upload point data to the GPU.
    pub fn upload_points(&mut self, device: &wgpu::Device, vertices: &[PointVertex]) {
        if vertices.is_empty() {
            self.vertex_buffer = None;
            self.n_points = 0;
            return;
        }

        self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Point Cloud Vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        self.n_points = vertices.len() as u32;
    }

    /// Upload points from position and color arrays.
    pub fn upload_from_arrays(
        &mut self,
        device: &wgpu::Device,
        positions: &[[f32; 3]],
        colors: &[f32],
    ) {
        let vertices: Vec<PointVertex> = positions
            .iter()
            .zip(colors.iter())
            .map(|(pos, &color)| PointVertex::new(*pos, color))
            .collect();

        self.upload_points(device, &vertices);
    }

    /// Render the point cloud.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>, bind_group: &'a wgpu::BindGroup) {
        if let Some(ref buffer) = self.vertex_buffer {
            if self.n_points > 0 {
                render_pass.set_pipeline(&self.pipeline);
                render_pass.set_bind_group(0, bind_group, &[]);
                render_pass.set_vertex_buffer(0, buffer.slice(..));
                render_pass.draw(0..self.n_points, 0..1);
            }
        }
    }

    /// Get the number of points.
    pub fn n_points(&self) -> u32 {
        self.n_points
    }

    /// Check if there are points to render.
    pub fn has_points(&self) -> bool {
        self.n_points > 0
    }
}
