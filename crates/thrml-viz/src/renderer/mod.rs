//! GPU rendering infrastructure for 3D visualization.
//!
//! Uses wgpu directly with egui's PaintCallback for custom 3D rendering.

mod point_cloud;

pub use point_cloud::{PointCloudRenderer, PointVertex};

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Uniform buffer data shared across renderers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ViewUniforms {
    /// View-projection matrix
    pub view_proj: [[f32; 4]; 4],
    /// Camera position in world space
    pub camera_pos: [f32; 3],
    /// Point size (for point cloud)
    pub point_size: f32,
    /// Color range min
    pub color_min: f32,
    /// Color range max
    pub color_max: f32,
    /// Padding
    pub _padding: [f32; 2],
}

impl Default for ViewUniforms {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_pos: [0.0, 0.0, 500.0],
            point_size: 4.0,
            color_min: 0.0,
            color_max: 1.0,
            _padding: [0.0, 0.0],
        }
    }
}

/// Combined sphere renderer for points, lines, and meshes.
pub struct SphereRenderer {
    pub point_cloud: PointCloudRenderer,
    pub uniforms: ViewUniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl SphereRenderer {
    /// Create a new sphere renderer.
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        // Create uniform buffer
        let uniforms = ViewUniforms::default();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Uniforms"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("View Uniforms Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Create bind group
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("View Uniforms Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create point cloud renderer
        let point_cloud = PointCloudRenderer::new(device, format, &bind_group_layout);

        Self {
            point_cloud,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
        }
    }

    /// Update the view-projection matrix.
    pub fn set_view_proj(&mut self, view_proj: [[f32; 4]; 4]) {
        self.uniforms.view_proj = view_proj;
    }

    /// Update the camera position.
    pub fn set_camera_pos(&mut self, pos: [f32; 3]) {
        self.uniforms.camera_pos = pos;
    }

    /// Set the point size.
    pub fn set_point_size(&mut self, size: f32) {
        self.uniforms.point_size = size;
    }

    /// Set the color range for normalization.
    pub fn set_color_range(&mut self, min: f32, max: f32) {
        self.uniforms.color_min = min;
        self.uniforms.color_max = max;
    }

    /// Upload uniforms to GPU.
    pub fn update_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }

    /// Render all components.
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        // Render point cloud
        self.point_cloud.render(render_pass, &self.uniform_bind_group);
    }

    /// Get a reference to the uniform bind group.
    pub fn uniform_bind_group(&self) -> &wgpu::BindGroup {
        &self.uniform_bind_group
    }
}

// wgpu buffer initialization helper
use wgpu::util::DeviceExt;

/// Callback for egui custom rendering.
pub struct SphereRenderCallback {
    pub renderer: Arc<SphereRenderer>,
}

impl eframe::egui_wgpu::CallbackTrait for SphereRenderCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        _callback_resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Update uniforms
        self.renderer.update_uniforms(queue);
        vec![]
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        _callback_resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        // SAFETY: The render pass is valid for the duration of this call.
        // We need to cast the lifetime to match what wgpu expects for our renderer.
        let render_pass: &mut wgpu::RenderPass<'_> = unsafe { std::mem::transmute(render_pass) };
        
        // Render the scene
        self.renderer.render(render_pass);
    }
}
