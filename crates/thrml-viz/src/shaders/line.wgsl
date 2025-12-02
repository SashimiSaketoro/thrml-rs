// Line rendering shader for ROOTS hierarchy visualization.
//
// Renders lines connecting tree nodes (internal → child centroids).

struct Uniforms {
    view_proj: mat4x4<f32>,
    line_color: vec4<f32>,
    line_width: f32,
    _padding: vec3<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform to clip space
    let world_pos = vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    
    // Use per-vertex color if provided, otherwise use uniform color
    let has_vertex_color = in.color.a > 0.0;
    out.color = select(uniforms.line_color, in.color, has_vertex_color);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
