// Point cloud rendering shader for TheSphere visualization.
//
// Renders points as circles with prominence-based coloring using the Viridis colormap.

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    point_size: f32,
    color_min: f32,
    color_max: f32,
    _padding: vec2<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color_value: f32,  // prominence or radius for coloring
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) point_center: vec2<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Viridis colormap (approximate)
fn viridis(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.267004, 0.004874, 0.329415);
    let c1 = vec3<f32>(0.282327, 0.140926, 0.457517);
    let c2 = vec3<f32>(0.253935, 0.265254, 0.529983);
    let c3 = vec3<f32>(0.206756, 0.371758, 0.553117);
    let c4 = vec3<f32>(0.163625, 0.471133, 0.558148);
    let c5 = vec3<f32>(0.127568, 0.566949, 0.550556);
    let c6 = vec3<f32>(0.134692, 0.658636, 0.517649);
    let c7 = vec3<f32>(0.266941, 0.748751, 0.440573);
    let c8 = vec3<f32>(0.477504, 0.821444, 0.318195);
    let c9 = vec3<f32>(0.741388, 0.873449, 0.149561);
    let c10 = vec3<f32>(0.993248, 0.906157, 0.143936);

    let t_clamped = clamp(t, 0.0, 1.0);
    let idx = t_clamped * 10.0;
    let i = u32(idx);
    let f = fract(idx);

    var colors = array<vec3<f32>, 11>(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10);
    
    if i >= 10u {
        return c10;
    }
    return mix(colors[i], colors[i + 1u], f);
}

@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform to clip space
    let world_pos = vec4<f32>(in.position, 1.0);
    out.clip_position = uniforms.view_proj * world_pos;
    
    // Normalize color value to [0, 1]
    let range = uniforms.color_max - uniforms.color_min;
    let t = select(0.5, (in.color_value - uniforms.color_min) / range, range > 0.0001);
    
    // Apply Viridis colormap
    let rgb = viridis(t);
    out.color = vec4<f32>(rgb, 1.0);
    
    // Store point center for potential circle rendering
    out.point_center = out.clip_position.xy / out.clip_position.w;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
