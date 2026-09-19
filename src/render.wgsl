struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) obj_type: f32, // 0.0 = Line, 1.0 = Node
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) obj_type: f32,
};

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Map screen coordinates (-1.0 to +1.0)
    out.clip_position = vec4<f32>(model.position, 0.0, 1.0);
    out.color = model.color;
    out.uv = model.uv;
    out.obj_type = model.obj_type;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    if (in.obj_type > 0.5) {
        // 🟢 HOLLOW NODE SDF (Perfectly smooth ring)
        let dist = distance(in.uv, vec2<f32>(0.5, 0.5));
        // Anti-aliased outer edge minus anti-aliased inner edge
        let ring = smoothstep(0.48, 0.45, dist) - smoothstep(0.35, 0.32, dist);
        
        // Add a faint inner glow
        let glow = smoothstep(0.5, 0.0, dist) * 0.3;
        
        let alpha = clamp(ring + glow, 0.0, 1.0);
        return vec4<f32>(in.color.rgb, in.color.a * alpha);
    } else {
        // 🔴 GLOWING LINE SDF
        let dist = abs(in.uv.y - 0.5);
        let core = smoothstep(0.15, 0.0, dist); 
        let glow = smoothstep(0.5, 0.1, dist) * 0.5;
        
        let alpha = clamp(core + glow, 0.0, 1.0);
        return vec4<f32>(in.color.rgb, in.color.a * alpha);
    }
}