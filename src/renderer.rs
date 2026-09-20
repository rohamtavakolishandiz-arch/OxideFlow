use wgpu::util::DeviceExt;
use std::borrow::Cow;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub obj_type: f32,
}

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { offset: 24, shader_location: 2, format: wgpu::VertexFormat::Float32x2 },
                wgpu::VertexAttribute { offset: 32, shader_location: 3, format: wgpu::VertexFormat::Float32 },
            ],
        }
    }
}

pub struct GpuRenderer {
    render_pipeline: wgpu::RenderPipeline,
}

impl GpuRenderer {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("render.wgsl"))),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, bind_group_layouts: &[], immediate_size: 0, 
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[Some(Vertex::desc())], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format, // Matches the Window Surface
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });

        Self { render_pipeline }
    }

    pub fn push_line(vertices: &mut Vec<Vertex>, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, color: [f32; 4], w: f32, h: f32) {
        let dx = x1 - x0; let dy = y1 - y0;
        let len = (dx * dx + dy * dy).sqrt();
        let nx = -dy / len * (thickness / 2.0); let ny = dx / len * (thickness / 2.0);

        let p0 = [((x0 + nx) / w) * 2.0 - 1.0, 1.0 - ((y0 + ny) / h) * 2.0];
        let p1 = [((x0 - nx) / w) * 2.0 - 1.0, 1.0 - ((y0 - ny) / h) * 2.0];
        let p2 = [((x1 - nx) / w) * 2.0 - 1.0, 1.0 - ((y1 - ny) / h) * 2.0];
        let p3 = [((x1 + nx) / w) * 2.0 - 1.0, 1.0 - ((y1 + ny) / h) * 2.0];

        vertices.push(Vertex { position: p0, color, uv: [0.0, 0.0], obj_type: 0.0 });
        vertices.push(Vertex { position: p1, color, uv: [0.0, 1.0], obj_type: 0.0 });
        vertices.push(Vertex { position: p2, color, uv: [1.0, 1.0], obj_type: 0.0 });
        vertices.push(Vertex { position: p0, color, uv: [0.0, 0.0], obj_type: 0.0 });
        vertices.push(Vertex { position: p2, color, uv: [1.0, 1.0], obj_type: 0.0 });
        vertices.push(Vertex { position: p3, color, uv: [1.0, 0.0], obj_type: 0.0 });
    }

    pub fn push_node(vertices: &mut Vec<Vertex>, cx: f32, cy: f32, radius: f32, color: [f32; 4], w: f32, h: f32) {
        let p0 = [((cx - radius) / w) * 2.0 - 1.0, 1.0 - ((cy - radius) / h) * 2.0];
        let p1 = [((cx - radius) / w) * 2.0 - 1.0, 1.0 - ((cy + radius) / h) * 2.0];
        let p2 = [((cx + radius) / w) * 2.0 - 1.0, 1.0 - ((cy + radius) / h) * 2.0];
        let p3 = [((cx + radius) / w) * 2.0 - 1.0, 1.0 - ((cy - radius) / h) * 2.0];

        vertices.push(Vertex { position: p0, color, uv: [0.0, 0.0], obj_type: 1.0 });
        vertices.push(Vertex { position: p1, color, uv: [0.0, 1.0], obj_type: 1.0 });
        vertices.push(Vertex { position: p2, color, uv: [1.0, 1.0], obj_type: 1.0 });
        vertices.push(Vertex { position: p0, color, uv: [0.0, 0.0], obj_type: 1.0 });
        vertices.push(Vertex { position: p2, color, uv: [1.0, 1.0], obj_type: 1.0 });
        vertices.push(Vertex { position: p3, color, uv: [1.0, 0.0], obj_type: 1.0 });
    }

    pub fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView, vertices: &[Vertex]) {
        if vertices.is_empty() { return; }
        
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"), contents: bytemuck::cast_slice(vertices), usage: wgpu::BufferUsages::VERTEX,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.05, b: 0.06, a: 1.0 }), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..vertices.len() as u32, 0..1);
        }
        queue.submit(Some(encoder.finish()));
    }
}