use wgpu::util::DeviceExt;
use std::sync::Arc;
use image::{RgbImage, Rgb}; 
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
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    render_pipeline: wgpu::RenderPipeline,
    target_texture: wgpu::Texture,
    output_buffer: wgpu::Buffer,
    pub width: u32,
    pub height: u32,
    padded_bytes_per_row: u32,
}

impl GpuRenderer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("render.wgsl"))),
        });

        // 🎯 Removed deprecated `push_constant_ranges`
        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { 
            label: None, 
            bind_group_layouts: &[],
            immediate_size: 0, 
        });
        
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SDF Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            // 🎯 Wrapped VertexBufferLayout in Some() 
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[Some(Vertex::desc())], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            // 🎯 Renamed multiview to multiview_mask
            depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview_mask: None, cache: None,
        });

        let texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            label: Some("Render Texture"),
            view_formats: &[],
        };
        let target_texture = device.create_texture(&texture_desc);

        let unpadded_bytes_per_row = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (unpadded_bytes_per_row + align - 1) & !(align - 1);

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: (padded_bytes_per_row * height) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { device, queue, render_pipeline, target_texture, output_buffer, width, height, padded_bytes_per_row }
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

    pub fn render_frame(&self, vertices: &[Vertex]) -> RgbImage {
        let vertex_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"), contents: bytemuck::cast_slice(vertices), usage: wgpu::BufferUsages::VERTEX,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let view = self.target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.04, g: 0.05, b: 0.06, a: 1.0 }), store: wgpu::StoreOp::Store },
                    depth_slice: None, // 🎯 Required in modern wgpu
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                multiview_mask: None, // 🎯 Required in modern wgpu
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            rpass.draw(0..vertices.len() as u32, 0..1);
        }

        // 🎯 ImageCopy explicitly renamed to TexelCopy
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.target_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &self.output_buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.padded_bytes_per_row), rows_per_image: Some(self.height) } },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );

        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = self.output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        
        // 🎯 Maintain replaced by PollType
        self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        // 🎯 get_mapped_range() now returns a Result
        let data = buffer_slice.get_mapped_range().expect("Failed to map buffer memory");
        let mut img = RgbImage::new(self.width, self.height);
        
        for y in 0..self.height {
            let row_start = (y * self.padded_bytes_per_row) as usize;
            for x in 0..self.width {
                let pixel_start = row_start + (x * 4) as usize;
                let r = data[pixel_start];
                let g = data[pixel_start + 1];
                let b = data[pixel_start + 2];
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        drop(data);
        self.output_buffer.unmap();
        img
    }
}