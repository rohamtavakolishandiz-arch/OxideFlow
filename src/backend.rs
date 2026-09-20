// src/backend.rs
use rayon::prelude::*;
use rand::Rng;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use std::borrow::Cow;

// ==========================================
// UNIFORM STRUCTS
// ==========================================
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulDims { m: u32, k: u32, n: u32, _padding: u32 }

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposeDims { rows: u32, cols: u32 }

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxDims { batch_size: u32, num_classes: u32, _pad1: u32, _pad2: u32 }

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MseDims { length: u32, _pad1: u32, _pad2: u32, _pad3: u32 }

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv2dDims {
    batch: u32, in_c: u32, h: u32, w: u32, out_c: u32, kh: u32, kw: u32,
    out_h: u32, out_w: u32, stride: u32, padding: u32, _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MaxPoolDims {
    batch: u32, c: u32, h: u32, w: u32, out_h: u32, out_w: u32, k: u32, _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AdamWConfig {
    pub length: u32, pub lr: f32, pub beta1: f32, pub beta2: f32,
    pub eps: f32, pub weight_decay: f32, pub bias_correction1: f32, pub bias_correction2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ReductionDims {
    length: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BroadcastDims { r1: u32, c1: u32, r2: u32, c2: u32 }

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchNormDims {
    batch: u32, c: u32, h: u32, w: u32,
    eps: f32, momentum: f32, is_training: u32, _pad: u32,
}

// ==========================================
// BACKEND TRAIT
// ==========================================
pub trait Backend: Clone + 'static {
    type Buffer: Clone + std::fmt::Debug;

    fn new() -> Self;
    fn from_data(device: &Self, data: Vec<f64>, shape: &[usize]) -> Self::Buffer;
    fn zeros(device: &Self, shape: &[usize]) -> Self::Buffer;
    fn randn(device: &Self, shape: &[usize]) -> Self::Buffer;

    fn add(device: &Self, a: &Self::Buffer, shape_a: &[usize], b: &Self::Buffer, shape_b: &[usize]) -> Result<(Self::Buffer, Vec<usize>), String>;
    fn sub(device: &Self, a: &Self::Buffer, b: &Self::Buffer, shape: &[usize]) -> Result<Self::Buffer, String>;
    fn matmul(device: &Self, a: &Self::Buffer, shape_a: &[usize], b: &Self::Buffer, shape_b: &[usize]) -> Result<Self::Buffer, String>;
    fn transpose(device: &Self, a: &Self::Buffer, shape: &[usize]) -> Result<Self::Buffer, String>;
    fn mul_elementwise(device: &Self, a: &Self::Buffer, b: &Self::Buffer, shape_a: &[usize], shape_b: &[usize]) -> Result<Self::Buffer, String>;

    fn relu(device: &Self, a: &Self::Buffer, shape: &[usize]) -> Self::Buffer;
    fn exp(device: &Self, a: &Self::Buffer, shape: &[usize]) -> Self::Buffer;
    fn sigmoid(device: &Self, a: &Self::Buffer, shape: &[usize]) -> Self::Buffer;
    fn tanh(device: &Self, a: &Self::Buffer, shape: &[usize]) -> Self::Buffer;
    fn softmax(device: &Self, a: &Self::Buffer, shape: &[usize]) -> Self::Buffer;
    fn mse_loss(device: &Self, a: &Self::Buffer, b: &Self::Buffer, shape: &[usize]) -> Result<Self::Buffer, String>;

    fn max(device: &Self, a: &Self::Buffer, shape: &[usize]) -> f64;
    fn sum(device: &Self, a: &Self::Buffer, shape: &[usize]) -> f64;

    fn conv2d(device: &Self, a: &Self::Buffer, sa: &[usize], w: &Self::Buffer, sw: &[usize], b: &Self::Buffer, sb: &[usize], stride: usize, padding: usize) -> Result<(Self::Buffer, Vec<usize>), String>;
    fn conv2d_backward(device: &Self, a: &Self::Buffer, sa: &[usize], go: &Self::Buffer, sgo: &[usize], w: &Self::Buffer, sw: &[usize], stride: usize, padding: usize) -> (Self::Buffer, Self::Buffer, Self::Buffer);
    fn maxpool2d(device: &Self, a: &Self::Buffer, shape: &[usize], kernel_size: usize) -> Result<(Self::Buffer, Vec<usize>), String>;
    fn maxpool2d_backward(device: &Self, a: &Self::Buffer, shape: &[usize], grad_out: &Self::Buffer, shape_go: &[usize], kernel_size: usize) -> Self::Buffer;

    // 🎯 The New Optimizer Route
    fn adamw_step(device: &Self, weight: &mut Self::Buffer, grad: &Self::Buffer, m: &mut Self::Buffer, v: &mut Self::Buffer, config: &AdamWConfig);

    fn batch_norm2d(
        device: &Self, input: &Self::Buffer, shape: &[usize],
        weight: &Self::Buffer, bias: &Self::Buffer,
        running_mean: &Self::Buffer, running_var: &Self::Buffer,
        is_training: bool, momentum: f64, eps: f64
    ) -> Result<Self::Buffer, String>;

    fn batch_norm2d_backward(
        device: &Self, input: &Self::Buffer, shape: &[usize],
        grad_out: &Self::Buffer, weight: &Self::Buffer, eps: f64
    ) -> (Self::Buffer, Self::Buffer, Self::Buffer);

    fn to_cpu(device: &Self, buffer: &Self::Buffer) -> Vec<f64>;
}

// ==========================================
// CPU BACKEND
// ==========================================
#[derive(Clone, Debug)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Buffer = Vec<f64>;

    fn new() -> Self { CpuBackend }
    fn from_data(_device: &Self, data: Vec<f64>, _shape: &[usize]) -> Self::Buffer { data }
    fn zeros(_device: &Self, shape: &[usize]) -> Self::Buffer { vec![0.0; shape.iter().product()] }

    fn randn(_device: &Self, shape: &[usize]) -> Self::Buffer {
        let mut rng = rand::thread_rng();
        let len: usize = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            data.push((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos());
        }
        data
    }

    fn add(_device: &Self, a: &Self::Buffer, shape_a: &[usize], b: &Self::Buffer, shape_b: &[usize]) -> Result<(Self::Buffer, Vec<usize>), String> {
        if shape_a == shape_b { return Ok((a.iter().zip(b.iter()).map(|(x, y)| x + y).collect(), shape_a.to_vec())); }
        if shape_a.len() == 2 && shape_b.len() == 2 {
            let (r1, c1) = (shape_a[0], shape_a[1]); let (r2, c2) = (shape_b[0], shape_b[1]);
            if c1 == c2 && r2 == 1 {
                let mut new_data = vec![0.0; r1 * c1];
                for i in 0..r1 { for j in 0..c1 { new_data[i * c1 + j] = a[i * c1 + j] + b[j]; } }
                return Ok((new_data, shape_a.to_vec()));
            }
            if c1 == c2 && r1 == 1 {
                let mut new_data = vec![0.0; r2 * c2];
                for i in 0..r2 { for j in 0..c2 { new_data[i * c2 + j] = a[j] + b[i * c2 + j]; } }
                return Ok((new_data, shape_b.to_vec()));
            }
        }
        Err("Cannot add or broadcast".to_string())
    }

    fn sub(_d: &Self, a: &Self::Buffer, b: &Self::Buffer, _s: &[usize]) -> Result<Self::Buffer, String> { Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()) }
    fn matmul(_d: &Self, a: &Self::Buffer, sa: &[usize], b: &Self::Buffer, sb: &[usize]) -> Result<Self::Buffer, String> {
        let (r, shared_dim, c) = (sa[0], sa[1], sb[1]);
        let mut out_data = vec![0.0; r * c];
        out_data.par_chunks_mut(c).enumerate().for_each(|(i, row_slice)| {
            for k in 0..shared_dim {
                let a_val = a[i * shared_dim + k];
                for j in 0..c { row_slice[j] += a_val * b[k * c + j]; }
            }
        });
        Ok(out_data)
    }
    fn transpose(_d: &Self, a: &Self::Buffer, shape: &[usize]) -> Result<Self::Buffer, String> {
        let (rows, cols) = (shape[0], shape[1]);
        let mut new_data = vec![0.0; rows * cols];
        for i in 0..rows { for j in 0..cols { new_data[j * rows + i] = a[i * cols + j]; } }
        Ok(new_data)
    }

    fn mul_elementwise(_d: &Self, a: &Self::Buffer, b: &Self::Buffer, _sa: &[usize], _sb: &[usize]) -> Result<Self::Buffer, String> { Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()) }
    fn relu(_d: &Self, a: &Self::Buffer, _shape: &[usize]) -> Self::Buffer { a.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect() }
    fn exp(_d: &Self, a: &Self::Buffer, _shape: &[usize]) -> Self::Buffer { a.iter().map(|v| v.exp()).collect() }
    fn sigmoid(_d: &Self, a: &Self::Buffer, _shape: &[usize]) -> Self::Buffer { a.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect() }
    fn tanh(_d: &Self, a: &Self::Buffer, _shape: &[usize]) -> Self::Buffer { a.iter().map(|&x| x.tanh()).collect() }
    fn softmax(_d: &Self, a: &Self::Buffer, shape: &[usize]) -> Self::Buffer {
        let (batch_size, num_classes) = (shape[0], shape[1]);
        let mut new_data = Vec::with_capacity(a.len());
        for i in 0..batch_size {
            let row = &a[i * num_classes..(i + 1) * num_classes];
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sums: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
            for &val in row { new_data.push((val - max_val).exp() / exp_sums); }
        }
        new_data
    }
    fn mse_loss(_d: &Self, a: &Self::Buffer, b: &Self::Buffer, _shape: &[usize]) -> Result<Self::Buffer, String> { Ok(vec![a.iter().zip(b.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / (a.len() as f64)]) }
    fn max(_d: &Self, a: &Self::Buffer, _shape: &[usize]) -> f64 { a.iter().cloned().fold(f64::NEG_INFINITY, f64::max) }
    fn sum(_d: &Self, a: &Self::Buffer, _shape: &[usize]) -> f64 { a.iter().sum() }
    fn conv2d(_d: &Self, a: &Self::Buffer, sa: &[usize], w: &Self::Buffer, sw: &[usize], b: &Self::Buffer, _sb: &[usize], stride: usize, padding: usize) -> Result<(Self::Buffer, Vec<usize>), String> {
        let (batch, in_c, h, w_dim) = (sa[0], sa[1], sa[2], sa[3]);
        let (out_c, kh, kw) = (sw[0], sw[2], sw[3]);
        let out_h = (h + 2 * padding - kh) / stride + 1;
        let out_w = (w_dim + 2 * padding - kw) / stride + 1;
        let mut out_data = vec![0.0; batch * out_c * out_h * out_w];
        
        out_data.par_chunks_mut(out_c * out_h * out_w).enumerate().for_each(|(b_idx, batch_slice)| {
            for oc in 0..out_c {
                let b_val = b[oc];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = b_val;
                        for ic in 0..in_c {
                            for kh_idx in 0..kh {
                                for kw_idx in 0..kw {
                                    let ih = (oh * stride + kh_idx) as isize - padding as isize;
                                    let iw = (ow * stride + kw_idx) as isize - padding as isize;
                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w_dim as isize {
                                        sum += a[b_idx * in_c * h * w_dim + ic * h * w_dim + (ih as usize) * w_dim + (iw as usize)] 
                                             * w[oc * in_c * kh * kw + ic * kh * kw + kh_idx * kw + kw_idx];
                                    }
                                }
                            }
                        }
                        batch_slice[oc * out_h * out_w + oh * out_w + ow] = sum;
                    }
                }
            }
        });
        Ok((out_data, vec![batch, out_c, out_h, out_w]))
    }
    fn conv2d_backward(_d: &Self, a: &Self::Buffer, sa: &[usize], go: &Self::Buffer, sgo: &[usize], w: &Self::Buffer, sw: &[usize], stride: usize, padding: usize) -> (Self::Buffer, Self::Buffer, Self::Buffer) {
        let (batch, in_c, h, w_dim) = (sa[0], sa[1], sa[2], sa[3]);
        let (out_c, kh, kw) = (sw[0], sw[2], sw[3]);
        let (out_h, out_w) = (sgo[2], sgo[3]);
        
        let batch_gradients: Vec<_> = (0..batch).into_par_iter().map(|b_idx| {
            let mut l_g_in = vec![0.0; in_c * h * w_dim]; let mut l_g_w = vec![0.0; w.len()]; let mut l_g_b = vec![0.0; out_c];
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let g = go[b_idx * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow];
                        l_g_b[oc] += g;
                        for ic in 0..in_c {
                            for kh_idx in 0..kh {
                                for kw_idx in 0..kw {
                                    let ih = (oh * stride + kh_idx) as isize - padding as isize;
                                    let iw = (ow * stride + kw_idx) as isize - padding as isize;
                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w_dim as isize {
                                        let in_idx = ic * h * w_dim + (ih as usize) * w_dim + (iw as usize);
                                        let w_idx = oc * in_c * kh * kw + ic * kh * kw + kh_idx * kw + kw_idx;
                                        l_g_in[in_idx] += w[w_idx] * g;
                                        l_g_w[w_idx] += a[b_idx * in_c * h * w_dim + in_idx] * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (l_g_in, l_g_w, l_g_b)
        }).collect();

        let mut final_g_in = Vec::with_capacity(batch * in_c * h * w_dim);
        let mut final_g_w = vec![0.0; w.len()]; let mut final_g_b = vec![0.0; out_c];
        for (g_in, g_w, g_b) in batch_gradients {
            final_g_in.extend(g_in);
            for i in 0..final_g_w.len() { final_g_w[i] += g_w[i]; }
            for i in 0..final_g_b.len() { final_g_b[i] += g_b[i]; }
        }
        (final_g_in, final_g_w, final_g_b)
    }
    fn maxpool2d(_d: &Self, a: &Self::Buffer, s: &[usize], k: usize) -> Result<(Self::Buffer, Vec<usize>), String> {
        let (batch, c, h, w) = (s[0], s[1], s[2], s[3]);
        let (out_h, out_w) = (h / k, w / k);
        let mut out_data = vec![0.0; batch * c * out_h * out_w];
        for b_idx in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f64::NEG_INFINITY;
                        for kh in 0..k {
                            for kw in 0..k {
                                let val = a[b_idx * c * h * w + ch * h * w + (oh * k + kh) * w + (ow * k + kw)];
                                if val > max_val { max_val = val; }
                            }
                        }
                        out_data[b_idx * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow] = max_val;
                    }
                }
            }
        }
        Ok((out_data, vec![batch, c, out_h, out_w]))
    }
    fn maxpool2d_backward(_d: &Self, a: &Self::Buffer, s: &[usize], go: &Self::Buffer, sgo: &[usize], k: usize) -> Self::Buffer {
        let (batch, c, h, w) = (s[0], s[1], s[2], s[3]);
        let (out_h, out_w) = (sgo[2], sgo[3]);
        let mut grad_in = vec![0.0; a.len()];
        for b_idx in 0..batch {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f64::NEG_INFINITY; let mut max_idx = 0;
                        for kh in 0..k {
                            for kw in 0..k {
                                let idx = b_idx * c * h * w + ch * h * w + (oh * k + kh) * w + (ow * k + kw);
                                if a[idx] > max_val { max_val = a[idx]; max_idx = idx; }
                            }
                        }
                        grad_in[max_idx] += go[b_idx * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow];
                    }
                }
            }
        }
        grad_in
    }

    fn adamw_step(_d: &Self, weight: &mut Self::Buffer, grad: &Self::Buffer, m: &mut Self::Buffer, v: &mut Self::Buffer, config: &AdamWConfig) {
        for j in 0..weight.len() {
            let g = grad[j] as f64;
            let w = weight[j] as f64;
            m[j] = (config.beta1 as f64) * m[j] + (1.0 - config.beta1 as f64) * g;
            v[j] = (config.beta2 as f64) * v[j] + (1.0 - config.beta2 as f64) * g * g;
            let m_hat = m[j] / (config.bias_correction1 as f64);
            let v_hat = v[j] / (config.bias_correction2 as f64);
            weight[j] = w - (config.lr as f64) * (m_hat / (v_hat.sqrt() + (config.eps as f64)) + (config.weight_decay as f64) * w);
        }
    }

    fn batch_norm2d(
        _device: &Self, _input: &Self::Buffer, _shape: &[usize],
        _weight: &Self::Buffer, _bias: &Self::Buffer,
        _running_mean: &Self::Buffer, _running_var: &Self::Buffer,
        _is_training: bool, _momentum: f64, _eps: f64
    ) -> Result<Self::Buffer, String> {
        unimplemented!("BatchNorm2d for CpuBackend is not implemented yet")
    }

    fn batch_norm2d_backward(
        _d: &Self, _i: &Self::Buffer, _s: &[usize], _go: &Self::Buffer, _w: &Self::Buffer, _eps: f64
    ) -> (Self::Buffer, Self::Buffer, Self::Buffer) {
        unimplemented!("BatchNorm backward on CPU is not implemented yet")
    }

    fn to_cpu(_device: &Self, buffer: &Self::Buffer) -> Vec<f64> { buffer.clone() }
}

// ==========================================
// WGPU BACKEND (GPU)
// ==========================================

// 🎯 The Buffer Pool: Maps a requested byte size to a list of free, reusable VRAM buffers
pub type BufferPool = Arc<RwLock<HashMap<wgpu::BufferAddress, Vec<Arc<wgpu::Buffer>>>>>;

#[derive(Clone, Debug)]
pub struct WgpuBackend {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipelines: Arc<RwLock<HashMap<&'static str, Arc<wgpu::ComputePipeline>>>>,
    pub pool: BufferPool, // <-- NEW: The Memory Allocator
}

#[derive(Clone, Debug)]
pub struct WgpuBuffer { 
    pub buffer: Arc<wgpu::Buffer>, 
    pub length: usize,
    pub pool: BufferPool, // <-- NEW: Buffers carry a reference to their pool
}

// 🎯 THE MAGIC: When a tensor is dropped, intercept it!
impl Drop for WgpuBuffer {
    fn drop(&mut self) {
        // If this is the last reference to this specific buffer...
        if Arc::strong_count(&self.buffer) == 1 {
            let size = self.buffer.size();
            if let Ok(mut pool) = self.pool.write() {
                // ...clone the Arc into the pool! 
                // This keeps the wgpu::Buffer alive on the GPU instead of destroying it.
                pool.entry(size).or_default().push(self.buffer.clone());
            }
        }
    }
}

impl WgpuBackend {

    pub fn with_surface(window: Arc<winit::window::Window>) -> (Self, wgpu::Surface<'static>, wgpu::SurfaceConfiguration) {
        let (device, queue, surface, config) = pollster::block_on(async {
            let instance = wgpu::Instance::default();
            
            // Create the surface from the winit window
            let surface = instance.create_surface(window.clone()).unwrap();
            
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            }).await.unwrap();
            
            let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
            
            let size = window.inner_size();
            let caps = surface.get_capabilities(&adapter);
            let format = caps.formats[0]; // Pick the preferred format for the display
            
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto, // <--- ADD THIS LINE
            };
            surface.configure(&device, &config);
            
            (device, queue, surface, config)
        });
        
        let backend = Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            pool: Arc::new(RwLock::new(HashMap::new())), // Active Memory Pool
        };
        
        (backend, surface, config)
    }
    // 🎯 Fast VRAM Allocator
    fn allocate(&self, size: wgpu::BufferAddress) -> Arc<wgpu::Buffer> {
        // 1. Check if we have a recycled buffer of the exact size
        if let Ok(mut pool) = self.pool.write() {
            if let Some(vec) = pool.get_mut(&size) {
                if let Some(recycled_buffer) = vec.pop() {
                    return recycled_buffer;
                }
            }
        }
        
        // 2. If the pool is empty for this size, allocate a fresh one from the OS
        Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled VRAM Tensor"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    fn get_pipeline(&self, entry_point: &'static str) -> Arc<wgpu::ComputePipeline> {
        if let Some(pipe) = self.pipelines.read().unwrap().get(entry_point) {
            return pipe.clone();
        }
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Global Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
        });
        let pipe = Arc::new(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point), layout: None, module: &shader, entry_point: Some(entry_point), compilation_options: Default::default(), cache: None,
        }));
        self.pipelines.write().unwrap().insert(entry_point, pipe.clone());
        pipe
    }

    fn dispatch_binary(&self, a: &WgpuBuffer, b: &WgpuBuffer, entry_point: &'static str) -> WgpuBuffer {
        let compute_pipeline = self.get_pipeline(entry_point);
        let out_buffer = <WgpuBackend as Backend>::zeros(self, &[a.length]);
        let length_data: [u32; 1] = [a.length as u32];
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims"), contents: bytemuck::cast_slice(&length_data), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((a.length as u32) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        out_buffer
    }

    fn dispatch_unary(&self, a: &WgpuBuffer, entry_point: &'static str) -> WgpuBuffer {
        let compute_pipeline = self.get_pipeline(entry_point);
        let out_buffer = <WgpuBackend as Backend>::zeros(self, &[a.length]);
        let length_data: [u32; 1] = [a.length as u32];
        let dims_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dims"), contents: bytemuck::cast_slice(&length_data), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((a.length as u32) + 63) / 64, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
        out_buffer
    }
}

impl Backend for WgpuBackend {
    type Buffer = WgpuBuffer;

    fn new() -> Self {
        let (device, queue) = pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
            adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap()
        });
        Self { 
            device: Arc::new(device), 
            queue: Arc::new(queue), 
            pipelines: Arc::new(RwLock::new(HashMap::new())),
            pool: Arc::new(RwLock::new(HashMap::new())), // Initialize empty pool
        }
    }

    fn from_data(device: &Self, data: Vec<f64>, _shape: &[usize]) -> Self::Buffer {
        let data_f32: Vec<f32> = data.into_iter().map(|x| x as f32).collect();
        let size = (data_f32.len() * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        
        // 🎯 Allocate from pool
        let buffer = device.allocate(size);
        
        // 🎯 Write data rapidly over the queue
        device.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&data_f32));
        
        WgpuBuffer { buffer, length: data_f32.len(), pool: device.pool.clone() }
    }

    fn zeros(device: &Self, shape: &[usize]) -> Self::Buffer {
        let len: usize = shape.iter().product();
        let size = (len * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        
        // 🎯 Allocate from pool
        let buffer = device.allocate(size);

        // 🎯 Fast hardware-level memory clear (overwrites garbage from recycled buffers)
        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.clear_buffer(&buffer, 0, None);
        device.queue.submit(Some(encoder.finish()));
        
        WgpuBuffer { buffer, length: len, pool: device.pool.clone() }
    }

    // ... Keep the rest of the functions exactly as they are!

    fn randn(device: &Self, shape: &[usize]) -> Self::Buffer { Self::from_data(device, CpuBackend::randn(&CpuBackend, shape), shape) }

    fn to_cpu(device: &Self, buffer: &Self::Buffer) -> Vec<f64> {
        let size = (buffer.length * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"), size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&buffer.buffer, 0, &staging_buffer, 0, size);
        device.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        receiver.recv().unwrap().unwrap();

        let data_result = buffer_slice.get_mapped_range().unwrap();
        let result_f32: &[f32] = bytemuck::cast_slice(&data_result);
        result_f32.iter().map(|&x| x as f64).collect()
    }

    fn add(device: &Self, a: &Self::Buffer, shape_a: &[usize], b: &Self::Buffer, shape_b: &[usize]) -> Result<(Self::Buffer, Vec<usize>), String> {
        if shape_a == shape_b {
            return Ok((device.dispatch_binary(a, b, "add_forward"), shape_a.to_vec()));
        }
    
        if shape_a.len() == 2 && shape_b.len() == 2 {
            let (r1, c1) = (shape_a[0], shape_a[1]);
            let (r2, c2) = (shape_b[0], shape_b[1]);
            
            if (c1 == c2 && (r1 == 1 || r2 == 1)) || (r1 == r2 && (c1 == 1 || c2 == 1)) {
                let out_shape = vec![r1.max(r2), c1.max(c2)];
                let out_buffer = Self::zeros(device, &out_shape);
                let dims = BroadcastDims { r1: r1 as u32, c1: c1 as u32, r2: r2 as u32, c2: c2 as u32 };
                
                let compute_pipeline = device.get_pipeline("add_broadcast_forward");
                let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Broadcast Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
                });
    
                let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: &compute_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: b.buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: out_buffer.buffer.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
                    ],
                });
    
                let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    cpass.set_pipeline(&compute_pipeline);
                    cpass.set_bind_group(0, &bind_group, &[]);
                    let out_len = (out_shape[0] * out_shape[1]) as u32;
                    cpass.dispatch_workgroups((out_len + 63) / 64, 1, 1);
                }
                device.queue.submit(Some(encoder.finish()));
                
                return Ok((out_buffer, out_shape));
            }
        }
        Err("Cannot add or broadcast GPU tensors".to_string())
    }
    fn sub(device: &Self, a: &Self::Buffer, b: &Self::Buffer, _shape: &[usize]) -> Result<Self::Buffer, String> { Ok(device.dispatch_binary(a, b, "sub_forward")) }
    fn mul_elementwise(device: &Self, a: &Self::Buffer, b: &Self::Buffer, _sa: &[usize], _sb: &[usize]) -> Result<Self::Buffer, String> { Ok(device.dispatch_binary(a, b, "mul_forward")) }
    fn relu(device: &Self, a: &Self::Buffer, _shape: &[usize]) -> Self::Buffer { device.dispatch_unary(a, "relu_forward") }
    fn sigmoid(device: &Self, a: &Self::Buffer, _shape: &[usize]) -> Self::Buffer { device.dispatch_unary(a, "sigmoid_forward") }
    fn tanh(device: &Self, a: &Self::Buffer, _shape: &[usize]) -> Self::Buffer { device.dispatch_unary(a, "tanh_forward") }

    fn exp(device: &Self, a: &Self::Buffer, _shape: &[usize]) -> Self::Buffer {
        // Exp seamlessly piggybacks off our existing Unary helper!
        device.dispatch_unary(a, "exp_forward") 
    }

    fn max(device: &Self, a: &Self::Buffer, _shape: &[usize]) -> f64 {
        let dims = ReductionDims { length: a.length as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let out_buffer = Self::zeros(device, &[1]); // Tiny 1-element buffer
        let compute_pipeline = device.get_pipeline("max_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Red Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        
        // 🎯 We only pull 4 bytes (1 float) back across the PCIe bus!
        Self::to_cpu(device, &out_buffer)[0]
    }

    fn sum(device: &Self, a: &Self::Buffer, _shape: &[usize]) -> f64 {
        let dims = ReductionDims { length: a.length as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let out_buffer = Self::zeros(device, &[1]); // Tiny 1-element buffer
        let compute_pipeline = device.get_pipeline("sum_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Red Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        
        // 🎯 We only pull 4 bytes (1 float) back across the PCIe bus!
        Self::to_cpu(device, &out_buffer)[0]
    }

    fn matmul(device: &Self, a: &Self::Buffer, shape_a: &[usize], b: &Self::Buffer, shape_b: &[usize]) -> Result<Self::Buffer, String> {
        let dims = MatmulDims { m: shape_a[0] as u32, k: shape_a[1] as u32, n: shape_b[1] as u32, _padding: 0 };
        let out_buffer = Self::zeros(device, &[dims.m as usize, dims.n as usize]);
        let compute_pipeline = device.get_pipeline("matmul_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matmul Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((dims.n + 15) / 16, (dims.m + 15) / 16, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        Ok(out_buffer)
    }

    fn transpose(device: &Self, a: &Self::Buffer, shape: &[usize]) -> Result<Self::Buffer, String> {
        let dims = TransposeDims { rows: shape[0] as u32, cols: shape[1] as u32 };
        let out_buffer = Self::zeros(device, &[dims.cols as usize, dims.rows as usize]);
        let compute_pipeline = device.get_pipeline("transpose_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Transpose Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((dims.cols + 15) / 16, (dims.rows + 15) / 16, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        Ok(out_buffer)
    }

    fn softmax(device: &Self, a: &Self::Buffer, shape: &[usize]) -> Self::Buffer {
        let dims = SoftmaxDims { batch_size: shape[0] as u32, num_classes: shape[1] as u32, _pad1: 0, _pad2: 0 };
        let out_buffer = Self::zeros(device, shape);
        let compute_pipeline = device.get_pipeline("softmax_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Softmax Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((dims.batch_size + 63) / 64, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        out_buffer
    }

    fn mse_loss(device: &Self, a: &Self::Buffer, b: &Self::Buffer, _shape: &[usize]) -> Result<Self::Buffer, String> {
        let dims = MseDims { length: a.length as u32, _pad1: 0, _pad2: 0, _pad3: 0 };
        let out_buffer = Self::zeros(device, &[1]);
        let compute_pipeline = device.get_pipeline("mse_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MSE Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(1, 1, 1); 
        }
        device.queue.submit(Some(encoder.finish()));
        Ok(out_buffer)
    }

    fn adamw_step(device: &Self, weight: &mut Self::Buffer, grad: &Self::Buffer, m: &mut Self::Buffer, v: &mut Self::Buffer, config: &AdamWConfig) {
        let compute_pipeline = device.get_pipeline("adamw_step");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("AdamW Dims"), contents: bytemuck::cast_slice(&[*config]), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: weight.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: m.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: v.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((config.length + 63) / 64, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
    }

    fn conv2d(device: &Self, a: &Self::Buffer, sa: &[usize], w: &Self::Buffer, sw: &[usize], b: &Self::Buffer, _sb: &[usize], stride: usize, padding: usize) -> Result<(Self::Buffer, Vec<usize>), String> {
        let (batch, in_c, h, w_dim) = (sa[0], sa[1], sa[2], sa[3]);
        let (out_c, kh, kw) = (sw[0], sw[2], sw[3]);
        let out_h = (h + 2 * padding - kh) / stride + 1; let out_w = (w_dim + 2 * padding - kw) / stride + 1;
        let dims = Conv2dDims {
            batch: batch as u32, in_c: in_c as u32, h: h as u32, w: w_dim as u32, out_c: out_c as u32, kh: kh as u32, kw: kw as u32,
            out_h: out_h as u32, out_w: out_w as u32, stride: stride as u32, padding: padding as u32, _pad: 0,
        };
        let out_shape = vec![batch, out_c, out_h, out_w];
        let out_buffer = Self::zeros(device, &out_shape);
        let compute_pipeline = device.get_pipeline("conv2d_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Conv2d Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((((batch * out_c * out_h * out_w) as u32) + 63) / 64, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        Ok((out_buffer, out_shape))
    }

    fn conv2d_backward(device: &Self, a: &Self::Buffer, sa: &[usize], go: &Self::Buffer, sgo: &[usize], w: &Self::Buffer, sw: &[usize], stride: usize, padding: usize) -> (Self::Buffer, Self::Buffer, Self::Buffer) {
        let (batch, in_c, h, w_dim) = (sa[0], sa[1], sa[2], sa[3]);
        let (out_c, kh, kw) = (sw[0], sw[2], sw[3]);
        let (out_h, out_w) = (sgo[2], sgo[3]);
        let dims = Conv2dDims {
            batch: batch as u32, in_c: in_c as u32, h: h as u32, w: w_dim as u32, out_c: out_c as u32, kh: kh as u32, kw: kw as u32,
            out_h: out_h as u32, out_w: out_w as u32, stride: stride as u32, padding: padding as u32, _pad: 0,
        };
        let grad_input = Self::zeros(device, sa); let grad_weight = Self::zeros(device, sw); let grad_bias = Self::zeros(device, &[sw[0]]);
        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Conv BW Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM });

        let pipe_bias = device.get_pipeline("conv2d_backward_bias");
        let pipe_weight = device.get_pipeline("conv2d_backward_weight");
        let pipe_input = device.get_pipeline("conv2d_backward_input");

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let bind_bias = device.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe_bias.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: go.buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: grad_bias.buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: dims_buffer.as_entire_binding() }] });
            cpass.set_pipeline(&pipe_bias); cpass.set_bind_group(0, &bind_bias, &[]);
            cpass.dispatch_workgroups(((out_c as u32) + 63) / 64, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let bind_weight = device.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe_weight.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: go.buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: grad_weight.buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() }] });
            cpass.set_pipeline(&pipe_weight); cpass.set_bind_group(0, &bind_weight, &[]);
            cpass.dispatch_workgroups((((out_c * in_c * kh * kw) as u32) + 63) / 64, 1, 1);
        }
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let bind_input = device.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipe_input.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: w.buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: go.buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 2, resource: grad_input.buffer.as_entire_binding() }, wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() }] });
            cpass.set_pipeline(&pipe_input); cpass.set_bind_group(0, &bind_input, &[]);
            cpass.dispatch_workgroups((((batch * in_c * h * w_dim) as u32) + 63) / 64, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        (grad_input, grad_weight, grad_bias)
    }

    fn maxpool2d(device: &Self, a: &Self::Buffer, s: &[usize], k: usize) -> Result<(Self::Buffer, Vec<usize>), String> {
        let (batch, c, h, w) = (s[0], s[1], s[2], s[3]);
        let (out_h, out_w) = (h / k, w / k);
        let dims = MaxPoolDims { batch: batch as u32, c: c as u32, h: h as u32, w: w as u32, out_h: out_h as u32, out_w: out_w as u32, k: k as u32, _pad: 0 };
        let out_shape = vec![batch, c, out_h, out_w];
        let out_buffer = Self::zeros(device, &out_shape);
        let compute_pipeline = device.get_pipeline("maxpool2d_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("MaxPool Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((((batch * c * out_h * out_w) as u32) + 63) / 64, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        Ok((out_buffer, out_shape))
    }

    fn maxpool2d_backward(device: &Self, a: &Self::Buffer, s: &[usize], go: &Self::Buffer, sgo: &[usize], k: usize) -> Self::Buffer {
        let (batch, c, h, w) = (s[0], s[1], s[2], s[3]);
        let (out_h, out_w) = (sgo[2], sgo[3]);
        let dims = MaxPoolDims { batch: batch as u32, c: c as u32, h: h as u32, w: w as u32, out_h: out_h as u32, out_w: out_w as u32, k: k as u32, _pad: 0 };
        let grad_in = Self::zeros(device, s);
        let compute_pipeline = device.get_pipeline("maxpool2d_backward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("MP Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM });
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: a.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: go.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grad_in.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline); cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((((batch * c * out_h * out_w) as u32) + 63) / 64, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        grad_in
    }

    fn batch_norm2d(
        device: &Self, input: &Self::Buffer, shape: &[usize],
        weight: &Self::Buffer, bias: &Self::Buffer,
        running_mean: &Self::Buffer, running_var: &Self::Buffer,
        is_training: bool, momentum: f64, eps: f64
    ) -> Result<Self::Buffer, String> {
        let (batch, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let dims = BatchNormDims {
            batch: batch as u32, c: c as u32, h: h as u32, w: w as u32,
            eps: eps as f32, momentum: momentum as f32, 
            is_training: if is_training { 1 } else { 0 }, _pad: 0,
        };
        
        let out_buffer = Self::zeros(device, shape);
        let compute_pipeline = device.get_pipeline("batch_norm2d_forward");

        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BN Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: weight.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bias.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: running_mean.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: running_var.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: out_buffer.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            // Dispatch 1 thread per channel
            cpass.dispatch_workgroups(((c as u32) + 63) / 64, 1, 1); 
        }
        device.queue.submit(Some(encoder.finish()));
        
        Ok(out_buffer)
    }

    fn batch_norm2d_backward(
        device: &Self, input: &Self::Buffer, shape: &[usize],
        grad_out: &Self::Buffer, weight: &Self::Buffer, eps: f64
    ) -> (Self::Buffer, Self::Buffer, Self::Buffer) {
        let (batch, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let dims = BatchNormDims {
            batch: batch as u32, c: c as u32, h: h as u32, w: w as u32,
            eps: eps as f32, momentum: 0.0, is_training: 0, _pad: 0,
        };
        
        let grad_in = Self::zeros(device, shape);
        let grad_w = Self::zeros(device, &[c]);
        let grad_b = Self::zeros(device, &[c]);
        
        let compute_pipeline = device.get_pipeline("batch_norm2d_backward");
        let dims_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BN BW Dims"), contents: bytemuck::cast_slice(&[dims]), usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: grad_out.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: weight.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grad_in.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grad_w.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grad_b.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dims_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = device.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(((c as u32) + 63) / 64, 1, 1);
        }
        device.queue.submit(Some(encoder.finish()));
        
        (grad_in, grad_w, grad_b)
    }
    
}