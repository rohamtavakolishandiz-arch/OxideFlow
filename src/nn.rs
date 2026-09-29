// src/nn.rs
use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::backend::Backend;
use crate::error::{OxideError, Result}; // 🛡️ Import the new Error Architecture
use rand::Rng;

pub trait Module<B: Backend> {
    // 🛡️ Forward pass is now completely fallible and safe
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>>;
    fn parameters(&self) -> Vec<Variable<B>> { vec![] }
}

// ==========================================
// Blocks & Containers
// ==========================================

pub struct Sequential<B: Backend> { 
    pub layers: Vec<Box<dyn Module<B>>> 
}

impl<B: Backend> Sequential<B> {
    pub fn new(layers: Vec<Box<dyn Module<B>>>) -> Self { 
        Self { layers } 
    }
}

impl<B: Backend> Module<B> for Sequential<B> {
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>> {
        let mut current = input.clone();
        for layer in &self.layers {
            // ? instantly bubbles up any NaN or Shape mismatches from deeper layers
            current = layer.forward(&current, is_training)?; 
        }
        Ok(current)
    }

    fn parameters(&self) -> Vec<Variable<B>> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

pub struct Residual<B: Backend> { 
    pub block: Sequential<B> 
}

impl<B: Backend> Residual<B> {
    pub fn new(layers: Vec<Box<dyn Module<B>>>) -> Self {
        Self { block: Sequential::new(layers) }
    }
}

impl<B: Backend> Module<B> for Residual<B> {
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>> {
        let fx = self.block.forward(input, is_training)?;
        input.add(&fx) 
    }
    
    fn parameters(&self) -> Vec<Variable<B>> { 
        self.block.parameters() 
    }
}

// ==========================================
// Layers with Parameters
// ==========================================

pub struct Conv2d<B: Backend> {
    pub weight: Variable<B>,
    pub bias: Variable<B>,
    pub stride: usize,
    pub padding: usize,
}

impl<B: Backend> Conv2d<B> {
    pub fn new(device: B, in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, zero_init: bool) -> Self {
        let len = out_channels * in_channels * kernel_size * kernel_size;
        
        let data = if zero_init {
            // 🚀 ZERO-INITIALIZATION: Starts the network in a pure, dormant state
            vec![0.0; len]
        } else {
            let fan_in = in_channels * kernel_size * kernel_size; 
            let kaiming_scale = (2.0 / fan_in as f64).sqrt();
            let mut rng = rand::thread_rng();
            let mut d = Vec::with_capacity(len);
            for _ in 0..len {
                let u1: f64 = rng.gen_range(1e-10..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                d.push(z0 * kaiming_scale);
            }
            d
        };
        
        let weight_tensor = Tensor::from_data(device.clone(), data, vec![out_channels, in_channels, kernel_size, kernel_size])
            .expect("Fatal Startup Error: Failed to allocate Conv2d weights in VRAM.");
        let bias_tensor = Tensor::zeros(device.clone(), vec![out_channels]);
        
        Self {
            weight: Variable::new(weight_tensor),
            bias: Variable::new(bias_tensor),
            stride,
            padding,
        }
    }
}

impl<B: Backend> Module<B> for Conv2d<B> {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> {
        input.conv2d(&self.weight, &self.bias, self.stride, self.padding)
    }
    fn parameters(&self) -> Vec<Variable<B>> { vec![self.weight.clone(), self.bias.clone()] }
}

pub struct BatchNorm2d<B: Backend> {
    pub weight: Variable<B>, 
    pub bias: Variable<B>,   
    pub running_mean: Tensor<B>,
    pub running_var: Tensor<B>,
    pub momentum: f64,
    pub eps: f64,
}

impl<B: Backend> BatchNorm2d<B> {
    pub fn new(device: B, num_features: usize) -> Self {
        let weight_tensor = Tensor::from_data(device.clone(), vec![1.0; num_features], vec![num_features])
            .expect("Fatal Startup Error: Failed to allocate BatchNorm weights.");
        let bias_tensor = Tensor::zeros(device.clone(), vec![num_features]);
        let running_mean = Tensor::zeros(device.clone(), vec![num_features]);
        let running_var = Tensor::from_data(device.clone(), vec![1.0; num_features], vec![num_features])
            .expect("Fatal Startup Error: Failed to allocate BatchNorm variance.");
        
        Self {
            weight: Variable::new(weight_tensor),
            bias: Variable::new(bias_tensor),
            running_mean,
            running_var,
            momentum: 0.1,
            eps: 1e-5,
        }
    }
}

impl<B: Backend> Module<B> for BatchNorm2d<B> {
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>> {
        input.batch_norm2d(
            &self.weight, 
            &self.bias, 
            &self.running_mean, 
            &self.running_var, 
            is_training, 
            self.momentum, 
            self.eps
        )
    }
    
    fn parameters(&self) -> Vec<Variable<B>> { 
        vec![self.weight.clone(), self.bias.clone()] 
    }
}

pub struct Linear<B: Backend> {
    pub weight: Variable<B>,
    pub bias: Variable<B>,
}

impl<B: Backend> Linear<B> {
    pub fn new(device: B, in_features: usize, out_features: usize, zero_init: bool) -> Self {
        let len = in_features * out_features;
        
        let data = if zero_init {
            vec![0.0; len]
        } else {
            let kaiming_scale = (2.0 / in_features as f64).sqrt();
            let mut rng = rand::thread_rng();
            let mut d = Vec::with_capacity(len);
            for _ in 0..len {
                let u1: f64 = rng.gen_range(1e-10..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                d.push(z0 * kaiming_scale);
            }
            d
        };
        
        let weight_tensor = Tensor::from_data(device.clone(), data, vec![in_features, out_features])
            .expect("Fatal Startup Error: Failed to allocate Linear weights.");
        let bias_tensor = Tensor::zeros(device.clone(), vec![1, out_features]);
        
        Self {
            weight: Variable::new(weight_tensor),
            bias: Variable::new(bias_tensor),
        }
    }
}

impl<B: Backend> Module<B> for Linear<B> {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> {
        input.matmul(&self.weight)?.add(&self.bias)
    }
    fn parameters(&self) -> Vec<Variable<B>> { vec![self.weight.clone(), self.bias.clone()] }
}

// ==========================================
// Utility Layers & Activations
// ==========================================

pub struct Flatten;

impl<B: Backend> Module<B> for Flatten {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> {
        let in_shape = &input.data.borrow().shape;
        if in_shape.len() < 2 {
            return Err(OxideError::InvalidDimension { 
                axis: 2, 
                shape: in_shape.clone() 
            });
        }
        let batch_size = in_shape[0];
        let flat_features: usize = in_shape[1..].iter().product(); 
        input.reshape(vec![batch_size, flat_features])
    }
}

pub struct MaxPool2d { 
    pub kernel_size: usize 
}

impl<B: Backend> Module<B> for MaxPool2d {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> {
        input.maxpool2d(self.kernel_size)
    }
}

pub struct Dropout { 
    pub p: f64 
}

impl<B: Backend> Module<B> for Dropout {
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>> {
        if is_training {
            // Safely propagate errors from the autograd dropout implementation
            Ok(input.dropout(self.p)?) 
        } else {
            Ok(input.clone()) 
        }
    }
}

pub struct ReLU;
impl<B: Backend> Module<B> for ReLU {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> { Ok(input.relu()) }
}

pub struct Sigmoid;
impl<B: Backend> Module<B> for Sigmoid {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> { Ok(input.sigmoid()) }
}

pub struct Tanh;
impl<B: Backend> Module<B> for Tanh {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> { Ok(input.tanh()) }
}

pub struct Softmax;
impl<B: Backend> Module<B> for Softmax {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> { Ok(input.softmax()) }
}

// ==========================================
// 👁️ NCA PHYSICS: Fixed Perception Layer
// ==========================================
pub struct Perception2d<B: Backend> {
    pub weight: Variable<B>,
    pub bias: Variable<B>,
}

impl<B: Backend> Perception2d<B> {
    pub fn new(device: B, channels: usize) -> Self {
        let out_channels = channels * 3;
        let len = out_channels * channels * 3 * 3;
        let mut data = vec![0.0; len];
        
        for c in 0..channels {
            let out_id = c * 3;
            let out_x  = c * 3 + 1;
            let out_y  = c * 3 + 2;
            
            let idx = |o: usize, i: usize, dy: usize, dx: usize| {
                o * (channels * 9) + i * 9 + dy * 3 + dx
            };

            // 1. Identity Filter (Preserved exactly)
            data[idx(out_id, c, 1, 1)] = 1.0;

            // 2. Sobel X Filter (Normalized by 8.0)
            data[idx(out_x, c, 0, 0)] = -1.0 / 8.0;
            data[idx(out_x, c, 1, 0)] = -2.0 / 8.0;
            data[idx(out_x, c, 2, 0)] = -1.0 / 8.0;
            data[idx(out_x, c, 0, 2)] =  1.0 / 8.0;
            data[idx(out_x, c, 1, 2)] =  2.0 / 8.0;
            data[idx(out_x, c, 2, 2)] =  1.0 / 8.0;

            // 3. Sobel Y Filter (Normalized by 8.0)
            data[idx(out_y, c, 0, 0)] = -1.0 / 8.0;
            data[idx(out_y, c, 0, 1)] = -2.0 / 8.0;
            data[idx(out_y, c, 0, 2)] = -1.0 / 8.0;
            data[idx(out_y, c, 2, 0)] =  1.0 / 8.0;
            data[idx(out_y, c, 2, 1)] =  2.0 / 8.0;
            data[idx(out_y, c, 2, 2)] =  1.0 / 8.0;
        }

        let weight_tensor = Tensor::from_data(device.clone(), data, vec![out_channels, channels, 3, 3])
            .expect("Fatal Startup Error: Failed to allocate Perception weights.");
        let bias_tensor = Tensor::zeros(device.clone(), vec![out_channels]);

        Self {
            weight: Variable::new_constant(weight_tensor),
            bias: Variable::new_constant(bias_tensor),
        }
    }
}

impl<B: Backend> Module<B> for Perception2d<B> {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> {
        input.conv2d(&self.weight, &self.bias, 1, 1)
    }
    
    fn parameters(&self) -> Vec<Variable<B>> {
        vec![] 
    }
}

// ==========================================
// 🧬 NCA PHYSICS: Asynchronous Cellular Firing
// ==========================================
pub struct AsynchronousFire {
    pub drop_probability: f64,
}

impl AsynchronousFire {
    pub fn new(drop_probability: f64) -> Self {
        Self { drop_probability }
    }
}

impl<B: Backend> Module<B> for AsynchronousFire {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>> {
        // 🛡️ We ignore `_is_training` and force dropout every single frame!
        // This kills the synchronized "glider" exploit.
        Ok(input.dropout(self.drop_probability)?)
    }
}