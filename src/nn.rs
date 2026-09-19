// src/nn.rs
use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::backend::Backend;
use rand::Rng;

pub trait Module<B: Backend> {
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>, String>;
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
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>, String> {
        let mut current = input.clone();
        for layer in &self.layers {
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
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>, String> {
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
    pub fn new(device: B, in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let len = out_channels * in_channels * kernel_size * kernel_size;
        let fan_in = in_channels * kernel_size * kernel_size; 
        let kaiming_scale = (2.0 / fan_in as f64).sqrt();
        
        // 🎯 Kaiming Init on CPU before allocating to the Backend
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data.push(z0 * kaiming_scale);
        }
        
        let weight_tensor = Tensor::from_data(device.clone(), data, vec![out_channels, in_channels, kernel_size, kernel_size]).unwrap();
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
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>, String> {
        input.conv2d(&self.weight, &self.bias, self.stride, self.padding)
    }
    fn parameters(&self) -> Vec<Variable<B>> { vec![self.weight.clone(), self.bias.clone()] }
}

pub struct Linear<B: Backend> {
    pub weight: Variable<B>,
    pub bias: Variable<B>,
}

impl<B: Backend> Linear<B> {
    pub fn new(device: B, in_features: usize, out_features: usize) -> Self {
        let len = in_features * out_features;
        let kaiming_scale = (2.0 / in_features as f64).sqrt();
        
        // 🎯 Kaiming Init on CPU before allocating to the Backend
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data.push(z0 * kaiming_scale);
        }
        
        let weight_tensor = Tensor::from_data(device.clone(), data, vec![in_features, out_features]).unwrap();
        let bias_tensor = Tensor::zeros(device.clone(), vec![1, out_features]);
        
        Self {
            weight: Variable::new(weight_tensor),
            bias: Variable::new(bias_tensor),
        }
    }
}

impl<B: Backend> Module<B> for Linear<B> {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>, String> {
        input.matmul(&self.weight)?.add(&self.bias)
    }
    fn parameters(&self) -> Vec<Variable<B>> { vec![self.weight.clone(), self.bias.clone()] }
}

// ==========================================
// Utility Layers & Activations
// ==========================================

pub struct Flatten;

impl<B: Backend> Module<B> for Flatten {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>, String> {
        let in_shape = &input.data.borrow().shape;
        if in_shape.len() < 2 {
            return Err("Cannot flatten a tensor with less than 2 dimensions".to_string());
        }
        let batch_size = in_shape[0];
        let flat_features: usize = in_shape[1..].iter().product(); 
        Ok(input.reshape(vec![batch_size, flat_features]))
    }
}

pub struct MaxPool2d { 
    pub kernel_size: usize 
}

impl<B: Backend> Module<B> for MaxPool2d {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>, String> {
        input.maxpool2d(self.kernel_size)
    }
}

pub struct Dropout { 
    pub p: f64 
}

impl<B: Backend> Module<B> for Dropout {
    fn forward(&self, input: &Variable<B>, is_training: bool) -> Result<Variable<B>, String> {
        if is_training {
            Ok(input.dropout(self.p)) 
        } else {
            Ok(input.clone()) 
        }
    }
}

pub struct ReLU;
impl<B: Backend> Module<B> for ReLU {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>, String> { Ok(input.relu()) }
}

pub struct Sigmoid;
impl<B: Backend> Module<B> for Sigmoid {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>, String> { Ok(input.sigmoid()) }
}

pub struct Tanh;
impl<B: Backend> Module<B> for Tanh {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>, String> { Ok(input.tanh()) }
}

pub struct Softmax;
impl<B: Backend> Module<B> for Softmax {
    fn forward(&self, input: &Variable<B>, _is_training: bool) -> Result<Variable<B>, String> { Ok(input.softmax()) }
}