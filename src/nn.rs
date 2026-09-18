use crate::autograd::Variable;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize, Deserializer, Serializer};
use std::fs::File;
use std::io::{Read, Write};

#[typetag::serde(tag = "type")]
pub trait Module {
    // 🎯 UPGRADED: Added is_training flag to context-switch behaviors
    fn forward(&self, input: &Variable, is_training: bool) -> Result<Variable, String>;
    
    fn parameters(&self) -> Vec<Variable> { vec![] }
}

// 🎯 NEW: Flattens N-Dimensional tensors into 2D [Batch, Features]
#[derive(Serialize, Deserialize)]
pub struct Flatten;

#[typetag::serde]
impl Module for Flatten {
    fn forward(&self, input: &Variable, _is_training: bool) -> Result<Variable, String> {
        let in_shape = &input.data.borrow().shape;
        
        if in_shape.len() < 2 {
            return Err("Cannot flatten a tensor with less than 2 dimensions".to_string());
        }

        let batch_size = in_shape[0];
        // Multiply the rest of the dimensions together (e.g., 1 * 28 * 28 = 784)
        let flat_features: usize = in_shape[1..].iter().product(); 
        
        Ok(input.reshape(vec![batch_size, flat_features]))
    }
}

pub struct Conv2d {
    pub weight: Variable,
    pub bias: Variable,
    pub stride: usize,
    pub padding: usize,
}

#[derive(Serialize, Deserialize)]
struct Conv2dState {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    padding: usize,
}

impl Serialize for Conv2d {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let state = Conv2dState {
            weight: self.weight.data.borrow().clone(),
            bias: self.bias.data.borrow().clone(),
            stride: self.stride,
            padding: self.padding,
        };
        state.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Conv2d {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        let state = Conv2dState::deserialize(deserializer)?;
        Ok(Conv2d {
            weight: Variable::new(state.weight),
            bias: Variable::new(state.bias),
            stride: state.stride,
            padding: state.padding,
        })
    }
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let mut weight_tensor = Tensor::randn(vec![out_channels, in_channels, kernel_size, kernel_size]);
        
        // 4D Kaiming Initialization for spatial filters
        let fan_in = in_channels * kernel_size * kernel_size; 
        let kaiming_scale = (2.0 / fan_in as f64).sqrt();
        for val in weight_tensor.data.iter_mut() { *val *= kaiming_scale; }
        
        Self {
            weight: Variable::new(weight_tensor),
            bias: Variable::new(Tensor::zeros(vec![out_channels])),
            stride,
            padding,
        }
    }
}

#[typetag::serde]
impl Module for Conv2d {
    fn forward(&self, input: &Variable, _is_training: bool) -> Result<Variable, String> {
        input.conv2d(&self.weight, &self.bias, self.stride, self.padding)
    }
    fn parameters(&self) -> Vec<Variable> { vec![self.weight.clone(), self.bias.clone()] }
}

#[derive(Serialize, Deserialize)]
pub struct MaxPool2d {
    pub kernel_size: usize,
}

#[typetag::serde]
impl Module for MaxPool2d {
    fn forward(&self, input: &Variable, _is_training: bool) -> Result<Variable, String> {
        input.maxpool2d(self.kernel_size)
    }
}

// ==========================================
// Blocks & Containers
// ==========================================

#[derive(Serialize, Deserialize)]
pub struct Sequential { pub layers: Vec<Box<dyn Module>> }

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self { Self { layers } }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut json = String::new();
        file.read_to_string(&mut json).map_err(|e| e.to_string())?;
        let model: Sequential = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        Ok(model)
    }
}

#[typetag::serde]
impl Module for Sequential {
    fn forward(&self, input: &Variable, is_training: bool) -> Result<Variable, String> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current, is_training)?;
        }
        Ok(current)
    }

    fn parameters(&self) -> Vec<Variable> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct Residual { pub block: Sequential }

impl Residual {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { block: Sequential::new(layers) }
    }
}

#[typetag::serde]
impl Module for Residual {
    fn forward(&self, input: &Variable, is_training: bool) -> Result<Variable, String> {
        let fx = self.block.forward(input, is_training)?;
        input.add(&fx) 
    }
    fn parameters(&self) -> Vec<Variable> { self.block.parameters() }
}

// ==========================================
// Layers & Activations
// ==========================================

pub struct Linear {
    pub weight: Variable,
    pub bias: Variable,
}

#[derive(Serialize, Deserialize)]
struct LayerState {
    weight: Tensor,
    bias: Tensor,
}

impl Serialize for Linear {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        let state = LayerState {
            weight: self.weight.data.borrow().clone(),
            bias: self.bias.data.borrow().clone(),
        };
        state.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Linear {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
        let state = LayerState::deserialize(deserializer)?;
        Ok(Linear {
            weight: Variable::new(state.weight),
            bias: Variable::new(state.bias),
        })
    }
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weight_tensor = Tensor::randn(vec![in_features, out_features]);
        let kaiming_scale = (2.0 / in_features as f64).sqrt();
        for val in weight_tensor.data.iter_mut() { *val *= kaiming_scale; }
        Self {
            weight: Variable::new(weight_tensor),
            bias: Variable::new(Tensor::zeros(vec![1, out_features])),
        }
    }
}

#[typetag::serde]
impl Module for Linear {
    fn forward(&self, input: &Variable, _is_training: bool) -> Result<Variable, String> {
        input.matmul(&self.weight)?.add(&self.bias)
    }
    fn parameters(&self) -> Vec<Variable> { vec![self.weight.clone(), self.bias.clone()] }
}

// 🎯 NEW: The Dropout Layer!
#[derive(Serialize, Deserialize)]
pub struct Dropout { pub p: f64 }

#[typetag::serde]
impl Module for Dropout {
    fn forward(&self, input: &Variable, is_training: bool) -> Result<Variable, String> {
        if is_training {
            Ok(input.dropout(self.p)) // Destroys random nodes during training
        } else {
            Ok(input.clone()) // Transparent pass-through during testing
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ReLU;
#[typetag::serde]
impl Module for ReLU {
    fn forward(&self, input: &Variable, _is_training: bool) -> Result<Variable, String> { Ok(input.relu()) }
}

#[derive(Serialize, Deserialize)]
pub struct Sigmoid;
#[typetag::serde]
impl Module for Sigmoid {
    fn forward(&self, input: &Variable, _is_training: bool) -> Result<Variable, String> { Ok(input.sigmoid()) }
}

#[derive(Serialize, Deserialize)]
pub struct Tanh;
#[typetag::serde]
impl Module for Tanh {
    fn forward(&self, input: &Variable, _is_training: bool) -> Result<Variable, String> { Ok(input.tanh()) }
}

#[derive(Serialize, Deserialize)]
pub struct Softmax;
#[typetag::serde]
impl Module for Softmax {
    fn forward(&self, input: &Variable, _is_training: bool) -> Result<Variable, String> { Ok(input.softmax()) }
}