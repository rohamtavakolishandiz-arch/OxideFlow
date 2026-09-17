use crate::autograd::Variable;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize, Deserializer, Serializer};
use std::fs::File;
use std::io::{Read, Write};

/// Trait مشترک با قابلیت سریالایز شدن خودکار توسط typetag
#[typetag::serde(tag = "type")]
pub trait Module {
    fn forward(&self, input: &Variable) -> Result<Variable, String>;
    
    fn parameters(&self) -> Vec<Variable> {
        vec![] 
    }
}

#[derive(Serialize, Deserialize)]
pub struct Residual {
    pub block: Sequential,
}

impl Residual {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self {
            block: Sequential::new(layers),
        }
    }
}

#[typetag::serde]
impl Module for Residual {
    fn forward(&self, input: &Variable) -> Result<Variable, String> {
        // 1. Pass input through the inner layers: F(x)
        let fx = self.block.forward(input)?;
        
        // 2. Add the original input back to the result: F(x) + x
        input.add(&fx) 
    }

    fn parameters(&self) -> Vec<Variable> {
        self.block.parameters()
    }
}
/// نمایانگر یک لایه خطی
pub struct Linear {
    pub weight: Variable,
    pub bias: Variable,
}

// ساختار واسط برای جلوگیری از ذخیره کل گراف محاسباتی (فقط وزن‌ها ذخیره می‌شوند)
#[derive(Serialize, Deserialize)]
struct LayerState {
    weight: Tensor,
    bias: Tensor,
}

// پیاده‌سازی سفارشی Serialize
impl Serialize for Linear {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let state = LayerState {
            weight: self.weight.data.borrow().clone(),
            bias: self.bias.data.borrow().clone(),
        };
        state.serialize(serializer)
    }
}

// پیاده‌سازی سفارشی Deserialize
impl<'de> Deserialize<'de> for Linear {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let state = LayerState::deserialize(deserializer)?;
        Ok(Linear {
            weight: Variable::new(state.weight),
            bias: Variable::new(state.bias),
        })
    }
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // 1. Generate standard random weights from a normal distribution
        let mut weight_tensor = Tensor::randn(vec![in_features, out_features]);
        
        // 2. Apply Kaiming (He) scale to stabilize variance across deep layers
        let kaiming_scale = (2.0 / in_features as f64).sqrt();
        for val in weight_tensor.data.iter_mut() {
            *val *= kaiming_scale;
        }

        let weight = Variable::new(weight_tensor);
        
        // 3. Biases safely start at zero
        let bias = Variable::new(Tensor::zeros(vec![1, out_features]));
        
        Self { weight, bias }
    }
}

#[typetag::serde]
impl Module for Linear {
    fn forward(&self, input: &Variable) -> Result<Variable, String> {
        let matmul_result = input.matmul(&self.weight)?;
        matmul_result.add(&self.bias)
    }

    fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

// --- توابع فعال‌سازی ---

#[derive(Serialize, Deserialize)]
pub struct ReLU;
#[typetag::serde]
impl Module for ReLU {
    fn forward(&self, input: &Variable) -> Result<Variable, String> { Ok(input.relu()) }
}

#[derive(Serialize, Deserialize)]
pub struct Sigmoid;
#[typetag::serde]
impl Module for Sigmoid {
    fn forward(&self, input: &Variable) -> Result<Variable, String> { Ok(input.sigmoid()) }
}

#[derive(Serialize, Deserialize)]
pub struct Tanh;
#[typetag::serde]
impl Module for Tanh {
    fn forward(&self, input: &Variable) -> Result<Variable, String> { Ok(input.tanh()) }
}

#[derive(Serialize, Deserialize)]
pub struct Softmax;

#[typetag::serde]
impl Module for Softmax {
    fn forward(&self, input: &Variable) -> Result<Variable, String> {
        Ok(input.softmax())
    }
}
// --- کانتینر ترتیبی ---

#[derive(Serialize, Deserialize)]
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    /// ذخیره‌سازی کل مدل
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize model: {}", e))?;
        let mut file = File::create(path).map_err(|e| format!("File Error: {}", e))?;
        file.write_all(json.as_bytes()).map_err(|e| format!("Write Error: {}", e))?;
        Ok(())
    }

    /// بارگذاری مدل به عنوان یک شیء جدید
    pub fn load(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("File Error: {}", e))?;
        let mut json = String::new();
        file.read_to_string(&mut json).map_err(|e| format!("Read Error: {}", e))?;
        let model = Sequential::new(vec![
            // 1. Projection layer (2D input -> 64D feature space)
            Box::new(Linear::new(2, 64)),
            Box::new(ReLU),
        
            // 2. Deep Residual Blocks (preserves gradient flow across deep layers)
            Box::new(Residual::new(vec![
                Box::new(Linear::new(64, 64)),
                Box::new(ReLU),
                Box::new(Linear::new(64, 64)),
                Box::new(ReLU),
            ])),
            Box::new(Residual::new(vec![
                Box::new(Linear::new(64, 64)),
                Box::new(ReLU),
                Box::new(Linear::new(64, 64)),
                Box::new(ReLU),
            ])),
        
            // 3. Classification Head (64D -> 2 output classes)
            Box::new(Linear::new(64, 2)),
        ]);
        Ok(model)
    }
}

#[typetag::serde]
impl Module for Sequential {
    fn forward(&self, input: &Variable) -> Result<Variable, String> {
        let mut current = input.clone();
        for layer in &self.layers {
            current = layer.forward(&current)?;
        }
        Ok(current)
    }

    fn parameters(&self) -> Vec<Variable> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}