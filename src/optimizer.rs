use crate::autograd::Variable;
use crate::tensor::Tensor;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write};

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn decay_lr(&mut self, factor: f64);
    
    // 🎯 NEW: Global gradient clipping to stabilize deep ResNets
    fn clip_grads(&self, max_norm: f64);
}

#[derive(Serialize, Deserialize)]
pub struct AdamWState {
    pub lr: f64,
    pub t: usize,
    pub m: Vec<Tensor>,
    pub v: Vec<Tensor>,
}

// ==========================================
// SGD Implementation
// ==========================================

pub struct SGD {
    parameters: Vec<Variable>,
    learning_rate: f64,
    momentum: f64,
    velocities: Vec<Tensor>,
}

impl SGD {
    pub fn new(parameters: Vec<Variable>, learning_rate: f64, momentum: f64) -> Self {
        let velocities = parameters.iter().map(|p| {
            Tensor::zeros(p.data.borrow().shape.clone())
        }).collect();

        Self {
            parameters,
            learning_rate,
            momentum,
            velocities,
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for (param, velocity) in self.parameters.iter().zip(self.velocities.iter_mut()) {
            let mut p = param.data.borrow_mut();
            let g = param.grad.borrow();

            for i in 0..p.data.len() {
                velocity.data[i] = (self.momentum * velocity.data[i]) + g.data[i];
                p.data[i] -= self.learning_rate * velocity.data[i];
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.parameters {
            let mut grad = param.grad.borrow_mut();
            for val in grad.data.iter_mut() {
                *val = 0.0;
            }
        }
    }
    
    fn decay_lr(&mut self, factor: f64) {
        self.learning_rate *= factor;
    }

    fn clip_grads(&self, max_norm: f64) {
        clip_global_norm(&self.parameters, max_norm);
    }
}

// ==========================================
// AdamW Implementation
// ==========================================

pub struct AdamW {
    pub params: Vec<Variable>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    t: usize,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    pub fn new(params: Vec<Variable>, lr: f64, weight_decay: f64) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        
        for p in &params {
            let shape = p.data.borrow().shape.clone();
            m.push(Tensor::zeros(shape.clone()));
            v.push(Tensor::zeros(shape));
        }
    
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay, 
            t: 0,
            m,
            v,
        }
    }

    /// 🎯 Saves the optimizer's momentum and variance state
    pub fn save(&self, path: &str) -> Result<(), String> {
        let state = AdamWState {
            lr: self.lr,
            t: self.t,
            m: self.m.clone(),
            v: self.v.clone(),
        };
        
        let json = serde_json::to_string_pretty(&state)
            .map_err(|e| format!("Failed to serialize optimizer: {}", e))?;
        let mut file = File::create(path).map_err(|e| format!("File Error: {}", e))?;
        file.write_all(json.as_bytes()).map_err(|e| format!("Write Error: {}", e))?;
        Ok(())
    }

    /// 🎯 Loads the state back into the optimizer to prevent gradient shocks
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        let mut file = File::open(path).map_err(|e| format!("File Error: {}", e))?;
        let mut json = String::new();
        file.read_to_string(&mut json).map_err(|e| format!("Read Error: {}", e))?;
        
        let state: AdamWState = serde_json::from_str(&json)
            .map_err(|e| format!("Parse Error: {}", e))?;
        
        // Overwrite current state with the loaded state
        self.lr = state.lr;
        self.t = state.t;
        self.m = state.m;
        self.v = state.v;
        Ok(())
    }
}

impl Optimizer for AdamW {
    fn zero_grad(&mut self) {
        for p in &self.params {
            let mut grad = p.grad.borrow_mut();
            for val in grad.data.iter_mut() {
                *val = 0.0;
            }
        }
    }

    fn step(&mut self) {
        self.t += 1;
        
        // 🎯 FIX: Calculate bias correction ONCE per step, not inside the inner loop!
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..self.params.len() {
            let mut weight = self.params[i].data.borrow_mut();
            let grad = self.params[i].grad.borrow();
            
            let m_tensor = &mut self.m[i];
            let v_tensor = &mut self.v[i];

            // 🎯 FIX: Smart weight decay masking. 
            // If the tensor is 1D or [1, N], it is a bias. Do not decay biases!
            let is_bias = weight.shape.len() == 1 || (weight.shape.len() == 2 && weight.shape[0] == 1);
            let current_wd = if is_bias { 0.0 } else { self.weight_decay };

            for j in 0..weight.data.len() {
                let g = grad.data[j];
                let w = weight.data[j];

                m_tensor.data[j] = self.beta1 * m_tensor.data[j] + (1.0 - self.beta1) * g;
                v_tensor.data[j] = self.beta2 * v_tensor.data[j] + (1.0 - self.beta2) * g * g;

                let m_hat = m_tensor.data[j] / bias_correction1;
                let v_hat = v_tensor.data[j] / bias_correction2;

                weight.data[j] = w - self.lr * (m_hat / (v_hat.sqrt() + self.eps) + current_wd * w);
            }
        }
    }
    
    fn decay_lr(&mut self, factor: f64) {
        self.lr *= factor;
    }

    fn clip_grads(&self, max_norm: f64) {
        clip_global_norm(&self.params, max_norm);
    }
}

// ==========================================
// Helper Functions
// ==========================================

/// 🎯 NEW: Calculates the global norm across all parameters and scales gradients if they exceed max_norm.
fn clip_global_norm(parameters: &Vec<Variable>, max_norm: f64) {
    let mut total_norm = 0.0;
    
    // 1. Calculate the L2 norm of all gradients combined
    for p in parameters {
        let grad = p.grad.borrow();
        for &g in &grad.data {
            total_norm += g * g;
        }
    }
    total_norm = total_norm.sqrt();
    
    // 2. Scale gradients down if they explode
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for p in parameters {
            let mut grad = p.grad.borrow_mut();
            for g in grad.data.iter_mut() {
                *g *= scale;
            }
        }
    }
}