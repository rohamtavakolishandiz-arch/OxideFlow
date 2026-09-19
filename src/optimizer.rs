// src/optimizer.rs
use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::backend::Backend;

pub trait Optimizer<B: Backend> {
    fn step(&mut self);
    fn zero_grad(&mut self);
    fn decay_lr(&mut self, factor: f64);
    fn clip_grads(&self, max_norm: f64);
}

// ==========================================
// SGD Implementation
// ==========================================

pub struct SGD<B: Backend> {
    parameters: Vec<Variable<B>>,
    learning_rate: f64,
    momentum: f64,
    velocities: Vec<Tensor<B>>,
}

impl<B: Backend> SGD<B> {
    pub fn new(parameters: Vec<Variable<B>>, learning_rate: f64, momentum: f64) -> Self {
        let velocities = parameters.iter().map(|p| {
            let shape = p.data.borrow().shape.clone();
            let device = p.data.borrow().device.clone();
            Tensor::zeros(device, shape)
        }).collect();

        Self {
            parameters,
            learning_rate,
            momentum,
            velocities,
        }
    }
}

impl<B: Backend> Optimizer<B> for SGD<B> {
    fn step(&mut self) {
        for (param, velocity) in self.parameters.iter().zip(self.velocities.iter_mut()) {
            let shape = param.data.borrow().shape.clone();
            let device = param.data.borrow().device.clone();

            // Pull current state to CPU
            let cpu_p = B::to_cpu(&device, &param.data.borrow().data);
            let cpu_g = B::to_cpu(&device, &param.grad.borrow().data);
            let mut cpu_v = B::to_cpu(&device, &velocity.data);
            let mut new_p = cpu_p.clone();

            for i in 0..cpu_p.len() {
                cpu_v[i] = (self.momentum * cpu_v[i]) + cpu_g[i];
                new_p[i] -= self.learning_rate * cpu_v[i];
            }

            // Push updated state back to device
            *param.data.borrow_mut() = Tensor::from_data(device.clone(), new_p, shape.clone()).unwrap();
            *velocity = Tensor::from_data(device, cpu_v, shape).unwrap();
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.parameters {
            param.zero_grad();
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

pub struct AdamW<B: Backend> {
    pub params: Vec<Variable<B>>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    t: usize,
    m: Vec<Tensor<B>>,
    v: Vec<Tensor<B>>,
}

impl<B: Backend> AdamW<B> {
    pub fn new(params: Vec<Variable<B>>, lr: f64, weight_decay: f64) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        
        for p in &params {
            let shape = p.data.borrow().shape.clone();
            let device = p.data.borrow().device.clone();
            m.push(Tensor::zeros(device.clone(), shape.clone()));
            v.push(Tensor::zeros(device, shape));
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

    pub fn save(&self, _path: &str) -> Result<(), String> {
        Err("Saving optimizer state is temporarily disabled during Backend Refactor".to_string())
    }

    pub fn load(&mut self, _path: &str) -> Result<(), String> {
        Err("Loading optimizer state is temporarily disabled during Backend Refactor".to_string())
    }
}

impl<B: Backend> Optimizer<B> for AdamW<B> {
    fn zero_grad(&mut self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn step(&mut self) {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..self.params.len() {
            let p_var = &self.params[i];
            let shape = p_var.data.borrow().shape.clone();
            let device = p_var.data.borrow().device.clone();

            let is_bias = shape.len() == 1 || (shape.len() == 2 && shape[0] == 1);
            let current_wd = if is_bias { 0.0 } else { self.weight_decay };

            // 🎯 Package hyperparameters for the GPU
            let config = crate::backend::AdamWConfig {
                length: shape.iter().product::<usize>() as u32,
                lr: self.lr as f32,
                beta1: self.beta1 as f32,
                beta2: self.beta2 as f32,
                eps: self.eps as f32,
                weight_decay: current_wd as f32,
                bias_correction1: bias_correction1 as f32,
                bias_correction2: bias_correction2 as f32,
            };

            let mut weight = p_var.data.borrow_mut();
            let grad = p_var.grad.borrow();

            // 🎯 Delegate execution entirely to the hardware Backend
            B::adamw_step(
                &device,
                &mut weight.data,
                &grad.data,
                &mut self.m[i].data,
                &mut self.v[i].data,
                &config
            );
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

fn clip_global_norm<B: Backend>(parameters: &Vec<Variable<B>>, max_norm: f64) {
    let mut total_norm = 0.0;
    
    // 1. Calculate the L2 norm of all gradients combined (Pulling to CPU)
    for p in parameters {
        let device = p.grad.borrow().device.clone();
        let cpu_g = B::to_cpu(&device, &p.grad.borrow().data);
        for &g in &cpu_g {
            total_norm += g * g;
        }
    }
    total_norm = total_norm.sqrt();
    
    // 2. Scale gradients down if they explode
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for p in parameters {
            let device = p.grad.borrow().device.clone();
            let shape = p.grad.borrow().shape.clone();
            let mut cpu_g = B::to_cpu(&device, &p.grad.borrow().data);
            
            for g in cpu_g.iter_mut() {
                *g *= scale;
            }
            *p.grad.borrow_mut() = Tensor::from_data(device, cpu_g, shape).unwrap();
        }
    }
}