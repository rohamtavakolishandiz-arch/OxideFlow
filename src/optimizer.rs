// src/optimizer.rs
use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::backend::Backend;
use crate::error::Result; // 🛡️ Import the safe error architecture

pub trait Optimizer<B: Backend> {
    fn step(&mut self) -> Result<()>;
    fn zero_grad(&mut self);
    fn decay_lr(&mut self, factor: f64);
    fn clip_grads(&self, max_norm: f64) -> Result<()>;
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
    fn step(&mut self) -> Result<()> {
        // 🛡️ INTERCEPT: Clean gradients before applying to SGD
        check_and_sanitize_nans(&self.parameters)?;

        for (param, velocity) in self.parameters.iter().zip(self.velocities.iter_mut()) {
            let shape = param.data.borrow().shape.clone();
            let device = param.data.borrow().device.clone();

            let cpu_p = B::to_cpu(&device, &param.data.borrow().data);
            let cpu_g = B::to_cpu(&device, &param.grad.borrow().data);
            let mut cpu_v = B::to_cpu(&device, &velocity.data);
            let mut new_p = cpu_p.clone();

            for i in 0..cpu_p.len() {
                cpu_v[i] = (self.momentum * cpu_v[i]) + cpu_g[i];
                new_p[i] -= self.learning_rate * cpu_v[i];
            }

            // 🛡️ Safe Propagation (No unwrap!)
            *param.data.borrow_mut() = Tensor::from_data(device.clone(), new_p, shape.clone())?;
            *velocity = Tensor::from_data(device, cpu_v, shape)?;
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
    
    fn decay_lr(&mut self, factor: f64) {
        self.learning_rate *= factor;
    }

    fn clip_grads(&self, max_norm: f64) -> Result<()> {
        clip_global_norm(&self.parameters, max_norm)
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

    pub fn save(&self, path: &str) -> Result<()> {
        if let Some(first_param) = self.params.first() {
            let device = first_param.data.borrow().device.clone();
            crate::checkpoint::save_adamw_state(&device, self.t, &self.m, &self.v, path)?;
        }
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        if let Some(first_param) = self.params.first() {
            let device = first_param.data.borrow().device.clone();
            crate::checkpoint::load_adamw_state(&device, &mut self.t, &mut self.m, &mut self.v, path)?;
        }
        Ok(())
    }
}

impl<B: Backend> Optimizer<B> for AdamW<B> {
    fn zero_grad(&mut self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn step(&mut self) -> Result<()> {
        self.t += 1;
        
        // 🛡️ INTERCEPT: Clean gradients before applying to AdamW
        check_and_sanitize_nans(&self.params)?;

        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for i in 0..self.params.len() {
            let p_var = &self.params[i];
            let shape = p_var.data.borrow().shape.clone();
            let device = p_var.data.borrow().device.clone();

            let is_bias = shape.len() == 1 || (shape.len() == 2 && shape[0] == 1);
            let current_wd = if is_bias { 0.0 } else { self.weight_decay };

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

            B::adamw_step(
                &device,
                &mut weight.data,
                &grad.data,
                &mut self.m[i].data,
                &mut self.v[i].data,
                &config
            );
        }
        Ok(())
    }
    
    fn decay_lr(&mut self, factor: f64) {
        self.lr *= factor;
    }

    fn clip_grads(&self, max_norm: f64) -> Result<()> {
        clip_global_norm(&self.params, max_norm)
    }
}

// ==========================================
// Helper Functions (Safe Interceptors)
// ==========================================

// 🛡️ NEW: Detects NaN/Infinity explosions and neutralizes them to 0.0 before they poison the model
fn check_and_sanitize_nans<B: Backend>(parameters: &Vec<Variable<B>>) -> Result<()> {
    for p in parameters {
        let mut grad = p.grad.borrow_mut();
        let device = grad.device.clone();
        
        let cpu_grad = B::to_cpu(&device, &grad.data);
        let mut is_corrupted = false;
        let mut safe_data = Vec::with_capacity(cpu_grad.len());
        
        for &g in &cpu_grad {
            if g.is_nan() || g.is_infinite() {
                is_corrupted = true;
                safe_data.push(0.0); // Neutralize
            } else {
                safe_data.push(g);
            }
        }

        if is_corrupted {
            println!("⚠️ [OXIDEFLOW GUARD]: NaN/Infinity detected in gradients! Neutralizing...");
            *grad = Tensor::from_data(device, safe_data, grad.shape.clone())?;
        }
    }
    Ok(())
}

fn clip_global_norm<B: Backend>(parameters: &Vec<Variable<B>>, max_norm: f64) -> Result<()> {
    let mut total_norm = 0.0;
    
    for p in parameters {
        let device = p.grad.borrow().device.clone();
        let cpu_g = B::to_cpu(&device, &p.grad.borrow().data);
        for &g in &cpu_g {
            total_norm += g * g;
        }
    }
    total_norm = total_norm.sqrt();
    
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for p in parameters {
            let device = p.grad.borrow().device.clone();
            let shape = p.grad.borrow().shape.clone();
            let mut cpu_g = B::to_cpu(&device, &p.grad.borrow().data);
            
            for g in cpu_g.iter_mut() {
                *g *= scale;
            }
            // 🛡️ Safe Propagation
            *p.grad.borrow_mut() = Tensor::from_data(device, cpu_g, shape)?;
        }
    }
    Ok(())
}

// ==========================================
// Learning Rate Schedulers
// ==========================================

pub trait Scheduler<B: Backend> {
    fn step(&mut self, optimizer: &mut dyn Optimizer<B>, epoch: usize);
}

pub struct StepScheduler {
    pub drop_every: usize,
    pub factor: f64,
}

impl<B: Backend> Scheduler<B> for StepScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer<B>, epoch: usize) {
        // Drop the learning rate by `factor` at regular epoch intervals
        if epoch > 0 && epoch % self.drop_every == 0 {
            optimizer.decay_lr(self.factor);
        }
    }
}

pub struct CosineAnnealingScheduler {
    pub initial_lr: f64,
    pub total_epochs: usize,
}

impl<B: Backend> Scheduler<B> for CosineAnnealingScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer<B>, epoch: usize) {
        if epoch == 0 || epoch > self.total_epochs {
            return;
        }
        
        let pi = std::f64::consts::PI;
        
        // Calculate the theoretical LR of the previous step and the current step
        let prev_lr = self.initial_lr * 0.5 * (1.0 + (((epoch - 1) as f64 * pi) / self.total_epochs as f64).cos());
        let new_lr = self.initial_lr * 0.5 * (1.0 + ((epoch as f64 * pi) / self.total_epochs as f64).cos());
        
        // Use the ratio as the decay factor to modify the optimizer's internal state
        if prev_lr > 1e-15 {
            let factor = new_lr / prev_lr;
            optimizer.decay_lr(factor);
        }
    }
}