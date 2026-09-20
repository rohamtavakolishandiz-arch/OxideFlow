// src/autograd.rs
use crate::tensor::Tensor;
use crate::backend::Backend;
use rand::Rng;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Op<B: Backend> {
    None,
    Add(Variable<B>, Variable<B>),
    Matmul(Variable<B>, Variable<B>),
    ReLU(Variable<B>),
    Sigmoid(Variable<B>),
    Tanh(Variable<B>),
    MSE(Variable<B>, Variable<B>),
    CrossEntropy(Variable<B>, Variable<B>),
    Softmax(Variable<B>),
    Dropout(Variable<B>, Tensor<B>), 
    Reshape(Variable<B>, Vec<usize>),
    Conv2d(Variable<B>, Variable<B>, Variable<B>, usize, usize),
    MaxPool2d(Variable<B>, usize),
    BatchNorm2d(Variable<B>, Variable<B>, Variable<B>, f64),
}

#[derive(Clone, Debug)]
pub struct Variable<B: Backend> {
    pub data: Rc<RefCell<Tensor<B>>>,
    pub grad: Rc<RefCell<Tensor<B>>>,
    pub creator: Rc<Op<B>>,
}

impl<B: Backend> Variable<B> {
    pub fn new(tensor: Tensor<B>) -> Self {
        let shape = tensor.shape.clone();
        let device = tensor.device.clone();
        Self {
            data: Rc::new(RefCell::new(tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::None),
        }
    }

    // 🎯 NEW: Skip full gradient allocation for constants (inputs/targets)
    pub fn new_constant(tensor: Tensor<B>) -> Self {
        let device = tensor.device.clone();
        Self {
            data: Rc::new(RefCell::new(tensor)),
            // Allocate a tiny 1-element dummy buffer to satisfy the type system without VRAM overhead
            grad: Rc::new(RefCell::new(Tensor::zeros(device, vec![1]))), 
            creator: Rc::new(Op::None),
        }
    }

    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.data) as usize
    }

    pub fn add(&self, other: &Variable<B>) -> Result<Self, String> {
        let result_tensor = self.data.borrow().add(&other.data.borrow())?;
        let shape = result_tensor.shape.clone();
        let device = result_tensor.device.clone();
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::Add(self.clone(), other.clone())),
        })
    }

    pub fn matmul(&self, other: &Variable<B>) -> Result<Self, String> {
        let result_tensor = self.data.borrow().matmul(&other.data.borrow())?;
        let shape = result_tensor.shape.clone();
        let device = result_tensor.device.clone();
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::Matmul(self.clone(), other.clone())),
        })
    }

    pub fn zero_grad(&self) {
        let device = self.grad.borrow().device.clone();
        let shape = self.grad.borrow().shape.clone();
        // Reset gradients by replacing the buffer with a fresh zeros buffer via the backend
        *self.grad.borrow_mut() = Tensor::zeros(device, shape);
    }

    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo<B: Backend>(v: &Variable<B>, topo: &mut Vec<Variable<B>>, visited: &mut HashSet<usize>) {
            let id = v.id();
            if !visited.contains(&id) {
                visited.insert(id);
                match &*v.creator {
                    Op::Add(a, b) | Op::Matmul(a, b) | Op::MSE(a, b) | Op::CrossEntropy(a, b) => {
                        build_topo(a, topo, visited);
                        build_topo(b, topo, visited);
                    }
                    Op::ReLU(a) | Op::Sigmoid(a) | Op::Tanh(a) | Op::Softmax(a) | 
                    Op::Dropout(a, _) | Op::Reshape(a, _) | Op::MaxPool2d(a, _) => {
                        build_topo(a, topo, visited);
                    }
                    Op::Conv2d(a, b, c, _, _) => {
                        build_topo(a, topo, visited);
                        build_topo(b, topo, visited);
                        build_topo(c, topo, visited);
                    }
                    Op::BatchNorm2d(a, b, c, _) => { // <-- Add the comma and underscore
                        build_topo(a, topo, visited);
                        build_topo(b, topo, visited);
                        build_topo(c, topo, visited);
                    }
                    Op::None => {}
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut topo, &mut visited);

        // Prime the final loss node with a gradient of 1.0. 
        // We do this by creating a CPU vector of [1.0] and passing it to the backend.
        let device = self.data.borrow().device.clone();
        let shape = self.data.borrow().shape.clone();
        *self.grad.borrow_mut() = Tensor::from_data(device, vec![1.0], shape).unwrap();

        for v in topo.into_iter().rev() {
            v._backward_step();
        }
    }

    fn _backward_step(&self) {
        let grad_value = self.grad.borrow().clone();

        match &*self.creator {
            Op::Add(parent_a, parent_b) => {
                let shape_a = &parent_a.data.borrow().shape;
                let shape_b = &parent_b.data.borrow().shape;

                let grad_a = Self::collapse_grad(&grad_value, shape_a);
                let grad_b = Self::collapse_grad(&grad_value, shape_b);

                Self::update_grad(parent_a, &grad_a);
                Self::update_grad(parent_b, &grad_b);
            }
            Op::Matmul(parent_a, parent_b) => {
                let b_transposed = parent_b.data.borrow().transpose().unwrap();
                let a_grad_update = grad_value.matmul(&b_transposed).unwrap();
                Self::update_grad(parent_a, &a_grad_update);

                let a_transposed = parent_a.data.borrow().transpose().unwrap();
                let b_grad_update = a_transposed.matmul(&grad_value).unwrap();
                Self::update_grad(parent_b, &b_grad_update);
            }
            Op::ReLU(parent) => {
                // 🎯 To keep math in the backend, we write a derivative helper or pull to CPU temporarily.
                // For this refactor step, we pull to CPU to calculate the mask, then send it back.
                let parent_data = parent.data.borrow();
                let device = parent_data.device.clone();
                let shape = parent_data.shape.clone();
                
                let cpu_data = B::to_cpu(&device, &parent_data.data);
                let cpu_grad = B::to_cpu(&device, &grad_value.data);
                
                let mut new_grad = vec![0.0; cpu_data.len()];
                for i in 0..cpu_data.len() {
                    if cpu_data[i] > 0.0 { new_grad[i] = cpu_grad[i]; }
                }
                
                let grad_update = Tensor::from_data(device, new_grad, shape).unwrap();
                Self::update_grad(parent, &grad_update);
            }
            Op::MSE(pred, target) => {
                let p_data = pred.data.borrow();
                let t_data = target.data.borrow();
                let device = p_data.device.clone();
                
                let cpu_p = B::to_cpu(&device, &p_data.data);
                let cpu_t = B::to_cpu(&device, &t_data.data);
                let cpu_g = B::to_cpu(&device, &grad_value.data)[0];
                let n = cpu_p.len() as f64;

                let mut new_grad = vec![0.0; cpu_p.len()];
                for i in 0..cpu_p.len() {
                    new_grad[i] = (2.0 / n) * (cpu_p[i] - cpu_t[i]) * cpu_g;
                }
                
                let grad_update = Tensor::from_data(device, new_grad, p_data.shape.clone()).unwrap();
                Self::update_grad(pred, &grad_update);
            }
            Op::CrossEntropy(pred, target) => {
                let p_data = pred.data.borrow();
                let t_data = target.data.borrow();
                let device = p_data.device.clone();
                
                let cpu_p = B::to_cpu(&device, &p_data.data);
                let cpu_t = B::to_cpu(&device, &t_data.data);
                let cpu_g = B::to_cpu(&device, &grad_value.data)[0];
                
                let batch_size = p_data.shape[0];
                let num_classes = p_data.shape[1];
                let mut new_grad = vec![0.0; cpu_p.len()];

                for i in 0..batch_size {
                    let mut max_val = f64::NEG_INFINITY;
                    for j in 0..num_classes {
                        let val = cpu_p[i * num_classes + j];
                        if val > max_val { max_val = val; }
                    }

                    let mut sum_exp = 0.0;
                    for j in 0..num_classes {
                        sum_exp += (cpu_p[i * num_classes + j] - max_val).exp();
                    }

                    for j in 0..num_classes {
                        let idx = i * num_classes + j;
                        let prob = (cpu_p[idx] - max_val).exp() / sum_exp;
                        new_grad[idx] = (prob - cpu_t[idx]) * (cpu_g / batch_size as f64);
                    }
                }
                
                let grad_update = Tensor::from_data(device, new_grad, p_data.shape.clone()).unwrap();
                Self::update_grad(pred, &grad_update);
            }
            Op::Sigmoid(parent) => {
                let out_data = self.data.borrow();
                let device = out_data.device.clone();
                let cpu_y = B::to_cpu(&device, &out_data.data);
                let cpu_g = B::to_cpu(&device, &grad_value.data);
                
                let mut new_grad = vec![0.0; cpu_y.len()];
                for i in 0..cpu_y.len() {
                    new_grad[i] = cpu_g[i] * cpu_y[i] * (1.0 - cpu_y[i]);
                }
                
                let grad_update = Tensor::from_data(device, new_grad, out_data.shape.clone()).unwrap();
                Self::update_grad(parent, &grad_update);
            }
            Op::Tanh(parent) => {
                let out_data = self.data.borrow();
                let device = out_data.device.clone();
                let cpu_y = B::to_cpu(&device, &out_data.data);
                let cpu_g = B::to_cpu(&device, &grad_value.data);
                
                let mut new_grad = vec![0.0; cpu_y.len()];
                for i in 0..cpu_y.len() {
                    new_grad[i] = cpu_g[i] * (1.0 - cpu_y[i] * cpu_y[i]);
                }
                
                let grad_update = Tensor::from_data(device, new_grad, out_data.shape.clone()).unwrap();
                Self::update_grad(parent, &grad_update);
            }
            Op::Softmax(parent) => {
                let out_data = self.data.borrow();
                let device = out_data.device.clone();
                let cpu_y = B::to_cpu(&device, &out_data.data);
                let cpu_g = B::to_cpu(&device, &grad_value.data);
                
                let batch_size = out_data.shape[0];
                let num_classes = out_data.shape[1];
                let mut new_grad = vec![0.0; cpu_y.len()];

                for i in 0..batch_size {
                    let mut dot_product = 0.0;
                    for k in 0..num_classes {
                        let idx = i * num_classes + k;
                        dot_product += cpu_g[idx] * cpu_y[idx];
                    }

                    for j in 0..num_classes {
                        let idx = i * num_classes + j;
                        new_grad[idx] = cpu_y[idx] * (cpu_g[idx] - dot_product);
                    }
                }
                
                let grad_update = Tensor::from_data(device, new_grad, out_data.shape.clone()).unwrap();
                Self::update_grad(parent, &grad_update);
            }
            Op::Dropout(parent, mask) => {
                let grad_update = grad_value.mul_elementwise(mask).unwrap();
                Self::update_grad(parent, &grad_update);
            }
            Op::Reshape(parent, original_shape) => {
                let reshaped_grad = grad_value.reshape(original_shape.clone()).unwrap();
                Self::update_grad(parent, &reshaped_grad);
            }
            Op::Conv2d(input, weight, bias, stride, padding) => {
                let (grad_in, grad_w, grad_b) = input.data.borrow().conv2d_backward(
                    &grad_value,
                    &weight.data.borrow(),
                    *stride,
                    *padding,
                );
                Self::update_grad(input, &grad_in);
                Self::update_grad(weight, &grad_w);
                Self::update_grad(bias, &grad_b);
            }
            Op::MaxPool2d(parent, kernel_size) => {
                let parent_grad = parent.data.borrow().maxpool2d_backward(&grad_value, *kernel_size);
                Self::update_grad(parent, &parent_grad);
            }
            Op::BatchNorm2d(input, weight, bias, eps) => {
                let (grad_in, grad_w, grad_b) = input.data.borrow().batch_norm2d_backward(
                    &grad_value,
                    &weight.data.borrow(),
                    *eps,
                );
                Self::update_grad(input, &grad_in);
                Self::update_grad(weight, &grad_w);
                Self::update_grad(bias, &grad_b);
            }
            Op::None => {}
        }
    }

    fn update_grad(var: &Variable<B>, grad_update: &Tensor<B>) {
        // We use the abstract Tensor::add to update gradients generically
        let current_grad = var.grad.borrow().clone();
        *var.grad.borrow_mut() = current_grad.add(grad_update).unwrap();
    }

    fn collapse_grad(grad: &Tensor<B>, target_shape: &Vec<usize>) -> Tensor<B> {
        if &grad.shape == target_shape {
            return grad.clone();
        }
        if grad.shape.len() == 2 && target_shape.len() == 2 {
            let (r, c) = (grad.shape[0], grad.shape[1]);
            let (tr, tc) = (target_shape[0], target_shape[1]);
            if tr == 1 && tc == c {
                let device = grad.device.clone();
                let cpu_grad = B::to_cpu(&device, &grad.data);
                
                let mut new_data = vec![0.0; c];
                for i in 0..r {
                    for j in 0..c {
                        new_data[j] += cpu_grad[i * c + j];
                    }
                }
                return Tensor::from_data(device, new_data, target_shape.clone()).unwrap();
            }
        }
        panic!("Cannot collapse gradient from shape {:?} to {:?}", grad.shape, target_shape);
    }

    pub fn relu(&self) -> Self {
        let result_tensor = self.data.borrow().relu();
        let shape = result_tensor.shape.clone();
        let device = result_tensor.device.clone();
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::ReLU(self.clone())),
        }
    }

    pub fn mse_loss(&self, target: &Variable<B>) -> Result<Self, String> {
        let result_tensor = self.data.borrow().mse_loss(&target.data.borrow())?;
        let device = result_tensor.device.clone();
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, vec![1]))),
            creator: Rc::new(Op::MSE(self.clone(), target.clone())),
        })
    }

    pub fn softmax(&self) -> Self {
        let result_tensor = self.data.borrow().softmax(); 
        let shape = result_tensor.shape.clone();
        let device = result_tensor.device.clone();
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::Softmax(self.clone())),
        }
    }

    pub fn cross_entropy_loss(&self, target: &Variable<B>) -> Result<Variable<B>, String> {
        let logits = self.data.borrow();
        let target_bound = target.data.borrow();
        let device = logits.device.clone();
        
        let cpu_logits = B::to_cpu(&device, &logits.data);
        let cpu_targets = B::to_cpu(&device, &target_bound.data);
        
        let batch_size = logits.shape[0];
        let num_classes = logits.shape[1];
        let epsilon = 1e-15;
        let mut loss_sum = 0.0;

        for i in 0..batch_size {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..num_classes {
                let val = cpu_logits[i * num_classes + j];
                if val > max_val { max_val = val; }
            }
            let mut sum_exp = 0.0;
            for j in 0..num_classes {
                sum_exp += (cpu_logits[i * num_classes + j] - max_val).exp();
            }
            for j in 0..num_classes {
                let idx = i * num_classes + j;
                let prob = (cpu_logits[idx] - max_val).exp() / sum_exp;
                let p_clamped = prob.max(epsilon).min(1.0 - epsilon);
                loss_sum += -cpu_targets[idx] * p_clamped.ln();
            }
        }

        let avg_loss = loss_sum / (batch_size as f64);
        let loss_tensor = Tensor::from_data(device.clone(), vec![avg_loss], vec![1])?;
        Ok(Self {
            data: Rc::new(RefCell::new(loss_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, vec![1]))),
            creator: Rc::new(Op::CrossEntropy(self.clone(), target.clone())),
        })
    }

    pub fn sigmoid(&self) -> Self {
        let result_tensor = self.data.borrow().sigmoid();
        let shape = result_tensor.shape.clone();
        let device = result_tensor.device.clone();
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::Sigmoid(self.clone())),
        }
    }

    pub fn tanh(&self) -> Self {
        let result_tensor = self.data.borrow().tanh();
        let shape = result_tensor.shape.clone();
        let device = result_tensor.device.clone();
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::Tanh(self.clone())),
        }
    }

    pub fn dropout(&self, p: f64) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (1.0 - p); 
        
        let in_tensor = self.data.borrow();
        let device = in_tensor.device.clone();
        let cpu_in = B::to_cpu(&device, &in_tensor.data);

        let mut out_data = Vec::with_capacity(cpu_in.len());
        let mut mask_data = Vec::with_capacity(cpu_in.len());

        for &val in &cpu_in {
            if rng.gen_range(0.0..1.0) >= p {
                out_data.push(val * scale);
                mask_data.push(scale);
            } else {
                out_data.push(0.0);
                mask_data.push(0.0);
            }
        }

        let out_tensor = Tensor::from_data(device.clone(), out_data, in_tensor.shape.clone()).unwrap();
        let mask_tensor = Tensor::from_data(device.clone(), mask_data, in_tensor.shape.clone()).unwrap();

        Self {
            data: Rc::new(RefCell::new(out_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, in_tensor.shape.clone()))),
            creator: Rc::new(Op::Dropout(self.clone(), mask_tensor)),
        }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let original_shape = self.data.borrow().shape.clone();
        let reshaped_tensor = self.data.borrow().reshape(new_shape).unwrap();
        let current_shape = reshaped_tensor.shape.clone();
        let device = reshaped_tensor.device.clone();

        Self {
            data: Rc::new(RefCell::new(reshaped_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, current_shape))),
            creator: Rc::new(Op::Reshape(self.clone(), original_shape)),
        }
    }

    pub fn conv2d(
        &self,
        weight: &Variable<B>,
        bias: &Variable<B>,
        stride: usize,
        padding: usize,
    ) -> Result<Self, String> {
        let result = self.data.borrow().conv2d(
            &weight.data.borrow(),
            &bias.data.borrow(),
            stride,
            padding,
        )?;
        let shape = result.shape.clone();
        let device = result.device.clone();
        Ok(Self {
            data: Rc::new(RefCell::new(result)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::Conv2d(
                self.clone(),
                weight.clone(),
                bias.clone(),
                stride,
                padding,
            )),
        })
    }

    pub fn maxpool2d(&self, kernel_size: usize) -> Result<Self, String> {
        let result = self.data.borrow().maxpool2d(kernel_size)?;
        let shape = result.shape.clone();
        let device = result.device.clone();
        Ok(Self {
            data: Rc::new(RefCell::new(result)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::MaxPool2d(self.clone(), kernel_size)),
        })
    }
    pub fn batch_norm2d(
        &self,
        weight: &Variable<B>,
        bias: &Variable<B>,
        running_mean: &Tensor<B>,
        running_var: &Tensor<B>,
        is_training: bool,
        momentum: f64,
        eps: f64,
    ) -> Result<Self, String> {
        let result_tensor = self.data.borrow().batch_norm2d(
            &weight.data.borrow(),
            &bias.data.borrow(),
            running_mean,
            running_var,
            is_training,
            momentum,
            eps,
        )?;
        let shape = result_tensor.shape.clone();
        let device = result_tensor.device.clone();
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(device, shape))),
            creator: Rc::new(Op::BatchNorm2d(self.clone(), weight.clone(), bias.clone(), eps)),
        })
    }
}