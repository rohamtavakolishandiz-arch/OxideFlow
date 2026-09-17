use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashSet;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub enum Op {
    None,
    Add(Variable, Variable),
    Matmul(Variable, Variable),
    ReLU(Variable),
    Sigmoid(Variable),
    Tanh(Variable),
    MSE(Variable, Variable), 
    CrossEntropy(Variable, Variable),
    Softmax(Variable),
}

#[derive(Clone, Debug)]
pub struct Variable {
    pub data: Rc<RefCell<Tensor>>,
    pub grad: Rc<RefCell<Tensor>>,
    pub creator: Rc<Op>,
}

impl Variable {

    pub fn new(tensor: Tensor) -> Self {
        let shape = tensor.shape.clone();
        Self {
            data: Rc::new(RefCell::new(tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::None),
        }
    }

    /// Unique identifier for topological sorting based on memory allocation
    pub fn id(&self) -> usize {
        Rc::as_ptr(&self.data) as usize
    }

    pub fn add(&self, other: &Variable) -> Result<Self, String> {
        let result_tensor = self.data.borrow().add(&other.data.borrow())?;
        let shape = result_tensor.shape.clone();
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::Add(self.clone(), other.clone())),
        })
    }

    pub fn matmul(&self, other: &Variable) -> Result<Self, String> {
        let result_tensor = self.data.borrow().matmul(&other.data.borrow())?;
        let shape = result_tensor.shape.clone();
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::Matmul(self.clone(), other.clone())),
        })
    }

    pub fn zero_grad(&self) {
        self.grad.borrow_mut().data.iter_mut().for_each(|g| *g = 0.0);
    }

    /// 🎯 Topologically sorted backward pass (PyTorch standard)
    pub fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(v: &Variable, topo: &mut Vec<Variable>, visited: &mut HashSet<usize>) {
            let id = v.id();
            if !visited.contains(&id) {
                visited.insert(id);
                match &*v.creator {
                    Op::Add(a, b) | Op::Matmul(a, b) | Op::MSE(a, b) | Op::CrossEntropy(a, b) => {
                        build_topo(a, topo, visited);
                        build_topo(b, topo, visited);
                    },
                    // 🎯 FIX: Added Softmax to the topological sort
                    Op::ReLU(a) | Op::Sigmoid(a) | Op::Tanh(a) | Op::Softmax(a) => {
                        build_topo(a, topo, visited);
                    },
                    Op::None => {}
                }
                topo.push(v.clone());
            }
        }

        build_topo(self, &mut topo, &mut visited);

        self.grad.borrow_mut().data.iter_mut().for_each(|g| *g = 1.0);

        for v in topo.into_iter().rev() {
            v._backward_step();
        }
    }

    /// Processes a single node's gradients
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
                let parent_data = parent.data.borrow();
                let mut parent_grad_update = Tensor::zeros(parent_data.shape.clone());

                for i in 0..parent_data.data.len() {
                    if parent_data.data[i] > 0.0 {
                        parent_grad_update.data[i] = grad_value.data[i];
                    }
                }
                Self::update_grad(parent, &parent_grad_update);
            }
            Op::MSE(pred, target) => {
                let p_data = pred.data.borrow();
                let t_data = target.data.borrow();
                let n = p_data.data.len() as f64;
                let g = grad_value.data[0];

                let mut pred_grad_update = Tensor::zeros(p_data.shape.clone());
                for i in 0..p_data.data.len() {
                    pred_grad_update.data[i] = (2.0 / n) * (p_data.data[i] - t_data.data[i]) * g;
                }
                Self::update_grad(pred, &pred_grad_update);
            }
            Op::CrossEntropy(pred, target) => {
                let p_data = pred.data.borrow();
                let t_data = target.data.borrow();
                let batch_size = p_data.shape[0];
                let num_classes = p_data.shape[1];
                let g = grad_value.data[0];

                let mut pred_grad_update = Tensor::zeros(p_data.shape.clone());

                for i in 0..batch_size {
                    let mut max_val = f64::NEG_INFINITY;
                    for j in 0..num_classes {
                        let val = p_data.data[i * num_classes + j];
                        if val > max_val { max_val = val; }
                    }

                    let mut sum_exp = 0.0;
                    for j in 0..num_classes {
                        sum_exp += (p_data.data[i * num_classes + j] - max_val).exp();
                    }

                    for j in 0..num_classes {
                        let idx = i * num_classes + j;
                        let prob = (p_data.data[idx] - max_val).exp() / sum_exp;
                        pred_grad_update.data[idx] = (prob - t_data.data[idx]) * (g / batch_size as f64);
                    }
                }
                Self::update_grad(pred, &pred_grad_update);
            }
            Op::Sigmoid(parent) => {
                let out_data = self.data.borrow();
                let mut parent_grad_update = Tensor::zeros(out_data.shape.clone());
                for i in 0..out_data.data.len() {
                    let y = out_data.data[i];
                    parent_grad_update.data[i] = grad_value.data[i] * y * (1.0 - y);
                }
                Self::update_grad(parent, &parent_grad_update);
            }
            Op::Tanh(parent) => {
                let out_data = self.data.borrow();
                let mut parent_grad_update = Tensor::zeros(out_data.shape.clone());
                for i in 0..out_data.data.len() {
                    let y = out_data.data[i];
                    parent_grad_update.data[i] = grad_value.data[i] * (1.0 - y * y);
                }
                Self::update_grad(parent, &parent_grad_update);
            }

            Op::Softmax(parent) => {
                let out_data = self.data.borrow();
                let mut parent_grad = Tensor::zeros(out_data.shape.clone());
                let batch_size = out_data.shape[0];
                let num_classes = out_data.shape[1];

                for i in 0..batch_size {
                    let mut dot_product = 0.0;
                    for k in 0..num_classes {
                        let idx = i * num_classes + k;
                        dot_product += grad_value.data[idx] * out_data.data[idx];
                    }
                    
                    for j in 0..num_classes {
                        let idx = i * num_classes + j;
                        let y_j = out_data.data[idx];
                        let g_j = grad_value.data[idx];
                        parent_grad.data[idx] = y_j * (g_j - dot_product);
                    }
                }
                Self::update_grad(parent, &parent_grad);
            }
            Op::None => {}
        }
    }

    /// 🎯 In-place gradient update to prevent memory allocation bottlenecks
    fn update_grad(var: &Variable, grad_update: &Tensor) {
        let mut current_grad = var.grad.borrow_mut();
        for i in 0..current_grad.data.len() {
            current_grad.data[i] += grad_update.data[i];
        }
    }

    pub fn relu(&self) -> Self {
        let result_tensor = self.data.borrow().relu();
        let shape = result_tensor.shape.clone();
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::ReLU(self.clone())),
        }
    }

    pub fn mse_loss(&self, target: &Variable) -> Result<Self, String> {
        let result_tensor = self.data.borrow().mse_loss(&target.data.borrow())?;
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(vec![1]))),
            creator: Rc::new(Op::MSE(self.clone(), target.clone())),
        })
    }

    pub fn softmax(&self) -> Self {
        let result_tensor = self.data.borrow().softmax(); // Uses the correct per-row math from tensor.rs
        let shape = result_tensor.shape.clone();
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::Softmax(self.clone())),
        }
    }

    pub fn cross_entropy_loss(&self, target: &Variable) -> Result<Variable, String> {
        let logits = self.data.borrow();
        let target_bound = target.data.borrow();
        let batch_size = logits.shape[0];
        let num_classes = logits.shape[1];
        let epsilon = 1e-15;
        let mut loss_sum = 0.0;

        for i in 0..batch_size {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..num_classes {
                let val = logits.data[i * num_classes + j];
                if val > max_val { max_val = val; }
            }
            let mut sum_exp = 0.0;
            for j in 0..num_classes {
                sum_exp += (logits.data[i * num_classes + j] - max_val).exp();
            }
            for j in 0..num_classes {
                let idx = i * num_classes + j;
                let prob = (logits.data[idx] - max_val).exp() / sum_exp;
                let p_clamped = prob.max(epsilon).min(1.0 - epsilon);
                loss_sum += -target_bound.data[idx] * p_clamped.ln();
            }
        }

        let avg_loss = loss_sum / (batch_size as f64);
        let loss_tensor = Tensor::from_data(vec![avg_loss], vec![1])?;
        Ok(Self {
            data: Rc::new(RefCell::new(loss_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(vec![1]))),
            creator: Rc::new(Op::CrossEntropy(self.clone(), target.clone())),
        })
    }

    pub fn sigmoid(&self) -> Self {
        let result_tensor = self.data.borrow().sigmoid();
        let shape = result_tensor.shape.clone();
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::Sigmoid(self.clone())),
        }
    }

    pub fn tanh(&self) -> Self {
        let result_tensor = self.data.borrow().tanh();
        let shape = result_tensor.shape.clone();
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::Tanh(self.clone())),
        }
    }

    fn collapse_grad(grad: &Tensor, target_shape: &Vec<usize>) -> Tensor {
        if &grad.shape == target_shape {
            return grad.clone();
        }
        if grad.shape.len() == 2 && target_shape.len() == 2 {
            let (r, c) = (grad.shape[0], grad.shape[1]);
            let (tr, tc) = (target_shape[0], target_shape[1]);
            if tr == 1 && tc == c {
                let mut new_data = vec![0.0; c];
                for i in 0..r {
                    for j in 0..c {
                        new_data[j] += grad.data[i * c + j];
                    }
                }
                return Tensor {
                    data: new_data,
                    shape: target_shape.clone(),
                };
            }
        }
        panic!("Cannot collapse gradient from shape {:?} to {:?}", grad.shape, target_shape);
    }

    
}