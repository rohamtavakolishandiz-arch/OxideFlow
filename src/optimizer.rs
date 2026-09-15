use crate::autograd::Variable;
use crate::tensor::Tensor;

/// Trait مشترک برای تمام الگوریتم‌های بهینه‌سازی
pub trait Optimizer {
    fn zero_grad(&self);
    fn step(&mut self);
}

// ==========================================
// SGD Implementation
// ==========================================

pub struct SGD {
    pub params: Vec<Variable>,
    pub lr: f64, // Make sure this is f64 to match your Tensors!
}

impl SGD {
    pub fn new(params: Vec<Variable>, lr: f64) -> Self {
        Self { params, lr }
    }
}

impl Optimizer for SGD {
    fn zero_grad(&self) {
        for p in &self.params {
            let mut grad = p.grad.borrow_mut();
            for val in grad.data.iter_mut() {
                *val = 0.0;
            }
        }
    }

    fn step(&mut self) {
        for p in &self.params {
            let mut weight = p.data.borrow_mut();
            let grad = p.grad.borrow();

            for i in 0..weight.data.len() {
                weight.data[i] -= self.lr * grad.data[i];
            }
        }
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
    pub fn new(params: Vec<Variable>, lr: f64) -> Self {
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
            weight_decay: 0.01,
            t: 0,
            m,
            v,
        }
    }
}

impl Optimizer for AdamW {
    fn zero_grad(&self) {
        for p in &self.params {
            let mut grad = p.grad.borrow_mut();
            for val in grad.data.iter_mut() {
                *val = 0.0;
            }
        }
    }

    fn step(&mut self) {
        self.t += 1;
        
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        for i in 0..self.params.len() {
            let mut weight = self.params[i].data.borrow_mut();
            let grad = self.params[i].grad.borrow();
            
            let m_tensor = &mut self.m[i];
            let v_tensor = &mut self.v[i];

            for j in 0..weight.data.len() {
                let g = grad.data[j];
                let w = weight.data[j];

                m_tensor.data[j] = beta1 * m_tensor.data[j] + (1.0 - beta1) * g;
                v_tensor.data[j] = beta2 * v_tensor.data[j] + (1.0 - beta2) * g * g;

                let m_hat = m_tensor.data[j] / (1.0 - beta1.powi(self.t as i32));
                let v_hat = v_tensor.data[j] / (1.0 - beta2.powi(self.t as i32));

                weight.data[j] = w - lr * (m_hat / (v_hat.sqrt() + eps) + wd * w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_sgd_optimizer() {
        // ۱. ساخت یک وزن (مثلاً 2.0)
        let weight = Variable::new(Tensor::from_data(vec![2.0], vec![1]).unwrap());
        
        // ۲. ساخت یک تارگت یا جواب واقعی (مثلاً 5.0)
        let target = Variable::new(Tensor::from_data(vec![5.0], vec![1]).unwrap());

        // ۳. ساخت بهینه‌ساز و معرفی وزن‌ها به آن (با نرخ یادگیری 0.1)
        let mut  optimizer = SGD::new(vec![weight.clone()], 0.1);

        // ۴. حلقه آموزش (۵ دور)
        for _ in 0..5 {
            // صفر کردن گرادیان‌های قبلی
            optimizer.zero_grad();

            // حرکت به جلو و محاسبه خطا
            let loss = weight.mse_loss(&target).unwrap();

            // انتشار به عقب (محاسبه گرادیان)
            loss.backward();

            // آپدیت وزن‌ها توسط بهینه‌ساز
            optimizer.step();
        }

        // بعد از ۵ دور آموزش، وزن ما که 2.0 بود باید به جواب واقعی (5.0) نزدیک شده باشد.
        // با این تنظیمات به حدود 3.7 تا 4.0 می‌رسد.
        let updated_weight = weight.data.borrow().data[0];
        assert!(updated_weight > 2.0 && updated_weight < 5.0);
    }
}