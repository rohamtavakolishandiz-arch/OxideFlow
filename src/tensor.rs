use std::fmt;
use crate::backend::Backend;

#[derive(Clone, Debug)]
pub struct Tensor<B: Backend> {
    pub data: B::Buffer,
    pub shape: Vec<usize>,
    pub device: B,
}

impl<B: Backend> Tensor<B> {
    pub fn from_data(device: B, data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(format!("Shape mismatch: expected {} elements, but got {}", expected_len, data.len()));
        }
        // 🎯 Route allocation to the hardware backend
        let buffer = B::from_data(&device, data, &shape);
        Ok(Self { data: buffer, shape, device })
    }

    pub fn zeros(device: B, shape: Vec<usize>) -> Self {
        let buffer = B::zeros(&device, &shape);
        Self { data: buffer, shape, device }
    }

    pub fn randn(device: B, shape: Vec<usize>) -> Self {
        let buffer = B::randn(&device, &shape);
        Self { data: buffer, shape, device }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let expected_len: usize = new_shape.iter().product();
        let current_len: usize = self.shape.iter().product();
        if expected_len != current_len {
            return Err(format!("Cannot reshape tensor of {} elements into shape {:?}", current_len, new_shape));
        }
        Ok(Self {
            data: self.data.clone(), 
            shape: new_shape,
            device: self.device.clone(),
        })
    }

    // ==========================================
    // 🎯 Delegated Math Operations 
    // All math is handled by the generic B hardware
    // ==========================================

    pub fn add(&self, other: &Tensor<B>) -> Result<Self, String> {
        let (out_data, out_shape) = B::add(&self.device, &self.data, &self.shape, &other.data, &other.shape)?;
        Ok(Self { data: out_data, shape: out_shape, device: self.device.clone() })
    }

    pub fn sub(&self, other: &Tensor<B>) -> Result<Self, String> {
        let out_data = B::sub(&self.device, &self.data, &other.data, &self.shape)?;
        Ok(Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() })
    }

    pub fn matmul(&self, other: &Tensor<B>) -> Result<Self, String> {
        let out_data = B::matmul(&self.device, &self.data, &self.shape, &other.data, &other.shape)?;
        Ok(Self { data: out_data, shape: vec![self.shape[0], other.shape[1]], device: self.device.clone() })
    }

    pub fn transpose(&self) -> Result<Self, String> {
        let out_data = B::transpose(&self.device, &self.data, &self.shape)?;
        Ok(Self { data: out_data, shape: vec![self.shape[1], self.shape[0]], device: self.device.clone() })
    }

    pub fn relu(&self) -> Self {
        let out_data = B::relu(&self.device, &self.data, &self.shape);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }

    pub fn mse_loss(&self, target: &Tensor<B>) -> Result<Self, String> {
        let out_data = B::mse_loss(&self.device, &self.data, &target.data, &self.shape)?;
        Ok(Self { data: out_data, shape: vec![1], device: self.device.clone() })
    }

    pub fn max(&self) -> f64 {
        B::max(&self.device, &self.data, &self.shape)
    }

    pub fn exp(&self) -> Self {
        let out_data = B::exp(&self.device, &self.data, &self.shape);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }

    pub fn sum(&self) -> f64 {
        B::sum(&self.device, &self.data, &self.shape)
    }

    pub fn sigmoid(&self) -> Self {
        let out_data = B::sigmoid(&self.device, &self.data, &self.shape);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }

    pub fn tanh(&self) -> Self {
        let out_data = B::tanh(&self.device, &self.data, &self.shape);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }

    pub fn mul_elementwise(&self, other: &Tensor<B>) -> Result<Self, String> {
        let out_data = B::mul_elementwise(&self.device, &self.data, &other.data, &self.shape, &other.shape)?;
        Ok(Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() })
    }

    pub fn softmax(&self) -> Self {
        let out_data = B::softmax(&self.device, &self.data, &self.shape);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }

    pub fn conv2d(&self, weight: &Tensor<B>, bias: &Tensor<B>, stride: usize, padding: usize) -> Result<Self, String> {
        let (out_data, out_shape) = B::conv2d(
            &self.device, &self.data, &self.shape,
            &weight.data, &weight.shape,
            &bias.data, &bias.shape,
            stride, padding
        )?;
        Ok(Self { data: out_data, shape: out_shape, device: self.device.clone() })
    }

    pub fn conv2d_backward(&self, grad_out: &Tensor<B>, weight: &Tensor<B>, stride: usize, padding: usize) -> (Tensor<B>, Tensor<B>, Tensor<B>) {
        let (grad_in_data, grad_w_data, grad_b_data) = B::conv2d_backward(
            &self.device, &self.data, &self.shape,
            &grad_out.data, &grad_out.shape,
            &weight.data, &weight.shape,
            stride, padding
        );
        (
            Tensor { data: grad_in_data, shape: self.shape.clone(), device: self.device.clone() },
            Tensor { data: grad_w_data, shape: weight.shape.clone(), device: self.device.clone() },
            Tensor { data: grad_b_data, shape: vec![weight.shape[0]], device: self.device.clone() }
        )
    }

    pub fn maxpool2d(&self, kernel_size: usize) -> Result<Self, String> {
        let (out_data, out_shape) = B::maxpool2d(&self.device, &self.data, &self.shape, kernel_size)?;
        Ok(Self { data: out_data, shape: out_shape, device: self.device.clone() })
    }

    pub fn maxpool2d_backward(&self, grad_out: &Tensor<B>, kernel_size: usize) -> Self {
        let out_data = B::maxpool2d_backward(&self.device, &self.data, &self.shape, &grad_out.data, &grad_out.shape, kernel_size);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }
}

impl<B: Backend> fmt::Display for Tensor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 🎯 We require backends to provide a way to sync data to CPU RAM for debugging
        let cpu_data = B::to_cpu(&self.device, &self.data);
        
        writeln!(f, "Tensor(shape: {:?}) [", self.shape)?;
        if self.shape.len() == 2 {
            let cols = self.shape[1];
            for (i, val) in cpu_data.iter().enumerate() {
                if i % cols == 0 {
                    write!(f, "  [")?;
                }
                write!(f, "{:.4}, ", val)?;
                if (i + 1) % cols == 0 {
                    writeln!(f, "]")?;
                }
            }
        } else {
            writeln!(f, "  {:?}", cpu_data)?;
        }
        write!(f, "]")
    }
}