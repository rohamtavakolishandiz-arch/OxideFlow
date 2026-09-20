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
        if self.shape != other.shape {
            return Err(format!("Shape mismatch in sub: {:?} != {:?}", self.shape, other.shape));
        }
        let out_data = B::sub(&self.device, &self.data, &other.data, &self.shape)?;
        Ok(Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() })
    }

    pub fn matmul(&self, other: &Tensor<B>) -> Result<Self, String> {
        // 1. Verify Rank 2 (Matrices)
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(format!(
                "Matmul requires exactly 2D tensors, but got shapes {:?} and {:?}", 
                self.shape, other.shape
            ));
        }
        
        // 2. Verify Inner Dimension Compatibility (K == K)
        if self.shape[1] != other.shape[0] {
            return Err(format!(
                "Matmul inner dimensions mismatch: {} (cols of A) != {} (rows of B)", 
                self.shape[1], other.shape[0]
            ));
        }
    
        // 3. Dispatch to Hardware safely
        let out_data = B::matmul(&self.device, &self.data, &self.shape, &other.data, &other.shape)?;
        
        Ok(Self { 
            data: out_data, 
            shape: vec![self.shape[0], other.shape[1]], 
            device: self.device.clone() 
        })
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
        if self.shape != target.shape {
            return Err(format!("Shape mismatch in mse_loss: target {:?} != pred {:?}", target.shape, self.shape));
        }
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
        if self.shape != other.shape {
            return Err(format!("Shape mismatch in mul_elementwise: {:?} != {:?}", self.shape, other.shape));
        }
        let out_data = B::mul_elementwise(&self.device, &self.data, &other.data, &self.shape, &other.shape)?;
        Ok(Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() })
    }

    pub fn softmax(&self) -> Self {
        let out_data = B::softmax(&self.device, &self.data, &self.shape);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }

    pub fn conv2d(&self, weight: &Tensor<B>, bias: &Tensor<B>, stride: usize, padding: usize) -> Result<Self, String> {
        // 1. Verify 4D Ranks
        if self.shape.len() != 4 || weight.shape.len() != 4 {
            return Err(format!(
                "Conv2d requires 4D tensors, but got input {:?} and weight {:?}", 
                self.shape, weight.shape
            ));
        }
    
        let (in_c, h, w_dim) = (self.shape[1], self.shape[2], self.shape[3]);
        let (out_c, w_in_c, kh, kw) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
    
        // 2. Verify Channel Consistency
        if in_c != w_in_c {
            return Err(format!(
                "Conv2d channel mismatch: input has {} channels, but weight expects {}", 
                in_c, w_in_c
            ));
        }
    
        // 3. Verify Bias Shape
        if bias.shape.len() != 1 || bias.shape[0] != out_c {
            return Err(format!(
                "Conv2d bias mismatch: expected shape [{}], but got {:?}", 
                out_c, bias.shape
            ));
        }
    
        // 4. Prevent Divide-by-Zero
        if stride == 0 {
            return Err("Conv2d stride cannot be zero".to_string());
        }
    
        // 5. Prevent usize Underflow (Kernel must fit inside padded input)
        let padded_h = h + 2 * padding;
        let padded_w = w_dim + 2 * padding;
        if kh > padded_h || kw > padded_w {
            return Err(format!(
                "Conv2d kernel size ({}x{}) is larger than padded input ({}x{})", 
                kh, kw, padded_h, padded_w
            ));
        }
    
        // 6. Safe Dispatch
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

    pub fn batch_norm2d(
        &self,
        weight: &Tensor<B>,
        bias: &Tensor<B>,
        running_mean: &Tensor<B>,
        running_var: &Tensor<B>,
        is_training: bool,
        momentum: f64,
        eps: f64,
    ) -> Result<Self, String> {
        if self.shape.len() != 4 {
            return Err(format!("BatchNorm2d requires 4D tensor [N, C, H, W], got {:?}", self.shape));
        }
        if weight.shape.len() != 1 || weight.shape[0] != self.shape[1] {
            return Err(format!("BatchNorm2d weight shape mismatch: expected [{}], got {:?}", self.shape[1], weight.shape));
        }
        if bias.shape.len() != 1 || bias.shape[0] != self.shape[1] {
            return Err(format!("BatchNorm2d bias shape mismatch: expected [{}], got {:?}", self.shape[1], bias.shape));
        }

        let out_data = B::batch_norm2d(
            &self.device,
            &self.data,
            &self.shape,
            &weight.data,
            &bias.data,
            &running_mean.data,
            &running_var.data,
            is_training,
            momentum,
            eps,
        )?;

        Ok(Self {
            data: out_data,
            shape: self.shape.clone(),
            device: self.device.clone(),
        })
    }

    pub fn batch_norm2d_backward(&self, grad_out: &Tensor<B>, weight: &Tensor<B>, eps: f64) -> (Tensor<B>, Tensor<B>, Tensor<B>) {
        let (grad_in, grad_w, grad_b) = B::batch_norm2d_backward(
            &self.device, 
            &self.data, 
            &self.shape,
            &grad_out.data, 
            &weight.data, 
            eps
        );
        (
            Tensor { data: grad_in, shape: self.shape.clone(), device: self.device.clone() },
            Tensor { data: grad_w, shape: weight.shape.clone(), device: self.device.clone() },
            Tensor { data: grad_b, shape: weight.shape.clone(), device: self.device.clone() }
        )
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