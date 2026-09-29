use std::fmt;
use crate::backend::Backend;
use crate::error::{OxideError, Result}; // Import the new Error Architecture

#[derive(Clone, Debug)]
pub struct Tensor<B: Backend> {
    pub data: B::Buffer,
    pub shape: Vec<usize>,
    pub device: B,
}

impl<B: Backend> Tensor<B> {
    pub fn from_data(device: B, data: Vec<f64>, shape: Vec<usize>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(OxideError::MathError(format!(
                "Buffer length mismatch: expected {} elements, but got {}", 
                expected_len, data.len()
            )));
        }
        // Route allocation to the hardware backend
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

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let expected_len: usize = new_shape.iter().product();
        let current_len: usize = self.shape.iter().product();
        if expected_len != current_len {
            return Err(OxideError::MathError(format!(
                "Cannot reshape tensor of {} elements into shape {:?}", 
                current_len, new_shape
            )));
        }
        Ok(Self {
            data: self.data.clone(), 
            shape: new_shape,
            device: self.device.clone(),
        })
    }

    // ==========================================
    // Delegated Math Operations 
    // ==========================================

    pub fn add(&self, other: &Tensor<B>) -> Result<Self> {
        // Let the hardware backend handle both identical shapes and 2D broadcasting
        let (out_data, out_shape) = B::add(&self.device, &self.data, &self.shape, &other.data, &other.shape)
            .map_err(OxideError::GpuError)?;
        
        Ok(Self { data: out_data, shape: out_shape, device: self.device.clone() })
    }

    pub fn sub(&self, other: &Tensor<B>) -> Result<Self> {
        if self.shape != other.shape {
            return Err(OxideError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        let out_data = B::sub(&self.device, &self.data, &other.data, &self.shape)
            .map_err(OxideError::GpuError)?;
        Ok(Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() })
    }

    pub fn matmul(&self, other: &Tensor<B>) -> Result<Self> {
        // 1. Verify Rank 2 (Matrices)
        if self.shape.len() != 2 {
            return Err(OxideError::InvalidDimension { axis: 2, shape: self.shape.clone() });
        }
        if other.shape.len() != 2 {
            return Err(OxideError::InvalidDimension { axis: 2, shape: other.shape.clone() });
        }
        
        // 2. Verify Inner Dimension Compatibility (K == K)
        if self.shape[1] != other.shape[0] {
            return Err(OxideError::MathError(format!(
                "Matmul inner dimensions mismatch: {} (cols of A) != {} (rows of B)", 
                self.shape[1], other.shape[0]
            )));
        }
    
        // 3. Dispatch to Hardware safely
        let out_data = B::matmul(&self.device, &self.data, &self.shape, &other.data, &other.shape)
            .map_err(OxideError::GpuError)?;
        
        Ok(Self { 
            data: out_data, 
            shape: vec![self.shape[0], other.shape[1]], 
            device: self.device.clone() 
        })
    }

    pub fn transpose(&self) -> Result<Self> {
        let out_data = B::transpose(&self.device, &self.data, &self.shape)
            .map_err(OxideError::GpuError)?;
        Ok(Self { data: out_data, shape: vec![self.shape[1], self.shape[0]], device: self.device.clone() })
    }

    pub fn relu(&self) -> Self {
        let out_data = B::relu(&self.device, &self.data, &self.shape);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }

    pub fn mse_loss(&self, target: &Tensor<B>) -> Result<Self> {
        if self.shape != target.shape {
            return Err(OxideError::ShapeMismatch {
                expected: target.shape.clone(),
                actual: self.shape.clone(),
            });
        }
        let out_data = B::mse_loss(&self.device, &self.data, &target.data, &self.shape)
            .map_err(OxideError::GpuError)?;
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

    pub fn mul_elementwise(&self, other: &Tensor<B>) -> Result<Self> {
        if self.shape != other.shape {
            return Err(OxideError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        let out_data = B::mul_elementwise(&self.device, &self.data, &other.data, &self.shape, &other.shape)
            .map_err(OxideError::GpuError)?;
        Ok(Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() })
    }

    pub fn softmax(&self) -> Self {
        let out_data = B::softmax(&self.device, &self.data, &self.shape);
        Self { data: out_data, shape: self.shape.clone(), device: self.device.clone() }
    }

    pub fn conv2d(&self, weight: &Tensor<B>, bias: &Tensor<B>, stride: usize, padding: usize) -> Result<Self> {
        if self.shape.len() != 4 {
            return Err(OxideError::InvalidDimension { axis: 4, shape: self.shape.clone() });
        }
        if weight.shape.len() != 4 {
            return Err(OxideError::InvalidDimension { axis: 4, shape: weight.shape.clone() });
        }
    
        let (in_c, h, w_dim) = (self.shape[1], self.shape[2], self.shape[3]);
        let (out_c, w_in_c, kh, kw) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
    
        if in_c != w_in_c {
            return Err(OxideError::MathError(format!(
                "Conv2d channel mismatch: input has {} channels, but weight expects {}", 
                in_c, w_in_c
            )));
        }
    
        if bias.shape.len() != 1 || bias.shape[0] != out_c {
            return Err(OxideError::MathError(format!(
                "Conv2d bias mismatch: expected shape [{}], but got {:?}", 
                out_c, bias.shape
            )));
        }
    
        if stride == 0 {
            return Err(OxideError::MathError("Conv2d stride cannot be zero".to_string()));
        }
    
        let padded_h = h + 2 * padding;
        let padded_w = w_dim + 2 * padding;
        if kh > padded_h || kw > padded_w {
            return Err(OxideError::MathError(format!(
                "Conv2d kernel size ({}x{}) is larger than padded input ({}x{})", 
                kh, kw, padded_h, padded_w
            )));
        }
    
        let (out_data, out_shape) = B::conv2d(
            &self.device, &self.data, &self.shape,
            &weight.data, &weight.shape,
            &bias.data, &bias.shape,
            stride, padding
        ).map_err(OxideError::GpuError)?;
        
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

    pub fn maxpool2d(&self, kernel_size: usize) -> Result<Self> {
        let (out_data, out_shape) = B::maxpool2d(&self.device, &self.data, &self.shape, kernel_size)
            .map_err(OxideError::GpuError)?;
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
    ) -> Result<Self> {
        if self.shape.len() != 4 {
            return Err(OxideError::InvalidDimension { axis: 4, shape: self.shape.clone() });
        }
        if weight.shape.len() != 1 || weight.shape[0] != self.shape[1] {
            return Err(OxideError::MathError(format!("BatchNorm2d weight shape mismatch: expected [{}], got {:?}", self.shape[1], weight.shape)));
        }
        if bias.shape.len() != 1 || bias.shape[0] != self.shape[1] {
            return Err(OxideError::MathError(format!("BatchNorm2d bias shape mismatch: expected [{}], got {:?}", self.shape[1], bias.shape)));
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
        ).map_err(OxideError::GpuError)?;

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

    pub fn apply_alive_mask_clamp(&self, alpha_idx: usize, loss: &Tensor<B>, seed: &Tensor<B>, do_clamp: bool) -> Result<Self> {
        // Pass the boolean down to the hardware backend
        let out_data = B::apply_alive_mask_clamp(&self.device, &self.data, &self.shape, alpha_idx, &loss.data, &seed.data, do_clamp);
        Ok(Self { 
            data: out_data, 
            shape: self.shape.clone(), 
            device: self.device.clone() 
        })
    }

    pub fn cellular_mask(device: B, shape: &[usize], drop_prob: f64, seed: u32) -> Result<Self> {
        let data = B::generate_cellular_mask(&device, shape, drop_prob as f32, seed);
        Ok(Self { 
            data, 
            shape: shape.to_vec(), 
            device 
        })
    }
    
}

impl<B: Backend> fmt::Display for Tensor<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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