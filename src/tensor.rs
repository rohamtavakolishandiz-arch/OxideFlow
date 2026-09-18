use std::fmt;
use rand::Rng;
use serde::{Deserialize, Serialize};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn from_data(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();
        if data.len() != expected_len {
            return Err(format!(
                "Shape mismatch: expected {} elements, but got {}",
                expected_len,
                data.len()
            ));
        }
        Ok(Self { data, shape })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    pub fn add(&self, other: &Tensor) -> Result<Self, String> {
        if self.shape == other.shape {
            let new_data = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect();

            return Ok(Self {
                data: new_data,
                shape: self.shape.clone(),
            });
        }

        if self.shape.len() == 2 && other.shape.len() == 2 {
            let (r1, c1) = (self.shape[0], self.shape[1]);
            let (r2, c2) = (other.shape[0], other.shape[1]);

            if c1 == c2 && r2 == 1 {
                let mut new_data = vec![0.0; r1 * c1];
                for i in 0..r1 {
                    for j in 0..c1 {
                        new_data[i * c1 + j] = self.data[i * c1 + j] + other.data[j];
                    }
                }
                return Ok(Self {
                    data: new_data,
                    shape: self.shape.clone(),
                });
            }
            
            if c1 == c2 && r1 == 1 {
                let mut new_data = vec![0.0; r2 * c2];
                for i in 0..r2 {
                    for j in 0..c2 {
                        new_data[i * c2 + j] = self.data[j] + other.data[i * c2 + j];
                    }
                }
                return Ok(Self {
                    data: new_data,
                    shape: other.shape.clone(),
                });
            }
        }

        Err(format!(
            "Cannot add or broadcast tensors of shapes {:?} and {:?}",
            self.shape, other.shape
        ))
    }

    pub fn sub(&self, other: &Tensor) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Cannot subtract tensors of different shapes: {:?} and {:?}",
                self.shape, other.shape
            ));
        }

        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Self, String> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Matmul requires 2D tensors".to_string());
        }
        if self.shape[1] != other.shape[0] {
            return Err(format!("Matmul shape mismatch: {:?} x {:?}", self.shape, other.shape));
        }

        let r = self.shape[0];
        let shared_dim = self.shape[1];
        let c = other.shape[1];

        let mut out_data = vec![0.0; r * c];

        // 🚀 Rayon Magic: Process every row of the output matrix on a different CPU core!
        out_data.par_chunks_mut(c).enumerate().for_each(|(i, row_slice)| {
            for k in 0..shared_dim {
                let a_val = self.data[i * shared_dim + k];
                for j in 0..c {
                    row_slice[j] += a_val * other.data[k * c + j];
                }
            }
        });

        Ok(Self { data: out_data, shape: vec![r, c] })
    }

    pub fn transpose(&self) -> Result<Self, String> {
        if self.shape.len() != 2 {
            return Err("Transpose currently only supports 2D tensors.".to_string());
        }

        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut new_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                new_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Ok(Self {
            data: new_data,
            shape: vec![cols, rows],
        })
    }

    pub fn relu(&self) -> Self {
        let new_data = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn mse_loss(&self, target: &Tensor) -> Result<Self, String> {
        if self.shape != target.shape {
            return Err("Shapes must match for MSE loss.".to_string());
        }

        let n = self.data.len() as f64;
        let sum_sq: f64 = self
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        Ok(Self {
            data: vec![sum_sq / n],
            shape: vec![1],
        })
    }

    /// 🎯 FIX: Implemented Box-Muller transform for true Gaussian distribution
    pub fn randn(shape: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let len: usize = shape.iter().product();
        
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            // Generate two uniform variables, avoiding ln(0) by starting at 1e-10
            let u1: f64 = rng.gen_range(1e-10..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            
            // Box-Muller transform creates standard normal distribution N(0, 1)
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data.push(z0);
        }
            
        Self { data, shape }
    }

    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn exp(&self) -> Tensor {
        let new_data = self.data.iter().map(|v| v.exp()).collect();
        Tensor {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn sigmoid(&self) -> Self {
        let new_data = self
            .data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn tanh(&self) -> Self {
        let new_data = self
            .data
            .iter()
            .map(|&x| x.tanh())
            .collect();

        Self {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    pub fn mul_elementwise(&self, other: &Tensor) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err("Cannot multiply tensors of different shapes element-wise.".to_string());
        }

        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }

    pub fn softmax(&self) -> Self {
        let mut new_data = Vec::with_capacity(self.data.len());
        let batch_size = self.shape[0];
        let num_classes = self.shape[1];
    
        for i in 0..batch_size {
            let start = i * num_classes;
            let end = start + num_classes;
            let row = &self.data[start..end];
            
            // Find max for numerical stability (prevents NaN overflow)
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sums: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
            
            for &val in row {
                new_data.push((val - max_val).exp() / exp_sums);
            }
        }
        
        Self { 
            data: new_data, 
            shape: self.shape.clone() 
        }
    }

    /// 🎯 Reshapes the tensor dynamically. 
    /// Fails if the new shape doesn't perfectly fit the amount of data.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, String> {
        let expected_len: usize = new_shape.iter().product();
        if expected_len != self.data.len() {
            return Err(format!(
                "Cannot reshape tensor of {} elements into shape {:?}", 
                self.data.len(), new_shape
            ));
        }
        
        Ok(Self {
            data: self.data.clone(), // Fast clone, memory size stays identical
            shape: new_shape,
        })
    }

    /// 🎯 Forward Convolution: Slides filters across the 4D image tensor
    pub fn conv2d(&self, weight: &Tensor, bias: &Tensor, stride: usize, padding: usize) -> Result<Self, String> {
        let (b, in_c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (out_c, _, kh, kw) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
        
        let out_h = (h + 2 * padding - kh) / stride + 1;
        let out_w = (w + 2 * padding - kw) / stride + 1;
        
        let mut out_data = vec![0.0; b * out_c * out_h * out_w];
        let batch_size_out = out_c * out_h * out_w;
        
        // 🚀 Process every image in the batch simultaneously across all CPU cores
        out_data.par_chunks_mut(batch_size_out).enumerate().for_each(|(batch, batch_slice)| {
            for oc in 0..out_c {
                let b_val = bias.data[oc];
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = b_val;
                        for ic in 0..in_c {
                            for kh_idx in 0..kh {
                                for kw_idx in 0..kw {
                                    let ih = (oh * stride + kh_idx) as isize - padding as isize;
                                    let iw = (ow * stride + kw_idx) as isize - padding as isize;
                                    
                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                        let in_val = self.data[batch * in_c * h * w + ic * h * w + (ih as usize) * w + (iw as usize)];
                                        let w_val = weight.data[oc * in_c * kh * kw + ic * kh * kw + kh_idx * kw + kw_idx];
                                        sum += in_val * w_val;
                                    }
                                }
                            }
                        }
                        batch_slice[oc * out_h * out_w + oh * out_w + ow] = sum;
                    }
                }
            }
        });
        
        Ok(Self { data: out_data, shape: vec![b, out_c, out_h, out_w] })
    }

    /// 🎯 Backward Convolution: Routes gradients back to the image, the weights, and the biases
    /// 🎯 Multi-Threaded Backward Convolution (Map-Reduce)
    pub fn conv2d_backward(&self, grad_out: &Tensor, weight: &Tensor, stride: usize, padding: usize) -> (Tensor, Tensor, Tensor) {
        let (b, in_c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (out_c, _, kh, kw) = (weight.shape[0], weight.shape[1], weight.shape[2], weight.shape[3]);
        let (out_h, out_w) = (grad_out.shape[2], grad_out.shape[3]);
        
        // 🚀 MAP: Each thread computes the gradients for one specific image independently
        let batch_gradients: Vec<_> = (0..b).into_par_iter().map(|batch| {
            let mut local_g_in = vec![0.0; in_c * h * w];
            let mut local_g_w = vec![0.0; weight.data.len()];
            let mut local_g_b = vec![0.0; out_c];
            
            for oc in 0..out_c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let g = grad_out.data[batch * out_c * out_h * out_w + oc * out_h * out_w + oh * out_w + ow];
                        local_g_b[oc] += g;
                        
                        for ic in 0..in_c {
                            for kh_idx in 0..kh {
                                for kw_idx in 0..kw {
                                    let ih = (oh * stride + kh_idx) as isize - padding as isize;
                                    let iw = (ow * stride + kw_idx) as isize - padding as isize;
                                    
                                    if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                        let in_idx = ic * h * w + (ih as usize) * w + (iw as usize);
                                        let w_idx = oc * in_c * kh * kw + ic * kh * kw + kh_idx * kw + kw_idx;
                                        
                                        local_g_in[in_idx] += weight.data[w_idx] * g;
                                        local_g_w[w_idx] += self.data[batch * in_c * h * w + in_idx] * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            (local_g_in, local_g_w, local_g_b)
        }).collect();

        // 🚀 REDUCE: Safely stitch the multi-threaded results back together
        let mut final_grad_input = Vec::with_capacity(b * in_c * h * w);
        let mut final_grad_weight = vec![0.0; weight.data.len()];
        let mut final_grad_bias = vec![0.0; out_c];

        for (g_in, g_w, g_b) in batch_gradients {
            final_grad_input.extend(g_in);
            for i in 0..final_grad_weight.len() { final_grad_weight[i] += g_w[i]; }
            for i in 0..final_grad_bias.len() { final_grad_bias[i] += g_b[i]; }
        }

        (
            Tensor { data: final_grad_input, shape: self.shape.clone() },
            Tensor { data: final_grad_weight, shape: weight.shape.clone() },
            Tensor { data: final_grad_bias, shape: vec![out_c] }
        )
    }

    /// 🎯 Forward MaxPool2d (Shrinks image by taking max of blocks)
    pub fn maxpool2d(&self, kernel_size: usize) -> Result<Self, String> {
        let (b, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let out_h = h / kernel_size;
        let out_w = w / kernel_size;
        let mut out_data = vec![0.0; b * c * out_h * out_w];

        for batch in 0..b {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f64::NEG_INFINITY;
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let val = self.data[batch * c * h * w + ch * h * w + (oh * kernel_size + kh) * w + (ow * kernel_size + kw)];
                                if val > max_val { max_val = val; }
                            }
                        }
                        out_data[batch * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow] = max_val;
                    }
                }
            }
        }
        Ok(Self { data: out_data, shape: vec![b, c, out_h, out_w] })
    }

    /// 🎯 Backward MaxPool2d (Routes gradient only to the maximum pixel)
    pub fn maxpool2d_backward(&self, grad_out: &Tensor, kernel_size: usize) -> Tensor {
        let (b, c, h, w) = (self.shape[0], self.shape[1], self.shape[2], self.shape[3]);
        let (out_h, out_w) = (grad_out.shape[2], grad_out.shape[3]);
        let mut grad_in = vec![0.0; self.data.len()];

        for batch in 0..b {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_idx = 0;
                        
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let idx = batch * c * h * w + ch * h * w + (oh * kernel_size + kh) * w + (ow * kernel_size + kw);
                                if self.data[idx] > max_val {
                                    max_val = self.data[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                        // Add gradient ONLY to the pixel that won the max pool
                        grad_in[max_idx] += grad_out.data[batch * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow];
                    }
                }
            }
        }
        Tensor { data: grad_in, shape: self.shape.clone() }
    }
    
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tensor(shape: {:?}) [", self.shape)?;
        if self.shape.len() == 2 {
            let cols = self.shape[1];
            for (i, val) in self.data.iter().enumerate() {
                if i % cols == 0 {
                    write!(f, "  [")?;
                }
                write!(f, "{:.4}, ", val)?;
                if (i + 1) % cols == 0 {
                    writeln!(f, "]")?;
                }
            }
        } else {
            writeln!(f, "  {:?}", self.data)?;
        }
        write!(f, "]")
    }
}