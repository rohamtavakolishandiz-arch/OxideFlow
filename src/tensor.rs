use std::fmt;
use rand::Rng;
use serde::{Deserialize, Serialize};

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
            return Err("Matmul currently only supports 2D tensors.".to_string());
        }

        let (r1, c1) = (self.shape[0], self.shape[1]);
        let (r2, c2) = (other.shape[0], other.shape[1]);

        if c1 != r2 {
            return Err(format!(
                "Incompatible shapes for matmul: {:?} x {:?}",
                self.shape, other.shape
            ));
        }

        let mut new_data = vec![0.0; r1 * c2];
        
        // 🚀 UPGRADE: Swapped loop order to i, k, j for Cache-Friendly Memory Access.
        for i in 0..r1 {
            for k in 0..c1 {
                let a_val = self.data[i * c1 + k];
                for j in 0..c2 {
                    new_data[i * c2 + j] += a_val * other.data[k * c2 + j];
                }
            }
        }

        Ok(Self {
            data: new_data,
            shape: vec![r1, c2],
        })
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