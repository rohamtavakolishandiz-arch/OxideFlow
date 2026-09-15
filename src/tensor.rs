use std::fmt;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// ساخت یک تانسور جدید از روی داده‌های خام با بررسی صحت ابعاد
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

    /// ساخت یک تانسور پر از صفر (مناسب برای مقداردهی اولیه گرادیان‌ها)
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        Self {
            data: vec![0.0; len],
            shape,
        }
    }

    /// جمع عنصر به عنصر دو تانسور
    /// جمع دو تانسور با پشتیبانی از Broadcasting برای بایاس‌ها
    pub fn add(&self, other: &Tensor) -> Result<Self, String> {
        // 1. Strict Match: If shapes are identical, do standard element-wise addition
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

        // 2. Broadcasting: [B, N] + [1, N] (Self is batch, Other is bias)
        if self.shape.len() == 2 && other.shape.len() == 2 {
            let (r1, c1) = (self.shape[0], self.shape[1]);
            let (r2, c2) = (other.shape[0], other.shape[1]);

            // If columns match and the second tensor is just 1 row
            if c1 == c2 && r2 == 1 {
                let mut new_data = vec![0.0; r1 * c1];
                for i in 0..r1 {
                    for j in 0..c1 {
                        // Add the bias (other.data[j]) to every row of the batch
                        new_data[i * c1 + j] = self.data[i * c1 + j] + other.data[j];
                    }
                }
                return Ok(Self {
                    data: new_data,
                    shape: self.shape.clone(),
                });
            }
            
            // Symmetry: [1, N] + [B, N] (Self is bias, Other is batch)
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

        // 3. Fallback error if shapes are entirely incompatible
        Err(format!(
            "Cannot add or broadcast tensors of shapes {:?} and {:?}",
            self.shape, other.shape
        ))
    }

    /// ضرب ماتریسی (فقط برای تانسورهای دو بعدی)
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
        for i in 0..r1 {
            for j in 0..c2 {
                let mut sum = 0.0;
                for k in 0..c1 {
                    sum += self.data[i * c1 + k] * other.data[k * c2 + j];
                }
                new_data[i * c2 + j] = sum;
            }
        }

        Ok(Self {
            data: new_data,
            shape: vec![r1, c2],
        })
    }

    /// محاسبه ترانهاده ماتریس دو بعدی
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

    /// اعمال تابع فعال‌سازی ReLU به صورت عضو به عضو
    pub fn relu(&self) -> Self {
        let new_data = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 }) // اعداد منفی صفر می‌شوند
            .collect();

        Self {
            data: new_data,
            shape: self.shape.clone(), // ابعاد تغییری نمی‌کنند
        }
    }

    /// محاسبه تابع زیان میانگین مربعات خطا (MSE)
    pub fn mse_loss(&self, target: &Tensor) -> Result<Self, String> {
        if self.shape != target.shape {
            return Err("Shapes must match for MSE loss.".to_string());
        }

        let n = self.data.len() as f64;
        
        // محاسبه میانگین مربعات اختلاف‌ها
        let sum_sq: f64 = self
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        // خروجی یک تانسور اسکالر (تک عضوی) با ابعاد [1] است
        Ok(Self {
            data: vec![sum_sq / n],
            shape: vec![1],
        })
    }
    /// ساخت یک تانسور با مقادیر تصادفی (برای مقداردهی اولیه وزن‌ها)
    pub fn randn(shape: Vec<usize>) -> Self {
        let mut rng = rand::thread_rng();
        let len: usize = shape.iter().product();
        
        let data: Vec<f64> = (0..len)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
            
        Self { data, shape }
    }

    /// پیدا کردن بزرگ‌ترین عنصر تانسور (برای پایداری عددی Softmax)
    pub fn max(&self) -> f64 {
        self.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// محاسبه e به توان تک‌تک عناصر
    pub fn exp(&self) -> Tensor {
        let new_data = self.data.iter().map(|v| v.exp()).collect();
        Tensor {
            data: new_data,
            shape: self.shape.clone(),
        }
    }

    /// مجموع تمام عناصر تانسور
    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    /// اعمال تابع فعال‌سازی Sigmoid به صورت عضو به عضو
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

    /// اعمال تابع فعال‌سازی Tanh به صورت عضو به عضو
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

    /// ضرب عضو به عضو دو تانسور (بسیار مهم برای محاسبه گرادیان‌ها در backprop)
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
}

// پیاده‌سازی Display برای چاپ خواناتر در ترمینال (اختیاری اما بسیار کاربردی)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let tensor = Tensor::zeros(vec![2, 3]);
        assert_eq!(tensor.data.len(), 6);
        assert_eq!(tensor.data, vec![0.0; 6]);
    }

    #[test]
    fn test_from_data_success() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::from_data(data, shape).unwrap();
        assert_eq!(tensor.data.len(), 4);
    }

    #[test]
    fn test_from_data_error() {
        let data = vec![1.0, 2.0, 3.0]; // ۳ داده
        let shape = vec![2, 2]; // به ۴ داده نیاز دارد
        let tensor = Tensor::from_data(data, shape);
        assert!(tensor.is_err()); // تست موفق است اگر برنامه ارور بدهد
    }

    #[test]
    fn test_add_success() {
        let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let t3 = t1.add(&t2).unwrap();
        
        assert_eq!(t3.data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_mismatch_error() {
        let t1 = Tensor::zeros(vec![2, 2]); // ۴ داده
        let t2 = Tensor::zeros(vec![3]);    // ۳ داده
        let result = t1.add(&t2);
        
        assert!(result.is_err()); // باید ارور بدهد چون ابعاد برابر نیستند
    }

    #[test]
    fn test_matmul_success() {
        // ماتریس ۲ در ۳
        let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        // ماتریس ۳ در ۲
        let t2 = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        
        let t3 = t1.matmul(&t2).unwrap();
        
        // خروجی باید یک ماتریس ۲ در ۲ باشد
        assert_eq!(t3.shape, vec![2, 2]);
        // نتایج محاسبه شده
        assert_eq!(t3.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_display_tensor() {
        let t = Tensor::from_data(vec![1.1, 2.0, 3.55, 4.0, 5.123, 6.0], vec![2, 3]).unwrap();
        // این دستور تانسور را با فرمت جدیدی که نوشتیم چاپ می‌کند
        println!("{}", t);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::from_data(vec![0.0, 2.0, -2.0], vec![3]).unwrap();
        let result = t.sigmoid();
        
        // Sigmoid(0) = 0.5
        assert!((result.data[0] - 0.5).abs() < 1e-6);
        assert!(result.data[1] > 0.88); // Sigmoid(2) ≈ 0.8807
        assert!(result.data[2] < 0.12); // Sigmoid(-2) ≈ 0.1192
    }

    #[test]
    fn test_tanh() {
        let t = Tensor::from_data(vec![0.0, 1.0, -1.0], vec![3]).unwrap();
        let result = t.tanh();
        
        // Tanh(0) = 0.0
        assert!((result.data[0] - 0.0).abs() < 1e-6);
        assert!(result.data[1] > 0.76); // Tanh(1) ≈ 0.7615
        assert!(result.data[2] < -0.76); // Tanh(-1) ≈ -0.7615
    }

    #[test]
    fn test_mul_elementwise() {
        let t1 = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = Tensor::from_data(vec![0.5, 2.0, -1.0, 0.0], vec![2, 2]).unwrap();
        let t3 = t1.mul_elementwise(&t2).unwrap();
        
        assert_eq!(t3.data, vec![0.5, 4.0, -3.0, 0.0]);
    }
}