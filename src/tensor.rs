use std::fmt;

/// ساختار اصلی تانسور برای نگهداری داده‌های خام و ابعاد آن‌ها
#[derive(Clone, Debug, PartialEq)]
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
    pub fn add(&self, other: &Tensor) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err("Cannot add tensors of different shapes.".to_string());
        }

        let new_data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
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
}