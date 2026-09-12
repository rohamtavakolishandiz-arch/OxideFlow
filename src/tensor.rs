#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let total_elements: usize = shape.iter().product();
        let data = vec![0.0; total_elements];
        Self { data, shape }
    }

    /// ساخت تانسور از روی داده‌های ورودی با بررسی صحت ابعاد
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        let expected_length: usize = shape.iter().product();
        
        if data.len() != expected_length {
            return Err(format!(
                "Dimension Mismatch: Expected {} elements, but got {}",
                expected_length,
                data.len()
            ));
        }
        
        Ok(Self { data, shape })
    }

    /// جمع دو تانسور با بررسی یکسان بودن ابعاد
    pub fn add(&self, other: &Tensor) -> Result<Self, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Shape Mismatch: Cannot add tensors of shape {:?} and {:?}",
                self.shape, other.shape
            ));
        }

        // ایجاد وکتور جدید با ظرفیت از پیش تعیین شده برای سرعت بیشتر
        let mut new_data = Vec::with_capacity(self.data.len());
        
        // پیمایش با یک حلقه ساده و جمع درایه‌های متناظر
        for i in 0..self.data.len() {
            new_data.push(self.data[i] + other.data[i]);
        }

        Ok(Self {
            data: new_data,
            shape: self.shape.clone(),
        })
    }

    /// ضرب ماتریسی دو تانسور (در حال حاضر فقط برای دو بعدی)
    pub fn matmul(&self, other: &Tensor) -> Result<Self, String> {
        // ۱. بررسی اینکه هر دو تانسور حتماً دو بعدی (ماتریس) باشند
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err("Matmul currently only supports 2D tensors.".to_string());
        }

        let m = self.shape[0]; // تعداد سطرهای ماتریس اول
        let n = self.shape[1]; // تعداد ستون‌های ماتریس اول
        let p = other.shape[1]; // تعداد ستون‌های ماتریس دوم

        // ۲. قانون طلایی ضرب ماتریس: ستون‌های اولی باید با سطرهای دومی برابر باشد
        if n != other.shape[0] {
            return Err(format!(
                "Matmul Shape Mismatch: {}x{} cannot be multiplied with {}x{}",
                m, n, other.shape[0], p
            ));
        }

        // ایجاد آرایه خروجی پر از صفر با ظرفیت مناسب (ابعاد m * p)
        let mut new_data = vec![0.0; m * p];

        // ۳. حلقه‌های تو در تو برای محاسبه ضرب ماتریسی
        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n {
                    // پیدا کردن جایگاه دقیق در آرایه خطی
                    let self_idx = i * n + k;
                    let other_idx = k * p + j;
                    sum += self.data[self_idx] * other.data[other_idx];
                }
                new_data[i * p + j] = sum;
            }
        }

        Ok(Self {
            data: new_data,
            shape: vec![m, p],
        })
    }

    /// محاسبه ترانهاده ماتریس دو بعدی (جایگزینی سطرها و ستون‌ها)
    pub fn transpose(&self) -> Result<Self, String> {
        if self.shape.len() != 2 {
            return Err("Transpose currently only supports 2D tensors.".to_string());
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        
        let mut new_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for j in 0..cols {
                let original_idx = i * cols + j;
                let transposed_idx = j * rows + i;
                new_data[transposed_idx] = self.data[original_idx];
            }
        }

        Ok(Self {
            data: new_data,
            shape: vec![cols, rows],
        })
    }
}

use std::fmt;

// پیاده‌سازی قابلیت چاپ سفارشی برای تانسور
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.shape.len() == 2 {
            let rows = self.shape[0];
            let cols = self.shape[1];
            writeln!(f, "Tensor {}x{}:", rows, cols)?;
            
            for i in 0..rows {
                write!(f, "[ ")?;
                for j in 0..cols {
                    // چاپ اعداد با ۴ رقم اعشار برای زیبایی و دقت
                    let idx = i * cols + j;
                    write!(f, "{:.4}  ", self.data[idx])?;
                }
                writeln!(f, "]")?;
            }
            Ok(())
        } else if self.shape.len() == 1 {
            writeln!(f, "Tensor 1D ({}):", self.shape[0])?;
            write!(f, "[ ")?;
            for val in &self.data {
                write!(f, "{:.4}  ", val)?;
            }
            write!(f, "]")
        } else {
            // برای تانسورهای ۳ بعدی و بالاتر، فعلاً فقط ابعاد را چاپ می‌کنیم
            write!(f, "Tensor with shape {:?}", self.shape)
        }
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