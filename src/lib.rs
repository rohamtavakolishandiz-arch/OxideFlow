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
}