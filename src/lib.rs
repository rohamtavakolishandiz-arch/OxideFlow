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
}