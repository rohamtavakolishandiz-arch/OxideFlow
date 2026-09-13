use crate::autograd::Variable;
use crate::tensor::Tensor;

/// نمایانگر یک لایه خطی (Fully Connected) در شبکه عصبی
pub struct Linear {
    pub weight: Variable,
    pub bias: Variable,
}

impl Linear {
    /// سازنده لایه جدید با مشخص کردن ابعاد ورودی و خروجی
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // برای شروع، وزن‌ها را با یک عدد کوچک (مثلاً 0.1) مقداردهی می‌کنیم
        // در آینده اینجا از اعداد تصادفی (Random) استفاده خواهیم کرد
        let w_len = in_features * out_features;
        let w_data = vec![0.1; w_len];
        let weight = Variable::new(Tensor::from_data(w_data, vec![in_features, out_features]).unwrap());

        // بایاس‌ها معمولاً در ابتدا با صفر مقداردهی می‌شوند
        let b_data = vec![0.0; out_features];
        let bias = Variable::new(Tensor::from_data(b_data, vec![1, out_features]).unwrap());

        Self { weight, bias }
    }

    /// محاسبه خروجی لایه: y = xW + b
    pub fn forward(&self, input: &Variable) -> Result<Variable, String> {
        let matmul_result = input.matmul(&self.weight)?;
        let output = matmul_result.add(&self.bias)?;
        Ok(output)
    }

    /// خروجی دادن تمام پارامترهای لایه برای ارسال به بهینه‌ساز
    pub fn parameters(&self) -> Vec<Variable> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}