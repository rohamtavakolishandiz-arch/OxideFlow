use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use crate::tensor::Tensor;
use crate::autograd::Variable;

/// نمایانگر یک لایه خطی (Fully Connected) در شبکه عصبی
pub struct Linear {
    pub weight: Variable,
    pub bias: Variable,
}

/// ساختاری ساده برای ذخیره‌سازی وزن و بایاس یک لایه
#[derive(Serialize, Deserialize)]
pub struct LayerState {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// سازنده لایه جدید با مشخص کردن ابعاد ورودی و خروجی
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // حالا وزن‌ها با اعداد تصادفی مقداردهی می‌شوند
        let weight = Variable::new(Tensor::randn(vec![in_features, out_features]));

        // بایاس‌ها معمولاً در ابتدا با صفر مقداردهی می‌شوند
        let bias = Variable::new(Tensor::zeros(vec![1, out_features]));

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

    /// ذخیره وضعیت لایه روی فایل
    pub fn save(&self, path: &str) -> Result<(), String> {
        let state = LayerState {
            weight: self.weight.data.borrow().clone(),
            bias: self.bias.data.borrow().clone(),
        };

        let json = serde_json::to_string_pretty(&state)
            .map_err(|e| format!("Failed to serialize layer: {}", e))?;

        let mut file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        file.write_all(json.as_bytes())
            .map_err(|e| format!("Failed to write to file: {}", e))?;

        Ok(())
    }

    /// بارگذاری وضعیت لایه از روی فایل
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        let mut file = File::open(path)
            .map_err(|e| format!("Failed to open file: {}", e))?;

        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|e| format!("Failed to read file: {}", e))?;

        let state: LayerState = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse JSON: {}", e))?;

        *self.weight.data.borrow_mut() = state.weight;
        *self.bias.data.borrow_mut() = state.bias;

        Ok(())
    }
}