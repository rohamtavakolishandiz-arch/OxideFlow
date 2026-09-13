use crate::autograd::Variable;

/// ساختار بهینه‌ساز گرادیان کاهشی تصادفی (SGD)
pub struct SGD {
    parameters: Vec<Variable>, // لیست تمام وزن‌هایی که باید آپدیت شوند
    learning_rate: f64,
}

impl SGD {
    /// سازنده بهینه‌ساز
    pub fn new(parameters: Vec<Variable>, learning_rate: f64) -> Self {
        Self {
            parameters,
            learning_rate,
        }
    }

    /// متد اصلی برای آپدیت کردن وزن‌ها
    pub fn step(&self) {
        for param in &self.parameters {
            // ۱. خواندن داده‌های فعلی و گرادیان‌ها
            let mut data = param.data.borrow_mut();
            let grad = param.grad.borrow();

            // ۲. آپدیت کردن تک‌تک اعضای تانسور بر اساس فرمول SGD
            for i in 0..data.data.len() {
                data.data[i] -= self.learning_rate * grad.data[i];
            }
        }
    }

    /// صفر کردن تمام گرادیان‌ها برای دور بعدی آموزش
    pub fn zero_grad(&self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_sgd_optimizer() {
        // ۱. ساخت یک وزن (مثلاً 2.0)
        let weight = Variable::new(Tensor::from_data(vec![2.0], vec![1]).unwrap());
        
        // ۲. ساخت یک تارگت یا جواب واقعی (مثلاً 5.0)
        let target = Variable::new(Tensor::from_data(vec![5.0], vec![1]).unwrap());

        // ۳. ساخت بهینه‌ساز و معرفی وزن‌ها به آن (با نرخ یادگیری 0.1)
        let optimizer = SGD::new(vec![weight.clone()], 0.1);

        // ۴. حلقه آموزش (۵ دور)
        for _ in 0..5 {
            // صفر کردن گرادیان‌های قبلی
            optimizer.zero_grad();

            // حرکت به جلو و محاسبه خطا
            let loss = weight.mse_loss(&target).unwrap();

            // انتشار به عقب (محاسبه گرادیان)
            loss.backward();

            // آپدیت وزن‌ها توسط بهینه‌ساز
            optimizer.step();
        }

        // بعد از ۵ دور آموزش، وزن ما که 2.0 بود باید به جواب واقعی (5.0) نزدیک شده باشد.
        // با این تنظیمات به حدود 3.7 تا 4.0 می‌رسد.
        let updated_weight = weight.data.borrow().data[0];
        assert!(updated_weight > 2.0 && updated_weight < 5.0);
    }
}