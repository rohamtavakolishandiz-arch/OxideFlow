use std::cell::RefCell;
use std::rc::Rc;
use crate::tensor::Tensor;

#[derive(Clone, Debug)]
pub enum Op {
    None,
    Add(Variable, Variable),
    Matmul(Variable, Variable),
    ReLU(Variable),
    // عملیات جدید: پیش‌بینی شبکه و جواب واقعی
    MSE(Variable, Variable), 
}

#[derive(Clone, Debug)]
pub struct Variable {
    pub data: Rc<RefCell<Tensor>>,
    pub grad: Rc<RefCell<Tensor>>,
    pub creator: Rc<Op>,
}

impl Variable {
    /// سازنده یک متغیر جدید از روی یک تانسور
    pub fn new(tensor: Tensor) -> Self {
        let shape = tensor.shape.clone();
        Self {
            data: Rc::new(RefCell::new(tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::None),
        }
    }

    /// جمع دو متغیر
    pub fn add(&self, other: &Variable) -> Result<Self, String> {
        let result_tensor = self.data.borrow().add(&other.data.borrow())?;
        let shape = result_tensor.shape.clone();
        
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::Add(self.clone(), other.clone())),
        })
    }

    /// ضرب ماتریسی دو متغیر
    pub fn matmul(&self, other: &Variable) -> Result<Self, String> {
        let result_tensor = self.data.borrow().matmul(&other.data.borrow())?;
        let shape = result_tensor.shape.clone();
        
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::Matmul(self.clone(), other.clone())),
        })
    }

    /// صفر کردن گرادیان‌ها با استفاده از Iterator (روش استاندارد راست)
    pub fn zero_grad(&self) {
        self.grad.borrow_mut().data.iter_mut().for_each(|g| *g = 0.0);
    }

    /// شروع فرآیند انتشار به عقب
    pub fn backward(&self) {
        // تنظیم گرادیان اولیه گره خروجی به ۱.۰
        self.grad.borrow_mut().data.iter_mut().for_each(|g| *g = 1.0);
        self._backward();
    }

    /// تابع داخلی پیمایش گراف
    fn _backward(&self) {
        let grad_value = self.grad.borrow().clone();

        match &*self.creator {
            Op::Add(parent_a, parent_b) => {
                Self::update_grad(parent_a, &grad_value);
                Self::update_grad(parent_b, &grad_value);

                parent_a._backward();
                parent_b._backward();
            }
            Op::Matmul(parent_a, parent_b) => {
                // محاسبات والد اول (A)
                let b_transposed = parent_b.data.borrow().transpose().unwrap();
                let a_grad_update = grad_value.matmul(&b_transposed).unwrap();
                Self::update_grad(parent_a, &a_grad_update);

                // محاسبات والد دوم (B)
                let a_transposed = parent_a.data.borrow().transpose().unwrap();
                let b_grad_update = a_transposed.matmul(&grad_value).unwrap();
                Self::update_grad(parent_b, &b_grad_update);

                parent_a._backward();
                parent_b._backward();
            }
            Op::None => {}
            Op::ReLU(parent) => {
                let parent_data = parent.data.borrow();
                let mut parent_grad_update = Tensor::zeros(parent_data.shape.clone());

                // برای هر عنصر چک می‌کنیم
                for i in 0..parent_data.data.len() {
                    // اگر داده اصلی مثبت بوده، گرادیان از این گره عبور می‌کند (ضرب در ۱)
                    // اگر منفی بوده، گرادیان صفر می‌شود (ضرب در ۰)
                    if parent_data.data[i] > 0.0 {
                        parent_grad_update.data[i] = grad_value.data[i];
                    } else {
                        parent_grad_update.data[i] = 0.0;
                    }
                }

                Self::update_grad(parent, &parent_grad_update);
                parent._backward();
            }
            Op::MSE(pred, target) => {
                let p_data = pred.data.borrow();
                let t_data = target.data.borrow();
                let n = p_data.data.len() as f64;
                
                // گرادیانی که از مراحل بالاتر آمده (برای گره نهایی همیشه ۱.۰ است)
                let g = grad_value.data[0];

                let mut pred_grad_update = Tensor::zeros(p_data.shape.clone());
                
                // اعمال فرمول مشتق: (2/n) * (pred - target) * g
                for i in 0..p_data.data.len() {
                    pred_grad_update.data[i] = (2.0 / n) * (p_data.data[i] - t_data.data[i]) * g;
                }

                // ما فقط گرادیان را به متغیر پیش‌بینی (وزن‌های شبکه ما) برمی‌گردانیم.
                // جواب‌های واقعی (target) مقادیر ثابتی هستند و نیازی به دریافت گرادیان و آپدیت شدن ندارند.
                Self::update_grad(pred, &pred_grad_update);
                
                pred._backward();
                // target._backward() فراخوانی می‌شود اما چون از نوع Op::None است اتفاقی نمی‌افتد
                target._backward();
            }
        }
    }

    /// یک تابع کمکی برای تمیز کردن منطق آپدیت گرادیان‌ها
    fn update_grad(var: &Variable, grad_update: &Tensor) {
        let mut current_grad = var.grad.borrow_mut();
        let new_grad = current_grad.add(grad_update).unwrap();
        *current_grad = new_grad;
    }

    /// اعمال تابع ReLU و اتصال به گراف
    pub fn relu(&self) -> Self {
        let result_tensor = self.data.borrow().relu();
        let shape = result_tensor.shape.clone();
        
        Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(Tensor::zeros(shape))),
            creator: Rc::new(Op::ReLU(self.clone())), // ذخیره متغیر فعلی به عنوان والد
        }
    }

    /// محاسبه خطای MSE و اتصال به گراف
    pub fn mse_loss(&self, target: &Variable) -> Result<Self, String> {
        let result_tensor = self.data.borrow().mse_loss(&target.data.borrow())?;
        
        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            // گرادیان خطا همیشه در ابتدا [0.0] است
            grad: Rc::new(RefCell::new(Tensor::zeros(vec![1]))),
            creator: Rc::new(Op::MSE(self.clone(), target.clone())),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_creation_and_mutation() {
        // ۱. ساخت یک تانسور ساده با دو عدد
        let tensor = Tensor::from_data(vec![1.0, 2.0], vec![2]).unwrap();
        
        // ۲. قرار دادن تانسور در جعبه Variable
        let var = Variable::new(tensor);

        // ۳. خواندن اطلاعات (استفاده از borrow)
        // وقتی borrow می‌کنیم، به محتوای داخل جعبه دسترسی فقط-خواندنی داریم
        let data_length = var.data.borrow().data.len();
        assert_eq!(data_length, 2);

        // بررسی اینکه گرادیان اولیه واقعاً صفر است
        assert_eq!(var.grad.borrow().data, vec![0.0, 0.0]);

        // ۴. تغییر دادن اطلاعات (استفاده از borrow_mut)
        // فرض کن در طول آموزش هوش مصنوعی، می‌خواهیم خطای محاسبه شده را به گرادیان اضافه کنیم
        {
            let mut mutable_grad = var.grad.borrow_mut();
            mutable_grad.data[0] = 0.5; // تغییر گرادیان عدد اول
            mutable_grad.data[1] = 0.8; // تغییر گرادیان عدد دوم
        } // این براکت (Scope) باعث می‌شود جعبه اینجا بسته شود و اجازه تغییرات تمام شود

        // ۵. بررسی اینکه آیا تغییرات با موفقیت در Variable اصلی اعمال شده است؟
        assert_eq!(var.grad.borrow().data, vec![0.5, 0.8]);
    }

    #[test]
    fn test_computational_graph_add() {
        // ۱. ساخت دو متغیر پایه
        let t1 = Tensor::from_data(vec![1.0, 2.0], vec![2]).unwrap();
        let var_a = Variable::new(t1);

        let t2 = Tensor::from_data(vec![3.0, 4.0], vec![2]).unwrap();
        let var_b = Variable::new(t2);

        // ۲. جمع کردن آن‌ها
        let var_c = var_a.add(&var_b).unwrap();

        // ۳. بررسی مقدار داده شده
        assert_eq!(var_c.data.borrow().data, vec![4.0, 6.0]);

        // ۴. بررسی تاریخچه عملیات (آیا والدین ثبت شده‌اند؟)
        if let Op::Add(parent_a, parent_b) = &*var_c.creator {
            // آیا والد اول همون var_a است؟
            assert_eq!(parent_a.data.borrow().data, vec![1.0, 2.0]);
            // آیا والد دوم همون var_b است؟
            assert_eq!(parent_b.data.borrow().data, vec![3.0, 4.0]);
        } else {
            panic!("Creator is not an Add operation!"); // اگر تاریخچه اشتباه بود، تست را متوقف کن
        }
    }

    #[test]
    fn test_backpropagation_add() {
        // ۱. ساخت متغیرها: x = 2, y = 3
        let x = Variable::new(Tensor::from_data(vec![2.0], vec![1]).unwrap());
        let y = Variable::new(Tensor::from_data(vec![3.0], vec![1]).unwrap());
        
        // ۲. عملیات جلو: z = x + y
        let z = x.add(&y).unwrap();
        
        // ۳. عملیات عقب: محاسبه مشتقات
        z.backward();
        
        // ۴. بررسی نتایج! مشتق z نسبت به x و y باید ۱ باشد.
        assert_eq!(x.grad.borrow().data, vec![1.0]);
        assert_eq!(y.grad.borrow().data, vec![1.0]);
    }

    #[test]
    fn test_backpropagation_matmul() {
        // ۱. ساخت ماتریس A (۱ سطر، ۲ ستون)
        let a_tensor = Tensor::from_data(vec![2.0, 3.0], vec![1, 2]).unwrap();
        let a = Variable::new(a_tensor);
        
        // ۲. ساخت ماتریس B (۲ سطر، ۱ ستون)
        let b_tensor = Tensor::from_data(vec![4.0, 5.0], vec![2, 1]).unwrap();
        let b = Variable::new(b_tensor);
        
        // ۳. عملیات حرکت به جلو (Forward Pass)
        let c = a.matmul(&b).unwrap();
        
        // بررسی مقدار خروجی (آیا ضرب درست کار کرده است؟)
        assert_eq!(c.data.borrow().data, vec![23.0]);
        assert_eq!(c.data.borrow().shape, vec![1, 1]);
        
        // ۴. عملیات انتشار به عقب (Backward Pass)
        c.backward();
        
        // ۵. بررسی گرادیان‌ها!
        // گرادیان a باید برابر با ترانهاده b باشد
        assert_eq!(a.grad.borrow().data, vec![4.0, 5.0]);
        // گرادیان b باید برابر با ترانهاده a باشد
        assert_eq!(b.grad.borrow().data, vec![2.0, 3.0]);
    }

    #[test]
    fn test_relu_forward_and_backward() {
        // ۱. ساخت یک متغیر با ترکیبی از اعداد مثبت و منفی
        let x = Variable::new(Tensor::from_data(vec![-2.0, 0.5, 3.0, -1.0], vec![4]).unwrap());
        
        // ۲. اعمال ReLU (مسیر رفت)
        let y = x.relu();
        
        // اعداد منفی باید صفر شده باشند
        assert_eq!(y.data.borrow().data, vec![0.0, 0.5, 3.0, 0.0]);
        
        // ۳. انتشار به عقب
        y.backward();
        
        // ۴. بررسی گرادیان‌ها
        // چون گرادیان اولیه y کلاً ۱.۰ است، بعد از برگشت به x:
        // جایی که x منفی بوده -> گرادیان باید صفر شود
        // جایی که x مثبت بوده -> گرادیان همان ۱.۰ می‌ماند
        assert_eq!(x.grad.borrow().data, vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_mse_loss_backward() {
        // ۱. پیش‌بینی شبکه ما (مثلاً خروجی یک لایه)
        let pred = Variable::new(Tensor::from_data(vec![2.0, 4.0], vec![2]).unwrap());
        
        // ۲. جواب واقعی که انتظار داشتیم
        let target = Variable::new(Tensor::from_data(vec![2.0, 2.0], vec![2]).unwrap());
        
        // ۳. محاسبه خطا
        // اختلاف‌ها: (2-2)=0 و (4-2)=2
        // توان دو: 0 و 4
        // میانگین: (0 + 4) / 2 = 2.0
        let loss = pred.mse_loss(&target).unwrap();
        assert_eq!(loss.data.borrow().data, vec![2.0]);
        
        // ۴. انتشار به عقب
        loss.backward();
        
        // ۵. بررسی گرادیان روی pred
        // مشتق برای عضو اول: (2/2) * (2 - 2) = 0.0
        // مشتق برای عضو دوم: (2/2) * (4 - 2) = 2.0
        assert_eq!(pred.grad.borrow().data, vec![0.0, 2.0]);
    }
}