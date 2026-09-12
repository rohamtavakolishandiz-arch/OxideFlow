use std::cell::RefCell;
use std::rc::Rc;
use crate::tensor::Tensor;

/// ذخیره نوع عملیاتی که این متغیر را به وجود آورده است
#[derive(Clone, Debug)]
pub enum Op {
    None, 
    Add(Variable, Variable),
    // عملیات ضرب ماتریسی دو متغیر
    Matmul(Variable, Variable),
}

/// ساختار Variable برای نگهداری تانسور و گرادیان آن در گراف محاسباتی
#[derive(Clone, Debug)]
pub struct Variable {
    // استفاده از Rc و RefCell برای مدیریت مالکیت اشتراکی و تغییرات داخلی
    pub data: Rc<RefCell<Tensor>>,
    pub grad: Rc<RefCell<Tensor>>,
    // ذخیره تاریخچه تولد این متغیر
    pub creator: Rc<Op>,
}

impl Variable {
    pub fn new(tensor: Tensor) -> Self {
        let grad = Tensor::zeros(tensor.shape.clone());
        
        Self {
            data: Rc::new(RefCell::new(tensor)),
            grad: Rc::new(RefCell::new(grad)),
            creator: Rc::new(Op::None), // هیچ والدی ندارد
        }
    }

    /// جمع دو متغیر و اتصال آن‌ها در گراف محاسباتی
    pub fn add(&self, other: &Variable) -> Result<Self, String> {
        // ۱. دسترسی موقت به تانسورهای خام (فقط-خواندنی)
        let t1 = self.data.borrow();
        let t2 = other.data.borrow();

        // ۲. محاسبه حاصل‌جمع با استفاده از متد خام تانسورها
        let result_tensor = t1.add(&t2)?;

        // ۳. ساخت متغیر جدید و ثبت والدین آن در گراف
        let result_grad = Tensor::zeros(result_tensor.shape.clone());

        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(result_grad)),
            // ۴. اینجا جادوی گراف اتفاق می‌افتد: ذخیره والدین!
            creator: Rc::new(Op::Add(self.clone(), other.clone())),
        })
    }

    /// ضرب ماتریسی دو متغیر و اتصال آن‌ها در گراف
    pub fn matmul(&self, other: &Variable) -> Result<Self, String> {
        let t1 = self.data.borrow();
        let t2 = other.data.borrow();

        // محاسبه ضرب با استفاده از متد خام تانسورها
        let result_tensor = t1.matmul(&t2)?;
        let result_grad = Tensor::zeros(result_tensor.shape.clone());

        Ok(Self {
            data: Rc::new(RefCell::new(result_tensor)),
            grad: Rc::new(RefCell::new(result_grad)),
            // ثبت این متغیر به عنوان فرزند حاصل از ضرب
            creator: Rc::new(Op::Matmul(self.clone(), other.clone())),
        })
    }

    /// شروع فرآیند انتشار به عقب از این متغیر
    pub fn backward(&self) {
        // ۱. وقتی از یک گره، انتشار به عقب را شروع می‌کنیم، گرادیان خودش همیشه ۱ است.
        // مثلاً مشتق یک متغیر نسبت به خودش برابر یک است.
        {
            let mut grad = self.grad.borrow_mut();
            // تمام درایه‌های ماتریس گرادیان را به ۱ تغییر می‌دهیم
            for i in 0..grad.data.len() {
                grad.data[i] = 1.0;
            }
        } // قفل RefCell اینجا باز می‌شود

        // ۲. فراخوانی تابع داخلی برای پخش کردن گرادیان‌ها در کل گراف
        self._backward();
    }

    /// تابع داخلی که گراف را به سمت عقب پیمایش می‌کند
    /// تابع داخلی که گراف را به سمت عقب پیمایش می‌کند
    fn _backward(&self) {
        let grad_value = self.grad.borrow().clone(); // گرادیان این گره چقدر است؟
        
        // نگاه می‌کنیم ببینیم این گره چطور متولد شده
        match &*self.creator {
            Op::Add(parent_a, parent_b) => {
                // ... (کدهای قبلی بخش Add که دست نخورده باقی می‌ماند) ...
                {
                    let mut a_grad = parent_a.grad.borrow_mut();
                    let t_grad = a_grad.add(&grad_value).unwrap();
                    *a_grad = t_grad;
                }
                {
                    let mut b_grad = parent_b.grad.borrow_mut();
                    let t_grad = b_grad.add(&grad_value).unwrap();
                    *b_grad = t_grad;
                }
                parent_a._backward();
                parent_b._backward();
            }
            
            // ---> این بخش جدید است که دقیقاً قبل از Op::None اضافه می‌شود <---
            Op::Matmul(parent_a, parent_b) => {
                // --- محاسبه گرادیان برای والد اول (A) ---
                {
                    let b_data = parent_b.data.borrow();
                    let b_transposed = b_data.transpose().unwrap(); // B^T
                    let a_grad_update = grad_value.matmul(&b_transposed).unwrap(); // dC * B^T
                    
                    let mut a_grad = parent_a.grad.borrow_mut();
                    let t_grad = a_grad.add(&a_grad_update).unwrap();
                    *a_grad = t_grad;
                }
                
                // --- محاسبه گرادیان برای والد دوم (B) ---
                {
                    let a_data = parent_a.data.borrow();
                    let a_transposed = a_data.transpose().unwrap(); // A^T
                    let b_grad_update = a_transposed.matmul(&grad_value).unwrap(); // A^T * dC
                    
                    let mut b_grad = parent_b.grad.borrow_mut();
                    let t_grad = b_grad.add(&b_grad_update).unwrap();
                    *b_grad = t_grad;
                }
                
                parent_a._backward();
                parent_b._backward();
            }
            // ---> پایان بخش جدید <---

            Op::None => {
                // این گره والدی ندارد (متغیر پایه است)
            }
        }
    }

    /// صفر کردن گرادیان این متغیر (برای استفاده در حلقه‌های آموزش)
    pub fn zero_grad(&self) {
        let mut grad = self.grad.borrow_mut();
        // تمام مقادیر گرادیان را به صفر تغییر می‌دهیم
        for i in 0..grad.data.len() {
            grad.data[i] = 0.0;
        }
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
}