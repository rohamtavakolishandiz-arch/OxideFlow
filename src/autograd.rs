use std::cell::RefCell;
use std::rc::Rc;
use crate::tensor::Tensor;

/// ذخیره نوع عملیاتی که این متغیر را به وجود آورده است
#[derive(Clone, Debug)]
pub enum Op {
    // متغیرهای پایه که دستی ساخته می‌شوند و والدی ندارند
    None, 
    // حاصل‌جمع دو متغیر (والد اول و والد دوم)
    Add(Variable, Variable),
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
}