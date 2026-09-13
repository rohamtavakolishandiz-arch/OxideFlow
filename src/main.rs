use oxide_flow::autograd::Variable;
use oxide_flow::optimizer::SGD;
use oxide_flow::tensor::Tensor;

fn main() {
    println!("--- Starting OxideFlow Training ---");

    // ۱. داده‌های آموزشی (Training Data)
    // ورودی‌ها: [1.0], [2.0], [3.0]
    let inputs = vec![
        Variable::new(Tensor::from_data(vec![1.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![2.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![3.0], vec![1, 1]).unwrap()),
    ];

    // جواب‌های واقعی (تارگت‌ها): [2.0], [4.0], [6.0]
    let targets = vec![
        Variable::new(Tensor::from_data(vec![2.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![4.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![6.0], vec![1, 1]).unwrap()),
    ];

    // ۲. تعریف مدل (وزن‌های شبکه)
    // یک وزن تصادفی اولیه (مثلاً 0.5) در نظر می‌گیریم. 
    // هدف این است که شبکه این 0.5 را به 2.0 برساند.
    let weight = Variable::new(Tensor::from_data(vec![0.5], vec![1, 1]).unwrap());

    // ۳. تنظیمات بهینه‌ساز (Optimizer)
    // وزن را به بهینه‌ساز می‌دهیم تا آن را آپدیت کند. نرخ یادگیری را 0.01 می‌گذاریم.
    let optimizer = SGD::new(vec![weight.clone()], 0.01);

    let epochs = 50; // تعداد دفعات آموزش

    // ۴. حلقه آموزش (Training Loop)
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;

        for (x, y) in inputs.iter().zip(targets.iter()) {
            // مرحله A: صفر کردن گرادیان‌های قبلی
            optimizer.zero_grad();

            // مرحله B: حرکت به جلو (پیش‌بینی = ورودی ضربدر وزن)
            let prediction = x.matmul(&weight).unwrap();

            // مرحله C: محاسبه خطا (MSE)
            let loss = prediction.mse_loss(&y).unwrap();
            epoch_loss += loss.data.borrow().data[0];

            // مرحله D: انتشار به عقب (محاسبه تقصیرِ وزن در این خطا)
            loss.backward();

            // مرحله E: آپدیت کردن وزن
            optimizer.step();
        }

        // هر ۱۰ مرحله، وضعیت خطا و وزن را چاپ می‌کنیم
        if epoch % 10 == 0 {
            let current_weight = weight.data.borrow().data[0];
            println!(
                "Epoch {:2}: Loss = {:.4} | Current Weight = {:.4}",
                epoch,
                epoch_loss / 3.0, // میانگین خطا برای ۳ داده
                current_weight
            );
        }
    }

    println!("--- Training Finished ---");

    // ۵. تست کردن شبکه (Inference)
    // حالا از شبکه می‌پرسیم: اگر ورودی 5.0 باشد، خروجی چیست؟
    let test_input = Variable::new(Tensor::from_data(vec![5.0], vec![1, 1]).unwrap());
    let test_prediction = test_input.matmul(&weight).unwrap();

    println!(
        "\nTest Input: 5.0 -> Prediction: {:.4} (Expected: 10.0)",
        test_prediction.data.borrow().data[0]
    );
}