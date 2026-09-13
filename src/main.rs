use oxide_flow::autograd::Variable;
use oxide_flow::nn::Linear;
use oxide_flow::optimizer::SGD;
use oxide_flow::tensor::Tensor;

fn main() {
    println!("--- Starting OxideFlow Training: XOR Problem ---");

    // ۱. داده‌های آموزشی دروازه XOR
    // ورودی‌ها: [0,0], [0,1], [1,0], [1,1]
    let inputs = vec![
        Variable::new(Tensor::from_data(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Variable::new(Tensor::from_data(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        Variable::new(Tensor::from_data(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        Variable::new(Tensor::from_data(vec![1.0, 1.0], vec![1, 2]).unwrap()),
    ];

    // جواب‌های واقعی: [0], [1], [1], [0]
    let targets = vec![
        Variable::new(Tensor::from_data(vec![0.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![1.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![1.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![0.0], vec![1, 1]).unwrap()),
    ];

    // ۲. تعریف معماری شبکه چند لایه (MLP)
    let layer1 = Linear::new(2, 4); // ۲ ورودی -> ۴ نود مخفی
    let layer2 = Linear::new(4, 1); // ۴ نود مخفی -> ۱ خروجی

    // جمع‌آوری پارامترهای هر دو لایه برای بهینه‌ساز
    let mut all_parameters = layer1.parameters();
    all_parameters.extend(layer2.parameters());

    // نرخ یادگیری را کمی بالاتر می‌بریم چون مسئله پیچیده‌تر است
    let optimizer = SGD::new(all_parameters, 0.1); 

    let epochs = 1000; // حل مسئله غیرخطی نیاز به تکرار بیشتری دارد

    // ۳. حلقه آموزش
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;

        for (x, y) in inputs.iter().zip(targets.iter()) {
            optimizer.zero_grad();

            // --- مسیر رفت (Forward Pass) با ساختار چند لایه ---
            // مرحله ۱: عبور از لایه اول
            let out1 = layer1.forward(x).unwrap();
            
            // مرحله ۲: اعمال تابع غیرخطی (بسیار مهم برای XOR)
            let activated1 = out1.relu();
            
            // مرحله ۳: عبور از لایه نهایی
            let prediction = layer2.forward(&activated1).unwrap();

            // --- محاسبه خطا و مسیر برگشت ---
            let loss = prediction.mse_loss(y).unwrap();
            epoch_loss += loss.data.borrow().data[0];

            loss.backward();
            optimizer.step();
        }

        if epoch % 100 == 0 {
            println!("Epoch {:4}: Loss = {:.4}", epoch, epoch_loss / 4.0);
        }
    }

    println!("\n--- Training Finished. Testing the Model ---");

    // ۴. تست نهایی شبکه
    for (i, x) in inputs.iter().enumerate() {
        let out1 = layer1.forward(x).unwrap();
        let activated1 = out1.relu();
        let prediction = layer2.forward(&activated1).unwrap();
        
        let expected = targets[i].data.borrow().data[0];
        let actual = prediction.data.borrow().data[0];
        
        // اگر خروجی بزرگتر از 0.5 بود یعنی شبکه عدد 1 را پیش‌بینی کرده است
        let binary_pred = if actual > 0.5 { 1.0 } else { 0.0 };

        println!(
            "Input: {:?} -> Raw Output: {:.4} | Prediction: {} (Expected: {})",
            x.data.borrow().data,
            actual,
            binary_pred,
            expected
        );
    }
}