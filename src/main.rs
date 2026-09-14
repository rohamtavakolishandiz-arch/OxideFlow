use oxide_flow::autograd::Variable;
use oxide_flow::nn::Linear;
use oxide_flow::optimizer::SGD;
use oxide_flow::tensor::Tensor;

fn main() {
    println!("--- Starting OxideFlow Training: XOR Problem ---");

    let inputs = vec![
        Variable::new(Tensor::from_data(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Variable::new(Tensor::from_data(vec![0.0, 1.0], vec![1, 2]).unwrap()),
        Variable::new(Tensor::from_data(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        Variable::new(Tensor::from_data(vec![1.0, 1.0], vec![1, 2]).unwrap()),
    ];

    let targets = vec![
        Variable::new(Tensor::from_data(vec![0.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![1.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![1.0], vec![1, 1]).unwrap()),
        Variable::new(Tensor::from_data(vec![0.0], vec![1, 1]).unwrap()),
    ];

    let layer1 = Linear::new(2, 4);
    let layer2 = Linear::new(4, 1);

    let mut all_parameters = layer1.parameters();
    all_parameters.extend(layer2.parameters());

    let optimizer = SGD::new(all_parameters, 0.1); 
    let epochs = 1000;

    println!("Training in progress...");
    for _epoch in 1..=epochs {
        for (x, y) in inputs.iter().zip(targets.iter()) {
            optimizer.zero_grad();

            let out1 = layer1.forward(x).unwrap();
            let activated1 = out1.relu();
            let prediction = layer2.forward(&activated1).unwrap();

            let loss = prediction.mse_loss(y).unwrap();
            loss.backward();
            optimizer.step();
        }
    }
    
    // ---------------------------------------------------------
    // بخش ۱: ذخیره مدل بعد از اتمام آموزش
    // ---------------------------------------------------------
    println!("Training Finished. Saving layers to disk...");
    layer1.save("layer1.json").unwrap();
    layer2.save("layer2.json").unwrap();
    println!("Model successfully saved to 'layer1.json' and 'layer2.json'.\n");


    // ---------------------------------------------------------
    // بخش ۲: بارگذاری مدل در یک شبکه کاملاً جدید
    // ---------------------------------------------------------
    println!("--- Loading Model from disk for Inference ---");
    let mut test_layer1 = Linear::new(2, 4);
    let mut test_layer2 = Linear::new(4, 1);

    // جایگزین کردن وزن‌های تصادفی با وزن‌های آموزش‌دیده
    test_layer1.load("layer1.json").unwrap();
    test_layer2.load("layer2.json").unwrap();
    println!("Model loaded! Running predictions:\n");

    // ---------------------------------------------------------
    // بخش ۳: تست کردن مدلی که لود شده است
    // ---------------------------------------------------------
    for (i, x) in inputs.iter().enumerate() {
        // توجه کن که اینجا از test_layer استفاده می‌کنیم
        let out1 = test_layer1.forward(x).unwrap();
        let activated1 = out1.relu();
        let prediction = test_layer2.forward(&activated1).unwrap();
        
        let expected = targets[i].data.borrow().data[0];
        let actual = prediction.data.borrow().data[0];
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