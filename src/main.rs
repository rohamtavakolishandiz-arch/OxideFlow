use oxide_flow::autograd::Variable;
use oxide_flow::nn::{Linear, Sequential, Module, Tanh};
use oxide_flow::optimizer::{AdamW, SGD, Optimizer};
use oxide_flow::tensor::Tensor;
use oxide_flow::data::DataLoader;

fn main() {
    println!("--- Starting OxideFlow Multi-Class Classification Test ---");

    // ۱. داده‌های ورودی (ابعاد: 1x2)
    let inputs = vec![
        Variable::new(Tensor::from_data(vec![2.0, 0.1], vec![1, 2]).unwrap()),   // کلاس ۰
        Variable::new(Tensor::from_data(vec![1.5, -0.2], vec![1, 2]).unwrap()),  // کلاس ۰
        Variable::new(Tensor::from_data(vec![0.1, 2.0], vec![1, 2]).unwrap()),   // کلاس ۱
        Variable::new(Tensor::from_data(vec![-0.1, 1.8], vec![1, 2]).unwrap()),  // کلاس ۱
        Variable::new(Tensor::from_data(vec![-1.5, -1.2], vec![1, 2]).unwrap()), // کلاس ۲
        Variable::new(Tensor::from_data(vec![-2.0, -0.8], vec![1, 2]).unwrap()), // کلاس ۲
    ];

    // ۲. تارگت‌ها به صورت One-Hot Encoded (ابعاد: 1x3)
    let targets = vec![
        Variable::new(Tensor::from_data(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap()), // کلاس ۰
        Variable::new(Tensor::from_data(vec![1.0, 0.0, 0.0], vec![1, 3]).unwrap()), // کلاس ۰
        Variable::new(Tensor::from_data(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap()), // کلاس ۱
        Variable::new(Tensor::from_data(vec![0.0, 1.0, 0.0], vec![1, 3]).unwrap()), // کلاس ۱
        Variable::new(Tensor::from_data(vec![0.0, 0.0, 1.0], vec![1, 3]).unwrap()), // کلاس ۲
        Variable::new(Tensor::from_data(vec![0.0, 0.0, 1.0], vec![1, 3]).unwrap()), // کلاس ۲
    ];

    // ۳. ساخت مدل با معماری جدید ماژولار
    let model = Sequential::new(vec![
        Box::new(Linear::new(2, 6)),
        Box::new(Tanh), // استفاده از تابع فعال‌سازی صریح بین لایه‌ها
        Box::new(Linear::new(6, 3)),
    ]);

    // Initialize BOTH optimizers
    // AdamW gets a standard learning rate
    let mut adam_optimizer = AdamW::new(model.parameters(), 0.01);
    
    // SGD gets a very small learning rate for fine-tuning
    let mut sgd_optimizer = SGD::new(model.parameters(), 0.001); 

    let dataloader = DataLoader::new(inputs, targets, 2, true);

    let epochs = 500;

    // ... (Keep your DataLoader initialization the same) ...

    println!("Training with True Matrix Mini-batching...");
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut total_samples = 0;

        let current_optimizer: &mut dyn Optimizer = if epoch <= 400 {
            &mut adam_optimizer
        } else {
            &mut sgd_optimizer
        };

        // batched_x and batched_y are now single [B, N] matrices!
        for (batched_x, batched_y) in dataloader.iter() {
            current_optimizer.zero_grad();

            // 🎯 ONE forward pass for the entire batch
            let logits = model.forward(&batched_x).unwrap();
            
            // 🎯 ONE loss calculation (averaged across the batch)
            let loss = logits.cross_entropy_loss(&batched_y).unwrap();
            
            // 🎯 ONE backward pass that propagates matrix-level gradients
            loss.backward();
            
            // Step the optimizer
            current_optimizer.step();
            
            // Track loss metrics
            let current_batch_size = batched_x.data.borrow().shape[0];
            epoch_loss += loss.data.borrow().data[0] * current_batch_size as f64;
            total_samples += current_batch_size;
        }

        if epoch % 50 == 0 {
            let opt_name = if epoch <= 400 { "AdamW" } else { "SGD  " };
            println!("Epoch {:3} [{}]: Average Loss = {:.4}", epoch, opt_name, epoch_loss / total_samples as f64);
        }
    }
}