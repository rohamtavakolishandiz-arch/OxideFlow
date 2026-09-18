use oxide_flow::autograd::Variable;
use oxide_flow::nn::{Conv2d, MaxPool2d, Flatten, Linear, Sequential, Module, ReLU, Dropout}; 
use oxide_flow::optimizer::{AdamW, Optimizer};
use oxide_flow::data::DataLoader;
use oxide_flow::cifar::CifarDataset;
use std::time::Instant;

fn main() {
    let model_path = "cifar10_model.json";
    let optim_path = "cifar10_optim.json";
    
    println!("--- Phase 4 Very Hard Test: CIFAR-10 Multi-Threaded CNN ---");
    
    println!("Loading CIFAR-10 Binaries (3-Channel RGB)...");
    // Load 20,000 training images
    let train_data = CifarDataset::load(&[
        "cifar-10-batches-bin/data_batch_1.bin",
        "cifar-10-batches-bin/data_batch_2.bin",
    ]).unwrap();
    
    // Load 10,000 test images
    let test_data = CifarDataset::load(&[
        "cifar-10-batches-bin/test_batch.bin"
    ]).unwrap();

    println!(">> Initializing OxideFlow Deep Vision Architecture...");
    let model = Sequential::new(vec![
        // Block 1: [Batch, 3, 32, 32] -> [Batch, 16, 16, 16]
        Box::new(Conv2d::new(3, 16, 3, 1, 1)),
        Box::new(ReLU),
        Box::new(Conv2d::new(16, 16, 3, 1, 1)),
        Box::new(ReLU),
        Box::new(MaxPool2d { kernel_size: 2 }), // 32x32 shrinks to 16x16
        Box::new(Dropout { p: 0.2 }),
        
        // Block 2: [Batch, 16, 16, 16] -> [Batch, 32, 8, 8]
        Box::new(Conv2d::new(16, 32, 3, 1, 1)),
        Box::new(ReLU),
        Box::new(Conv2d::new(32, 32, 3, 1, 1)),
        Box::new(ReLU),
        Box::new(MaxPool2d { kernel_size: 2 }), // 16x16 shrinks to 8x8
        Box::new(Dropout { p: 0.3 }),
        
        // Flatten 32 channels * 8 * 8 = 2048 features
        Box::new(Flatten),
        
        // Classification Head
        Box::new(Linear::new(2048, 256)),
        Box::new(ReLU),
        Box::new(Dropout { p: 0.4 }),
        Box::new(Linear::new(256, 10)),
    ]);
    
    let mut adam_optimizer = AdamW::new(model.parameters(), 0.001, 0.0);

    let train_loader = DataLoader::new(train_data.inputs, train_data.targets, 64, true);
    let test_loader = DataLoader::new(test_data.inputs, test_data.targets, 100, false);
    
    let epochs = 25;
    let mut best_test_acc = 0.0;

    for epoch in 1..=epochs {
        println!("\n[Epoch {}] Training Spatial Filters...", epoch);
        let start_time = Instant::now();
        
        let mut train_loss = 0.0;
        let mut train_samples = 0;
        
        for (batched_x, batched_y) in train_loader.iter() {
            adam_optimizer.zero_grad();
            let logits = model.forward(&batched_x, true).unwrap();
            let loss = logits.cross_entropy_loss(&batched_y).unwrap();
            
            loss.backward();
            adam_optimizer.clip_grads(1.0);
            adam_optimizer.step();
            
            let batch_size = batched_x.data.borrow().shape[0];
            train_loss += loss.data.borrow().data[0] * batch_size as f64;
            train_samples += batch_size;
            
            // Print progress indicator (convolutions take heavy compute!)
            if train_samples % 1600 == 0 { 
                print!("█"); std::io::Write::flush(&mut std::io::stdout()).unwrap(); 
            }
        }

        let mut correct = 0;
        let mut test_samples = 0;

        for (batched_x, batched_y) in test_loader.iter() {
            let logits = model.forward(&batched_x, false).unwrap();
            let p_data = logits.data.borrow();
            let t_data = batched_y.data.borrow();
            let batch_size = batched_x.data.borrow().shape[0];
            test_samples += batch_size;

            for i in 0..batch_size {
                let (mut max_p_idx, mut max_p_val) = (0, f64::NEG_INFINITY);
                let (mut max_t_idx, mut max_t_val) = (0, f64::NEG_INFINITY);
                for j in 0..10 {
                    if p_data.data[i * 10 + j] > max_p_val { max_p_val = p_data.data[i * 10 + j]; max_p_idx = j; }
                    if t_data.data[i * 10 + j] > max_t_val { max_t_val = t_data.data[i * 10 + j]; max_t_idx = j; }
                }
                if max_p_idx == max_t_idx { correct += 1; }
            }
        }

        let test_acc = (correct as f64 / test_samples as f64) * 100.0;
        let elapsed = start_time.elapsed().as_secs();
        
        print!("\n> Train Loss: {:.4} | Test Accuracy: {:.2}% | Time: {}s", 
               train_loss / train_samples as f64, test_acc, elapsed);

        if test_acc > best_test_acc {
            best_test_acc = test_acc;
            model.save(model_path).unwrap();
            adam_optimizer.save(optim_path).unwrap();
            println!(" 🌟 [New Best CIFAR-10 Model!]");
        } else {
            println!();
        }
    }
}