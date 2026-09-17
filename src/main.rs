use oxide_flow::autograd::Variable;
use oxide_flow::nn::{Linear, Sequential, Module, ReLU, Residual}; 
use oxide_flow::optimizer::{AdamW, SGD, Optimizer};
use oxide_flow::tensor::Tensor;
use oxide_flow::data::DataLoader;
use std::f64::consts::PI;
use rand::Rng;
use plotters::prelude::*;
use std::path::Path;
use std::fs;
use std::process::Command;

/// Generates the complex Two Spirals dataset for testing highly non-linear classification
fn generate_spirals(n_samples: usize) -> (Vec<Variable>, Vec<Variable>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..n_samples {
        let class_label = i % 2;
        
        let n = (i / 2) as f64 / (n_samples / 2) as f64; 
        let theta = 3.0 * PI * n.sqrt() + (class_label as f64 * PI);
        let r = 2.0 * theta + PI;
        
        let x = (r * theta.cos() + rng.gen_range(-1.0..1.0)) * 0.1;
        let y = (r * theta.sin() + rng.gen_range(-1.0..1.0)) * 0.1;

        inputs.push(Variable::new(Tensor::from_data(vec![x, y], vec![1, 2]).unwrap()));
        
        let target_vec = if class_label == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
        targets.push(Variable::new(Tensor::from_data(target_vec, vec![1, 2]).unwrap()));
    }
    
    (inputs, targets)
}

/// Generates Level 2: Concentric Circles
fn generate_circles(n_samples: usize) -> (Vec<Variable>, Vec<Variable>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..n_samples {
        let class_label = i % 2;
        
        // Class 0: Inner circle (radius 0.0 to 0.4)
        // Class 1: Outer ring (radius 0.7 to 1.0)
        let radius = if class_label == 0 {
            rng.gen_range(0.0..0.4)
        } else {
            rng.gen_range(0.7..1.0)
        };
        
        let angle = rng.gen_range(0.0..(2.0 * std::f64::consts::PI));
        
        let x = radius * angle.cos();
        let y = radius * angle.sin();

        inputs.push(Variable::new(Tensor::from_data(vec![x, y], vec![1, 2]).unwrap()));
        
        let target_vec = if class_label == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
        targets.push(Variable::new(Tensor::from_data(target_vec, vec![1, 2]).unwrap()));
    }
    
    (inputs, targets)
}

/// Generates Level 3: Interlocking Half-Moons
fn generate_moons(n_samples: usize) -> (Vec<Variable>, Vec<Variable>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..n_samples {
        let class_label = i % 2;
        
        // Generate a random angle between 0 and PI
        let angle = rng.gen_range(0.0..std::f64::consts::PI);
        
        // Add random Gaussian-like noise to make the moons thick
        let noise_x = rng.gen_range(-0.1..0.1);
        let noise_y = rng.gen_range(-0.1..0.1);

        let (mut x, mut y) = (angle.cos(), angle.sin());

        // Shift the second moon down and to the right, and flip it upside down
        if class_label == 1 {
            x = 1.0 - x;
            y = 0.5 - y;
        }

        // Apply noise and scale down slightly to fit the plotter
        x = (x + noise_x) * 0.5;
        y = (y + noise_y) * 0.5;

        inputs.push(Variable::new(Tensor::from_data(vec![x, y], vec![1, 2]).unwrap()));
        
        let target_vec = if class_label == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
        targets.push(Variable::new(Tensor::from_data(target_vec, vec![1, 2]).unwrap()));
    }
    
    (inputs, targets)
}

/// Generates Level 1: Linearly Separable Blobs
fn generate_blobs(n_samples: usize) -> (Vec<Variable>, Vec<Variable>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..n_samples {
        let class_label = i % 2;
        
        // Class 0 centers around (-0.5, -0.5), Class 1 centers around (0.5, 0.5)
        let center_x = if class_label == 0 { -0.5 } else { 0.5 };
        let center_y = if class_label == 0 { -0.5 } else { 0.5 };
        
        // Add random uniform noise to scatter the points into a "blob"
        let x = center_x + rng.gen_range(-0.3..0.3);
        let y = center_y + rng.gen_range(-0.3..0.3);

        inputs.push(Variable::new(Tensor::from_data(vec![x, y], vec![1, 2]).unwrap()));
        
        let target_vec = if class_label == 0 { vec![1.0, 0.0] } else { vec![0.0, 1.0] };
        targets.push(Variable::new(Tensor::from_data(target_vec, vec![1, 2]).unwrap()));
    }
    
    (inputs, targets)
}

fn visualize_decision_boundary(
    model: &Sequential, 
    inputs: &[Variable], 
    targets: &[Variable],
    epoch: usize
) -> Result<(), Box<dyn std::error::Error>> {
    
    // Ensure the frames directory exists
    fs::create_dir_all("frames")?;

    // Format the filename so it sorts alphabetically (e.g., frame_00100.png)
    let filename = format!("frames/frame_{:05}.png", epoch);
    
    let root = BitMapBackend::new(&filename, (800, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(format!("Decision Boundary - Epoch {}", epoch), ("sans-serif", 30))
        .margin(10)
        .build_cartesian_2d(-1.5f64..1.5f64, -1.5f64..1.5f64)?;

    chart.configure_mesh().draw()?;

    let grid_size = 200;
    for i in 0..grid_size {
        for j in 0..grid_size {
            let x = -1.5 + (3.0 * i as f64 / grid_size as f64);
            let y = -1.5 + (3.0 * j as f64 / grid_size as f64);
            let grid_point = Variable::new(Tensor::from_data(vec![x, y], vec![1, 2]).unwrap());
            
            if let Ok(pred) = model.forward(&grid_point) {
                let p_data = pred.data.borrow();
                let color = if p_data.data[0] > p_data.data[1] {
                    RGBColor(200, 220, 255) 
                } else {
                    RGBColor(255, 200, 200) 
                };
                chart.draw_series(std::iter::once(Rectangle::new(
                    [(x, y), (x + 0.015, y + 0.015)],
                    color.filled(),
                )))?;
            }
        }
    }

    for (input, target) in inputs.iter().zip(targets.iter()) {
        let x = input.data.borrow().data[0];
        let y = input.data.borrow().data[1];
        let is_class_0 = target.data.borrow().data[0] > 0.5;
        let color = if is_class_0 { &BLUE } else { &RED };
        chart.draw_series(std::iter::once(Circle::new((x, y), 3, color.filled())))?;
    }

    root.present()?;
    Ok(())
}

fn main() {
    let model_path = "oxide_model.json";
    let optim_path = "oxide_optim.json";
    
    println!("--- Starting OxideFlow Two Spirals Challenge ---");
    let (inputs, targets) = generate_spirals(1000);

    let mut model;
    let mut adam_optimizer;
    
    // 1. Move frame_count initialization to the top
    let mut frame_count = 1; 

    if Path::new(model_path).exists() && Path::new(optim_path).exists() {
        println!(">> Loading existing model and optimizer state from disk...");
        model = Sequential::load(model_path).unwrap();
        
        adam_optimizer = AdamW::new(model.parameters(), 0.005, 0.0);
        adam_optimizer.load(optim_path).unwrap();
        
        // 2. Count existing frames to pick up exactly where the last run left off
        if let Ok(entries) = std::fs::read_dir("frames") {
            frame_count = entries.filter_map(Result::ok).count() + 1;
        }
    } else {
        println!(">> No checkpoint found. Initializing a fresh Deep ResNet...");
        
        // 3. Only wipe the frames folder if we are starting completely fresh
        std::fs::remove_dir_all("frames").ok(); 
        std::fs::create_dir_all("frames").unwrap();
        
        model = Sequential::new(vec![
            Box::new(Linear::new(2, 128)),
            Box::new(ReLU),
            Box::new(Residual::new(vec![
                Box::new(Linear::new(128, 128)),
                Box::new(ReLU),
                Box::new(Linear::new(128, 128)),
                Box::new(ReLU),
            ])),
            Box::new(Residual::new(vec![
                Box::new(Linear::new(128, 128)),
                Box::new(ReLU),
                Box::new(Linear::new(128, 128)),
                Box::new(ReLU),
            ])),
            Box::new(Linear::new(128, 2)),
        ]);
        adam_optimizer = AdamW::new(model.parameters(), 0.005, 0.0);
    }

    let dataloader = DataLoader::new(inputs.clone(), targets.clone(), 32, true);
    
    // You can lower this to 1000 or 2000 per run now that you can resume it!
    let epochs = 5000;

    println!("Training OxideFlow on Two Spirals...");
    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0;
        let mut total_samples = 0;

        for (batched_x, batched_y) in dataloader.iter() {
            adam_optimizer.zero_grad();

            let logits = model.forward(&batched_x).unwrap();
            let loss = logits.cross_entropy_loss(&batched_y).unwrap();
            
            loss.backward();
            
            adam_optimizer.clip_grads(1.0);
            adam_optimizer.step();
            
            let current_batch_size = batched_x.data.borrow().shape[0];
            epoch_loss += loss.data.borrow().data[0] * current_batch_size as f64;
            total_samples += current_batch_size;
        }

        // Print loss frequently, but only save frames every 50 epochs
        if epoch % 50 == 0 {
            println!("Epoch {:4} [AdamW]: Average Loss = {:.4}", epoch, epoch_loss / total_samples as f64);
        }

        // Save a frame every 50 epochs (yields exactly 100 frames for a 10-second video)
        if epoch % 50 == 0 {
            println!("Saving frame {}...", frame_count);
            visualize_decision_boundary(&model, &inputs, &targets, frame_count).unwrap();
            frame_count += 1;
        }

        // Slow down the learning rate decay so it doesn't stall out early
        if epoch % 500 == 0 {
            adam_optimizer.decay_lr(0.9);
            println!(">> Decayed learning rate by 10% at epoch {}", epoch);
        }
    }

    // 🎯 NEW: Save the progress so we can pick up where we left off!
    println!("Saving checkpoints...");
    model.save(model_path).unwrap();
    adam_optimizer.save(optim_path).unwrap();

    let status = Command::new("ffmpeg")
        .args([
            "-y", 
            "-framerate", "10", 
            // 🎯 FIX 2: Use standard sequential formatting instead of glob
            "-i", "frames/frame_%05d.png", 
            "-c:v", "libx264", 
            "-pix_fmt", "yuv420p", 
            "training_evolution.mp4" 
        ])
        .status();

    match status {
        Ok(s) if s.success() => println!("Successfully created training_evolution.mp4!"),
        _ => println!("Failed to create MP4. Make sure FFmpeg is installed and added to your PATH."),
    }
}