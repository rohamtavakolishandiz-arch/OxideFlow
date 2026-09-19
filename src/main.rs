use oxide_flow::backend::{Backend, WgpuBackend};
use oxide_flow::nn::{Module, Sequential, Linear, ReLU, Sigmoid};
use oxide_flow::optimizer::{Optimizer, AdamW};
use oxide_flow::autograd::Variable;
use oxide_flow::tensor::Tensor;
use oxide_flow::renderer::{GpuRenderer, Vertex};
use std::fs;
use std::time::Instant;

fn get_node_y(index: usize, total_nodes: usize, center_y: f32, spacing: f32) -> f32 {
    let total_height = ((total_nodes - 1) as f32) * spacing;
    let start_y = center_y - (total_height / 2.0);
    start_y + (index as f32 * spacing)
}

fn main() {
    println!("==================================================");
    println!("🎥 OXIDEFLOW: 1080p GPU ARCHITECT");
    println!("==================================================");

    let device = WgpuBackend::new();
    fs::create_dir_all("frames").expect("Failed to create frames");

    // Initialize the new HD Hardware Renderer!
    let width = 1920.0;
    let height = 1080.0;
    let renderer = GpuRenderer::new(device.device.clone(), device.queue.clone(), width as u32, height as u32);

    let batch_size = 64;
    let mut inputs_vec = Vec::with_capacity(batch_size * 6);
    let mut targets_vec = Vec::with_capacity(batch_size * 5);

    for _ in 0..batch_size {
        let v: Vec<f64> = (0..6).map(|_| (rand::random::<f64>() * 2.0) - 1.0).collect();
        inputs_vec.extend_from_slice(&v);
        targets_vec.push((v[0] * v[1]).sin().abs()); 
        targets_vec.push((v[2] + v[3]).cos().abs()); 
        targets_vec.push((v[4] * v[5]).tanh().abs()); 
        targets_vec.push(v[0].max(v[5]).abs());
        targets_vec.push((v[1] - v[4]).abs() / 2.0);
    }

    let inputs = Variable::new(Tensor { data: WgpuBackend::from_data(&device, inputs_vec, &[batch_size, 6]), shape: vec![batch_size, 6], device: device.clone() });
    let targets = Variable::new(Tensor { data: WgpuBackend::from_data(&device, targets_vec, &[batch_size, 5]), shape: vec![batch_size, 5], device: device.clone() });

    let model = Sequential::new(vec![
        Box::new(Linear::new(device.clone(), 6, 10)),
        Box::new(ReLU),
        Box::new(Linear::new(device.clone(), 10, 10)),
        Box::new(ReLU),
        Box::new(Linear::new(device.clone(), 10, 5)),
        Box::new(Sigmoid),
    ]);

    let mut optimizer = AdamW::new(model.parameters(), 0.01, 0.0);

    println!(">> 🚀 Igniting HD Render Pipeline...");
    let total_epochs = 5000;
    let mut frame_idx = 0;

    for epoch in 1..=total_epochs {
        let start = Instant::now();
        optimizer.zero_grad();

        let preds = model.forward(&inputs, true).unwrap();
        let loss = preds.mse_loss(&targets).unwrap(); 
        loss.backward();
        optimizer.step();

        if epoch % 5 == 0 || epoch == 1 {
            let params = model.parameters();
            let w1 = WgpuBackend::to_cpu(&device, &params[0].data.borrow().data);
            let w2 = WgpuBackend::to_cpu(&device, &params[2].data.borrow().data);
            let w3 = WgpuBackend::to_cpu(&device, &params[4].data.borrow().data);

            let mut vertices: Vec<Vertex> = Vec::new();

            // Screen Layout adjustments for 1080p
            let layer_xs = [400.0, 773.0, 1146.0, 1520.0];
            let layer_sizes = [6, 10, 10, 5];
            let spacing = 80.0;
            let center_y = 540.0; // Half of 1080

            // 1. QUEUE STRINGS
            let weights = [&w1, &w2, &w3];
            for l in 0..3 {
                let in_nodes = layer_sizes[l];
                let out_nodes = layer_sizes[l + 1];
                let x0 = layer_xs[l];
                let x1 = layer_xs[l + 1];

                for i in 0..in_nodes {
                    for j in 0..out_nodes {
                        let y0 = get_node_y(i, in_nodes, center_y, spacing);
                        let y1 = get_node_y(j, out_nodes, center_y, spacing);
                        
                        let weight_mag = weights[l][i * out_nodes + j].abs() as f32;
                        
                        if weight_mag > 0.05 {
                            let activity = (weight_mag * 1.5).clamp(0.0, 1.0);
                            let r = activity;
                            let g = (1.0 - activity) * 0.8 + 0.1;
                            
                            // Line thickness dynamically scales slightly with weight strength!
                            let thickness = 2.0 + (activity * 3.0);
                            GpuRenderer::push_line(&mut vertices, x0, y0, x1, y1, thickness, [r, g, 0.0, 1.0], width, height);
                        }
                    }
                }
            }

            // 2. QUEUE NODES
            for l in 0..4 {
                let nodes = layer_sizes[l];
                let x = layer_xs[l];
                for i in 0..nodes {
                    let y = get_node_y(i, nodes, center_y, spacing);
                    // Radius = 20 pixels, neon green
                    GpuRenderer::push_node(&mut vertices, x, y, 20.0, [0.0, 1.0, 0.4, 1.0], width, height); 
                }
            }

            // 3. EXECUTE GPU RENDER PASS
            let frame = renderer.render_frame(&vertices);
            frame.save(format!("frames/architect_{:04}.png", frame_idx)).unwrap();
            
            let loss_val = WgpuBackend::to_cpu(&device, &loss.data.borrow().data)[0];
            println!("Epoch {} | Loss: {:.5} | Time: {:.2}s", epoch, loss_val, start.elapsed().as_secs_f64());
            frame_idx += 1;
        }
    }
    println!(">> ✅ 1080p RENDER FINISHED!");
}