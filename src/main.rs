use oxide_flow::backend::{Backend, WgpuBackend};
use oxide_flow::nn::{Module, Sequential, Linear, ReLU, Sigmoid};
use oxide_flow::optimizer::{Optimizer, AdamW};
use oxide_flow::autograd::Variable;
use oxide_flow::tensor::Tensor;
use oxide_flow::renderer::GpuRenderer;
use std::sync::Arc;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn get_node_y(index: usize, total_nodes: usize, center_y: f32, spacing: f32) -> f32 {
    let total_height = ((total_nodes - 1) as f32) * spacing;
    let start_y = center_y - (total_height / 2.0);
    start_y + (index as f32 * spacing)
}

fn main() {
    println!("==================================================");
    println!("🎥 OXIDEFLOW: LIVE INTERACTIVE RENDERER");
    println!("==================================================");

    // 1. Initialize Window & Event Loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll); // Poll keeps the loop spinning as fast as possible
    
    let window = Arc::new(WindowBuilder::new()
        .with_title("OxideFlow Live Topology")
        .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080))
        .build(&event_loop).unwrap());

    // 2. Initialize VRAM Backend linked to the Window
    let (device, surface, mut config) = WgpuBackend::with_surface(window.clone());
    let renderer = GpuRenderer::new(&device.device, config.format);

    // 3. Build Neural Network 
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

    let inputs = Variable::new(Tensor::from_data(device.clone(), inputs_vec, vec![batch_size, 6]).unwrap());
    let targets = Variable::new(Tensor::from_data(device.clone(), targets_vec, vec![batch_size, 5]).unwrap());

    let model = Sequential::new(vec![
        Box::new(Linear::new(device.clone(), 6, 10)), Box::new(ReLU),
        Box::new(Linear::new(device.clone(), 10, 10)), Box::new(ReLU),
        Box::new(Linear::new(device.clone(), 10, 5)), Box::new(Sigmoid),
    ]);

    let mut optimizer = AdamW::new(model.parameters(), 0.01, 0.0);
    let mut epoch = 0;

    // 4. Enter Live Training & Rendering Loop
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(physical_size) => {
                    if physical_size.width > 0 && physical_size.height > 0 {
                        config.width = physical_size.width;
                        config.height = physical_size.height;
                        surface.configure(&device.device, &config);
                    }
                    config.width = physical_size.width;
                    config.height = physical_size.height;
                    surface.configure(&device.device, &config);
                }
                WindowEvent::RedrawRequested => {
                    // -> TRAIN ONE STEP <-
                    epoch += 1;
                    optimizer.zero_grad();
                    let preds = model.forward(&inputs, true).unwrap();
                    let loss = preds.mse_loss(&targets).unwrap(); 
                    loss.backward();
                    optimizer.step();

                    if epoch % 50 == 0 {
                        let loss_val = WgpuBackend::to_cpu(&device, &loss.data.borrow().data)[0];
                        println!("Epoch {} | Live Loss: {:.5}", epoch, loss_val);
                    }

                    // -> EXTRACT WEIGHTS <-
                    let params = model.parameters();
                    let w1 = WgpuBackend::to_cpu(&device, &params[0].data.borrow().data);
                    let w2 = WgpuBackend::to_cpu(&device, &params[2].data.borrow().data);
                    let w3 = WgpuBackend::to_cpu(&device, &params[4].data.borrow().data);

                    // -> BUILD VISUAL TOPOLOGY <-
                    let mut vertices = Vec::new();
                    let w = config.width as f32;
                    let h = config.height as f32;
                    let layer_xs = [w * 0.2, w * 0.4, w * 0.6, w * 0.8];
                    let layer_sizes = [6, 10, 10, 5];
                    let spacing = h * 0.08;
                    let center_y = h / 2.0;

                    let weights = [&w1, &w2, &w3];
                    for l in 0..3 {
                        let in_nodes = layer_sizes[l];
                        let out_nodes = layer_sizes[l + 1];
                        for i in 0..in_nodes {
                            for j in 0..out_nodes {
                                let y0 = get_node_y(i, in_nodes, center_y, spacing);
                                let y1 = get_node_y(j, out_nodes, center_y, spacing);
                                let weight_mag = weights[l][i * out_nodes + j].abs() as f32;
                                
                                if weight_mag > 0.05 {
                                    let activity = (weight_mag * 1.5).clamp(0.0, 1.0);
                                    let thickness = 2.0 + (activity * 3.0);
                                    GpuRenderer::push_line(&mut vertices, layer_xs[l], y0, layer_xs[l + 1], y1, thickness, [activity, (1.0 - activity) * 0.8 + 0.1, 0.0, 1.0], w, h);
                                }
                            }
                        }
                    }

                    for l in 0..4 {
                        let nodes = layer_sizes[l];
                        for i in 0..nodes {
                            let y = get_node_y(i, nodes, center_y, spacing);
                            GpuRenderer::push_node(&mut vertices, layer_xs[l], y, 20.0, [0.0, 1.0, 0.4, 1.0], w, h); 
                        }
                    }

                    // -> RENDER TO SCREEN <-
                    let frame = match surface.get_current_texture() {
                        // Extract the frame if it was successful or suboptimal
                        wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                        // If the surface was lost, outdated, or timed out, just skip this frame
                        _ => return, 
                    };
                    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                    
                    renderer.render(&device.device, &device.queue, &view, &vertices);
                    
                    // In wgpu v30+, presentation is routed through the Queue rather than the frame
                    device.queue.present(frame);
                }
                _ => {}
            },
            Event::AboutToWait => {
                // Request the next frame immediately for 60FPS
                window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}