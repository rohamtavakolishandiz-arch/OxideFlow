// src/main.rs
use std::sync::Arc;
use rand::Rng;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use oxide_flow::config::TrainingConfig;
use oxide_flow::env::Environment;
use oxide_flow::dashboard::Dashboard;

use oxide_flow::autograd::Variable;
use oxide_flow::backend::{Backend, WgpuBackend};
use oxide_flow::checkpoint::save_state_dict;
use oxide_flow::optimizer::AdamW;
use oxide_flow::renderer::GpuRenderer;
use oxide_flow::tensor::Tensor;
use oxide_flow::nn::Module;

use oxide_flow::replay::{ReplayBuffer, Transition};
use oxide_flow::dqn::{build_dqn_model, sync_target_network, train_bellman_step};
use oxide_flow::visuals::render_frame;
use oxide_flow::flappy::FlappyEnv;

use clap::{Parser, ValueEnum};
use oxide_flow::checkpoint::load_state_dict;

#[derive(Parser, Debug)]
#[command(author, version, about = "OxideFlow - Custom Deep Learning Framework")]
struct Args {
    /// The mode to run the engine in (train or eval)
    #[arg(short, long, default_value = "train")]
    mode: Mode,

    /// Path to weights file (used in eval mode)
    #[arg(short, long, default_value = "flappy_checkpoint.bin")]
    weights: String,

    /// Rendering quality (full, low, headless)
    #[arg(short, long, default_value = "full")]
    render: RenderMode,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum Mode {
    Train,
    Eval,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum RenderMode {
    Full,
    Low,
    Headless,
}



fn main() {
    // 1. Parse CLI Arguments
    let args = Args::parse();
    println!(">> Starting OxideFlow in {:?} mode (Render: {:?})...", args.mode, args.render);

    // 2. Load Config
    let config_data = TrainingConfig::load_or_default("config.toml");

    // 3. Setup Window & WGPU
    let mut dashboard = if args.render == RenderMode::Headless {
        Some(Dashboard::new())
    } else {
        None
    };

    let event_loop = EventLoop::new().unwrap();
    
    // Create window, but keep it invisible if the user wants Headless mode!
    let window = Arc::new(WindowBuilder::new()
        .with_title(format!("OxideFlow - {:?}", args.mode))
        .with_inner_size(winit::dpi::LogicalSize::new(1200, 600))
        .with_visible(args.render != RenderMode::Headless) 
        .build(&event_loop).unwrap());
    
    let (device, surface, mut config) = WgpuBackend::with_surface(window.clone());
    let renderer = GpuRenderer::new(&device.device, config.format);

    // 2. Setup RL Environment & Brain
    let model = build_dqn_model(device.clone(), &config_data.network);
    let target_model = build_dqn_model(device.clone(), &config_data.network);
    
    // --> NEW: Safe Weight Loading
    if args.mode == Mode::Eval {
        println!(">> 🧠 Loading neural network weights from: {}", args.weights);
        if let Err(e) = load_state_dict(&device, &model.parameters(), &args.weights) {
            eprintln!("\n>> ❌ OXIDEFLOW ERROR: Failed to load weights.");
            eprintln!(">> Details: {}", e);
            eprintln!(">> Hint: Did you run `--mode train` first to generate the file?");
            std::process::exit(1);
        }
    }

    let mut optimizer = AdamW::new(model.parameters(), config_data.hyperparams.learning_rate, 0.0);
    
    let mut env = FlappyEnv::new();
    let mut memory = ReplayBuffer::new(config_data.hyperparams.memory_capacity);
    
    let batch_size = config_data.hyperparams.batch_size;
    let gamma = config_data.hyperparams.gamma;
    let mut epsilon = config_data.hyperparams.epsilon_start;
    let epsilon_min = config_data.hyperparams.epsilon_min;
    let epsilon_decay = config_data.hyperparams.epsilon_decay;
    
    let mut state = env.get_state();
    let mut steps = 0;
    let mut episode = 1;
    let mut rng = rand::thread_rng();

    // 3. The Main Lifecycle
    let _ = event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
    
        match event {
            Event::WindowEvent { event: WindowEvent::Resized(new_size), .. } => {
                if new_size.width > 0 && new_size.height > 0 {
                    config.width = new_size.width;
                    config.height = new_size.height;
                    surface.configure(&device.device, &config);
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                let _ = save_state_dict(&device, &model.parameters(), "flappy_final.bin");
                elwt.exit();
            }
            
            // NEW: Listen for the 'Q' key when the visual window is focused
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput {
                    event: winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(winit::keyboard::KeyCode::KeyQ),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                    ..
                },
                ..
            } => {
                println!("\n>> 'Q' pressed! Safely saving neural weights and shutting down...");
                let _ = save_state_dict(&device, &model.parameters(), "flappy_checkpoint.bin");
                elwt.exit();
            }
            
            Event::AboutToWait => {
                if let Some(dash) = &dashboard {
                    if dash.check_quit() {
                        let _ = save_state_dict(&device, &model.parameters(), "flappy_checkpoint.bin");
                        elwt.exit();
                        return;
                    }
                }
                
                // A. OBSERVE & ACT
                let state_var = Variable::new(Tensor::from_data(device.clone(), state.clone(), vec![1, 5]).unwrap());
                let q_data = WgpuBackend::to_cpu(&device, &model.forward(&state_var, false).unwrap().data.borrow().data);
                
                let action = if args.mode == Mode::Train && rng.gen_range(0.0..1.0) < epsilon { 
                    rng.gen_range(0..2) 
                } else if q_data[1] > q_data[0] { 1 } else { 0 };

                // ==========================================
                // B. EXPERIENCE
                // ==========================================
                // 1. Always step the game forward, no matter what mode we are in
                let (reward, is_alive) = env.step(action);
                let next_state = env.get_state();
                steps += 1;

                // 2. Only save memories if we are in Train mode
                if args.mode == Mode::Train {
                    memory.push(Transition { 
                        state: state.clone(), 
                        action, 
                        reward, 
                        next_state: next_state.clone(), 
                        done: !is_alive 
                    });
                }
                
                state = next_state;

                // ==========================================
                // C. LEARN
                // ==========================================
                if args.mode == Mode::Train && memory.is_ready(batch_size) {
                    let batch = memory.sample(batch_size);
                    
                    // NEW: Gracefully catch Bellman math or shape mismatch errors
                    if let Err(e) = train_bellman_step(&device, &model, &target_model, &mut optimizer, &batch, gamma) {
                        eprintln!("\n>> ❌ OXIDEFLOW FATAL ERROR: Bellman Backpropagation Failed!");
                        eprintln!(">> Details: {:?}", e);
                        eprintln!(">> Hint: If you changed network sizes in config.toml, delete your old .bin checkpoint file.");
                        elwt.exit();
                        return;
                    }
                }

                // ==========================================
                // D. ADAPT & RESET
                // ==========================================
                if !is_alive {
                    if args.mode == Mode::Train {
                        if epsilon > epsilon_min { epsilon *= epsilon_decay; }
                        
                        if episode % 10 == 0 {
                            sync_target_network(&device, &model, &target_model);
                            
                            // NEW: Only use println! if we are NOT in Headless mode
                            if dashboard.is_none() {
                                println!("Episode {:<4} | Score: {:<2} | Steps: {:<4} | Exploration: {:.1}%", episode, env.score(), steps, epsilon * 100.0);
                            }
                        }
                        
                        // NEW: Update the Dashboard every time the agent dies
                        if let Some(dash) = &mut dashboard {
                            dash.draw(episode, env.score(), epsilon, memory.len(), config_data.hyperparams.memory_capacity);
                        }

                        if episode % 100 == 0 {
                            let _ = save_state_dict(&device, &model.parameters(), "flappy_checkpoint.bin");
                        }
                    } else if dashboard.is_none() {
                        println!("Eval Episode {:<4} | Score: {:<2} | Survival Steps: {:<4}", episode, env.score(), steps);
                    }
                    
                    episode += 1;
                    steps = 0;
                    env.reset();
                    state = env.get_state();
                }

                // ==========================================
                // E. RENDER UI (Skip if Headless!)
                // ==========================================
                if args.render != RenderMode::Headless {
                    let is_low_quality = args.render == RenderMode::Low;
                    
                    let needs_reconfig = render_frame(
                        &device, &surface, &config, &renderer, &env, &state, 
                        &q_data, action, epsilon, memory.len(), batch_size, // 🎯 Changed to &q_data and action
                        is_low_quality 
                    );
                    
                    if needs_reconfig {
                        surface.configure(&device.device, &config);
                    }
                }
            }
            _ => {}
        }
    });
}