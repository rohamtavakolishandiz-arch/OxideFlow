// src/dqn.rs
use crate::autograd::Variable;
use crate::backend::{Backend, WgpuBackend};
use crate::nn::{Linear, Module, ReLU, Sequential};
use crate::optimizer::{AdamW, Optimizer};
use crate::tensor::Tensor;
use crate::replay::Transition;
use crate::config::NetworkConfig;
use anyhow::{Context, Result};

pub fn build_dqn_model(device: WgpuBackend, config: &NetworkConfig) -> Sequential<WgpuBackend> {
    Sequential::new(vec![
        Box::new(Linear::new(device.clone(), config.input_size, config.hidden_size, false)) as Box<dyn Module<WgpuBackend>>,
        Box::new(ReLU),
        Box::new(Linear::new(device.clone(), config.hidden_size, config.hidden_size, false)),
        Box::new(ReLU),
        Box::new(Linear::new(device, config.hidden_size, config.output_size, false)),
    ])
}

pub fn sync_target_network(
    device: &WgpuBackend,
    primary: &Sequential<WgpuBackend>,
    target: &Sequential<WgpuBackend>,
) {
    let p_params = primary.parameters();
    let t_params = target.parameters();
    for (p, t) in p_params.iter().zip(t_params.iter()) {
        let cpu_weights = WgpuBackend::to_cpu(device, &p.data.borrow().data);
        t.data.borrow_mut().data = WgpuBackend::from_data(device, cpu_weights, &p.data.borrow().shape);
    }
}

pub fn train_bellman_step(
    device: &WgpuBackend,
    model: &Sequential<WgpuBackend>,
    target_model: &Sequential<WgpuBackend>,
    optimizer: &mut AdamW<WgpuBackend>,
    batch: &[Transition],
    gamma: f64,
) -> Result<()> { // NEW: Returns a Result
    let batch_size = batch.len();
    let mut state_batch = Vec::with_capacity(batch_size * 5);
    let mut next_state_batch = Vec::with_capacity(batch_size * 5);
    
    for t in batch {
        state_batch.extend_from_slice(&t.state);
        next_state_batch.extend_from_slice(&t.next_state);
    }

    // NEW: Using `?` instead of `.unwrap()` safely passes errors up the chain
    let state_var_batch = Variable::new(Tensor::from_data(device.clone(), state_batch, vec![batch_size, 5])
        .context("Failed to create state batch tensor. Check input sizes.")?);
        
    let next_state_var_batch = Variable::new(Tensor::from_data(device.clone(), next_state_batch, vec![batch_size, 5])
        .context("Failed to create next_state batch tensor.")?);

    let q_pred = model.forward(&state_var_batch, true)
        .context("Forward pass failed on primary model.")?;
        
    let q_next = target_model.forward(&next_state_var_batch, false)
        .context("Forward pass failed on target model.")?;

    let mut target_q_data = WgpuBackend::to_cpu(device, &q_pred.data.borrow().data);
    let next_q_data = WgpuBackend::to_cpu(device, &q_next.data.borrow().data);

    for i in 0..batch_size {
        let t = &batch[i];
        let max_next_q = next_q_data[i * 2].max(next_q_data[i * 2 + 1]);
        let target_q = if t.done { t.reward as f64 } else { t.reward as f64 + (gamma * max_next_q) };
        target_q_data[i * 2 + t.action] = target_q;
    }

    let target_var = Variable::new_constant(Tensor::from_data(device.clone(), target_q_data, vec![batch_size, 2])?);
    let loss = q_pred.mse_loss(&target_var)?;
    
    optimizer.zero_grad();
    loss.backward()?;
    optimizer.step()?;
    
    Ok(()) // NEW: Tell Rust the math succeeded!
}