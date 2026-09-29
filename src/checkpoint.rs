// src/checkpoint.rs
use std::fs::File;
use std::io::{Result as IoResult, Error, ErrorKind};
use serde::{Serialize, Deserialize};
use crate::backend::Backend;
use crate::autograd::Variable;
use crate::tensor::Tensor;

// ==========================================
// SERDE DATA STRUCTURES
// ==========================================

#[derive(Serialize, Deserialize)]
pub struct CheckpointTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub version: u32,
    pub parameters: Vec<CheckpointTensor>,
}

#[derive(Serialize, Deserialize)]
pub struct AdamWCheckpoint {
    pub version: u32,
    pub t: usize,
    pub m_tensors: Vec<CheckpointTensor>,
    pub v_tensors: Vec<CheckpointTensor>,
}

// ==========================================
// MODEL WEIGHT CHECKPOINTING
// ==========================================

pub fn save_state_dict<B: Backend>(
    device: &B, 
    parameters: &[Variable<B>], 
    path: &str
) -> IoResult<()> {
    let mut checkpoint = ModelCheckpoint {
        version: 1, // Future-proofing format
        parameters: Vec::with_capacity(parameters.len()),
    };

    for param in parameters {
        let tensor = param.data.borrow();
        checkpoint.parameters.push(CheckpointTensor {
            shape: tensor.shape.clone(),
            data: B::to_cpu(device, &tensor.data),
        });
    }

    let file = File::create(path)?;
    bincode::serialize_into(file, &checkpoint)
        .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
    
    Ok(())
}

pub fn load_state_dict<B: Backend>(
    device: &B, 
    parameters: &[Variable<B>], 
    path: &str
) -> IoResult<()> {
    let file = File::open(path)?;
    let checkpoint: ModelCheckpoint = bincode::deserialize_from(file)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;

    if checkpoint.parameters.len() != parameters.len() {
        return Err(Error::new(ErrorKind::InvalidData, "Parameter count mismatch! Are you loading into the wrong model?"));
    }

    for (param, ckpt_tensor) in parameters.iter().zip(checkpoint.parameters.into_iter()) {
        let mut tensor = param.data.borrow_mut();
        
        if tensor.shape.len() != ckpt_tensor.shape.len() {
            return Err(Error::new(ErrorKind::InvalidData, "Dimension count mismatch! Corrupted file or wrong model."));
        }

        if tensor.shape != ckpt_tensor.shape {
            return Err(Error::new(ErrorKind::InvalidData, "Shape mismatch! The saved weights don't fit this layer."));
        }

        // Push directly back to VRAM
        tensor.data = B::from_data(device, ckpt_tensor.data, &ckpt_tensor.shape);
    }
    
    Ok(())
}

// ==========================================
// OPTIMIZER STATE CHECKPOINTING
// ==========================================

pub fn save_adamw_state<B: Backend>(
    device: &B, 
    t: usize,
    m_tensors: &[Tensor<B>], 
    v_tensors: &[Tensor<B>], 
    path: &str
) -> IoResult<()> {
    let mut m_ckpt = Vec::with_capacity(m_tensors.len());
    for tensor in m_tensors {
        m_ckpt.push(CheckpointTensor {
            shape: tensor.shape.clone(),
            data: B::to_cpu(device, &tensor.data),
        });
    }

    let mut v_ckpt = Vec::with_capacity(v_tensors.len());
    for tensor in v_tensors {
        v_ckpt.push(CheckpointTensor {
            shape: tensor.shape.clone(),
            data: B::to_cpu(device, &tensor.data),
        });
    }

    let checkpoint = AdamWCheckpoint {
        version: 1,
        t,
        m_tensors: m_ckpt,
        v_tensors: v_ckpt,
    };

    let file = File::create(path)?;
    bincode::serialize_into(file, &checkpoint)
        .map_err(|e| Error::new(ErrorKind::Other, e.to_string()))?;
    
    Ok(())
}

pub fn load_adamw_state<B: Backend>(
    device: &B, 
    t: &mut usize,
    m_tensors: &mut [Tensor<B>], 
    v_tensors: &mut [Tensor<B>], 
    path: &str
) -> IoResult<()> {
    let file = File::open(path)?;
    let checkpoint: AdamWCheckpoint = bincode::deserialize_from(file)
        .map_err(|e| Error::new(ErrorKind::InvalidData, e.to_string()))?;

    if checkpoint.m_tensors.len() != m_tensors.len() || checkpoint.v_tensors.len() != v_tensors.len() {
        return Err(Error::new(ErrorKind::InvalidData, "Optimizer parameter count mismatch! Are you loading into the wrong model?"));
    }

    *t = checkpoint.t;

    // Iterate through both m and v tensors simultaneously with the checkpoint data
    let tensors_iter = m_tensors.iter_mut().chain(v_tensors.iter_mut());
    let ckpt_tensors_iter = checkpoint.m_tensors.into_iter().chain(checkpoint.v_tensors.into_iter());

    for (tensor, ckpt_tensor) in tensors_iter.zip(ckpt_tensors_iter) {
        if tensor.shape.len() != ckpt_tensor.shape.len() {
            return Err(Error::new(ErrorKind::InvalidData, "Dimension count mismatch! Corrupted file or wrong model."));
        }

        if tensor.shape != ckpt_tensor.shape {
            return Err(Error::new(ErrorKind::InvalidData, "Shape mismatch! The saved optimizer states don't fit this layer."));
        }

        // Push directly back to VRAM
        tensor.data = B::from_data(device, ckpt_tensor.data, &ckpt_tensor.shape);
    }
    
    Ok(())
}