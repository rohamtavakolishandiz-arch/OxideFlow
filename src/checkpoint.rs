// src/checkpoint.rs
use std::fs::File;
use std::io::{Read, Write, Result as IoResult, Error, ErrorKind};
use crate::backend::Backend;
use crate::autograd::Variable;

pub fn save_state_dict<B: Backend>(
    device: &B, 
    parameters: &[Variable<B>], 
    path: &str
) -> IoResult<()> {
    let mut file = File::create(path)?;
    
    // 1. Write the total number of parameter tensors
    let num_params = parameters.len() as u64;
    file.write_all(&num_params.to_le_bytes())?;

    for param in parameters {
        let tensor = param.data.borrow();
        
        // 2. Write the Shape (Dimensions)
        let dims = tensor.shape.len() as u64;
        file.write_all(&dims.to_le_bytes())?;
        for &dim in &tensor.shape {
            file.write_all(&(dim as u64).to_le_bytes())?;
        }

        // 3. Pull VRAM data to CPU and write the raw float bytes
        let cpu_data = B::to_cpu(device, &tensor.data);
        for &val in &cpu_data {
            file.write_all(&val.to_le_bytes())?;
        }
    }
    
    Ok(())
}

pub fn load_state_dict<B: Backend>(
    device: &B, 
    parameters: &[Variable<B>], 
    path: &str
) -> IoResult<()> {
    let mut file = File::open(path)?;
    
    let mut num_params_bytes = [0u8; 8];
    file.read_exact(&mut num_params_bytes)?;
    let num_params = u64::from_le_bytes(num_params_bytes) as usize;

    if num_params != parameters.len() {
        return Err(Error::new(ErrorKind::InvalidData, "Parameter count mismatch! Are you loading into the wrong model?"));
    }

    for param in parameters {
        let mut dims_bytes = [0u8; 8];
        file.read_exact(&mut dims_bytes)?;
        let dims = u64::from_le_bytes(dims_bytes) as usize;

        let mut shape = Vec::with_capacity(dims);
        for _ in 0..dims {
            let mut dim_bytes = [0u8; 8];
            file.read_exact(&mut dim_bytes)?;
            shape.push(u64::from_le_bytes(dim_bytes) as usize);
        }

        let mut tensor = param.data.borrow_mut();
        if tensor.shape != shape {
            return Err(Error::new(ErrorKind::InvalidData, "Shape mismatch! The saved weights don't fit this layer."));
        }

        let num_elements = shape.iter().product::<usize>();
        let mut cpu_data = Vec::with_capacity(num_elements);
        
        for _ in 0..num_elements {
            let mut val_bytes = [0u8; 8];
            file.read_exact(&mut val_bytes)?;
            cpu_data.push(f64::from_le_bytes(val_bytes));
        }

        // 4. Push the loaded file data directly back to VRAM!
        tensor.data = B::from_data(device, cpu_data, &shape);
    }
    
    Ok(())
}