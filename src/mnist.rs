use std::fs::File;
use std::io::{self, Read};
use crate::tensor::Tensor;
use crate::autograd::Variable;
use crate::backend::Backend;

fn read_u32_be<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

pub struct MNISTDataset<B: Backend> {
    pub inputs: Vec<Variable<B>>,
    pub targets: Vec<Variable<B>>,
}

impl<B: Backend> MNISTDataset<B> {
    pub fn load(device: B, img_path: &str, lbl_path: &str, max_samples: Option<usize>) -> Result<Self, String> {
        let mut img_file = File::open(img_path).map_err(|e| format!("Failed to open image file: {}", e))?;
        let mut lbl_file = File::open(lbl_path).map_err(|e| format!("Failed to open label file: {}", e))?;

        let img_magic = read_u32_be(&mut img_file).map_err(|e| e.to_string())?;
        if img_magic != 2051 {
            return Err("Invalid MNIST image file magic number".to_string());
        }
        let num_images = read_u32_be(&mut img_file).map_err(|e| e.to_string())? as usize;
        let rows = read_u32_be(&mut img_file).map_err(|e| e.to_string())? as usize;
        let cols = read_u32_be(&mut img_file).map_err(|e| e.to_string())? as usize;
        let image_size = rows * cols; // 784

        let lbl_magic = read_u32_be(&mut lbl_file).map_err(|e| e.to_string())?;
        if lbl_magic != 2049 {
            return Err("Invalid MNIST label file magic number".to_string());
        }
        let _num_labels = read_u32_be(&mut lbl_file).map_err(|e| e.to_string())? as usize;

        let limit = max_samples.unwrap_or(num_images);
        let mut raw_images = vec![0u8; limit * image_size];
        img_file.read_exact(&mut raw_images).map_err(|e| format!("Image Read Error: {}", e))?;

        let mut raw_labels = vec![0u8; limit];
        lbl_file.read_exact(&mut raw_labels).map_err(|e| format!("Label Read Error: {}", e))?;

        let mut inputs = Vec::with_capacity(limit);
        let mut targets = Vec::with_capacity(limit);

        for i in 0..limit {
            let start = i * image_size;
            let end = start + image_size;
            
            // Normalize 0-255 pixels to 0.0-1.0 range
            let img_f64: Vec<f64> = raw_images[start..end]
                .iter()
                .map(|&p| p as f64 / 255.0)
                .collect();

            // 🎯 Allocate directly to the target device
            inputs.push(Variable::new(Tensor::from_data(device.clone(), img_f64, vec![1, 1, 28, 28]).unwrap()));

            // One-hot encode the 10 possible digit classes
            let label = raw_labels[i] as usize;
            let mut target_vec = vec![0.0; 10];
            target_vec[label] = 1.0;
            
            targets.push(Variable::new(Tensor::from_data(device.clone(), target_vec, vec![1, 10]).unwrap()));
        }

        Ok(Self { inputs, targets })
    }
}