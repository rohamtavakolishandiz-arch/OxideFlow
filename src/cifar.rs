use std::fs::File;
use std::io::Read;
use crate::tensor::Tensor;
use crate::autograd::Variable;

pub struct CifarDataset {
    pub inputs: Vec<Variable>,
    pub targets: Vec<Variable>,
}

impl CifarDataset {
    pub fn load(paths: &[&str]) -> Result<Self, String> {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for &path in paths {
            let mut file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).map_err(|e| e.to_string())?;

            // CIFAR-10 binary format: 1 byte label + 3072 bytes RGB data
            let num_images = buffer.len() / 3073;
            for i in 0..num_images {
                let start = i * 3073;
                let label = buffer[start] as usize;
                
                let mut img_data = vec![0.0; 3072];
                for j in 0..3072 {
                    // Normalize 0-255 pixels to 0.0-1.0
                    img_data[j] = buffer[start + 1 + j] as f64 / 255.0;
                }

                // 🎯 Shape: [1 Batch, 3 Channels (RGB), 32 Height, 32 Width]
                inputs.push(Variable::new(Tensor::from_data(img_data, vec![1, 3, 32, 32]).unwrap()));

                // One-hot encode targets (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
                let mut target_vec = vec![0.0; 10];
                target_vec[label] = 1.0;
                targets.push(Variable::new(Tensor::from_data(target_vec, vec![1, 10]).unwrap()));
            }
        }
        Ok(Self { inputs, targets })
    }
}