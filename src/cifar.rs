use std::fs::File;
use std::io::Read;

pub struct CifarDataset {
    pub inputs: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

impl CifarDataset {
    pub fn load(paths: &[&str]) -> Result<Self, String> {
        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for &path in paths {
            let mut file = File::open(path).map_err(|e| e.to_string())?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer).unwrap();

            let num_images = buffer.len() / 3073;
            for i in 0..num_images {
                let start = i * 3073;
                let label = buffer[start] as usize;
                
                let mut img_data = vec![0.0; 3072];
                for j in 0..3072 {
                    img_data[j] = buffer[start + 1 + j] as f64 / 255.0;
                }

                inputs.push(img_data); // Keep on CPU

                let mut target_vec = vec![0.0; 10];
                target_vec[label] = 1.0;
                targets.push(target_vec);
            }
        }
        Ok(Self { inputs, targets })
    }
}