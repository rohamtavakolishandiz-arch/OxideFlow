use std::fs::File;
use std::io::{self, Read};

fn read_u32_be<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

pub struct MNISTDataset {
    pub inputs: Vec<Vec<f64>>,
    pub targets: Vec<Vec<f64>>,
}

impl MNISTDataset {
    pub fn load(img_path: &str, lbl_path: &str, max_samples: Option<usize>) -> Result<Self, String> {
        let mut img_file = File::open(img_path).map_err(|e| format!("Failed to open image file: {}", e))?;
        let mut lbl_file = File::open(lbl_path).map_err(|e| format!("Failed to open label file: {}", e))?;

        let _img_magic = read_u32_be(&mut img_file).unwrap();
        let num_images = read_u32_be(&mut img_file).unwrap() as usize;
        let rows = read_u32_be(&mut img_file).unwrap() as usize;
        let cols = read_u32_be(&mut img_file).unwrap() as usize;
        let image_size = rows * cols; // 784

        let _lbl_magic = read_u32_be(&mut lbl_file).unwrap();
        let _num_labels = read_u32_be(&mut lbl_file).unwrap() as usize;

        let limit = max_samples.unwrap_or(num_images);
        let mut raw_images = vec![0u8; limit * image_size];
        img_file.read_exact(&mut raw_images).map_err(|e| e.to_string())?;

        let mut raw_labels = vec![0u8; limit];
        lbl_file.read_exact(&mut raw_labels).map_err(|e| e.to_string())?;

        let mut inputs = Vec::with_capacity(limit);
        let mut targets = Vec::with_capacity(limit);

        for i in 0..limit {
            let start = i * image_size;
            let end = start + image_size;
            
            // Keep on CPU
            let img_f64: Vec<f64> = raw_images[start..end].iter().map(|&p| p as f64 / 255.0).collect();
            inputs.push(img_f64);

            let label = raw_labels[i] as usize;
            let mut target_vec = vec![0.0; 10];
            target_vec[label] = 1.0;
            targets.push(target_vec);
        }

        Ok(Self { inputs, targets })
    }
}