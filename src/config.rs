// src/config.rs
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NetworkConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Hyperparameters {
    pub learning_rate: f64,
    pub gamma: f64,
    pub batch_size: usize,
    pub memory_capacity: usize,
    pub epsilon_start: f64,
    pub epsilon_min: f64,
    pub epsilon_decay: f64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingConfig {
    pub network: NetworkConfig,
    pub hyperparams: Hyperparameters,
}

impl TrainingConfig {
    pub fn load_or_default(path: &str) -> Self {
        // If the file exists, read and parse it
        if let Ok(contents) = fs::read_to_string(path) {
            println!(">> ⚙️ Loaded training configuration from {}", path);
            toml::from_str(&contents).expect("Failed to parse config file. Check your TOML syntax.")
        } else {
            // If it doesn't exist, create it with default Flappy Bird values
            println!(">> ⚙️ No config found. Generating default {}", path);
            let default_config = Self::default_config();
            let toml_string = toml::to_string(&default_config).unwrap();
            fs::write(path, toml_string).expect("Failed to write default config file");
            default_config
        }
    }

    fn default_config() -> Self {
        Self {
            network: NetworkConfig {
                input_size: 5,
                hidden_size: 64,
                output_size: 2,
            },
            hyperparams: Hyperparameters {
                learning_rate: 0.001,
                gamma: 0.99,
                batch_size: 64,
                memory_capacity: 10_000,
                epsilon_start: 1.0,
                epsilon_min: 0.05,
                epsilon_decay: 0.995,
            },
        }
    }
}