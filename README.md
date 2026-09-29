OxideFlow
 A modular, hardware-accelerated Deep Reinforcement Learning framework built from scratch in Rust.

 Note on Creation: OxideFlow was built collaboratively by a human developer using AI as a pair-programming partner. We believe in transparency—this project serves as a testament to how human creativity and AI thought-partners can build complex, hardware-accelerated systems in Rust.

 OxideFlow is a custom Machine Learning engine that bypasses heavy python wrappers. It features its own WGPU-based autograd engine for direct GPU/VRAM tensor math, a modular trait-based environment system, and a stunning cyberpunk-style visualizer. It is designed to be fast, safe, and completely game-agnostic.

 Key Features
 WGPU Autograd Engine: Custom backpropagation and tensor operations running directly on your GPU for maximum hardware speed.

 Game-Agnostic Environment Trait: Train AI on anything. If you can define the rules, OxideFlow can learn it. Just implement the state, action, and reward logic.

 Cyberpunk Visualizer: Watch the neural network learn in real-time with LiDAR raycasts, pulsating multi-layer brain synapses, and dynamic soft-max confidence HUDs.

 TUI Dashboard: A hacker-style ratatui terminal interface for monitoring max-speed headless training (zero UI rendering overhead).

 Robust Developer Experience: Fully dynamic TOML configuration, bulletproof error handling with anyhow, and safe memory exits.

Quick Start
 Ensure you have Rust installed, then clone the repository:

Bash
 git clone https://github.com/rohamtavakolishandiz-arch/OxideFlow.git
 cd OxideFlow
 Command Line Interface
 OxideFlow includes a professional CLI to control hardware rendering and agent behavior.

Watch the AI Train (Full Visuals):

 Bash
 cargo run --release -- --mode train --render full
 Train at Maximum Speed (Headless TUI):
 Runs the Bellman equations at maximum CPU/VRAM speed without spawning a window.

Bash
 cargo run --release -- --mode train --render headless
 Evaluate a Trained Brain:
 Loads the flappy_checkpoint.bin file and runs a deterministic evaluation using the learned weights.

Bash
 cargo run --release -- --mode eval --render full
 Run on Older Hardware (Low Quality):
 Skips the heavy neural network HUD rendering to save resources.

Bash
 cargo run --release -- --mode train --render low
 Configuration
 OxideFlow is fully configurable without recompiling the Rust binary. Open the config.toml file to adjust:

 Hyperparameters: Learning rate, epsilon decay, discount factor (gamma), batch sizes, and replay buffer capacity.

 Network Shape: Dynamically adjust the hidden layer sizes of the Deep Q-Network.

Controls
 Press Q: Whether you are in the visual window or the headless terminal dashboard, pressing Q will safely pause the simulation, serialize the neural weights to a .bin checkpoint file, and cleanly exit the program.

Build Your Own Environment
You are not limited to Flappy Bird. You can turn OxideFlow into a custom solver by implementing the Environment trait.

Rust
use oxide_flow::env::Environment;

pub struct MyCustomGame { /* ... */ }

impl Environment for MyCustomGame {
    fn reset(&mut self) { ... }
    fn step(&mut self, action: usize) -> (f32, bool) { ... }
    fn get_state(&self) -> Vec<f64> { ... }
    fn state_size(&self) -> usize { ... }
    fn action_space(&self) -> usize { ... }
    fn score(&self) -> u32 { ... }
}
Swap out the environment in main.rs, update your config.toml sizes, and let the engine solve your custom game!

## ⚡ Fuel the Engine

OxideFlow is 100% open-source and free to use. Building a custom hardware-accelerated autograd engine, trait-based environments, and a real-time cyberpunk rendering suite from scratch in Rust takes countless late-night coding sessions and copious amounts of caffeine. 

If this framework helped you train your own AI, saved you hours of debugging Python wrappers, or you simply believe in supporting high-performance Rust tools, consider dropping a crypto tip. 

*Your support directly fuels my compute cycles, hardware testing, and the next major feature update.*

** Drop a Crypto Coffee:**
* **Solana (SOL):** `E86tzDPwHYGuDNCSHEaaJzwmvb2ik4L8qnCULK9yuAQQ`

> **Note:** Make sure to double-check the network (like TRC20 for USDT) before sending!

*Not into crypto? No problem. Simply starring the repository ⭐ and sharing OxideFlow with other developers is a massive help and hugely appreciated!*
