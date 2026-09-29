// src/env.rs
pub trait Environment {
    /// Resets the game to its starting state.
    fn reset(&mut self);
    
    /// Advances the game by one frame based on the agent's action.
    /// Returns a tuple: (Reward for this step, Is the agent still alive?)
    fn step(&mut self, action: usize) -> (f32, bool);
    
    /// Returns the current state of the game as a flat array of numbers.
    fn get_state(&self) -> Vec<f64>;
    
    /// How many inputs does this game give to the neural network?
    fn state_size(&self) -> usize;
    
    /// How many possible actions can the agent take?
    fn action_space(&self) -> usize;
    
    /// Returns the current score (useful for logging progress).
    fn score(&self) -> u32;
}