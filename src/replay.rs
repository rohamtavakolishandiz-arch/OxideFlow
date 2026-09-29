// src/replay.rs
use std::collections::VecDeque;
use rand::seq::SliceRandom;

#[derive(Clone, Debug)]
pub struct Transition {
    pub state: Vec<f64>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f64>,
    pub done: bool,
}

pub struct ReplayBuffer {
    capacity: usize,
    memory: VecDeque<Transition>,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            memory: VecDeque::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.memory.len() >= self.capacity {
            self.memory.pop_front();
        }
        self.memory.push_back(transition);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        let mut rng = rand::thread_rng();
        let mut batch: Vec<Transition> = self.memory.iter().cloned().collect();
        batch.shuffle(&mut rng);
        batch.into_iter().take(batch_size).collect()
    }

    pub fn len(&self) -> usize {
        self.memory.len()
    }

    pub fn is_ready(&self, batch_size: usize) -> bool {
        self.memory.len() >= batch_size
    }
}