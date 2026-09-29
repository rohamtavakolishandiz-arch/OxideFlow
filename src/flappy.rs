use rand::Rng;
use crate::env::Environment;

const GRAVITY: f32 = 0.002;
const JUMP_STRENGTH: f32 = -0.03;
const PIPE_SPEED: f32 = 0.01;
const BIRD_X: f32 = 0.2;
const PIPE_WIDTH: f32 = 0.1;
const GAP_SIZE: f32 = 0.3;

#[derive(Clone, Debug)]
pub struct Pipe {
    pub x: f32,
    pub top_gap: f32,
    pub bottom_gap: f32,
    pub passed: bool,
}

pub struct FlappyEnv {
    pub bird_y: f32,
    pub bird_vel: f32,
    pub pipes: Vec<Pipe>,
    pub score: u32,
    pub is_alive: bool,
}

impl FlappyEnv {
    pub fn new() -> Self {
        let mut env = Self {
            bird_y: 0.5,
            bird_vel: 0.0,
            pipes: Vec::new(),
            score: 0,
            is_alive: true,
        };
        env.spawn_pipe(1.0);
        env
    }

    fn spawn_pipe(&mut self, start_x: f32) {
        let mut rng = rand::thread_rng();
        // The top of the gap can be anywhere from 10% to 60% down the screen
        let top_gap = rng.gen_range(0.1..0.6); 
        self.pipes.push(Pipe {
            x: start_x,
            top_gap,
            bottom_gap: top_gap + GAP_SIZE,
            passed: false,
        });
    }

    

}

impl Environment for FlappyEnv {
     fn reset(&mut self) {
        self.bird_y = 0.5;
        self.bird_vel = 0.0;
        self.pipes.clear();
        self.score = 0;
        self.is_alive = true;
        self.spawn_pipe(1.0);
    }

     fn step(&mut self, action: usize) -> (f32, bool) {
        if !self.is_alive {
            return (0.0, false);
        }

        // 1. Apply Action
        if action == 1 {
            self.bird_vel = JUMP_STRENGTH;
        }

        // 2. Apply Physics
        self.bird_vel += GRAVITY;
        self.bird_y += self.bird_vel;

        let mut reward = 0.01; // Small reward just for staying alive!

        // 3. Move Pipes & Check Score
        for pipe in self.pipes.iter_mut() {
            pipe.x -= PIPE_SPEED;
            
            // If the bird passes the right edge of the pipe, score a point
            if !pipe.passed && BIRD_X > pipe.x + PIPE_WIDTH {
                pipe.passed = true;
                self.score += 1;
                reward = 5.0; // Big incentive to clear obstacles!
            }

        }

        // 4. Clean up old pipes and spawn new ones
        if self.pipes[0].x < -PIPE_WIDTH {
            self.pipes.remove(0);
            self.spawn_pipe(1.0);
        }

        // 5. Collision Detection
        let mut crashed = false;
        
        // Hit the floor or ceiling
        if self.bird_y < 0.0 || self.bird_y > 1.0 {
            crashed = true;
        }

        // Hit a pipe
        for pipe in &self.pipes {
            let in_pipe_x = BIRD_X + 0.05 > pipe.x && BIRD_X - 0.05 < pipe.x + PIPE_WIDTH;
            let in_gap_y = self.bird_y > pipe.top_gap && self.bird_y < pipe.bottom_gap;
            
            if in_pipe_x && !in_gap_y {
                crashed = true;
            }
        }

        if crashed {
            self.is_alive = false;
            reward = -5.0; // Painful penalty for dying
        }

        (reward, self.is_alive)
    }

     fn get_state(&self) -> Vec<f64> {
        let next_pipe = self.pipes.iter().find(|p| !p.passed).unwrap_or(&self.pipes[0]);
        let gap_center = next_pipe.top_gap + (GAP_SIZE / 2.0);
        
        vec![
            self.bird_y as f64,
            (self.bird_vel * 20.0) as f64,       // Normalized velocity (~ -0.6 to +0.6)
            (next_pipe.x - BIRD_X) as f64,       // Relative horizontal distance
            (self.bird_y - gap_center) as f64,   // Relative vertical distance to hole!
            gap_center as f64,
        ]
    }

    fn state_size(&self) -> usize {
        5 // Flappy bird has 5 inputs
    }

    fn action_space(&self) -> usize {
        2 // Jump or Glide
    }

    fn score(&self) -> u32 {
        self.score
    }
}