use crate::autograd::Variable;
use rand::seq::SliceRandom;
use rand::thread_rng;
use crate::tensor::Tensor;

pub struct DataLoader {
    inputs: Vec<Variable>,
    targets: Vec<Variable>,
    batch_size: usize,
    shuffle: bool,
}

impl DataLoader {
    pub fn new(inputs: Vec<Variable>, targets: Vec<Variable>, batch_size: usize, shuffle: bool) -> Self {
        Self { inputs, targets, batch_size, shuffle }
    }

    /// Creates an iterator that yields batches of (Inputs, Targets)
    pub fn iter(&self) -> DataLoaderIterator {
        let mut indices: Vec<usize> = (0..self.inputs.len()).collect();
        
        if self.shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }

        DataLoaderIterator {
            inputs: self.inputs.clone(),
            targets: self.targets.clone(),
            indices,
            batch_size: self.batch_size,
            current_idx: 0,
        }
    }
}

pub struct DataLoaderIterator {
    inputs: Vec<Variable>,
    targets: Vec<Variable>,
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
}

impl Iterator for DataLoaderIterator {
    // 🎯 Now yields a single Variable for the entire batch's inputs and targets
    type Item = (Variable, Variable);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() {
            return None;
        }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let actual_batch_size = batch_indices.len();

        let mut inputs_data = Vec::new();
        let mut targets_data = Vec::new();

        // Extract and flatten the raw data from the selected indices
        for &i in batch_indices {
            inputs_data.extend_from_slice(&self.inputs[i].data.borrow().data);
            targets_data.extend_from_slice(&self.targets[i].data.borrow().data);
        }

        // Calculate the number of features dynamically
        let input_features = inputs_data.len() / actual_batch_size;
        let target_features = targets_data.len() / actual_batch_size;

        // Construct the single B x N matrix variables
        let batched_input = Variable::new(
            Tensor::from_data(inputs_data, vec![actual_batch_size, input_features]).unwrap()
        );
        let batched_target = Variable::new(
            Tensor::from_data(targets_data, vec![actual_batch_size, target_features]).unwrap()
        );

        self.current_idx = end_idx;
        Some((batched_input, batched_target))
    }
}