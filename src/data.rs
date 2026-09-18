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

    pub fn iter(&self) -> DataLoaderIterator<'_> {
        let mut indices: Vec<usize> = (0..self.inputs.len()).collect();
        
        if self.shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }

        DataLoaderIterator {
            inputs: &self.inputs,
            targets: &self.targets,
            indices,
            batch_size: self.batch_size,
            current_idx: 0,
        }
    }
}

pub struct DataLoaderIterator<'a> {
    inputs: &'a [Variable],
    targets: &'a [Variable],
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = (Variable, Variable);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() { return None; }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let actual_batch_size = batch_indices.len();

        let mut inputs_data = Vec::new();
        let mut targets_data = Vec::new();

        for &i in batch_indices {
            inputs_data.extend_from_slice(&self.inputs[i].data.borrow().data);
            targets_data.extend_from_slice(&self.targets[i].data.borrow().data);
        }

        // 🎯 UPGRADE: Dynamically inherit the true N-dimensional shape of the data!
        let mut input_shape = self.inputs[batch_indices[0]].data.borrow().shape.clone();
        input_shape[0] = actual_batch_size; 

        let mut target_shape = self.targets[batch_indices[0]].data.borrow().shape.clone();
        target_shape[0] = actual_batch_size;

        let batched_input = Variable::new(Tensor::from_data(inputs_data, input_shape).unwrap());
        let batched_target = Variable::new(Tensor::from_data(targets_data, target_shape).unwrap());

        self.current_idx = end_idx;
        Some((batched_input, batched_target))
    }
}