use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::backend::Backend;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct DataLoader<B: Backend> {
    inputs: Vec<Vec<f64>>,
    targets: Vec<Vec<f64>>,
    input_shape: Vec<usize>,
    target_shape: Vec<usize>,
    batch_size: usize,
    shuffle: bool,
    device: B,
}

impl<B: Backend> DataLoader<B> {
    pub fn new(
        inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, 
        input_shape: Vec<usize>, target_shape: Vec<usize>, 
        batch_size: usize, shuffle: bool, device: B
    ) -> Self {
        Self { inputs, targets, input_shape, target_shape, batch_size, shuffle, device }
    }

    pub fn iter(&self) -> DataLoaderIterator<'_, B> {
        let mut indices: Vec<usize> = (0..self.inputs.len()).collect();
        if self.shuffle {
            indices.shuffle(&mut thread_rng());
        }

        DataLoaderIterator {
            inputs: &self.inputs,
            targets: &self.targets,
            input_shape: &self.input_shape,
            target_shape: &self.target_shape,
            indices,
            batch_size: self.batch_size,
            current_idx: 0,
            device: self.device.clone(),
        }
    }
}

pub struct DataLoaderIterator<'a, B: Backend> {
    inputs: &'a [Vec<f64>],
    targets: &'a [Vec<f64>],
    input_shape: &'a [usize],
    target_shape: &'a [usize],
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
    device: B,
}

impl<'a, B: Backend> Iterator for DataLoaderIterator<'a, B> {
    type Item = (Variable<B>, Variable<B>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() { return None; }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let actual_batch_size = batch_indices.len();

        // Stitch memory safely on the CPU first
        let mut inputs_data = Vec::with_capacity(actual_batch_size * self.inputs[0].len());
        let mut targets_data = Vec::with_capacity(actual_batch_size * self.targets[0].len());

        for &i in batch_indices {
            inputs_data.extend_from_slice(&self.inputs[i]);
            targets_data.extend_from_slice(&self.targets[i]);
        }

        let mut final_in_shape = vec![actual_batch_size];
        final_in_shape.extend_from_slice(self.input_shape);
        
        let mut final_tg_shape = vec![actual_batch_size];
        final_tg_shape.extend_from_slice(self.target_shape);

        // One single GPU allocation per batch!
        let batched_input = Variable::new_constant(Tensor::from_data(self.device.clone(), inputs_data, final_in_shape).unwrap());
        let batched_target = Variable::new_constant(Tensor::from_data(self.device.clone(), targets_data, final_tg_shape).unwrap());

        self.current_idx = end_idx;
        Some((batched_input, batched_target))
    }
}