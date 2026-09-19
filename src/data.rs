use crate::autograd::Variable;
use crate::tensor::Tensor;
use crate::backend::Backend;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct DataLoader<B: Backend> {
    inputs: Vec<Variable<B>>,
    targets: Vec<Variable<B>>,
    batch_size: usize,
    shuffle: bool,
}

impl<B: Backend> DataLoader<B> {
    pub fn new(inputs: Vec<Variable<B>>, targets: Vec<Variable<B>>, batch_size: usize, shuffle: bool) -> Self {
        Self { inputs, targets, batch_size, shuffle }
    }

    pub fn iter(&self) -> DataLoaderIterator<'_, B> {
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

pub struct DataLoaderIterator<'a, B: Backend> {
    inputs: &'a [Variable<B>],
    targets: &'a [Variable<B>],
    indices: Vec<usize>,
    batch_size: usize,
    current_idx: usize,
}

impl<'a, B: Backend> Iterator for DataLoaderIterator<'a, B> {
    type Item = (Variable<B>, Variable<B>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.indices.len() { return None; }

        let end_idx = std::cmp::min(self.current_idx + self.batch_size, self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let actual_batch_size = batch_indices.len();

        let mut inputs_data = Vec::new();
        let mut targets_data = Vec::new();

        // 🎯 Dynamically extract the device from the first tensor in the batch
        let device = self.inputs[batch_indices[0]].data.borrow().device.clone();

        for &i in batch_indices {
            // Safely extract opaque hardware memory to CPU to stitch the batch together
            let cpu_in = B::to_cpu(&device, &self.inputs[i].data.borrow().data);
            let cpu_tg = B::to_cpu(&device, &self.targets[i].data.borrow().data);
            
            inputs_data.extend(cpu_in);
            targets_data.extend(cpu_tg);
        }

        let mut input_shape = self.inputs[batch_indices[0]].data.borrow().shape.clone();
        input_shape[0] = actual_batch_size; 

        let mut target_shape = self.targets[batch_indices[0]].data.borrow().shape.clone();
        target_shape[0] = actual_batch_size;

        // Push the fully assembled batch back to the target device
        let batched_input = Variable::new(Tensor::from_data(device.clone(), inputs_data, input_shape).unwrap());
        let batched_target = Variable::new(Tensor::from_data(device, targets_data, target_shape).unwrap());

        self.current_idx = end_idx;
        Some((batched_input, batched_target))
    }
}