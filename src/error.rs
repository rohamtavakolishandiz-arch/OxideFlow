use thiserror::Error;

/// The central Result type for the OxideFlow framework.
pub type Result<T> = std::result::Result<T, OxideError>;

#[derive(Error, Debug)]
pub enum OxideError {
    // TENSOR ERRORS (Math & Geometry)
    #[error("Tensor shape mismatch. Expected {expected:?}, but got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Dimension out of bounds. Axis {axis} is invalid for shape {shape:?}")]
    InvalidDimension {
        axis: usize,
        shape: Vec<usize>,
    },
    #[error("Mathematical error: {0}")]
    MathError(String),

    // AUTOGRAD ERRORS (Backpropagation)
    #[error("Graph detached: Attempted to call backward on a variable with no unrolled graph.")]
    DetachedGraph,
    #[error("Gradient explosion detected: NaN or Infinity found during backward pass.")]
    GradientExplosion,

    // BACKEND ERRORS (Wgpu / Hardware)
    #[error("GPU Backend Error: {0}")]
    GpuError(String),
    #[error("Out of VRAM: Failed to allocate tensor of size {bytes} bytes.")]
    OutOfMemory { bytes: usize },

    // IO ERRORS (Saving/Loading Models)
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
}