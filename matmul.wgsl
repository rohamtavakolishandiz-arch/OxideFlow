// We bind 4 buffers from the CPU to the GPU:
@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec4<u32>; // M (rows A), K (shared), N (cols B), padding

// We tell the GPU to organize the threads in 16x16 grids
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // global_id tells this specific thread which output cell it is calculating
    let row = global_id.y;
    let col = global_id.x;
    
    let M = dimensions.x;
    let K = dimensions.y;
    let N = dimensions.z;

    // If the thread is outside the matrix bounds, shut it down
    if (row >= M || col >= N) {
        return;
    }

    // Compute the dot product for this exact cell
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < K; i = i + 1u) {
        sum = sum + matrix_a[row * K + i] * matrix_b[i * N + col];
    }
    
    // Write the result to VRAM
    matrix_c[row * N + col] = sum;
}