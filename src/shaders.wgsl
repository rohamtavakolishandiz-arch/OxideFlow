@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// A simple struct to pass our array length to the GPU
struct Dimensions {
    length: u32,
}
@group(0) @binding(3) var<uniform> dims: Dimensions;

// 🚀 Execute 64 parallel threads per workgroup
@compute @workgroup_size(64)
fn add_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < dims.length) {
        out[i] = a[i] + b[i];
    }
}

@compute @workgroup_size(64)
fn sub_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < dims.length) {
        out[i] = a[i] - b[i];
    }
}

@compute @workgroup_size(64)
fn mul_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < dims.length) {
        out[i] = a[i] * b[i];
    }
}

// src/shaders.wgsl (Append this to the bottom)

struct MatmulDims {
    m: u32,
    k: u32,
    n: u32,
    _padding: u32,
}
@group(0) @binding(3) var<uniform> mm_dims: MatmulDims;

@compute @workgroup_size(16, 16)
fn matmul_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row < mm_dims.m && col < mm_dims.n) {
        var sum: f32 = 0.0;
        // Dot product of row A and column B
        for (var i: u32 = 0u; i < mm_dims.k; i = i + 1u) {
            sum = sum + a[row * mm_dims.k + i] * b[i * mm_dims.n + col];
        }
        out[row * mm_dims.n + col] = sum;
    }
}

// ==========================================
// UNARY OPERATIONS (Activations)
// ==========================================

@group(0) @binding(0) var<storage, read> unary_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> unary_out: array<f32>;
@group(0) @binding(2) var<uniform> unary_dims: Dimensions;

@compute @workgroup_size(64)
fn relu_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < unary_dims.length) {
        unary_out[i] = max(unary_in[i], 0.0);
    }
}

@compute @workgroup_size(64)
fn sigmoid_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < unary_dims.length) {
        unary_out[i] = 1.0 / (1.0 + exp(-unary_in[i]));
    }
}

@compute @workgroup_size(64)
fn tanh_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < unary_dims.length) {
        unary_out[i] = tanh(unary_in[i]);
    }
}

// ==========================================
// MATRIX TRANSPOSE
// ==========================================

struct TransposeDims {
    rows: u32,
    cols: u32,
}
@group(0) @binding(0) var<storage, read> t_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(2) var<uniform> t_dims: TransposeDims;

@compute @workgroup_size(16, 16)
fn transpose_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let r = global_id.y;
    let c = global_id.x;
    
    if (r < t_dims.rows && c < t_dims.cols) {
        // Read from row-major, write to column-major
        let in_idx = r * t_dims.cols + c;
        let out_idx = c * t_dims.rows + r;
        t_out[out_idx] = t_in[in_idx];
    }
}

// ==========================================
// SOFTMAX (Per-Row)
// ==========================================

struct SoftmaxDims {
    batch_size: u32,
    num_classes: u32,
    _pad1: u32,
    _pad2: u32,
}
@group(0) @binding(0) var<storage, read> sm_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> sm_out: array<f32>;
@group(0) @binding(2) var<uniform> sm_dims: SoftmaxDims;

@compute @workgroup_size(64)
fn softmax_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    
    if (batch_idx < sm_dims.batch_size) {
        let start = batch_idx * sm_dims.num_classes;
        
        // 1. Find max for numerical stability
        var max_val: f32 = sm_in[start];
        for (var i: u32 = 1u; i < sm_dims.num_classes; i = i + 1u) {
            let val = sm_in[start + i];
            if (val > max_val) {
                max_val = val;
            }
        }
        
        // 2. Compute sum of exponentials
        var exp_sum: f32 = 0.0;
        for (var i: u32 = 0u; i < sm_dims.num_classes; i = i + 1u) {
            exp_sum = exp_sum + exp(sm_in[start + i] - max_val);
        }
        
        // 3. Normalize and write to output
        for (var i: u32 = 0u; i < sm_dims.num_classes; i = i + 1u) {
            sm_out[start + i] = exp(sm_in[start + i] - max_val) / exp_sum;
        }
    }
}

// ==========================================
// MSE LOSS (Single-Threaded Reduction)
// ==========================================

struct MseDims {
    length: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}
@group(0) @binding(0) var<storage, read> mse_pred: array<f32>;
@group(0) @binding(1) var<storage, read> mse_target: array<f32>;
@group(0) @binding(2) var<storage, read_write> mse_out: array<f32>;
@group(0) @binding(3) var<uniform> mse_dims: MseDims;

// 🎯 Using 1 thread to sum small arrays keeps the data entirely in VRAM 
// without requiring complex atomic syncs across thread groups.
@compute @workgroup_size(1)
fn mse_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        var sum_sq: f32 = 0.0;
        for (var i: u32 = 0u; i < mse_dims.length; i = i + 1u) {
            let diff = mse_pred[i] - mse_target[i];
            sum_sq = sum_sq + (diff * diff);
        }
        mse_out[0] = sum_sq / f32(mse_dims.length);
    }
}

// ==========================================
// 2D CONVOLUTION
// ==========================================

struct Conv2dDims {
    batch: u32,
    in_c: u32,
    h: u32,
    w: u32,
    out_c: u32,
    kh: u32,
    kw: u32,
    out_h: u32,
    out_w: u32,
    stride: u32,
    padding: u32,
    _pad: u32, // Padded to 48 bytes (multiple of 16 for WGPU)
}

@group(0) @binding(0) var<storage, read> conv_in: array<f32>;
@group(0) @binding(1) var<storage, read> conv_w: array<f32>;
@group(0) @binding(2) var<storage, read> conv_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> conv_out: array<f32>;
@group(0) @binding(4) var<uniform> c_dims: Conv2dDims;

@compute @workgroup_size(64)
fn conv2d_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_out = c_dims.batch * c_dims.out_c * c_dims.out_h * c_dims.out_w;
    
    if (idx < total_out) {
        // 🎯 Decode 1D thread ID back into 4D coordinates (b, oc, oh, ow)
        let ow = idx % c_dims.out_w;
        var temp = idx / c_dims.out_w;
        let oh = temp % c_dims.out_h;
        temp = temp / c_dims.out_h;
        let oc = temp % c_dims.out_c;
        let b = temp / c_dims.out_c;

        var sum: f32 = conv_b[oc]; // Start with the bias

        for (var ic: u32 = 0u; ic < c_dims.in_c; ic = ic + 1u) {
            for (var kh_idx: u32 = 0u; kh_idx < c_dims.kh; kh_idx = kh_idx + 1u) {
                for (var kw_idx: u32 = 0u; kw_idx < c_dims.kw; kw_idx = kw_idx + 1u) {
                    
                    let ih_i32 = i32(oh * c_dims.stride + kh_idx) - i32(c_dims.padding);
                    let iw_i32 = i32(ow * c_dims.stride + kw_idx) - i32(c_dims.padding);

                    if (ih_i32 >= 0 && ih_i32 < i32(c_dims.h) && iw_i32 >= 0 && iw_i32 < i32(c_dims.w)) {
                        let ih = u32(ih_i32);
                        let iw = u32(iw_i32);
                        
                        let in_idx = b * c_dims.in_c * c_dims.h * c_dims.w + ic * c_dims.h * c_dims.w + ih * c_dims.w + iw;
                        let w_idx = oc * c_dims.in_c * c_dims.kh * c_dims.kw + ic * c_dims.kh * c_dims.kw + kh_idx * c_dims.kw + kw_idx;
                                  
                        sum = sum + conv_in[in_idx] * conv_w[w_idx];
                    }
                }
            }
        }
        conv_out[idx] = sum;
    }
}

// ==========================================
// 2D MAX POOLING
// ==========================================

struct MaxPoolDims {
    batch: u32,
    c: u32,
    h: u32,
    w: u32,
    out_h: u32,
    out_w: u32,
    k: u32,
    _pad: u32, // Padded to 32 bytes
}

@group(0) @binding(0) var<storage, read> mp_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> mp_out: array<f32>;
@group(0) @binding(2) var<uniform> mp_dims: MaxPoolDims;

@compute @workgroup_size(64)
fn maxpool2d_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_out = mp_dims.batch * mp_dims.c * mp_dims.out_h * mp_dims.out_w;
    
    if (idx < total_out) {
        let ow = idx % mp_dims.out_w;
        var temp = idx / mp_dims.out_w;
        let oh = temp % mp_dims.out_h;
        temp = temp / mp_dims.out_h;
        let ch = temp % mp_dims.c;
        let b = temp / mp_dims.c;

        // Initialize with the first element of the kernel window
        let first_idx = b * mp_dims.c * mp_dims.h * mp_dims.w + ch * mp_dims.h * mp_dims.w + (oh * mp_dims.k) * mp_dims.w + (ow * mp_dims.k);
        var max_val: f32 = mp_in[first_idx];

        for (var kh: u32 = 0u; kh < mp_dims.k; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < mp_dims.k; kw = kw + 1u) {
                let in_idx = b * mp_dims.c * mp_dims.h * mp_dims.w + ch * mp_dims.h * mp_dims.w + (oh * mp_dims.k + kh) * mp_dims.w + (ow * mp_dims.k + kw);
                let val = mp_in[in_idx];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        mp_out[idx] = max_val;
    }
}

// ==========================================
// MAXPOOL2D BACKWARD
// ==========================================

@group(0) @binding(0) var<storage, read> mp_in_b: array<f32>;
@group(0) @binding(1) var<storage, read> mp_grad_out: array<f32>;
@group(0) @binding(2) var<storage, read_write> mp_grad_in: array<f32>;
@group(0) @binding(3) var<uniform> mp_dims_b: MaxPoolDims;

@compute @workgroup_size(64)
fn maxpool2d_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_out = mp_dims_b.batch * mp_dims_b.c * mp_dims_b.out_h * mp_dims_b.out_w;

    if (idx < total_out) {
        let ow = idx % mp_dims_b.out_w;
        var temp = idx / mp_dims_b.out_w;
        let oh = temp % mp_dims_b.out_h;
        temp = temp / mp_dims_b.out_h;
        let ch = temp % mp_dims_b.c;
        let b = temp / mp_dims_b.c;

        var max_val: f32 = -9999999.0;
        var max_idx: u32 = 0u;

        for (var kh: u32 = 0u; kh < mp_dims_b.k; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < mp_dims_b.k; kw = kw + 1u) {
                let in_idx = b * mp_dims_b.c * mp_dims_b.h * mp_dims_b.w + ch * mp_dims_b.h * mp_dims_b.w + (oh * mp_dims_b.k + kh) * mp_dims_b.w + (ow * mp_dims_b.k + kw);
                let val = mp_in_b[in_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_idx;
                }
            }
        }
        // mp_grad_in is pre-allocated with zeros by the framework
        mp_grad_in[max_idx] = mp_grad_out[idx];
    }
}

// ==========================================
// CONV2D BACKWARD - BIAS
// ==========================================

@group(0) @binding(0) var<storage, read> grad_out_b: array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_bias: array<f32>;
@group(0) @binding(2) var<uniform> c_dims_b: Conv2dDims;

@compute @workgroup_size(64)
fn conv2d_backward_bias(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let oc = global_id.x;
    if (oc < c_dims_b.out_c) {
        var sum: f32 = 0.0;
        for (var b: u32 = 0u; b < c_dims_b.batch; b = b + 1u) {
            for (var oh: u32 = 0u; oh < c_dims_b.out_h; oh = oh + 1u) {
                for (var ow: u32 = 0u; ow < c_dims_b.out_w; ow = ow + 1u) {
                    let idx = b * c_dims_b.out_c * c_dims_b.out_h * c_dims_b.out_w + oc * c_dims_b.out_h * c_dims_b.out_w + oh * c_dims_b.out_w + ow;
                    sum = sum + grad_out_b[idx];
                }
            }
        }
        grad_bias[oc] = sum;
    }
}

// ==========================================
// CONV2D BACKWARD - WEIGHTS
// ==========================================

@group(0) @binding(0) var<storage, read> input_w: array<f32>;
@group(0) @binding(1) var<storage, read> grad_out_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_weight: array<f32>;
@group(0) @binding(3) var<uniform> c_dims_w: Conv2dDims;

@compute @workgroup_size(64)
fn conv2d_backward_weight(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_w = c_dims_w.out_c * c_dims_w.in_c * c_dims_w.kh * c_dims_w.kw;

    if (idx < total_w) {
        let kw = idx % c_dims_w.kw;
        var temp = idx / c_dims_w.kw;
        let kh = temp % c_dims_w.kh;
        temp = temp / c_dims_w.kh;
        let ic = temp % c_dims_w.in_c;
        let oc = temp / c_dims_w.in_c;

        var sum: f32 = 0.0;
        for (var b: u32 = 0u; b < c_dims_w.batch; b = b + 1u) {
            for (var oh: u32 = 0u; oh < c_dims_w.out_h; oh = oh + 1u) {
                for (var ow: u32 = 0u; ow < c_dims_w.out_w; ow = ow + 1u) {
                    let ih_i32 = i32(oh * c_dims_w.stride + kh) - i32(c_dims_w.padding);
                    let iw_i32 = i32(ow * c_dims_w.stride + kw) - i32(c_dims_w.padding);

                    if (ih_i32 >= 0 && ih_i32 < i32(c_dims_w.h) && iw_i32 >= 0 && iw_i32 < i32(c_dims_w.w)) {
                        let ih = u32(ih_i32);
                        let iw = u32(iw_i32);
                        let in_idx = b * c_dims_w.in_c * c_dims_w.h * c_dims_w.w + ic * c_dims_w.h * c_dims_w.w + ih * c_dims_w.w + iw;
                        let go_idx = b * c_dims_w.out_c * c_dims_w.out_h * c_dims_w.out_w + oc * c_dims_w.out_h * c_dims_w.out_w + oh * c_dims_w.out_w + ow;
                        sum = sum + input_w[in_idx] * grad_out_w[go_idx];
                    }
                }
            }
        }
        grad_weight[idx] = sum;
    }
}

// ==========================================
// CONV2D BACKWARD - INPUTS
// ==========================================

@group(0) @binding(0) var<storage, read> weight_i: array<f32>;
@group(0) @binding(1) var<storage, read> grad_out_i: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input: array<f32>;
@group(0) @binding(3) var<uniform> c_dims_i: Conv2dDims;

@compute @workgroup_size(64)
fn conv2d_backward_input(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_in = c_dims_i.batch * c_dims_i.in_c * c_dims_i.h * c_dims_i.w;

    if (idx < total_in) {
        let iw = idx % c_dims_i.w;
        var temp = idx / c_dims_i.w;
        let ih = temp % c_dims_i.h;
        temp = temp / c_dims_i.h;
        let ic = temp % c_dims_i.in_c;
        let b = temp / c_dims_i.in_c;

        var sum: f32 = 0.0;
        for (var oc: u32 = 0u; oc < c_dims_i.out_c; oc = oc + 1u) {
            for (var kh: u32 = 0u; kh < c_dims_i.kh; kh = kh + 1u) {
                for (var kw: u32 = 0u; kw < c_dims_i.kw; kw = kw + 1u) {
                    let oh_num = i32(ih) + i32(c_dims_i.padding) - i32(kh);
                    let ow_num = i32(iw) + i32(c_dims_i.padding) - i32(kw);

                    // Check if this input pixel contributed to a valid output pixel
                    if (oh_num >= 0 && oh_num % i32(c_dims_i.stride) == 0 && ow_num >= 0 && ow_num % i32(c_dims_i.stride) == 0) {
                        let oh = u32(oh_num) / c_dims_i.stride;
                        let ow = u32(ow_num) / c_dims_i.stride;

                        if (oh < c_dims_i.out_h && ow < c_dims_i.out_w) {
                            let go_idx = b * c_dims_i.out_c * c_dims_i.out_h * c_dims_i.out_w + oc * c_dims_i.out_h * c_dims_i.out_w + oh * c_dims_i.out_w + ow;
                            let w_idx = oc * c_dims_i.in_c * c_dims_i.kh * c_dims_i.kw + ic * c_dims_i.kh * c_dims_i.kw + kh * c_dims_i.kw + kw;
                            sum = sum + grad_out_i[go_idx] * weight_i[w_idx];
                        }
                    }
                }
            }
        }
        grad_input[idx] = sum;
    }
}

// ==========================================
// ADAMW OPTIMIZER
// ==========================================

struct AdamWConfig {
    length: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
}

@group(0) @binding(0) var<storage, read_write> adam_w: array<f32>;
@group(0) @binding(1) var<storage, read> adam_g: array<f32>;
@group(0) @binding(2) var<storage, read_write> adam_m: array<f32>;
@group(0) @binding(3) var<storage, read_write> adam_v: array<f32>;
@group(0) @binding(4) var<uniform> adam_cfg: AdamWConfig;

@compute @workgroup_size(64)
fn adamw_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < adam_cfg.length) {
        let g = adam_g[i];
        var w = adam_w[i];

        // 1. Update biased first and second moment estimates
        let new_m = adam_cfg.beta1 * adam_m[i] + (1.0 - adam_cfg.beta1) * g;
        let new_v = adam_cfg.beta2 * adam_v[i] + (1.0 - adam_cfg.beta2) * (g * g);

        adam_m[i] = new_m;
        adam_v[i] = new_v;

        // 2. Compute bias-corrected moments
        let m_hat = new_m / adam_cfg.bias_correction1;
        let v_hat = new_v / adam_cfg.bias_correction2;

        // 3. Update weights with weight decay
        w = w - adam_cfg.lr * (m_hat / (sqrt(v_hat) + adam_cfg.eps) + adam_cfg.weight_decay * w);
        adam_w[i] = w;
    }
}

// ==========================================
// EXPONENTIAL (Unary)
// ==========================================

@compute @workgroup_size(64)
fn exp_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i < unary_dims.length) {
        unary_out[i] = exp(unary_in[i]);
    }
}

// ==========================================
// GLOBAL REDUCTIONS (Max & Sum)
// ==========================================

struct ReductionDims {
    length: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> red_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> red_out: array<f32>;
@group(0) @binding(2) var<uniform> red_dims: ReductionDims;

// 🎯 We use a single thread to reduce the entire array. 
// For arrays under a few million elements, this is significantly faster 
// than the PCIe bus transfer latency to the CPU!
@compute @workgroup_size(1)
fn max_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        var max_val: f32 = -9999999.0;
        for (var i: u32 = 0u; i < red_dims.length; i = i + 1u) {
            let val = red_in[i];
            if (val > max_val) {
                max_val = val;
            }
        }
        red_out[0] = max_val;
    }
}

@compute @workgroup_size(1)
fn sum_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x == 0u) {
        var sum_val: f32 = 0.0;
        for (var i: u32 = 0u; i < red_dims.length; i = i + 1u) {
            sum_val = sum_val + red_in[i];
        }
        red_out[0] = sum_val;
    }
}

struct BroadcastDims {
    r1: u32,
    c1: u32,
    r2: u32,
    c2: u32,
}
@group(0) @binding(3) var<uniform> b_dims: BroadcastDims;

@compute @workgroup_size(64)
fn add_broadcast_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let out_r = max(b_dims.r1, b_dims.r2);
    let out_c = max(b_dims.c1, b_dims.c2);
    
    if (idx < out_r * out_c) {
        let row = idx / out_c;
        let col = idx % out_c;
        
        // Modulo arithmetic naturally handles the 1-dimension stretching
        let a_idx = (row % b_dims.r1) * b_dims.c1 + (col % b_dims.c1);
        let b_idx = (row % b_dims.r2) * b_dims.c2 + (col % b_dims.c2);
        
        out[idx] = a[a_idx] + b[b_idx];
    }
}

struct BatchNormDims {
    batch: u32,
    c: u32,
    h: u32,
    w: u32,
    eps: f32,
    momentum: f32,
    is_training: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> bn_in: array<f32>;
@group(0) @binding(1) var<storage, read> bn_weight: array<f32>; // Gamma
@group(0) @binding(2) var<storage, read> bn_bias: array<f32>;   // Beta
@group(0) @binding(3) var<storage, read_write> bn_run_mean: array<f32>;
@group(0) @binding(4) var<storage, read_write> bn_run_var: array<f32>;
@group(0) @binding(5) var<storage, read_write> bn_out: array<f32>;
@group(0) @binding(6) var<uniform> bn_dims: BatchNormDims;

@compute @workgroup_size(64)
fn batch_norm2d_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.x;
    
    if (c < bn_dims.c) {
        let spatial_size = bn_dims.h * bn_dims.w;
        let batch_size = bn_dims.batch;
        let num_elements = f32(batch_size * spatial_size);
        
        var mean: f32 = bn_run_mean[c];
        var variance: f32 = bn_run_var[c];

        if (bn_dims.is_training == 1u) {
            // 1. Calculate Mean for this channel
            var sum: f32 = 0.0;
            for (var b: u32 = 0u; b < batch_size; b = b + 1u) {
                let offset = b * bn_dims.c * spatial_size + c * spatial_size;
                for (var i: u32 = 0u; i < spatial_size; i = i + 1u) {
                    sum = sum + bn_in[offset + i];
                }
            }
            mean = sum / num_elements;
            
            // 2. Calculate Variance for this channel
            var var_sum: f32 = 0.0;
            for (var b: u32 = 0u; b < batch_size; b = b + 1u) {
                let offset = b * bn_dims.c * spatial_size + c * spatial_size;
                for (var i: u32 = 0u; i < spatial_size; i = i + 1u) {
                    let diff = bn_in[offset + i] - mean;
                    var_sum = var_sum + (diff * diff);
                }
            }
            variance = var_sum / num_elements;
            
            // 3. Update running stats directly in VRAM
            bn_run_mean[c] = (1.0 - bn_dims.momentum) * bn_run_mean[c] + bn_dims.momentum * mean;
            bn_run_var[c] = (1.0 - bn_dims.momentum) * bn_run_var[c] + bn_dims.momentum * variance;
        }

        // 4. Apply Normalization and Affine Transform (Gamma & Beta)
        let inv_std = 1.0 / sqrt(variance + bn_dims.eps);
        let gamma = bn_weight[c];
        let beta = bn_bias[c];
        
        for (var b: u32 = 0u; b < batch_size; b = b + 1u) {
            let offset = b * bn_dims.c * spatial_size + c * spatial_size;
            for (var i: u32 = 0u; i < spatial_size; i = i + 1u) {
                let idx = offset + i;
                bn_out[idx] = (bn_in[idx] - mean) * inv_std * gamma + beta;
            }
        }
    }
}

// ==========================================
// 2D BATCH NORMALIZATION BACKWARD
// ==========================================

@group(0) @binding(0) var<storage, read> bn_in_b: array<f32>;
@group(0) @binding(1) var<storage, read> bn_grad_out: array<f32>;
@group(0) @binding(2) var<storage, read> bn_weight_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> bn_grad_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> bn_grad_weight: array<f32>;
@group(0) @binding(5) var<storage, read_write> bn_grad_bias: array<f32>;
@group(0) @binding(6) var<uniform> bn_dims_b: BatchNormDims;

@compute @workgroup_size(64)
fn batch_norm2d_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.x;
    
    if (c < bn_dims_b.c) {
        let spatial_size = bn_dims_b.h * bn_dims_b.w;
        let batch_size = bn_dims_b.batch;
        let num_elements = f32(batch_size * spatial_size);

        // 1. Recompute batch mean for this channel
        var sum: f32 = 0.0;
        for (var b: u32 = 0u; b < batch_size; b = b + 1u) {
            let offset = b * bn_dims_b.c * spatial_size + c * spatial_size;
            for (var i: u32 = 0u; i < spatial_size; i = i + 1u) {
                sum = sum + bn_in_b[offset + i];
            }
        }
        let mean = sum / num_elements;

        // 2. Recompute batch variance for this channel
        var var_sum: f32 = 0.0;
        for (var b: u32 = 0u; b < batch_size; b = b + 1u) {
            let offset = b * bn_dims_b.c * spatial_size + c * spatial_size;
            for (var i: u32 = 0u; i < spatial_size; i = i + 1u) {
                let diff = bn_in_b[offset + i] - mean;
                var_sum = var_sum + (diff * diff);
            }
        }
        let variance = var_sum / num_elements;
        let inv_std = 1.0 / sqrt(variance + bn_dims_b.eps);
        let gamma = bn_weight_b[c];

        // 3. Compute gradients for Gamma (Weight) and Beta (Bias)
        var d_beta: f32 = 0.0;
        var d_gamma: f32 = 0.0;
        
        for (var b: u32 = 0u; b < batch_size; b = b + 1u) {
            let offset = b * bn_dims_b.c * spatial_size + c * spatial_size;
            for (var i: u32 = 0u; i < spatial_size; i = i + 1u) {
                let idx = offset + i;
                let x_hat = (bn_in_b[idx] - mean) * inv_std;
                let dy = bn_grad_out[idx];
                
                d_beta = d_beta + dy;
                d_gamma = d_gamma + (dy * x_hat);
            }
        }
        
        bn_grad_bias[c] = d_beta;
        bn_grad_weight[c] = d_gamma;

        // 4. Compute gradient for Inputs
        let scale = (gamma * inv_std) / num_elements;
        
        for (var b: u32 = 0u; b < batch_size; b = b + 1u) {
            let offset = b * bn_dims_b.c * spatial_size + c * spatial_size;
            for (var i: u32 = 0u; i < spatial_size; i = i + 1u) {
                let idx = offset + i;
                let x_hat = (bn_in_b[idx] - mean) * inv_std;
                let dy = bn_grad_out[idx];
                
                // The simplified backward formula for BatchNorm input
                bn_grad_in[idx] = scale * (num_elements * dy - d_beta - x_hat * d_gamma);
            }
        }
    }
}