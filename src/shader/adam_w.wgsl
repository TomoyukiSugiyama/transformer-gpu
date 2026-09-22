const SIZE: u32 = 256u;

struct Dims {
    len: u32,
    step: u32,
    _pad0: u32,
    _pad1: u32,

    beta1: f32,
    beta2: f32,
    lr: f32,
    eps: f32,

    weight_decay: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
}

@group(0) @binding(0)
var<storage, read> grad: array<f32>;

@group(0) @binding(1)
var<storage, read_write> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> m: array<f32>;

@group(0) @binding(3)
var<storage, read_write> v: array<f32>;

@group(0) @binding(4)
var<uniform> dims: Dims;


@compute @workgroup_size(SIZE)
fn adamw_step(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;

    if i >= dims.len {
        return;
    }

    let step_f = f32(dims.step);
    let beta1 = dims.beta1;
    let beta2 = dims.beta2;

    let g = grad[i];

    let new_m = beta1 * m[i]
        + (1.0 - beta1) * g;

    let new_v = beta2 * v[i]
        + (1.0 - beta2) * g * g;

    m[i] = new_m;
    v[i] = new_v;

    let m_hat = new_m / (1.0 - pow(beta1, step_f));
    let v_hat = new_v / (1.0 - pow(beta2, step_f));

    weight[i] -= dims.lr * (
        m_hat / (sqrt(v_hat) + dims.eps)
        + dims.weight_decay * weight[i]
    );
}