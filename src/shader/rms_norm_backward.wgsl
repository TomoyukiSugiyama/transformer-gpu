const WG: u32 = 128u;

struct Dims {
    d_model: u32,
    eps: f32
}

@group(0) @binding(0) var<storage, read>       dy: array<f32>;     // (seq × d_model)
@group(0) @binding(1) var<storage, read>       x: array<f32>;      // (seq × d_model)
@group(0) @binding(2) var<storage, read>       gamma: array<f32>;  // (d_model,)
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;     // (seq × d_model)
// Too many bindings of type StorageBuffers in Stage ShaderStages(COMPUTE), limit is 4, count was 5.
// @group(0) @binding(4) var<storage, read_write> dgamma_partial: array<f32>; // (seq × d_model)
@group(0) @binding(4) var<uniform>             dims: Dims;

var<workgroup> tile: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn rms_norm_backward(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let seq = wid.x; // seq: 0 ~ (seq_len - 1)
    let d_model = dims.d_model; // d_model: 1024
    let base = seq * d_model; // 0 ~ (seq_len - 1) * 1024

    var sum = 0.0;
    // i = 0 ~ (WG - 1)
    var i = lid.x;
    loop {
        if i >= d_model { break; }
        let v = x[base + i];
        sum += v * v;
        i += WG; // +256
    }
    // tile[0] ~ tile[WG-1] 各タイルに4個分の和が含まれる
    tile[lid.x] = sum;

    workgroupBarrier();

    // tree reduction で workgroup 内の sum を計算
    // stride = 128: tile[0]+=tile[128],tile[1]+=tile[129], ... ,tile[127]+=tile[255]
    // stride = 64: tile[0]+=tile[64],tile[1]+=tile[65], ... ,tile[63]+=tile[127]
    // ...
    // stride = 1: tile[0]+=tile[1]
    var stride = WG / 2u; // 256 / 2 = 128
    loop {
        if stride == 0u { break;}
        if lid.x < stride {
            tile[lid.x] += tile[lid.x + stride]; // 0 + 128 ~ 255 + 128
        }
        workgroupBarrier();
        stride /= 2u;
    }
    // sum(x^2) → rms
    let rms = sqrt(tile[0] / f32(d_model) + dims.eps);
    let inv_rms:f32 = 1.0 / rms;

    // sum(dy * gamma * x_hat) → dot
    tile[lid.x] = 0.0;
    workgroupBarrier();
    // j = 0 ~ (WG - 1)
    var j = lid.x;
    loop {
        if j >= d_model { break; }
        let v = x[base + j];
        let x_hat = v * inv_rms;
        tile[lid.x] += dy[base + j] * gamma[j] * x_hat;
        j += WG; // +256
    }

    workgroupBarrier();

    // tree reduction で workgroup 内の sum を計算
    var stride_dot = WG / 2u; // 256 / 2 = 128
    loop {
        if stride_dot == 0u { break; }
        if lid.x < stride_dot {
            tile[lid.x] += tile[lid.x + stride_dot]; // 0 + 128 ~ 255 + 128
        }
        workgroupBarrier();
        stride_dot /= 2u;
    }

    workgroupBarrier();
    let dot = tile[0] / f32(d_model);

    var k = lid.x;
    loop {
        if k >= d_model { break; }
        let x_hat = x[base + k] * inv_rms;
        dx[base + k] = gamma[k] * inv_rms * (dy[base + k] - x_hat * dot);
        // dgamma_partial[base + k] = dy[base + k] * x_hat;
        k += WG; // +256
    }
}