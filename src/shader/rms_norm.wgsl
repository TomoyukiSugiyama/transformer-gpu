const WG: u32 = 256u;

struct Dims {
    d_model: u32,
    eps: f32
}

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tile: array<f32, WG>;

@compute @workgroup_size(WG, 1, 1)
fn rms_norm(
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

    let rms = sqrt(tile[0] / f32(d_model) + dims.eps);

    var j = lid.x;
    loop {
        if j >= d_model { break; }
        out[base + j] = (x[base + j] / rms * gamma[j]);
        j += WG;
    }
}