use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

const BR: u32 = 64;
const MAX_D_HEAD: u32 = 128;
const TILE: u32 = 16u32; // before_flash_attention で利用する

pub fn attention(
    ctx: &GpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: u32,
    d_head: u32,
) -> (Vec<f32>, Vec<f32>) {
    assert!(
        d_head <= MAX_D_HEAD,
        "d_head={} exceeds MAX_D_HEAD={}",
        d_head,
        MAX_D_HEAD
    );
    assert_eq!(q.len(), (seq * d_head) as usize, "q must be seq×d_head");
    assert_eq!(k.len(), (seq * d_head) as usize, "k must be seq×d_head");
    assert_eq!(v.len(), (seq * d_head) as usize, "v must be seq×d_head");
    let out_size = (seq * d_head + seq) as u64 * 4;
    let fa_q = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fa_q"),
            contents: bytemuck::cast_slice(&q),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let fa_k = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fa_k"),
            contents: bytemuck::cast_slice(&k),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let fa_v = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fa_v"),
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let fa_score = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fa_score"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [seq, d_head, 0, 0];
    let fa_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fa_dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/flash_attention.wgsl"));
    let fa_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flash_attention"),
            layout: None,
            module: &module,
            entry_point: Some("flash_attention"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let fa_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &fa_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: fa_q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: fa_k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: fa_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: fa_score.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: fa_dims.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&fa_pipeline);
        pass.set_bind_group(0, &fa_bind_group, &[]);
        pass.dispatch_workgroups(seq.div_ceil(BR), 1, 1);
    }

    let buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&fa_score, 0, &buf_read, 0, out_size);
    ctx.queue.submit([encoder.finish()]);

    let slice = buf_read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let data = slice.get_mapped_range();
    let result = bytemuck::allocation::pod_collect_to_vec(&data);

    let o = result[..(seq * d_head) as usize].to_vec();
    let l = result[(seq * d_head) as usize..].to_vec();
    (o, l)
}

pub fn attention_backward(
    ctx: &GpuContext,
    do_: &[f32], // [seq, d_head]
    q: &[f32],   // [seq, d_head]
    k: &[f32],   // [seq, d_head]
    v: &[f32],   // [seq, d_head]
    o: &[f32],   // [seq, d_head]
    l: &[f32],   // [seq]
    seq: u32,
    d_head: u32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = (seq * d_head) as usize;

    // D[i] = rowsum(o[i] * do_[i]) を事前計算
    let d_vec: Vec<f32> = (0..seq as usize)
        .map(|i| {
            (0..d_head as usize)
                .map(|d| o[i * d_head as usize + d] * do_[i * d_head as usize + d])
                .sum()
        })
        .collect();

    let mut inputs = Vec::with_capacity(n * 4 + seq as usize * 2);
    inputs.extend_from_slice(do_);
    inputs.extend_from_slice(q);
    inputs.extend_from_slice(k);
    inputs.extend_from_slice(v);
    inputs.extend_from_slice(l);
    inputs.extend_from_slice(&d_vec);

    let buf_inputs = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("inputs"),
            contents: bytemuck::cast_slice(&inputs),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let zero_i32 = vec![0i32; n];
    let buf_dq = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dq"),
            contents: bytemuck::cast_slice(&zero_i32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let zero_f32 = vec![0.0f32; n];
    let buf_dk = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dk"),
            contents: bytemuck::cast_slice(&zero_f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
    let buf_dv = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dv"),
            contents: bytemuck::cast_slice(&zero_f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let dims_padded: [u32; 4] = [seq, d_head, 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx.device.create_shader_module(wgpu::include_wgsl!(
        "../shader/flash_attention_backward.wgsl"
    ));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attention_backward"),
            layout: None,
            module: &module,
            entry_point: Some("attention_backward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_inputs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_dq.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_dk.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_dv.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: buf_dims.as_entire_binding(),
            },
        ],
    });

    let byte_size = (n * 4) as u64;
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(seq, 1, 1);
    }

    let read_dq = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let read_dk = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let read_dv = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&buf_dq, 0, &read_dq, 0, byte_size);
    encoder.copy_buffer_to_buffer(&buf_dk, 0, &read_dk, 0, byte_size);
    encoder.copy_buffer_to_buffer(&buf_dv, 0, &read_dv, 0, byte_size);
    ctx.queue.submit([encoder.finish()]);

    // dq は i32 bits → f32 に変換
    let dq = read_gpu_buffer::<i32>(ctx, &read_dq)
        .iter()
        .map(|&x| f32::from_bits(x as u32))
        .collect();
    let dk = read_gpu_buffer::<f32>(ctx, &read_dk);
    let dv = read_gpu_buffer::<f32>(ctx, &read_dv);

    (dq, dk, dv)
}

fn read_gpu_buffer<T: bytemuck::Pod>(ctx: &GpuContext, buf: &wgpu::Buffer) -> Vec<T> {
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();
    let data = slice.get_mapped_range();
    bytemuck::allocation::pod_collect_to_vec(&data)
}

/// Flash Attention 導入前の実装、ベンチで利用
#[allow(dead_code)]
pub fn before_flash_attention(
    ctx: &GpuContext,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: u32,
    d_head: u32,
) -> Vec<f32> {
    assert!(
        d_head <= MAX_D_HEAD,
        "d_head={} exceeds MAX_D_HEAD={}",
        d_head,
        MAX_D_HEAD
    );
    assert_eq!(q.len(), (seq * d_head) as usize, "q must be seq×d_head");
    assert_eq!(k.len(), (seq * d_head) as usize, "k must be seq×d_head");
    assert_eq!(v.len(), (seq * d_head) as usize, "v must be seq×d_head");
    let byte_size = (seq * d_head * 4) as u64;
    let qkt_q = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("qkt_q"),
            contents: bytemuck::cast_slice(&q),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let qkt_k = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("qkt_k"),
            contents: bytemuck::cast_slice(&k),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let score = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("score"),
        size: (seq * seq * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [seq, d_head, 0, 0];
    let qkt_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("qkt_dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/attention.wgsl"));
    let qkt_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qkt"),
            layout: None,
            module: &module,
            entry_point: Some("qkt"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let qkt_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &qkt_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: qkt_q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: qkt_k.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: score.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: qkt_dims.as_entire_binding(),
            },
        ],
    });

    let dims_padded: [u32; 4] = [seq, 0, 0, 0];
    let sm_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sm_dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let sm_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_causal"),
            layout: None,
            module: &module,
            entry_point: Some("softmax_causal"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let sm_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &sm_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: score.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sm_dims.as_entire_binding(),
            },
        ],
    });

    let av_v = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("av_v"),
            contents: bytemuck::cast_slice(&v),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let av_o = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("av_o"),
        size: byte_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [seq, d_head, 0, 0];
    let av_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("av_dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let av_pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attn_v"),
            layout: None,
            module: &module,
            entry_point: Some("attn_v"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let av_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &av_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: score.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: av_v.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: av_o.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: av_dims.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&qkt_pipeline);
        pass.set_bind_group(0, &qkt_bind_group, &[]);
        pass.dispatch_workgroups(seq.div_ceil(TILE), seq.div_ceil(TILE), 1);

        pass.set_pipeline(&sm_pipeline);
        pass.set_bind_group(0, &sm_bind_group, &[]);
        pass.dispatch_workgroups(seq.div_ceil(64), 1, 1);

        pass.set_pipeline(&av_pipeline);
        pass.set_bind_group(0, &av_bind_group, &[]);
        pass.dispatch_workgroups(d_head.div_ceil(TILE), seq.div_ceil(TILE), 1);
    }

    let buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&av_o, 0, &buf_read, 0, byte_size as u64);
    ctx.queue.submit([encoder.finish()]);

    let slice = buf_read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let data = slice.get_mapped_range();
    bytemuck::allocation::pod_collect_to_vec(&data)
}

// CPU リファレンス
#[cfg(test)]
pub fn attention_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    d_head: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(q.len(), (seq * d_head) as usize, "q must be seq×d_head");
    assert_eq!(k.len(), (seq * d_head) as usize, "k must be seq×d_head");
    assert_eq!(v.len(), (seq * d_head) as usize, "v must be seq×d_head");

    // 1 / √d_k
    let scale = 1.0 / (d_head as f32).sqrt();

    // QK^T / √d_k , casual mask
    let mut scores = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            let s = (0..d_head)
                .map(|k_| q[i * d_head + k_] * k[j * d_head + k_])
                .sum::<f32>()
                * scale;
            scores[i * seq + j] = if j > i { -1e9 } else { s };
        }
    }

    // L[i] = m_i + log(l_i)
    let mut l_vec = vec![0.0f32; seq];

    // softmax
    for i in 0..seq {
        let row = &mut scores[i * seq..(i + 1) * seq];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
        l_vec[i] = max + sum.ln();
        for x in row.iter_mut() {
            *x = (*x - max).exp() / sum;
        }
    }

    // fa_score * V
    let mut out = vec![0.0f32; seq * d_head];
    for i in 0..seq {
        for d in 0..d_head {
            out[i * d_head + d] = (0..seq)
                .map(|j| scores[i * seq + j] * v[j * d_head + d])
                .sum();
        }
    }
    (out, l_vec)
}

#[cfg(test)]
pub fn attention_backward_cpu(
    do_: &[f32], // [seq, d_head]
    q: &[f32],   // [seq, d_head]
    k: &[f32],   // [seq, d_head]
    v: &[f32],   // [seq, d_head]
    o: &[f32],   // [seq, d_head]
    l: &[f32],   // [seq]
    seq: usize,
    d_head: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(q.len(), (seq * d_head) as usize, "q must be seq×d_head");
    assert_eq!(k.len(), (seq * d_head) as usize, "k must be seq×d_head");
    assert_eq!(v.len(), (seq * d_head) as usize, "v must be seq×d_head");

    // 1 / √d_k
    let scale = 1.0 / (d_head as f32).sqrt();

    // D[i] = rowsum(o[i] * do_[i])
    let mut d_vec = vec![0.0f32; seq];
    for i in 0..seq {
        d_vec[i] = (0..d_head)
            .map(|d| o[i * d_head + d] * do_[i * d_head + d])
            .sum()
    }
    // S[i,j] = Q[i]・K[j]^T * scale (causal mask)
    let mut scores = vec![f32::NEG_INFINITY; seq * seq];
    for i in 0..seq {
        for j in 0..=i {
            scores[i * seq + j] = (0..d_head)
                .map(|d| q[i * d_head + d] * k[j * d_head + d])
                .sum::<f32>()
                * scale;
        }
    }

    // softmax
    // P[i,j] = exp(S[i,j] - L[i])
    let mut p = vec![0.0f32; seq * seq];
    for i in 0..seq {
        for j in 0..=i {
            p[i * seq + j] = (scores[i * seq + j] - l[i]).exp()
        }
    }

    let mut dq = vec![0.0f32; seq * d_head];
    let mut dk = vec![0.0f32; seq * d_head];
    let mut dv = vec![0.0f32; seq * d_head];

    for i in 0..seq {
        for j in 0..=i {
            let p_ij = p[i * seq + j];

            // dV[j] += P[i,j] * do_[i]
            for d in 0..d_head {
                dv[j * d_head + d] += p_ij * do_[i * d_head + d];
            }

            // dS[i,j] = P[i,j] * (do_[i]・V[j] - D[i])
            let do_v_j: f32 = (0..d_head)
                .map(|d| do_[i * d_head + d] * v[j * d_head + d])
                .sum();
            let ds_ij = p_ij * (do_v_j - d_vec[i]);

            // dQ[i] += scale * dS[i,j] * K[j]
            for d in 0..d_head {
                dq[i * d_head + d] += scale * ds_ij * k[j * d_head + d];
            }
            // dK[j] += scale * dS[i,j] * Q[i]
            for d in 0..d_head {
                dk[j * d_head + d] += scale * ds_ij * q[i * d_head + d];
            }
        }
    }

    (dq, dk, dv)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{test_utils::assert_close, util::random_f32};

    #[test]
    fn test_softmax_row_sum() {
        let seq: usize = 1;
        let d_head: usize = 1;
        let q: Vec<f32> = vec![2.0];
        let k: Vec<f32> = vec![3.0];
        let v: Vec<f32> = vec![4.0];

        let (cpu_o, cpu_l) = attention_cpu(&q, &k, &v, seq, d_head);
        let ctx = GpuContext::new();
        let (gpu_o, gpu_l) = attention(&ctx, &q, &k, &v, seq as u32, d_head as u32);

        // 1 / √d_k
        // => 1.0 / √1 = 1.0
        // QK^T / √d_k , casual mask
        // => [2.0]*[3.0] = [6.0]
        // fa_score = softmax()
        // => 1.0
        // fa_score * V
        // => 1.0 * 4.0 = 4.0
        assert_eq!(cpu_o.len(), 1);
        assert_eq!(cpu_l.len(), 1);
        assert!((cpu_o[0] - 4.0).abs() < 1e-4);
        assert_eq!(gpu_o.len(), 1);
        assert_eq!(gpu_l.len(), 1);
        assert!((gpu_o[0] - 4.0).abs() < 1e-4);
        assert_close(&gpu_o, &cpu_o, 1e-4, 1e-5);
        assert_close(&gpu_l, &cpu_l, 1e-4, 1e-5);
    }

    #[test]
    fn test_backward_row() {
        let seq: usize = 1;
        let d_head: usize = 1;
        let do_: Vec<f32> = vec![5.0];
        let q: Vec<f32> = vec![2.0];
        let k: Vec<f32> = vec![3.0];
        let v: Vec<f32> = vec![4.0];
        let o: Vec<f32> = vec![6.0];
        let l: Vec<f32> = vec![7.0];

        // scale = 1 / √d_k
        // scale = 1

        // D[i] = rowsum(o[i] * do_[i])
        // D = 30.0

        // S[i,j] = Q[i]・K[j]^T * scale (causal mask)
        // S[i,j] = 6.0

        // softmax
        // P[i,j] = exp(S[i,j] - L[i])
        // P[i,j] = exp(-1) = 0.36787

        // dV[j] += P[i,j] * do_[i]
        // dV[j] = 0.36787*5.0 = 1.83935

        // dS[i,j] = P[i,j] * (do_[i]・V[j] - D[i])
        // dS[i,j] = 0.36787*(5.0*4.0-30) = -3.6787
        // dQ[i] += scale * dS[i,j] * K[j]
        // dQ[i] = -3.6787*3.0 = -11.0361
        // dK[j] += scale * dS[i,j] * Q[i]
        // dK[j] = -3.6787*2.0 = -7.3574
        let (exp_dq, exp_dk, exp_dv) = (vec![-11.0361], vec![-7.3574], vec![1.83935]);
        let (cpu_dq, cpu_dk, cpu_dv) =
            attention_backward_cpu(&do_, &q, &k, &v, &o, &l, seq, d_head);
        let ctx = GpuContext::new();
        let (gpu_dq, gpu_dk, gpu_dv) =
            attention_backward(&ctx, &do_, &q, &k, &v, &o, &l, seq as u32, d_head as u32);

        assert_eq!(cpu_dq.len(), 1);
        assert_eq!(cpu_dk.len(), 1);
        assert_eq!(cpu_dv.len(), 1);
        assert!(
            (cpu_dq[0] - exp_dq[0]).abs() < 1e-3,
            "cpu dq={},exp={}",
            cpu_dq[0],
            exp_dq[0]
        );
        assert!(
            (cpu_dk[0] - exp_dk[0]).abs() < 1e-3,
            "cpu dk={},exp={}",
            cpu_dk[0],
            exp_dk[0]
        );
        assert!(
            (cpu_dv[0] - exp_dv[0]).abs() < 1e-3,
            "cpu dv={},exp={}",
            cpu_dv[0],
            exp_dv[0]
        );

        assert_eq!(gpu_dq.len(), 1);
        assert_eq!(gpu_dk.len(), 1);
        assert_eq!(gpu_dv.len(), 1);

        assert!(
            (gpu_dq[0] - exp_dq[0]).abs() < 1e-3,
            "gpu dq={},exp={}",
            gpu_dq[0],
            exp_dq[0]
        );
        assert!(
            (gpu_dk[0] - exp_dk[0]).abs() < 1e-3,
            "gpu dk={},exp={}",
            gpu_dk[0],
            exp_dk[0]
        );
        assert!(
            (gpu_dv[0] - exp_dv[0]).abs() < 1e-3,
            "gpu dv={},exp={}",
            gpu_dv[0],
            exp_dv[0]
        );

        assert_close(&gpu_dq, &cpu_dq, 1e-4, 1e-5);
        assert_close(&gpu_dk, &cpu_dk, 1e-4, 1e-5);
        assert_close(&gpu_dv, &cpu_dv, 1e-4, 1e-5);
    }

    #[test]
    fn test_casual_mask() {
        let seq: usize = 2;
        let d_head: usize = 2;
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let k: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let v: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0];
        let (cpu_o, cpu_l) = attention_cpu(&q, &k, &v, seq, d_head);
        let ctx = GpuContext::new();
        let (gpu_o, gpu_l) = attention(&ctx, &q, &k, &v, seq as u32, d_head as u32);

        // q            k           v
        // | 1.0 0.0 | | 1.0 1.0 | | 0.0 1.0 |
        // | 0.0 1.0 | | 1.0 1.0 | | 1.0 1.0 |

        // fa_score (d_k = 2)
        // QK^T / √d_k , casual mask
        // | 0.7071 0      |
        // | 0.7071 0.7071 |

        // softmax
        // seq = 0: max = 0.7071 , sum = exp(0.7071 - 0.7071) + exp(-∞ - 0.7071) = 1.0
        // seq = 1: max = 0.7071 , sum = exp(0.7071 - 0.7071) + exp(0.7071 - 0.7071) = 2.0
        // exp(fa_score-max)/sum
        // | exp(0.7071-0.7071)/1.0  exp(-∞-0.7071)/1.0     |
        // | exp(0.7071-0.7071)/2.0  exp(0.7071-0.7071)/2.0 |
        // =
        // | 1.0  0.0 |
        // | 0.5  0.5 |

        // fa_score * v
        // | 0.0  1.0 |
        // | 0.5  1.0 |
        let exp: Vec<f32> = vec![0.0, 1.0, 0.5, 1.0];
        assert!((&cpu_o[0] - &exp[0]).abs() < 1e-4);
        assert!((&cpu_o[1] - &exp[1]).abs() < 1e-4);
        assert!((&cpu_o[2] - &exp[2]).abs() < 1e-4);
        assert!((&cpu_o[3] - &exp[3]).abs() < 1e-4);

        assert!(
            (&gpu_o[0] - &exp[0]).abs() < 1e-4,
            "out[0] = {}, exp[0] = {}",
            gpu_o[0],
            exp[0]
        );
        assert!(
            (&gpu_o[1] - &exp[1]).abs() < 1e-4,
            "out[1] = {}, exp[1] = {}",
            gpu_o[1],
            exp[1]
        );
        assert!(
            (&gpu_o[2] - &exp[2]).abs() < 1e-4,
            "out[2] = {}, exp[2] = {}",
            gpu_o[2],
            exp[2]
        );
        assert!(
            (&gpu_o[3] - &exp[3]).abs() < 1e-4,
            "out[3] = {}, exp[3] = {}",
            gpu_o[3],
            exp[3]
        );
        assert_close(&gpu_o, &cpu_o, 1e-4, 1e-5);
        assert_close(&gpu_l, &cpu_l, 1e-4, 1e-5);
    }

    #[test]
    fn test_casual_backward_mask() {
        let seq: usize = 2;
        let d_head: usize = 2;
        let do_: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let k: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let v: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0];
        let o: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let l: Vec<f32> = vec![1.0, 1.0];
        let (cpu_dq, cpu_dk, cpu_dv) =
            attention_backward_cpu(&do_, &q, &k, &v, &o, &l, seq, d_head);
        let ctx = GpuContext::new();
        let (gpu_dq, gpu_dk, gpu_dv) =
            attention_backward(&ctx, &do_, &q, &k, &v, &o, &l, seq as u32, d_head as u32);

        // scale = 1 / √d_k
        // scale = 0.70710

        // q            k           v
        // | 1.0 0.0 | | 1.0 1.0 | | 0.0 1.0 |
        // | 0.0 1.0 | | 1.0 1.0 | | 1.0 1.0 |

        // do_          o            l
        // | 1.0 1.0 | | 1.0 1.0 | | 1.0 1.0 |
        // | 1.0 1.0 | | 1.0 1.0 |

        // D[i] = rowsum(o[i] * do_[i])
        // D[0] = 1.0*1.0 + 1.0*1.0 = 2.0
        // D[1] = 1.0*1.0 + 1.0*1.0 = 2.0

        // S[i,j] = Q[i]・K[j]^T * scale (causal mask)
        // | 0.7071 -∞      |
        // | 0.7071 0.7071 |

        // softmax
        // P[i,j] = exp(S[i,j] - L[i])
        // | exp(0.7071-1) exp(-∞-1)     |
        // | exp(0.7071-1) exp(0.7071-1) |
        // =
        // | 0.74609 0       |
        // | 0.74609 0.74609 |

        // dV[j] += P[i,j] * do_[i]
        // dV[0][0] = P[0][0]*do_[0][0] + P[0][0]*do_[0][1]
        // dV[0][1] = P[1][0]*do_[1][0] + P[1][0]*do_[1][1]
        // | 0.74609+0.74609 0.74609+0.74609 |
        // = | 1.49218 1.49218 |
        // dV[1][0] = P[1][1]*do_[1][0]
        // dV[1][1] = P[1][1]*do_[1][1]
        // | 0.74609 0.74609 |
        //
        // dS[i,j] = P[i,j] * (do_[i]・V[j] - D[i])
        // do_[i]・V[j]
        // | 1.0 1.0 | | 0.0 1.0 |
        // | 1.0 1.0 | | 1.0 1.0 |
        // do_[0]・V[0] = 1.0
        // P[0][0] = 0.74609
        // D[0] = 2.0
        // dS[0,0] = 0.74609 * (1.0 - 2.0) = -0.74609
        //
        // do_[1]・V[0] = 1.0
        // do_[1]・V[1] = 2.0
        // P[1][0] = 0.74609
        // P[1][1] = 0.74609
        // D[1] = 4.0
        // dS[1,0] = 0.74609 * (1.0 - 2.0) = -0.74609
        // dS[1,1] = 0.74609 * (2.0 - 2.0) = 0.0

        // dQ[i] += scale * dS[i,j] * K[j]
        // dQ[0] = | (0.70710 * (-0.74609) * 1.0) (0.70710 * (-0.74609) * 1.0) |
        //       = | -0.52756 -0.52756 |
        // dQ[1] = | (0.70710 * (-0.74609) * 1.0) (0.70710 * (-0.74609) * 1.0) |
        //       + | (0.70710 * 0.0 * 1.0)        (0.70710 * 0.0 * 1.0)        |
        //       = | -0.52756 -0.52756 |

        // dK[j] += scale * dS[i,j] * Q[i]
        // dK[0] = | (0.70710 * (-0.74609) * 1.0) (0.70710 * (-0.74609) * 0.0) |
        //       + | (0.70710 * (-0.74609) * 0.0) (0.70710 * (-0.74609) * 1.0) |
        //       = | -0.52756 -0.52756 |
        // dK[1] = | (0.70710 * 0.0 * 0.0)        (0.70710 * 0.0 * 1.0) |
        //       = | 0.0 0.0 |

        let exp_dq: Vec<f32> = vec![-0.52756, -0.52756, -0.52756, -0.52756];
        let exp_dk: Vec<f32> = vec![-0.52756, -0.52756, 0.0, 0.0];
        let exp_dv: Vec<f32> = vec![1.49218, 1.49218, 0.74609, 0.74609];
        assert_eq!(l.len(), seq, "l must be seq");
        assert_eq!(o.len(), seq * d_head, "o must be seq×d_head");
        assert_eq!(do_.len(), seq * d_head, "do_ must be seq×d_head");
        assert_close(&cpu_dq, &exp_dq, 1e-4, 1e-5);
        assert_close(&cpu_dk, &exp_dk, 1e-4, 1e-5);
        assert_close(&cpu_dv, &exp_dv, 1e-4, 1e-5);

        assert_close(&gpu_dq, &exp_dq, 1e-4, 1e-5);
        assert_close(&gpu_dk, &exp_dk, 1e-4, 1e-5);
        assert_close(&gpu_dv, &exp_dv, 1e-4, 1e-5);

        assert_close(&gpu_dq, &cpu_dq, 1e-4, 1e-5);
        assert_close(&gpu_dk, &cpu_dk, 1e-4, 1e-5);
        assert_close(&gpu_dv, &cpu_dv, 1e-4, 1e-5);
    }

    #[test]
    fn test_attention_random() {
        let seq: usize = 1024;
        let d_head: usize = 64;
        let scale = 0.1f32;
        let q: Vec<f32> = random_f32(seq * d_head, 32, scale);
        let k: Vec<f32> = random_f32(seq * d_head, 33, scale);
        let v: Vec<f32> = random_f32(seq * d_head, 34, scale);
        let (cpu_o, cpu_l) = attention_cpu(&q, &k, &v, seq, d_head);
        let ctx = GpuContext::new();
        let (gpu_o, gpu_l) = attention(&ctx, &q, &k, &v, seq as u32, d_head as u32);

        assert_eq!(cpu_o.len(), seq * d_head);
        assert_eq!(gpu_o.len(), seq * d_head);
        assert_close(&gpu_o, &cpu_o, 1e-4, 1e-5);
        assert_close(&gpu_l, &cpu_l, 1e-4, 1e-5);
    }
    #[test]
    fn test_attention_backward_random() {
        let seq: usize = 1024;
        let d_head: usize = 64;
        let scale = (1.0 / d_head as f32).sqrt();
        let do_: Vec<f32> = random_f32(seq * d_head, 32, scale);
        let q: Vec<f32> = random_f32(seq * d_head, 33, scale);
        let k: Vec<f32> = random_f32(seq * d_head, 34, scale);
        let v: Vec<f32> = random_f32(seq * d_head, 35, scale);
        let o: Vec<f32> = random_f32(seq * d_head, 36, scale);
        let l: Vec<f32> = random_f32(seq, 37, scale);
        let (cpu_dq, cpu_dk, cpu_dv) =
            attention_backward_cpu(&do_, &q, &k, &v, &o, &l, seq, d_head);
        let ctx = GpuContext::new();
        let (gpu_dq, gpu_dk, gpu_dv) =
            attention_backward(&ctx, &do_, &q, &k, &v, &o, &l, seq as u32, d_head as u32);

        assert_close(&gpu_dq, &cpu_dq, 1e-2, 1e-4);
        assert_close(&gpu_dk, &cpu_dk, 1e-2, 1e-4);
        assert_close(&gpu_dv, &cpu_dv, 1e-2, 1e-4);
    }
    #[test]
    fn test_attention_non_power_of_two() {
        let seq: usize = 7;
        let d_head: usize = 65;
        let scale = 0.1f32;
        let q: Vec<f32> = random_f32(seq * d_head, 42, scale);
        let k: Vec<f32> = random_f32(seq * d_head, 43, scale);
        let v: Vec<f32> = random_f32(seq * d_head, 44, scale);
        let (cpu_o, cpu_l) = attention_cpu(&q, &k, &v, seq, d_head);
        let ctx = GpuContext::new();
        let (gpu_o, gpu_l) = attention(&ctx, &q, &k, &v, seq as u32, d_head as u32);

        assert_eq!(cpu_o.len(), seq * d_head);
        assert_eq!(gpu_o.len(), seq * d_head);
        assert_close(&gpu_o, &cpu_o, 1e-4, 1e-5);
        assert_close(&gpu_l, &cpu_l, 1e-4, 1e-5);
    }
}
