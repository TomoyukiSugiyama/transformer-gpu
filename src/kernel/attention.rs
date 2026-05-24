use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

const BR: u32 = 64;
const MAX_D_HEAD: u32 = 128;
const TILE: u32 = 16u32; // befor_flash_attention_gpu で利用する

pub fn attention_gpu(
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
        size: byte_size,
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
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&fa_score, 0, &buf_read, 0, byte_size);
    ctx.queue.submit([encoder.finish()]);

    let slice = buf_read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let data = slice.get_mapped_range();
    bytemuck::allocation::pod_collect_to_vec(&data)
}

/// Flash Attention 導入前の実装、ベンチで利用
#[allow(dead_code)]
pub fn before_flash_attention_gpu(
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
pub fn attention_cpu(q: &[f32], k: &[f32], v: &[f32], seq: usize, d_head: usize) -> Vec<f32> {
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

    // softmax
    for i in 0..seq {
        let row = &mut scores[i * seq..(i + 1) * seq];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|&x| (x - max).exp()).sum();
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
    out
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

        let cpu = attention_cpu(&q, &k, &v, seq, d_head);
        let ctx = GpuContext::new();
        let gpu = attention_gpu(&ctx, &q, &k, &v, seq as u32, d_head as u32);

        // 1 / √d_k
        // => 1.0 / √1 = 1.0
        // QK^T / √d_k , casual mask
        // => [2.0]*[3.0] = [6.0]
        // fa_score = softmax()
        // => 1.0
        // fa_score * V
        // => 1.0 * 4.0 = 4.0
        assert!((cpu[0] - 4.0).abs() < 1e-4);
        assert!((gpu[0] - 4.0).abs() < 1e-4);
        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_casual_mask() {
        let seq: usize = 2;
        let d_head: usize = 2;
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let k: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let v: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0];
        let cpu = attention_cpu(&q, &k, &v, seq, d_head);
        let ctx = GpuContext::new();
        let gpu = attention_gpu(&ctx, &q, &k, &v, seq as u32, d_head as u32);

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
        assert!((&cpu[0] - &exp[0]).abs() < 1e-4);
        assert!((&cpu[1] - &exp[1]).abs() < 1e-4);
        assert!((&cpu[2] - &exp[2]).abs() < 1e-4);
        assert!((&cpu[3] - &exp[3]).abs() < 1e-4);
        let exp: Vec<f32> = vec![0.0, 1.0, 0.5, 1.0];
        assert!(
            (&gpu[0] - &exp[0]).abs() < 1e-4,
            "out[0] = {}, exp[0] = {}",
            gpu[0],
            exp[0]
        );
        assert!(
            (&gpu[1] - &exp[1]).abs() < 1e-4,
            "out[1] = {}, exp[1] = {}",
            gpu[1],
            exp[1]
        );
        assert!(
            (&gpu[2] - &exp[2]).abs() < 1e-4,
            "out[2] = {}, exp[2] = {}",
            gpu[2],
            exp[2]
        );
        assert!(
            (&gpu[3] - &exp[3]).abs() < 1e-4,
            "out[3] = {}, exp[3] = {}",
            gpu[3],
            exp[3]
        );
        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_attention_random() {
        let seq: usize = 1024;
        let d_head: usize = 64;
        let q: Vec<f32> = random_f32(seq * d_head, 32);
        let k: Vec<f32> = random_f32(seq * d_head, 33);
        let v: Vec<f32> = random_f32(seq * d_head, 34);
        let cpu = attention_cpu(&q, &k, &v, seq, d_head);
        let ctx = GpuContext::new();
        let gpu = attention_gpu(&ctx, &q, &k, &v, seq as u32, d_head as u32);

        assert_eq!(cpu.len(), seq * d_head);
        assert_eq!(gpu.len(), seq * d_head);
        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }

    #[test]
    fn test_attention_non_power_of_two() {
        let seq: usize = 7;
        let d_head: usize = 65;
        let q: Vec<f32> = random_f32(seq * d_head, 42);
        let k: Vec<f32> = random_f32(seq * d_head, 43);
        let v: Vec<f32> = random_f32(seq * d_head, 44);
        let cpu = attention_cpu(&q, &k, &v, seq, d_head);
        let ctx = GpuContext::new();
        let gpu = attention_gpu(&ctx, &q, &k, &v, seq as u32, d_head as u32);

        assert_eq!(cpu.len(), seq * d_head);
        assert_eq!(gpu.len(), seq * d_head);
        assert_close(&gpu, &cpu, 1e-4, 1e-5);
    }
}
