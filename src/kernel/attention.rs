use wgpu::util::DeviceExt;

const TILE: u32 = 16;

fn attention_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: u32,
    d_head: u32,
) -> Vec<f32> {
    let buf_q = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Q"),
        contents: bytemuck::cast_slice(&q),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_k = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("K"),
        contents: bytemuck::cast_slice(&k),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("V"),
        contents: bytemuck::cast_slice(&v),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_o = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("O"),
        size: (seq * seq * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [seq, d_head, 0, 0];
    let buf_dims = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dims"),
        contents: bytemuck::cast_slice(&dims_padded),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let module = device.create_shader_module(wgpu::include_wgsl!("../shader/attention.wgsl"));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("attention"),
        layout: None,
        module: &module,
        entry_point: Some("attention"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_q.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_k.as_entire_binding(),
            },
            // wgpu::BindGroupEntry {
            //     binding: 2,
            //     resource: buf_v.as_entire_binding(),
            // },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_o.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_dims.as_entire_binding(),
            },
        ],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        pass.dispatch_workgroups(seq.div_ceil(TILE), seq.div_ceil(TILE), 1);
    }

    let buf_read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (seq * seq * 4) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_o, 0, &buf_read, 0, (seq * d_head * 4) as u64);
    queue.submit([encoder.finish()]);

    let slice = buf_read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let data = slice.get_mapped_range();
    bytemuck::allocation::pod_collect_to_vec(&data)
}
// CPU リファレンス
fn attention_cpu(q: &[f32], k: &[f32], v: &[f32], seq: usize, d_head: usize) -> Vec<f32> {
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

    // score * V
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
    use crate::test_utils::random_f32;

    fn gpu_context() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .unwrap()
    }

    #[test]
    fn test_softmax_row_sum() {
        let seq: usize = 1;
        let d_head: usize = 1;
        let q: Vec<f32> = vec![2.0];
        let k: Vec<f32> = vec![3.0];
        let v: Vec<f32> = vec![4.0];

        let out = attention_cpu(&q, &k, &v, seq, d_head);

        // 1 / √d_k
        // => 1.0 / √1 = 1.0
        // QK^T / √d_k , casual mask
        // => [2.0]*[3.0] = [6.0]
        // score = softmax()
        // => 1.0
        // score * V
        // => 1.0 * 4.0 = 4.0
        assert!(out[0] - 4.0 < 1e-4);
    }

    #[test]
    fn test_casual_mask() {
        let seq: usize = 2;
        let d_head: usize = 2;
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let k: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let v: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0];
        let out = attention_cpu(&q, &k, &v, seq, d_head);

        // q            k           v
        // | 1.0 0.0 | | 1.0 1.0 | | 0.0 1.0 |
        // | 0.0 1.0 | | 1.0 1.0 | | 1.0 1.0 |

        // score (d_k = 2)
        // QK^T / √d_k , casual mask
        // | 0.7071 0      |
        // | 0.7071 0.7071 |

        // softmax
        // seq = 0: max = 0.7071 , sum = exp(0.7071 - 0.7071) + exp(-∞ - 0.7071) = 1.0
        // seq = 1: max = 0.7071 , sum = exp(0.7071 - 0.7071) + exp(0.7071 - 0.7071) = 2.0
        // exp(score-max)/sum
        // | exp(0.7071-0.7071)/1.0  exp(-∞-0.7071)/1.0     |
        // | exp(0.7071-0.7071)/2.0  exp(0.7071-0.7071)/2.0 |
        // =
        // | 1.0  0.0 |
        // | 0.5  0.5 |

        // score * v
        // | 0.0  1.0 |
        // | 0.5  1.0 |
        let exp: Vec<f32> = vec![0.0, 1.0, 0.5, 1.0];
        assert!((&out[0] - &exp[0]).abs() < 1e-4);
        assert!((&out[1] - &exp[1]).abs() < 1e-4);
        assert!((&out[2] - &exp[2]).abs() < 1e-4);
        assert!((&out[3] - &exp[3]).abs() < 1e-4);
    }

    #[test]
    fn test_attention_random_cpu_reference() {
        let seq: usize = 1024;
        let d_head: usize = 64;
        let q: Vec<f32> = random_f32(seq * d_head, 32);
        let k: Vec<f32> = random_f32(seq * d_head, 33);
        let v: Vec<f32> = random_f32(seq * d_head, 34);
        let out = attention_cpu(&q, &k, &v, seq, d_head);

        assert_eq!(out.len(), seq * d_head)
    }

    #[test]
    fn test_attention_non_power_of_two() {
        let seq: usize = 7;
        let d_head: usize = 65;
        let q: Vec<f32> = random_f32(seq * d_head, 42);
        let k: Vec<f32> = random_f32(seq * d_head, 43);
        let v: Vec<f32> = random_f32(seq * d_head, 44);
        let out = attention_cpu(&q, &k, &v, seq, d_head);

        assert_eq!(out.len(), seq * d_head)
    }

    #[test]
    fn test_softmax_row_sum_gpu() {
        let seq: usize = 1;
        let d_head: usize = 1;
        let q: Vec<f32> = vec![2.0];
        let k: Vec<f32> = vec![3.0];
        let v: Vec<f32> = vec![4.0];
        let (device, queue) = gpu_context();
        let out = attention_gpu(&device, &queue, &q, &k, &v, seq as u32, d_head as u32);

        // 1 / √d_k
        // => 1.0 / √1 = 1.0
        // QK^T / √d_k , casual mask
        // => [2.0]*[3.0] = [6.0]
        // score = softmax()
        // => 1.0
        // score * V
        // => 1.0 * 4.0 = 4.0
        assert!((out[0] - 6.0).abs() < 1e-4);
    }


    #[test]
    fn test_casual_mask_gpu() {
        let seq: usize = 2;
        let d_head: usize = 2;
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let k: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let v: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0];
        let (device, queue) = gpu_context();
        let out = attention_gpu(&device, &queue, &q, &k, &v, seq as u32, d_head as u32);

        println!("{:?}",out);
        // q            k           v
        // | 1.0 0.0 | | 1.0 1.0 | | 0.0 1.0 |
        // | 0.0 1.0 | | 1.0 1.0 | | 1.0 1.0 |

        // score (d_k = 2)
        // QK^T / √d_k , casual mask
        // | 0.7071 0      |
        // | 0.7071 0.7071 |

        // softmax
        // seq = 0: max = 0.7071 , sum = exp(0.7071 - 0.7071) + exp(-∞ - 0.7071) = 1.0
        // seq = 1: max = 0.7071 , sum = exp(0.7071 - 0.7071) + exp(0.7071 - 0.7071) = 2.0
        // exp(score-max)/sum
        // | exp(0.7071-0.7071)/1.0  exp(-∞-0.7071)/1.0     |
        // | exp(0.7071-0.7071)/2.0  exp(0.7071-0.7071)/2.0 |
        // =
        // | 1.0  0.0 |
        // | 0.5  0.5 |

        // score * v
        // | 0.0  1.0 |
        // | 0.5  1.0 |
        let exp: Vec<f32> = vec![0.7071, 0.7071, 0.7071, 0.7071];
        assert!((&out[0] - &exp[0]).abs() < 1e-4);
        assert!((&out[1] - &exp[1]).abs() < 1e-4);
        assert!((&out[2] - &exp[2]).abs() < 1e-4);
        assert!((&out[3] - &exp[3]).abs() < 1e-4);
    }
}
