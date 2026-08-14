use crate::gpu_context::GpuContext;
use wgpu::util::DeviceExt;

const SIZE: u32 = 256;

pub fn embedding(
    ctx: &GpuContext,
    token_ids: &[u32],
    weight: &[f32],
    vocab_size: usize,
    d_model: usize,
) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        vocab_size * d_model,
        "weight length must be vocab_size * d_model"
    );

    for (pos, &id) in token_ids.iter().enumerate() {
        assert!(
            (id as usize) < vocab_size,
            "token_ids[{pos}]={} out of range",
            id
        );
    }
    let seq_len = token_ids.len() as u32;
    let size = (token_ids.len() * d_model) as u32;
    let byte_size = (size * 4) as u64;
    let buf_token_ids = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("token_ids"),
            contents: bytemuck::cast_slice(&token_ids),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_weight = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("weight"),
            contents: bytemuck::cast_slice(&weight),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [d_model as u32, seq_len as u32, 0, 0];
    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/embedding.wgsl"));
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embedding"),
            layout: None,
            module: &module,
            entry_point: Some("embedding"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_token_ids.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_weight.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_dims.as_entire_binding(),
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

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        pass.dispatch_workgroups(seq_len.div_ceil(SIZE), 1, 1);
    }

    let buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_read, 0, byte_size);
    ctx.queue.submit([encoder.finish()]);

    let slice = buf_read.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let data = slice.get_mapped_range();
    bytemuck::allocation::pod_collect_to_vec(&data)
}

pub fn embedding_backward(
    ctx: &GpuContext,
    dy: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
    d_model: usize,
) -> Vec<f32> {
    assert!(vocab_size > 0);
    assert!(d_model > 0);

    let seq_len = token_ids.len();

    assert_eq!(
        dy.len(),
        seq_len * d_model,
        "dy length must be seq_len * d_model"
    );

    for (pos, &id) in token_ids.iter().enumerate() {
        assert!(
            (id as usize) < vocab_size,
            "token_ids[{pos}]={} >= vocab_size={}",
            id,
            vocab_size
        );
    }

    if seq_len == 0 {
        return vec![0.0; vocab_size * d_model];
    }

    let size = vocab_size
        .checked_mul(d_model)
        .expect("vocab_size * d_model overflow");

    let byte_size = (size * std::mem::size_of::<f32>()) as u64;

    let buf_dy = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dy"),
            contents: bytemuck::cast_slice(dy),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let buf_token_ids = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("token_ids"),
            contents: bytemuck::cast_slice(token_ids),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let zeros = vec![0i32; size];

    let buf_dweight = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dweight"),
            contents: bytemuck::cast_slice(&zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let dims_padded: [u32; 4] = [d_model as u32, seq_len as u32, 0, 0];

    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/embedding_backward.wgsl"));

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("embedding_backward"),
            layout: None,
            module: &module,
            entry_point: Some("embedding_backward"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("embedding_backward_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_dy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_token_ids.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_dweight.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buf_dims.as_entire_binding(),
            },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("embedding_backward_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("embedding_backward_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        let total = (seq_len * d_model) as u32;

        pass.dispatch_workgroups(total.div_ceil(SIZE), 1, 1);
    }

    let buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dweight_read"),
        size: byte_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&buf_dweight, 0, &buf_read, 0, byte_size);

    ctx.queue.submit([encoder.finish()]);

    let slice = buf_read.slice(..);

    slice.map_async(wgpu::MapMode::Read, |_| {});

    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .unwrap();

    let data = slice.get_mapped_range();

    let bits: Vec<i32> = bytemuck::allocation::pod_collect_to_vec(&data);

    drop(data);
    buf_read.unmap();

    bits.into_iter()
        .map(|bits| f32::from_bits(bits as u32))
        .collect()
}

// CPU リファレンス
#[cfg(test)]
pub fn embedding_cpu(
    token_ids: &[u32],
    weight: &[f32],
    vocab_size: usize,
    d_model: usize,
) -> Vec<f32> {
    assert_eq!(
        weight.len(),
        vocab_size * d_model,
        "weight length must be vocab_size * d_model"
    );

    for (pos, &id) in token_ids.iter().enumerate() {
        assert!(
            (id as usize) < vocab_size,
            "token_ids[{pos}]={} out of range",
            id
        );
    }
    let scale = (d_model as f32).sqrt();

    token_ids
        .iter()
        .flat_map(|&id| {
            let start = id as usize * d_model;

            weight[start..start + d_model]
                .iter()
                .map(move |&x| x * scale)
        })
        .collect()
}

#[cfg(test)]
pub fn embedding_backward_cpu(
    dy: &[f32],
    token_ids: &[u32],
    vocab_size: usize,
    d_model: usize,
) -> Vec<f32> {
    assert!(vocab_size > 0);
    assert!(d_model > 0);

    let seq_len = token_ids.len();

    assert_eq!(
        dy.len(),
        seq_len * d_model,
        "dy length must be seq_len * d_model"
    );

    for (pos, &id) in token_ids.iter().enumerate() {
        assert!(
            (id as usize) < vocab_size,
            "token_ids[{pos}]={} >= vocab_size={}",
            id,
            vocab_size
        );
    }

    if seq_len == 0 {
        return vec![0.0; vocab_size * d_model];
    }


    let scale = (d_model as f32).sqrt();
    let mut dweight = vec![0.0f32; vocab_size * d_model];

    for (pos, &id) in token_ids.iter().enumerate() {
        let id = id as usize;

        assert!(
            id < vocab_size,
            "token id {} is out of range {}",
            id,
            vocab_size
        );

        let src = pos * d_model;
        let dst = id * d_model;

        for j in 0..d_model {
            dweight[dst + j] += dy[src + j] * scale;
        }
    }

    dweight
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::embedding::{embedding, embedding_backward, embedding_backward_cpu, embedding_cpu},
    };

    #[test]
    fn test_embedding() {
        let token_ids = vec![1, 2, 3, 4];
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let vocab_size = 5;
        let d_model = 2;

        let cpu = embedding_cpu(&token_ids, &weight, vocab_size, d_model);
        let ctx = GpuContext::new();
        let gpu = embedding(&ctx, &token_ids, &weight, vocab_size, d_model);

        let exp = vec![
            4.2426405, 5.656854, 7.071068, 8.485281, 9.899495, 11.313708, 12.7279215, 14.142136,
        ];

        assert_eq!(cpu, exp);
        assert_eq!(gpu, exp);
    }

    #[test]
    fn test_embedding_backward() {
        let token_ids = vec![1, 2, 3, 4];
        let dy = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let vocab_size = 5;
        let d_model = 2;

        let cpu = embedding_backward_cpu(&dy, &token_ids, vocab_size, d_model);
        let ctx = GpuContext::new();
        let gpu = embedding_backward(&ctx, &dy, &token_ids, vocab_size, d_model);

        // dweight[t*d + j] = dy[pos*d + j]
        // token_ids[pos] = t
        let exp = vec![
            0.0, 0.0, 4.2426405, 5.656854, 7.071068, 8.485281, 9.899495, 11.313708, 12.7279215,
            14.142136,
        ];

        assert_eq!(cpu, exp);
        assert_eq!(gpu, exp);
    }
}
