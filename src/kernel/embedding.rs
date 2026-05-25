use crate::gpu_context::GpuContext;
use wgpu::util::DeviceExt;

const SIZE: u32 = 256;

pub fn embedding_gpu(
    ctx: &GpuContext,
    token_ids: &[u32],
    weight: &[f32],
    d_model: usize,
) -> Vec<f32> {
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

    let dims_padded: [u32; 4] = [d_model as u32, 0, 0, 0];
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
            label: Some("residual_add"),
            layout: None,
            module: &module,
            entry_point: Some("residual_add"),
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

        pass.dispatch_workgroups(size.div_ceil(SIZE), 1, 1);
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

// CPU リファレンス
#[cfg(test)]
pub fn embedding_cpu(token_ids: &[u32], weight: &[f32], d_model: usize) -> Vec<f32> {
    token_ids
        .iter()
        .flat_map(|&id| {
            let start = id as usize * d_model;
            weight[start..start + d_model].iter().copied()
        })
        .collect()
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::embedding::{embedding_cpu, embedding_gpu},
    };

    #[test]
    fn test_embedding() {
        let token_ids = vec![1, 2, 3, 4];
        let weight = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let d_model = 2;

        let cpu = embedding_cpu(&token_ids, &weight, d_model);
        let ctx = GpuContext::new();
        let gpu = embedding_gpu(&ctx, &token_ids, &weight, d_model);

        let exp = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        assert_eq!(cpu, exp);
        assert_eq!(gpu, exp);
    }
}
