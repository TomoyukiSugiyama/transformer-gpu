use wgpu::util::DeviceExt;

use crate::gpu_context::GpuContext;

const SIZE: u32 = 256;

pub fn residual_add(ctx: &GpuContext, x1: &[f32], x2: &[f32]) -> Vec<f32> {
    assert_eq!(x1.len(), x2.len(), "x1 and x2 must have the same length");

    if x1.is_empty() {
        return Vec::new();
    }

    let size = x1.len();

    let size_u32 = u32::try_from(size).expect("residual size exceeds u32 range");

    let byte_size = size
        .checked_mul(std::mem::size_of::<f32>())
        .expect("residual buffer size overflow") as u64;

    let buf_x1 = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x1"),
            contents: bytemuck::cast_slice(x1),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let buf_x2 = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x2"),
            contents: bytemuck::cast_slice(x2),
            usage: wgpu::BufferUsages::STORAGE,
        });

    let buf_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: byte_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let dims_padded: [u32; 4] = [size_u32, 0, 0, 0];

    let buf_dims = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dims"),
            contents: bytemuck::cast_slice(&dims_padded),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("../shader/residual_add.wgsl"));

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
        label: Some("residual_add_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_x1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_x2.as_entire_binding(),
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("residual_add_encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("residual_add_pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        pass.dispatch_workgroups(size_u32.div_ceil(SIZE), 1, 1);
    }

    let buf_read = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("residual_add_read"),
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

    let result = bytemuck::allocation::pod_collect_to_vec(&data);

    drop(data);
    buf_read.unmap();

    result
}

// CPU リファレンス
#[cfg(test)]
pub fn residual_add_cpu(x1: &[f32], x2: &[f32]) -> Vec<f32> {
    assert_eq!(x1.len(), x2.len(), "x1 and x2 must have the same length");
    x1.iter().zip(x2.iter()).map(|(a, b)| *a + *b).collect()
}

#[cfg(test)]
mod test {
    use crate::{
        gpu_context::GpuContext,
        kernel::residual_add::{residual_add, residual_add_cpu},
    };

    #[test]
    fn test_residual_add() {
        let x1: Vec<f32> = vec![1.0, 1.0];
        let x2: Vec<f32> = vec![2.0, 2.0];

        let cpu = residual_add_cpu(&x1, &x2);
        let ctx = GpuContext::new();
        let gpu = residual_add(&ctx, &x1, &x2);
        let exp: Vec<f32> = vec![3.0, 3.0];

        assert_eq!(cpu, exp);
        assert_eq!(gpu, exp);
    }
}
